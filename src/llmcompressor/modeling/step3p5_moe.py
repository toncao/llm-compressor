from __future__ import annotations

# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Calibration module for Step3p5MoEMLP that unfuses MoELinear experts into
individual MLP modules with standard nn.Linear layers, enabling per-expert
activation capture for AWQ quantization.

The original Step3p5MoEMLP stores expert weights in fused MoELinear tensors
of shape (num_experts, out_features, in_features). This module decomposes
them into individual Step3p5ExpertMLP modules, each containing separate
gate_proj, up_proj, and down_proj Linear layers.
"""
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

from compressed_tensors.utils import patch_attr

if TYPE_CHECKING:
    from .configuration_step3p5 import Step3p5Config


def _sigmoid_routing_function(
    gating_output: torch.Tensor, topk: int, renormalize: bool
):
    gating_output = gating_output.float()
    gate_prob = torch.sigmoid(gating_output)
    gate_prob = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
    topk_prob, indices = torch.topk(gate_prob, k=topk, dim=1)
    expert_topk_weight = topk_prob
    if renormalize:
        expert_topk_weight = expert_topk_weight / torch.sum(
            expert_topk_weight, dim=-1, keepdim=True
        )
    return expert_topk_weight, indices


@MoECalibrationModule.register("Step3p5MoEMLP")
class CalibrateStep3p5MoEMLP(MoECalibrationModule):
    """
    Calibration version of Step3p5MoEMLP that unfuses MoELinear experts into
    individual MLP modules with standard nn.Linear layers.

    Supports two calibration modes:
      - calibrate_all_experts=True:  Every expert processes ALL tokens, and
        only the routed subset is used for the weighted sum. This ensures
        every expert's activations are captured regardless of routing.
      - calibrate_all_experts=False: Each expert only processes the tokens
        that are routed to it (standard sparse behavior).

    The routing logic (sigmoid / softmax / router-bias) and the
    routed_scaling_factor are preserved exactly as in the original module.

    MoELinear performs computation in float32 (``F.linear(x.float(), w.float())``).
    The unfused Step3p5ExpertMLP replicates this by casting inputs and weights
    to float32 for each linear operation and returning the result in the
    original hidden-state dtype.
    """

    is_permanent = True

    def __init__(
        self,
        original: "Step3p5MoEMLP",
        config: "Step3p5Config",
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.calibrate_all_experts = calibrate_all_experts
        self.routed_scaling_factor = getattr(
            config, "moe_router_scaling_factor", 1.0
        )
        self.need_fp32_gate = config.need_fp32_gate

        # ---- routing function ------------------------------------------------
        self.use_moe_router_bias = config.use_moe_router_bias
        if self.use_moe_router_bias:
            # Keep the learned bias on the same device / dtype as the original
            self.router_bias = original.router_bias
            self.custom_routing_function = self._router_bias_func
        elif getattr(config, "moe_router_activation", None) == "sigmoid":
            self.custom_routing_function = _sigmoid_routing_function
        else:
            self.custom_routing_function = None

        # ---- gate (router projection) ----------------------------------------
        self.gate = original.gate
        self.gate.weight.data = self.gate.weight.data.float()

        # ---- unfused experts --------------------------------------------------
        self.experts = SequentialStep3p5MoEExperts(config, original)

    # ------------------------------------------------------------------
    # Router-bias routing (mirrors original Step3p5MoEMLP.router_bias_func)
    # ------------------------------------------------------------------
    def _router_bias_func(
        self, gating_output: torch.Tensor, topk: int, renormalize: bool
    ):
        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_with_bias, k=topk, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        expert_topk_weight = topk_prob
        if renormalize:
            expert_topk_weight = expert_topk_weight / (
                torch.sum(expert_topk_weight, dim=-1, keepdim=True) + 1e-20
            )
        return expert_topk_weight, indices

    # ------------------------------------------------------------------
    # Forward – mirrors Step3p5MoEMLP.forward with unfused experts
    # ------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # ---- router logits ---------------------------------------------------
        if self.need_fp32_gate:
            router_logits = self.gate(hidden_states.float())
        else:
            router_logits = self.gate(hidden_states)

        # ---- routing weights & expert selection ------------------------------
        if self.custom_routing_function is not None:
            routing_weights, selected_experts = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True
            )
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )

        routing_weights = routing_weights * self.routed_scaling_factor

        # ---- expert computation ----------------------------------------------
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One-hot encode selected experts → (num_experts, top_k, num_tokens)
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                # Run ALL tokens through the expert so its activations are
                # captured by any hooks, then index only the routed tokens.
                expert_out = expert_layer(hidden_states)[top_x]
            else:
                expert_out = expert_layer(hidden_states[top_x])

            if len(top_x) > 0:
                current_hidden_states = (
                    expert_out * routing_weights[top_x, idx, None]
                )
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    current_hidden_states.to(hidden_states.dtype),
                )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original


class Step3p5ExpertMLP(nn.Module):
    """
    Single-expert MLP extracted from the fused MoELinear representation.

    Replicates the float32 computation of the original MoELinear:
        ``F.linear(x.float(), weight[expert_id].float())``
    so that calibration outputs match the original model numerically.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float | None = None,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x.float())
        gate = self.act_fn(self.gate_proj(x.float()))

        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj((gate * up).float())


class SequentialStep3p5MoEExperts(nn.ModuleList):
    """
    Unfuses the fused MoELinear expert weight tensors from Step3p5MoEMLP
    into individual Step3p5ExpertMLP modules with separate Linear layers,
    enabling per-expert activation capture for AWQ calibration.

    MoELinear stores:
        gate_proj.weight: (num_experts, moe_intermediate_size, hidden_size)
        up_proj.weight:   (num_experts, moe_intermediate_size, hidden_size)
        down_proj.weight: (num_experts, hidden_size, moe_intermediate_size)

    Each expert is split into a Step3p5ExpertMLP with:
        gate_proj.weight: (moe_intermediate_size, hidden_size)
        up_proj.weight:   (moe_intermediate_size, hidden_size)
        down_proj.weight: (hidden_size, moe_intermediate_size)
    """

    def __init__(self, config: "Step3p5Config", original_moe: "Step3p5MoEMLP"):
        num_experts = config.moe_num_experts
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        swiglu_limit = original_moe.limit

        with skip_weights_initialize():
            super().__init__(
                [
                    Step3p5ExpertMLP(
                        hidden_size, moe_intermediate_size, swiglu_limit
                    )
                    for _ in range(num_experts)
                ]
            )

        # Copy weights from the fused MoELinear parameters into individual
        # Linear layers.  MoELinear.weight has shape
        # (num_experts, out_features, in_features) which matches nn.Linear's
        # (out_features, in_features) per-expert slice.
        for i in range(num_experts):
            self[i].gate_proj.weight.data = (
                original_moe.gate_proj.weight[i].clone().contiguous().float()
            )
            self[i].up_proj.weight.data = (
                original_moe.up_proj.weight[i].clone().contiguous().float()
            )
            self[i].down_proj.weight.data = (
                original_moe.down_proj.weight[i].clone().contiguous().float()
            )
