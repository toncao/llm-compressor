from __future__ import annotations

# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# All rights reserved.
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
from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )


@MoECalibrationModule.register("Qwen3_5MoeSparseMoeBlock")
class CalibrateQwen3_5MoeSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of Qwen3_5MoeSparseMoeBlock that sends all tokens to all
    experts. Experts are unfused from 3D parameter tensors into individual MLP
    modules with Linear layers to enable per-expert activation capture.
    """

    is_permanent = True

    def __init__(
        self,
        original: Qwen3_5MoeSparseMoeBlock,
        config: Qwen3_5MoeConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        text_config = config.get_text_config()

        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok

        # gating
        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate

        # Unfuse fused 3D expert weights into individual MLP modules
        self.experts = SequentialQwen3_5MoeExperts(text_config, original.experts)

        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = torch.nn.functional.linear(hidden_states, self.gate.weight)

        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=-1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be
        # sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the
        # computation on each expert
        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[top_x]
            else:
                expert_out = expert_layer(hidden_states[top_x])

            # Index the correct hidden states and compute the expert hidden
            # state for the current expert. We need to make sure to multiply
            # the output hidden states by `routing_weights` on the
            # corresponding tokens (top-1 and top-2)
            if len(top_x) > 0:
                current_hidden_states = expert_out * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    current_hidden_states.to(hidden_states.dtype),
                )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            torch.nn.functional.sigmoid(self.shared_expert_gate(hidden_states))
            * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original


class SequentialQwen3_5MoeExperts(torch.nn.ModuleList):
    """
    Unfuses the fused 3D expert weight tensors from Qwen3_5MoeExperts into
    individual Qwen3_5MoeMLP modules with separate Linear layers, enabling
    per-expert activation capture for AWQ calibration.

    Qwen3_5MoeExperts stores:
        gate_up_proj: (num_experts, 2 * moe_intermediate_size, hidden_size)
        down_proj:    (num_experts, hidden_size, moe_intermediate_size)

    Each is split into a Qwen3_5MoeMLP with:
        gate_proj.weight: (moe_intermediate_size, hidden_size)
        up_proj.weight:   (moe_intermediate_size, hidden_size)
        down_proj.weight: (hidden_size, moe_intermediate_size)
    """

    def __init__(self, config, original):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeMLP,
        )

        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [
                    Qwen3_5MoeMLP(config, intermediate_size=intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]  # (2 * intermediate_size, hidden_size)
            down = original.down_proj[i]  # (hidden_size, intermediate_size)

            # gate_up stores [gate_proj; up_proj] stacked along dim 0
            self[i].gate_proj.weight.data = gate_up[:intermediate_size, :].clone().contiguous()
            self[i].up_proj.weight.data = gate_up[intermediate_size:, :].clone().contiguous()
            self[i].down_proj.weight.data = down.clone().contiguous()
