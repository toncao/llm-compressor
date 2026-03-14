# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule

if TYPE_CHECKING:
    from .configuration_nemotron_h import NemotronHConfig
    from .modeling_nemotron_h import NemotronHMoE


@MoECalibrationModule.register("NemotronHMoE")
class CalibrationNemotronHMoE(MoECalibrationModule):
    """
    Calibration version of NemotronHMoE that sends all tokens to all experts
    so that their activations are captured for AWQ calibration.

    The NemotronH MoE block uses:
    - A sigmoid-based TopkRouter (not softmax) with grouped expert selection
    - Optional latent dimension projections (fc1_latent_proj / fc2_latent_proj)
    - Shared experts that process all tokens unconditionally
    """

    is_permanent = False

    def __init__(
        self,
        original: NemotronHMoE,
        config: NemotronHConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate
        self.experts = original.experts
        self.shared_experts = original.shared_experts
        self.fc1_latent_proj = original.fc1_latent_proj
        self.fc2_latent_proj = original.fc2_latent_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        hidden_states = self.fc1_latent_proj(hidden_states)
        hidden_states = self._moe_calibration(hidden_states, topk_indices, topk_weights)
        hidden_states = self.fc2_latent_proj(hidden_states)

        hidden_states = hidden_states.view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    def _moe_calibration(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if self.calibrate_all_experts:
                # Send ALL tokens through the expert to capture full activations
                expert_output = expert(hidden_states)
                if token_indices.numel() > 0:
                    expert_weights = topk_weights[token_indices, weight_indices]
                    weighted_output = (
                        expert_output[token_indices] * expert_weights.unsqueeze(-1)
                    )
                    final_hidden_states.index_add_(
                        0, token_indices, weighted_output
                    )
            else:
                if token_indices.numel() > 0:
                    expert_weights = topk_weights[token_indices, weight_indices]
                    expert_input = hidden_states[token_indices]
                    expert_output = expert(expert_input)
                    weighted_output = expert_output * expert_weights.unsqueeze(-1)
                    final_hidden_states.index_add_(
                        0, token_indices, weighted_output
                    )
                else:
                    # No-op compute that still marks params as used
                    expert_dtype = expert.down_proj.weight.dtype
                    dummy_out = expert(
                        torch.zeros_like(hidden_states[0])
                        .unsqueeze(0)
                        .to(expert_dtype)
                    )
                    final_hidden_states = final_hidden_states + dummy_out

        return final_hidden_states.type(hidden_states.dtype)

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original