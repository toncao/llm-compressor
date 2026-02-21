from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaMoE,
    )


@MoECalibrationModule.register("GlmMoeDsaMoE")
class CalibrateGlmMoeDsaMoE(MoECalibrationModule):
    """
    Calibration version of GlmMoeDsaMoE that sends all tokens to all experts.
    Experts are unfused from 3D parameter tensors into individual MLP modules
    with Linear layers to enable per-expert activation capture.

    GlmMoeDsaNaiveMoe stores:
        gate_up_proj: (num_experts, 2 * moe_intermediate_size, hidden_size)
        down_proj:    (num_experts, hidden_size, moe_intermediate_size)

    Each is split into a GlmMoeDsaMLP with:
        gate_proj.weight: (moe_intermediate_size, hidden_size)
        up_proj.weight:   (moe_intermediate_size, hidden_size)
        down_proj.weight: (hidden_size, moe_intermediate_size)
    """

    is_permanent = True

    def __init__(
        self,
        original: "GlmMoeDsaMoE",
        config: "GlmMoeDsaConfig",
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config

        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

        # gating
        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate

        # Unfuse fused 3D expert weights into individual MLP modules
        self.experts = SequentialGlmMoeDsaExperts(config, original.experts)

        self.shared_experts = original.shared_experts

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape

        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        final_hidden_states = torch.zeros_like(hidden_states)

        # One-hot encode selected experts to create an expert mask
        # expert_mask shape: (num_experts, top_k, num_tokens)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.n_routed_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                # Run all tokens through the expert to gather activation stats,
                # but only use routed tokens for the final output
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                expert_out = expert_layer(hidden_states[token_idx])

            if len(token_idx) > 0:
                current_hidden_states = (
                    expert_out * topk_weights[token_idx, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )

        hidden_states = final_hidden_states.view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original


class SequentialGlmMoeDsaExperts(torch.nn.ModuleList):
    """
    Unfuses the fused 3D expert weight tensors from GlmMoeDsaNaiveMoe into
    individual GlmMoeDsaMLP modules with separate Linear layers, enabling
    per-expert activation capture for AWQ calibration.

    GlmMoeDsaNaiveMoe stores:
        gate_up_proj: (num_experts, 2 * moe_intermediate_size, hidden_size)
        down_proj:    (num_experts, hidden_size, moe_intermediate_size)

    Each is split into a GlmMoeDsaMLP with:
        gate_proj.weight: (moe_intermediate_size, hidden_size)
        up_proj.weight:   (moe_intermediate_size, hidden_size)
        down_proj.weight: (hidden_size, moe_intermediate_size)
    """

    def __init__(self, config, original):
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaMLP,
        )

        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [
                    GlmMoeDsaMLP(config, intermediate_size=intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]  # (2 * intermediate_size, hidden_size)
            down = original.down_proj[i]  # (hidden_size, intermediate_size)

            # gate_up stores [gate_proj; up_proj] stacked along dim 0
            self[i].gate_proj.weight.data = (
                gate_up[:intermediate_size, :].clone().contiguous()
            )
            self[i].up_proj.weight.data = (
                gate_up[intermediate_size:, :].clone().contiguous()
            )
            self[i].down_proj.weight.data = down.clone().contiguous()