from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers import Lfm2MoeConfig
    from transformers.models.lfm2_moe.modeling_lfm2_moe import (
        Lfm2MoeSparseMoeBlock,
    )


@MoECalibrationModule.register("Lfm2MoeSparseMoeBlock")
class CalibrateLfm2MoeSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of Lfm2MoeSparseMoeBlock that sends all tokens to all
    experts. Experts are unfused from 3D parameter tensors into individual MLP
    modules with Linear layers to enable per-expert activation capture.
    """

    is_permanent = True

    def __init__(
        self,
        original: "Lfm2MoeSparseMoeBlock",
        config: "Lfm2MoeConfig",
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        text_config = config.get_text_config()

        self.num_experts = text_config.num_experts
        self.top_k = text_config.num_experts_per_tok
        self.routed_scaling_factor = text_config.routed_scaling_factor
        self.norm_topk_prob = text_config.norm_topk_prob
        self.use_expert_bias = text_config.use_expert_bias

        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate

        if self.use_expert_bias:
            self.register_buffer("expert_bias", original.expert_bias.clone())

        # Unfuse fused 3D expert weights into individual MLP modules
        self.experts = SequentialLfm2MoeExperts(text_config, original.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # Lfm2Moe uses sigmoid routing (not softmax)
        routing_weights = router_logits.sigmoid()

        if self.use_expert_bias:
            scores_for_routing = routing_weights + self.expert_bias
            _, selected_experts = torch.topk(
                scores_for_routing, k=self.top_k, dim=-1
            )
            routing_weights = torch.gather(
                routing_weights, dim=1, index=selected_experts
            ).type_as(router_logits)
        else:
            routing_weights, selected_experts = torch.topk(
                routing_weights, k=self.top_k, dim=-1
            )

        if self.norm_topk_prob:
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-6
            )
        routing_weights = routing_weights * self.routed_scaling_factor

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One-hot encode selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                # Send ALL tokens through the expert (for activation capture),
                # but only use outputs for the routed tokens
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

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original


class SequentialLfm2MoeExperts(torch.nn.ModuleList):
    """
    Unfuses the fused 3D expert weight tensors from Lfm2MoeExperts into
    individual Lfm2MoeMLP modules with separate Linear layers, enabling
    per-expert activation capture for AWQ calibration.

    Lfm2MoeExperts stores:
        gate_up_proj: (num_experts, 2 * moe_intermediate_size, hidden_size)
        down_proj:    (num_experts, hidden_size, moe_intermediate_size)

    Each is split into an Lfm2MoeMLP with:
        w1.weight (gate):  (moe_intermediate_size, hidden_size)
        w3.weight (up):    (moe_intermediate_size, hidden_size)
        w2.weight (down):  (hidden_size, moe_intermediate_size)
    """

    def __init__(self, config, original):
        from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeMLP

        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [
                    Lfm2MoeMLP(config, intermediate_size=intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        for i in range(self.num_experts):
            # gate_up_proj[i]: (2 * intermediate_size, hidden_size)
            # After nn.functional.linear(x, gate_up_proj[i]).chunk(2, dim=-1):
            #   first half  → gate (w1)
            #   second half → up   (w3)
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]  # (hidden_size, intermediate_size)

            self[i].w1.weight.data = (
                gate_up[:intermediate_size, :].clone().contiguous()
            )
            self[i].w3.weight.data = (
                gate_up[intermediate_size:, :].clone().contiguous()
            )
            self[i].w2.weight.data = down.clone().contiguous()