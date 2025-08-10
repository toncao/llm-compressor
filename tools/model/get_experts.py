import torch
from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F

model_dir = "/mnt/LinuxDrive_1/huggingface/hub/gpt-oss-20b-bf16" 
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cpu", torch_dtype="auto")

L = 5  # layer to inspect
layer = model.model.layers[L]
ex = layer.mlp.experts

# Shapes:
# ex.gate_up_proj: [num_experts, in_hidden, 2*ff]
# ex.gate_up_proj_bias: [num_experts, 2*ff]
# ex.down_proj: [num_experts, ff, hidden]
# ex.down_proj_bias: [num_experts, hidden]

n_exp = ex.down_proj.shape[0]
hidden = ex.gate_up_proj.shape[1]
two_ff = ex.gate_up_proj.shape[2]
assert two_ff % 2 == 0
ff = two_ff // 2

def get_expert_params(i):
    # Weights are stored as [in, out], which matches using x @ W for forward.
    W_gate = ex.gate_up_proj[i, :, :ff]      # [hidden, ff]
    W_up   = ex.gate_up_proj[i, :, ff:]      # [hidden, ff]
    b_gate = ex.gate_up_proj_bias[i, :ff] if hasattr(ex, "gate_up_proj_bias") else None  # [ff]
    b_up   = ex.gate_up_proj_bias[i, ff:] if hasattr(ex, "gate_up_proj_bias") else None  # [ff]

    W_down = ex.down_proj[i, :, :]           # [ff, hidden]
    b_down = ex.down_proj_bias[i, :] if hasattr(ex, "down_proj_bias") else None          # [hidden]
    return W_gate, b_gate, W_up, b_up, W_down, b_down

# Example: view expert 0 shapes
W_gate, b_gate, W_up, b_up, W_down, b_down = get_expert_params(0)
print("Expert 0 shapes:")
print("  W_gate:", tuple(W_gate.shape), "b_gate:", None if b_gate is None else tuple(b_gate.shape))
print("  W_up  :", tuple(W_up.shape),   "b_up  :", None if b_up   is None else tuple(b_up.shape))
print("  W_down:", tuple(W_down.shape), "b_down:", None if b_down is None else tuple(b_down.shape))