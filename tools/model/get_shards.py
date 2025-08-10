from safetensors import safe_open
import os, json

model_dir = "/mnt/LinuxDrive_1/huggingface/hub/gpt-oss-20b-bf16" 

index_path = os.path.join(model_dir, "model.safetensors.index.json")
if os.path.exists(index_path):
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
else:
    # Fallback: scan shards in folder (will still list contained tensors)
    weight_map = {}
    for fn in sorted(os.listdir(model_dir)):
        if fn.endswith(".safetensors"):
            weight_map[f"__all_in__{fn}"] = fn

seen = set()
for tensor_name, shard in sorted(weight_map.items()):
    shard_path = os.path.join(model_dir, shard)
    if shard_path in seen:
        continue
    seen.add(shard_path)
    print("\n== Shard:", shard)
    with safe_open(shard_path, framework="pt") as f:
        for k in f.keys():
            print(k, tuple(f.get_tensor(k).shape))