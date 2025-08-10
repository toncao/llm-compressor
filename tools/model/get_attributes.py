import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------
# Utilities to inspect a model
# -------------------------------

def list_leaf_layers(model):
    leaves = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            leaves.append((name, m))
    return leaves

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def _shp(x):
    # Turn arbitrary tensors/containers into a readable shape record
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [_shp(xx) for xx in x]
    if isinstance(x, dict):
        return {k: _shp(v) for k, v in x.items()}
    # HuggingFace outputs and other objects: try common attributes
    for attr in ("shape", "sizes"):
        if hasattr(x, attr):
            try:
                s = getattr(x, attr)
                return tuple(s) if not isinstance(s, tuple) else s
            except Exception:
                pass
    return type(x).__name__

def run_model(model, example_input):
    # Call the model correctly for tensor/tuple/list/dict inputs
    if isinstance(example_input, dict):
        return model(**example_input)
    elif isinstance(example_input, (list, tuple)):
        return model(*example_input)
    else:
        return model(example_input)

def capture_io_shapes(model, example_input):
    # Register forward hooks on leaf layers to capture input/output shapes
    io = {}
    hooks = []

    def make_hook(name):
        def hook(m, inputs, output):
            # inputs is always a tuple from PyTorch
            io[name] = {
                "type": m.__class__.__name__,
                "in": _shp(inputs),
                "out": _shp(output),
            }
        return hook

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = run_model(model, example_input)

    for h in hooks:
        h.remove()
    return io

# -------------------------------
# Dummy input builder
# -------------------------------

def has_any_embedding(model):
    return any(isinstance(m, nn.Embedding) for _, m in list_leaf_layers(model))

def get_vocab_size(model, default=30522):
    # Try to infer vocab size from any embedding layers; fallback to a typical BERT vocab size.
    vocab_sizes = []
    for _, m in list_leaf_layers(model):
        if isinstance(m, nn.Embedding):
            vocab_sizes.append(getattr(m, "num_embeddings", None))
    vocab_sizes = [v for v in vocab_sizes if v is not None]
    return max(vocab_sizes) if vocab_sizes else default


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example 1: Vision model (no embeddings)
    MODEL_ID = "/mnt/LinuxDrive_1/huggingface/hub/gpt-oss-20b"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")

    # Print architecture
    print("Architecture:\n", model)

    # Leaf layers
    print("\nLeaf layers:")
    for name, m in list_leaf_layers(model):
        print(f"{name:35s} -> {m.__class__.__name__}")

    # Parameter counts
    total, trainable = count_params(model)
    print(f"\nTotal params: {total:,} | Trainable: {trainable:,}")

    # Capture I/O shapes per leaf layer
    shapes = capture_io_shapes(model, dummy)

    print("\nFirst 10 leaf layer I/O shapes:")
    for i, (name, rec) in enumerate(shapes.items()):
        if i >= 10:
            break
        print(f"{name:35s} ({rec['type']}) | in: {rec['in']} -> out: {rec['out']}")

    # Inspect first 10 parameter tensors
    print("\nFirst 10 parameter tensors:")
    for i, (name, p) in enumerate(model.named_parameters()):
        if i >= 10:
            break
        print(name, tuple(p.shape))

    # ---------------------------
    # Example 2 (optional): If you load a text model with embeddings,
    # this shows how the dummy input switches to LongTensor automatically.
    # Uncomment to test with a toy embedding model.

    # class TinyTextModel(nn.Module):
    #     def __init__(self, vocab=1000, emb=32, hid=64, num_classes=2):
    #         super().__init__()
    #         self.emb = nn.Embedding(vocab, emb)
    #         self.rnn = nn.GRU(emb, hid, batch_first=True)
    #         self.fc = nn.Linear(hid, num_classes)
    #     def forward(self, x):
    #         # x should be LongTensor of token ids [B, L]
    #         x = self.emb(x)
    #         _, h = self.rnn(x)
    #         return self.fc(h[-1])
    #
    # text_model = TinyTextModel()
    # text_dummy = build_dummy_input(text_model, batch_size=2, seq_len=20)  # LongTensor
    # print("\nText model dummy dtype:", text_dummy.dtype)  # should be torch.int64 (Long)
    # text_shapes = capture_io_shapes(text_model, text_dummy)
    # for name, rec in list(text_shapes.items())[:10]:
    #         print(f"{name:35s} ({rec['type']}) | in: {rec['in']} -> out: {rec['out']}")