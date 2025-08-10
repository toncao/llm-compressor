from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
model_id = "unsloth/gpt-oss-20b-BF16"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "nvidia/Llama-Nemotron-Post-Training-Dataset"
SUBSET = "default"
split = "train"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 1024

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, SUBSET, split=split, revision="refs/convert/parquet") \
    .shuffle(seed=42) \
    .remove_columns(['category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']) \
    .select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    conversations = example["input"]
    conversations.append({"role": "assistant", "content": example["output"]})
    return {"text": tokenizer.apply_chat_template(conversations, tokenize=False,)}

ds = ds.map(preprocess).select(range(NUM_CALIBRATION_SAMPLES))


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
config_groups = {
    "group_0": {
        "targets": ["Linear"],
        "weights": {
            "num_bits": 4,
            "type": "int",
            "symmetric": True,
            "strategy": "group",
            "group_size": 32,
            }
    }
}

# Configure the quantization algorithm to run.
recipe = [
            GPTQModifier(
                ignore=["lm_head", "re:.*router", "re:.*sinks"],
                config_groups=config_groups,
                ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    calibrate_moe_context=True,
    pipeline="sequential",
)

SAVE_DIR = "./gpt-oss-20b-BF16" + "-W4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
