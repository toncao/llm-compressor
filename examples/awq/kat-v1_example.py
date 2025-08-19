from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "Kwaipilot/KAT-V1-40B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "sqres/tulu3_qwen32b"
DATASET_SPLIT = "train"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT).shuffle(seed=42).select(range(1024))

def preprocess(example):
    #print(example["conversations"])
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"][0]["content"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)

# Configure the quantization algorithm to run.
recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)


# Save to disk compressed.
SAVE_DIR = "./KAT-V1-40B-AWQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)