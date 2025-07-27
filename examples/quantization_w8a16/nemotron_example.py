from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "nvidia/Llama-Nemotron-Post-Training-Dataset"
SUBSET = "default"
split = "train"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, SUBSET, split=split, revision="refs/convert/parquet") \
    .shuffle(seed=42) \
    .remove_columns(['category', 'license', 'reasoning', 'generator', 'used_in_training', 'version', 'system_prompt']) 

print(ds)

def preprocess(example):
    #print(example["conversations"])
    conversations = example["input"]
    conversations.append({"role": "assistant", "content": example["output"]})
    return {
        "text": tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
        )
    }


ds = ds.map(preprocess).select(range(NUM_CALIBRATION_SAMPLES))

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
recipe = GPTQModifier(targets="Linear", scheme="W8A16", ignore=["lm_head"])

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
)

# Save to disk compressed.
SAVE_DIR = "./Llama-3_3-Nemotron-Super-49B-v1_5-v2-GPTQ-8bit"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)