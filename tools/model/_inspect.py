from transformers import Glm4vForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig

import torch

import time

MODEL_ID = "/mnt/LinuxDrive_1/huggingface/hub/GLM-4.1V-9B-Thinking-AWQ-4bit"

model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)

print(model)
count = 0

for name, module in model.named_modules():
    print("MODULES", name, module)

    count += 1
    if count >= 25: break

