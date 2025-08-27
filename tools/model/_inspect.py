from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig

import torch

import time

MODEL_ID = "/mnt/LinuxDrive_1/huggingface/hub/Hermes-4-70B"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

print(model)