import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors import QUANTIZATION_CONFIG_NAME

def decompress_to_fp16(
    compressed_model_dir: str,
    output_dir: str,
    force_rebuild: bool = False,
):
    """
    Convert MXFP4 model to FP16 HF format

    Args:
        compressed_model_dir: Path to compressed model (MXFP4)
        skeleton_model_id: HF model ID for skeleton (must match architecture)
        output_dir: Where to save FP16 model
        force_rebuild: Rebuild even if output exists
    """

    # Prepare output
    os.makedirs(output_dir, exist_ok=True)
    if not force_rebuild and os.path.exists(output_dir):
        print(f"Output exists, skipping. Use force_rebuild=True to overwrite.")
        return

    # Option 1: Try loading as HF-compressed model
    print("Attempting HF-compressed load...")
    tokenizer = AutoTokenizer.from_pretrained(compressed_model_dir, use_fast=True)

    # Load with decompression
    model = AutoModelForCausalLM.from_pretrained(
        compressed_model_dir,
        attn_implementation="kernels-community/vllm-flash-attn3"
        torch_dtype="auto",
        device_map="auto",
    )

    # Save dense FP16
    model.save_pretrained(output_dir, save_safetensors=True, save_compressed=False)
    tokenizer.save_pretrained(output_dir)
    print(f"Successfully decompressed to {output_dir}")
    return None


if __name__ == "__main__":
    # Example usage
    decompress_to_fp16(
        compressed_model_dir="openai/gpt-oss-20b",
        output_dir="./gpt-oss-20b-bf16",
        force_rebuild=False,
    )