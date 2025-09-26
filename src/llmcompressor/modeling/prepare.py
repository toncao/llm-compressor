import tqdm
from compressed_tensors.utils import replace_module
from compressed_tensors.utils.offload import offload_to_weights_map
from transformers import PreTrainedModel

from llmcompressor.modeling.deepseek_v3 import replace as replace_deepseekv3
from llmcompressor.modeling.gpt_oss import replace_gpt_oss
from llmcompressor.modeling.llama4 import replace as replace_llama4
from llmcompressor.modeling.qwen3_moe import replace as replace_Qwen3MoE
from llmcompressor.utils.helpers import patch_attr

__all__ = ["replace_modules_for_calibration"]

# ---------------------- module replacements; permanent -------------------------
replacements = {
    "DeepseekV3MoE": replace_deepseekv3,
    "Llama4TextMoe": replace_llama4,
    "GptOssExperts": replace_gpt_oss,
}


def replace_modules_for_calibration(
    model: PreTrainedModel,
    calibrate_all_experts: bool = True,
) -> PreTrainedModel:
    for name, module in tqdm.tqdm(list(model.named_modules())):
        cls_name = module.__class__.__name__
        if cls_name in replacements:
            new_module = replacements[cls_name](
                config=model.config,
                module=module,
                calibrate_all_experts=calibrate_all_experts,
            )
            replace_module(model, name, new_module)

    return model


# ------------------- module replacements; during calibration --------------------


def update_qwen3_moe(model, stack, calibrate_all_experts):
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "Qwen3MoeDecoderLayer":
            # Optionally update the model.config to pass in other arguments
            stack.enter_context(
                patch_attr(
                    module,
                    "mlp",
                    replace_Qwen3MoE(
                        config=model.config,
                        module=module.mlp,
                        calibrate_all_experts=calibrate_all_experts,
                    ),
                )
            )


def update_gpt_oss(model: PreTrainedModel, stack):
    @contextlib.contextmanager
    def replace(mod_name, module, name, original):
        hook: AlignDevicesHook = original._hf_hook

        replacement = replace_gpt_oss(model.config, original)
        replace_offload_module(module, name, hook, replacement)
        del original

        yield

        restored = replacement.to_original()
        delattr(module, name)
        module.register_module(name, restored)
        # replace_offload_module(module, name, hook, restored)
        del replacement

    modules = list(model.named_modules())
    for name, module in tqdm.tqdm(modules, desc="Converting modules"):
        for child_name, child in list(module.named_children()):
            if child.__class__.__name__ == "GptOssExperts":
                stack.enter_context(replace(name, module, child_name, child))


moe_context = {
    "Qwen3MoeForCausalLM": update_qwen3_moe,
    "GptOssForCausalLM": update_gpt_oss,
}


def moe_calibration_context(
    model: PreTrainedModel,
    stack,
    calibrate_all_experts: bool = True,
):
    # Temporarily updates the MoE modules within the context
    # Once the context exists, parameter updates persist
    cls_name = model.__class__.__name__
    if cls_name in moe_context:
        moe_context.get(cls_name)(model, stack, calibrate_all_experts)
