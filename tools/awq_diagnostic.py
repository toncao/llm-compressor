#!/usr/bin/env python3
"""
AWQ Pre-Quantization Safety Diagnostic
=======================================

Run this on the ORIGINAL (unquantized) model BEFORE quantization to predict
whether AWQ will produce gibberish or elevated perplexity.

Checks performed:
  1. (1+w) RMSNorm zero-gain channels → normalization blow-up → gibberish
  2. (1+w) RMSNorm near-zero-gain channels → moderate scale amplification → high PPL
  3. Standard (w) RMSNorm with near-zero weights → similar but less severe
  4. bf16 precision cliffs that make the problem discrete/unpredictable
  5. w_mean analysis for duo_scaling safety
  6. Cross-check with stored precision vs fp32 precision

Root causes discovered:
  - Original model has norm weights = -1.0 in bf16 → effective gain = 0
  - AWQ activation hook captures norm output → x_mean[ch] = 0 for dead channels
  - Grid search computes raw_scale = x_mean^ratio → 0 for dead channels → clamped to 1e-4
  - Normalization: scales / sqrt(max * min) → sqrt(normal * 1e-4) ≈ 0.01
  - ALL channels amplified by ~100x, killing them in the (1+w) RMSNorm
  - duo_scaling amplifies spread further via w_mean division

Usage:
    python awq_diagnostic.py /path/to/model
    python awq_diagnostic.py /path/to/model --group-size 128
    python awq_diagnostic.py /path/to/model --verbose
    python awq_diagnostic.py /path/to/model --json results.json
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: pip install safetensors")
    sys.exit(1)


# ============================================================================
# Data classes for structured results
# ============================================================================


@dataclass
class NormChannelInfo:
    layer_idx: int
    norm_type: str  # "unit_plus_weight" (1+w) or "standard" (w)
    dtype: str
    hidden_size: int
    # Gain = (1+w) for unit_plus_weight, or w for standard
    exactly_zero_gain: int
    gain_lt_001: int
    gain_lt_01: int
    min_gain: float
    min_weight: float
    # Precision analysis
    next_representable_gain: float  # next bf16/fp16 step above 0
    is_trigger: bool  # has exactly-zero gain channel


@dataclass
class WMeanInfo:
    layer_idx: int
    w_mean_min: float
    w_mean_max: float
    w_mean_std: float
    channels_lt_001: int
    channels_lt_01: int


@dataclass
class LayerRisk:
    layer_idx: int
    norm_info: NormChannelInfo
    wmean_info: WMeanInfo | None
    # Predicted outcome
    will_blow_up: bool  # normalization blow-up → gibberish
    estimated_amplification: float  # scale amplification factor
    risk_level: str  # "CRITICAL", "HIGH", "MODERATE", "LOW", "SAFE"
    risk_reason: str


@dataclass
class DiagnosticResult:
    model_path: str
    model_type: str
    hidden_size: int
    num_layers: int
    num_experts: int | None
    norm_type: str
    stored_dtype: str
    group_size: int
    # Overall verdict
    verdict: str  # "GIBBERISH", "HIGH_PPL", "MODERATE_RISK", "SAFE"
    verdict_detail: str
    # Per-layer results
    layers: list[LayerRisk] = field(default_factory=list)
    # Recommendations
    recommendations: list[str] = field(default_factory=list)


# ============================================================================
# Helpers
# ============================================================================


def load_config(model_path: str) -> dict:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json in {model_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    return {
        "hidden_size": text_cfg.get("hidden_size"),
        "num_hidden_layers": text_cfg.get("num_hidden_layers"),
        "num_experts": text_cfg.get("num_experts"),
        "moe_intermediate_size": text_cfg.get("moe_intermediate_size"),
        "intermediate_size": text_cfg.get("intermediate_size"),
        "model_type": cfg.get("model_type", text_cfg.get("model_type", "unknown")),
        "architectures": cfg.get("architectures", []),
        "torch_dtype": cfg.get("torch_dtype", "unknown"),
    }


def load_safetensors_index(model_path: str) -> dict[str, str]:
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f).get("weight_map", {})
    weight_map = {}
    for st_file in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = st_file.name
    return weight_map


def load_tensor(model_path: str, weight_map: dict, name: str) -> torch.Tensor:
    shard = Path(model_path) / weight_map[name]
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def extract_layer_idx(key: str) -> int:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


# ============================================================================
# Detect norm type
# ============================================================================

# Known (1+w) RMSNorm architectures (weight initialized to 0, gain = 1+w)
UNIT_PLUS_WEIGHT_NORMS = {
    "Qwen3_5MoeRMSNorm",
    "Qwen3_5RMSNorm",
    "Qwen3_5MoeRMSNorm",
    "Qwen3NextRMSNorm",
    "Gemma3RMSNorm",
    "GemmaRMSNorm",
    "Gemma2RMSNorm",
    "Step3p5RMSNorm",
}

# Model types known to use (1+w) norms
UNIT_PLUS_WEIGHT_MODEL_TYPES = {
    "qwen3_5_moe",
    "qwen3_5",
    "qwen3_next",
    "gemma",
    "gemma2",
    "gemma3",
}


def detect_norm_type(config: dict, weight_map: dict, model_path: str) -> str:
    """
    Detect whether the model uses (1+w) or standard (w) RMSNorm.

    (1+w) norms: weight initialized to 0, effective gain = 1 + weight
    Standard norms: weight initialized to 1, effective gain = weight

    Heuristic: if the median norm weight is near 0, it's (1+w).
    If near 1, it's standard.
    """
    model_type = config.get("model_type", "")
    if model_type in UNIT_PLUS_WEIGHT_MODEL_TYPES:
        return "unit_plus_weight"

    # Heuristic: check first norm layer's weight distribution
    norm_keys = [k for k in weight_map if "layernorm.weight" in k or "norm.weight" in k]
    if not norm_keys:
        return "unknown"

    sample_key = norm_keys[0]
    w = load_tensor(model_path, weight_map, sample_key).float()
    median = w.median().item()

    if abs(median) < 0.3:
        return "unit_plus_weight"
    elif abs(median - 1.0) < 0.3:
        return "standard"
    else:
        return "unknown"


def get_precision_step(dtype: torch.dtype, near_value: float) -> float:
    """
    Get the step size between representable values near a given value.
    For bf16 near -1.0: step = 2^-7 = 0.0078125
    For fp16 near -1.0: step = 2^-10 = 0.000977
    """
    if dtype == torch.bfloat16:
        # bf16: 1 sign + 8 exp + 7 mantissa bits
        # Near 1.0: exponent = 0, step = 2^-7 = 0.0078125
        return 2**-7  # 0.0078125
    elif dtype == torch.float16:
        # fp16: 1 sign + 5 exp + 10 mantissa bits
        # Near 1.0: exponent = 0, step = 2^-10 = 0.000977
        return 2**-10  # 0.0009766
    else:
        return 0.0  # fp32 is effectively continuous for our purposes


# ============================================================================
# Check 1: Norm weight analysis
# ============================================================================


def analyze_norm_weights(
    model_path: str,
    weight_map: dict,
    norm_type: str,
) -> list[NormChannelInfo]:
    """
    Analyze all post_attention_layernorm weights for dangerous channels.
    """
    norm_keys = sorted(
        [k for k in weight_map if "post_attention_layernorm.weight" in k],
        key=extract_layer_idx,
    )

    if not norm_keys:
        # Try alternative names
        norm_keys = sorted(
            [k for k in weight_map if "post_feedforward_layernorm.weight" in k],
            key=extract_layer_idx,
        )

    results = []
    for key in norm_keys:
        layer_idx = extract_layer_idx(key)
        w = load_tensor(model_path, weight_map, key)
        dtype = w.dtype
        w_f32 = w.float()

        if norm_type == "unit_plus_weight":
            gain = 1.0 + w  # in stored precision
            gain_f32 = 1.0 + w_f32
        else:
            gain = w
            gain_f32 = w_f32

        exactly_zero = (gain == 0).sum().item()
        gain_lt_001 = (gain_f32.abs() < 0.01).sum().item()
        gain_lt_01 = (gain_f32.abs() < 0.1).sum().item()

        min_gain = gain_f32.min().item()
        min_w = w_f32.min().item()

        precision_step = get_precision_step(dtype, -1.0 if norm_type == "unit_plus_weight" else 0.0)

        results.append(NormChannelInfo(
            layer_idx=layer_idx,
            norm_type=norm_type,
            dtype=str(dtype),
            hidden_size=w.numel(),
            exactly_zero_gain=exactly_zero,
            gain_lt_001=gain_lt_001,
            gain_lt_01=gain_lt_01,
            min_gain=min_gain,
            min_weight=min_w,
            next_representable_gain=precision_step,
            is_trigger=exactly_zero > 0,
        ))

    return results


# ============================================================================
# Check 2: w_mean analysis for duo_scaling
# ============================================================================


def analyze_wmean_sample(
    model_path: str,
    weight_map: dict,
    layer_idx: int,
    group_size: int,
    config: dict,
) -> WMeanInfo | None:
    """Compute w_mean for one layer's expert weights."""
    # Try fused format
    fused_key = None
    for prefix in [
        f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj",
        f"model.layers.{layer_idx}.mlp.experts.gate_up_proj",
    ]:
        if prefix in weight_map:
            fused_key = prefix
            break

    if fused_key:
        gate_up = load_tensor(model_path, weight_map, fused_key).float()
        num_experts, fused_dim, hidden = gate_up.shape
        intermediate = fused_dim // 2

        total_count = 0
        total_sum = torch.zeros(hidden, dtype=torch.float64)

        for expert_idx in range(num_experts):
            for sub_w in (gate_up[expert_idx, :intermediate], gate_up[expert_idx, intermediate:]):
                w = sub_w.clone().reshape(-1, group_size).abs_()
                w.div_(w.amax(dim=1, keepdim=True) + 1e-6)
                w = w.reshape(sub_w.shape)
                total_count += w.size(0)
                total_sum += w.sum(0, dtype=torch.float64)

        w_mean = (total_sum / total_count).float()
        del gate_up
        gc.collect()

        return WMeanInfo(
            layer_idx=layer_idx,
            w_mean_min=w_mean.min().item(),
            w_mean_max=w_mean.max().item(),
            w_mean_std=w_mean.std().item(),
            channels_lt_001=(w_mean < 0.001).sum().item(),
            channels_lt_01=(w_mean < 0.01).sum().item(),
        )

    # Try unfused format
    gate_keys = [
        k for k in weight_map
        if f"layers.{layer_idx}.mlp.experts." in k
        and (k.endswith("gate_proj.weight") or k.endswith("up_proj.weight"))
    ]
    if not gate_keys:
        return None

    total_count = 0
    total_sum = None
    for key in gate_keys:
        w = load_tensor(model_path, weight_map, key).float()
        orig_shape = w.shape
        w = w.reshape(-1, group_size).abs_()
        w.div_(w.amax(dim=1, keepdim=True) + 1e-6)
        w = w.reshape(orig_shape)
        total_count += w.size(0)
        s = w.sum(0, dtype=torch.float64)
        total_sum = s if total_sum is None else total_sum + s
        del w

    if total_sum is None:
        return None

    w_mean = (total_sum / total_count).float()
    return WMeanInfo(
        layer_idx=layer_idx,
        w_mean_min=w_mean.min().item(),
        w_mean_max=w_mean.max().item(),
        w_mean_std=w_mean.std().item(),
        channels_lt_001=(w_mean < 0.001).sum().item(),
        channels_lt_01=(w_mean < 0.01).sum().item(),
    )


# ============================================================================
# Risk assessment per layer
# ============================================================================


def assess_layer_risk(
    norm_info: NormChannelInfo,
    wmean_info: WMeanInfo | None,
) -> LayerRisk:
    """
    Predict AWQ outcome for one layer based on norm weights and w_mean.
    """
    risk_level = "SAFE"
    risk_reason = ""
    will_blow_up = False
    estimated_amp = 1.0

    # --- Check 1: Exactly-zero gain (normalization blow-up) ---
    if norm_info.exactly_zero_gain > 0:
        # Zero gain → x_mean = 0 → raw_scale clamped to 1e-4
        # sqrt(normal_max * 1e-4) ≈ 0.01 → all scales × 100
        estimated_amp = 1.0 / (1e-4 * 0.3) ** 0.5  # ≈ 180
        will_blow_up = True
        risk_level = "CRITICAL"
        risk_reason = (
            f"{norm_info.exactly_zero_gain} channel(s) have effective gain = 0.0 exactly "
            f"(weight = -1.0 in {norm_info.dtype}). "
            f"x_mean = 0 → raw scale clamped to 1e-4 → normalization blow-up "
            f"amplifies ALL scales by ~{estimated_amp:.0f}x. "
            f"With (1+w) norm: (1+w)/{estimated_amp:.0f} - 1 ≈ -1 → kills hundreds of channels."
        )

    # --- Check 2: Near-zero gain (moderate amplification) ---
    elif norm_info.gain_lt_001 > 0:
        # Gain ≈ 0.004 → x_mean ≈ 0.004 * hidden_norm → small but not zero
        # raw_scale ≈ (0.004)^ratio → at ratio=0.5: 0.063
        # sqrt(0.3 * 0.063) ≈ 0.14 → amplification ≈ 7x
        min_gain = max(norm_info.min_gain, 1e-6)
        raw_scale_estimate = min_gain ** 0.5  # ratio=0.5
        estimated_amp = 1.0 / (raw_scale_estimate * 0.3) ** 0.5
        estimated_amp = min(estimated_amp, 200)

        if estimated_amp > 20:
            risk_level = "HIGH"
            risk_reason = (
                f"Min gain = {norm_info.min_gain:.6f} (next step above 0 is "
                f"{norm_info.next_representable_gain:.6f}). "
                f"Near-zero x_mean → estimated scale amplification ~{estimated_amp:.0f}x. "
                f"May cause significant perplexity increase."
            )
        elif estimated_amp > 5:
            risk_level = "MODERATE"
            risk_reason = (
                f"Min gain = {norm_info.min_gain:.6f}. "
                f"Mild scale amplification ~{estimated_amp:.1f}x possible."
            )
        else:
            risk_level = "LOW"
            risk_reason = f"Min gain = {norm_info.min_gain:.6f}. Amplification manageable."

    # --- Check 3: duo_scaling w_mean risk ---
    if wmean_info and wmean_info.channels_lt_001 > 0:
        old_level = risk_level
        if risk_level in ("SAFE", "LOW"):
            risk_level = "MODERATE"
        risk_reason += (
            f" Additionally, {wmean_info.channels_lt_001} channels have w_mean < 0.001, "
            f"which amplifies scales further with duo_scaling=True."
        )

    if risk_level == "SAFE":
        risk_reason = "No dangerous channels detected."

    return LayerRisk(
        layer_idx=norm_info.layer_idx,
        norm_info=norm_info,
        wmean_info=wmean_info,
        will_blow_up=will_blow_up,
        estimated_amplification=estimated_amp,
        risk_level=risk_level,
        risk_reason=risk_reason,
    )


# ============================================================================
# Overall verdict
# ============================================================================


def compute_verdict(layers: list[LayerRisk], norm_type: str) -> tuple[str, str, list[str]]:
    """Compute overall verdict and recommendations."""
    critical = [l for l in layers if l.risk_level == "CRITICAL"]
    high = [l for l in layers if l.risk_level == "HIGH"]
    moderate = [l for l in layers if l.risk_level == "MODERATE"]

    recommendations = []

    if critical:
        verdict = "GIBBERISH"
        trigger_layers = [l.layer_idx for l in critical]

        # Check: will grid search always trigger?
        # If n_grid=20, ratio=0 is tested first and has zero amplification.
        # But ratio=1/20=0.05 already causes blow-up for zero-gain channels.
        # Grid search selects the ratio with lowest quantization loss.
        # Since dead channels contribute 0 to the loss, ANY ratio > 0 is
        # as good as ratio=0 for them, and better for other channels.
        # So the grid search will almost certainly pick ratio > 0.

        # How many layers would NOT blow up because grid search picks ratio=0?
        # Unpredictable — depends on calibration data. Conservative: assume all trigger.

        detail = (
            f"{len(critical)} layers have exactly-zero gain channels that will trigger "
            f"AWQ normalization blow-up (layers: {trigger_layers}). "
            f"The grid search will almost certainly select ratio > 0 "
            f"(because it doesn't penalize norm weight precision loss), "
            f"amplifying ALL scales by ~100-200x. "
            f"In the (1+w) RMSNorm, (1+w)/scale - 1 ≈ -1, killing hundreds of "
            f"channels per affected layer. Output will be gibberish."
        )

        recommendations = [
            "IMMEDIATE: Use duo_scaling=False (reduces but does NOT eliminate the problem)",
            "FIX 1 - Replace AWQ normalization: scales / scales.median() instead of scales / sqrt(max*min)",
            "FIX 2 - Clamp x_mean: x_mean = x_mean.clamp(min=x_mean[x_mean > 0].min()) before scale computation",
            "FIX 3 - Clamp final scales: scales = scales.clamp(max=50) after normalization",
            "FIX 4 - Cast model to fp32 before calibration (eliminates bf16 precision cliff at -1.0)",
            "FIX 5 - Add norm-aware loss: include (1+w)/scales precision in grid search objective",
        ]

        if norm_type == "unit_plus_weight":
            recommendations.append(
                "FIX 6 - Pre-process: clamp original norm weights to prevent exact -1.0: "
                "w.clamp_(min=-1.0 + eps) before running AWQ"
            )

    elif high:
        verdict = "HIGH_PPL"
        detail = (
            f"{len(high)} layers have near-zero gain channels (gain < 0.01). "
            f"AWQ scales will be moderately amplified (~5-20x), compressing "
            f"norm gain for most channels. Expect noticeably elevated perplexity "
            f"compared to properly quantized models."
        )
        recommendations = [
            "Use duo_scaling=False to reduce amplification",
            "Clamp scales: scales = scales.clamp(max=20)",
            "Consider casting to fp32 before calibration",
        ]

    elif moderate:
        verdict = "MODERATE_RISK"
        detail = (
            f"{len(moderate)} layers have mild risk factors. "
            f"AWQ should work but may have slightly elevated perplexity."
        )
        recommendations = [
            "duo_scaling=True should be safe, but monitor perplexity",
            "If perplexity is higher than expected, try duo_scaling=False",
        ]

    else:
        verdict = "SAFE"
        detail = (
            "No dangerous patterns detected. AWQ with duo_scaling=True "
            "should work correctly."
        )
        recommendations = [
            "Proceed with AWQ quantization normally",
        ]

    return verdict, detail, recommendations


# ============================================================================
# Printing
# ============================================================================


def print_report(result: DiagnosticResult):
    """Print the full diagnostic report."""
    W = 80
    print(f"\n{'='*W}")
    print("AWQ PRE-QUANTIZATION SAFETY DIAGNOSTIC")
    print(f"{'='*W}")

    print(f"\n  Model:        {result.model_path}")
    print(f"  Type:         {result.model_type}")
    print(f"  Hidden size:  {result.hidden_size}")
    print(f"  Layers:       {result.num_layers}")
    print(f"  Experts:      {result.num_experts or 'N/A (dense)'}")
    print(f"  Norm type:    {result.norm_type}")
    print(f"  Stored dtype: {result.stored_dtype}")
    print(f"  Group size:   {result.group_size}")

    # --- Per-layer table ---
    print(f"\n{'='*W}")
    print("PER-LAYER ANALYSIS")
    print(f"{'='*W}\n")

    print(f"  {'Layer':>5s} {'Risk':<10s} {'Gain=0':>6s} {'Gain<.01':>8s} "
          f"{'Min Gain':>10s} {'Amp':>6s} {'w_min':>8s}")
    print(f"  {'-'*5} {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*6} {'-'*8}")

    for lr in result.layers:
        ni = lr.norm_info
        wm = lr.wmean_info

        risk_marker = {
            "CRITICAL": "***",
            "HIGH": "** ",
            "MODERATE": "*  ",
            "LOW": "   ",
            "SAFE": "   ",
        }.get(lr.risk_level, "   ")

        wm_min = f"{wm.w_mean_min:.4f}" if wm else "n/a"

        print(
            f"  {ni.layer_idx:>5d} {lr.risk_level:<10s} {ni.exactly_zero_gain:>6d} "
            f"{ni.gain_lt_001:>8d} {ni.min_gain:>10.6f} {lr.estimated_amplification:>6.0f} "
            f"{wm_min:>8s} {risk_marker}"
        )

    # --- Risk summary ---
    critical = [l for l in result.layers if l.risk_level == "CRITICAL"]
    high = [l for l in result.layers if l.risk_level == "HIGH"]

    if critical:
        print(f"\n  CRITICAL layers (will cause gibberish):")
        for lr in critical:
            print(f"    Layer {lr.layer_idx}: {lr.risk_reason}")

    if high:
        print(f"\n  HIGH risk layers (elevated perplexity):")
        for lr in high:
            print(f"    Layer {lr.layer_idx}: {lr.risk_reason}")

    # --- bf16 precision analysis ---
    if result.stored_dtype in ("torch.bfloat16", "bfloat16"):
        print(f"\n{'='*W}")
        print("BF16 PRECISION ANALYSIS")
        print(f"{'='*W}")
        print(f"""
  bf16 representable values near -1.0 (for (1+w) norm):
    ..., -1.0078 (gain=+0.0078), -1.0000 (gain=0.0), -0.9922 (gain=+0.0078), ...
                                  ^^^^^^^^
                                  CLIFF: one step is the difference between
                                  "safe" (gain=0.0078, amp~3x) and
                                  "catastrophic" (gain=0.0, amp~180x)

  The original model's fp32 training weights near -1.0 round to
  exactly -1.0 in bf16 storage, creating the trigger channels.
  This is a storage precision artifact, not a training bug.""")

    # --- Verdict ---
    print(f"\n{'='*W}")
    verdict_color = {
        "GIBBERISH": "🔴",
        "HIGH_PPL": "🟠",
        "MODERATE_RISK": "🟡",
        "SAFE": "🟢",
    }.get(result.verdict, "❓")
    print(f"VERDICT: {verdict_color} {result.verdict}")
    print(f"{'='*W}")
    print(f"\n  {result.verdict_detail}")

    print(f"\n  Recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"    {i}. {rec}")

    # --- Quick reference for specific settings ---
    print(f"\n{'='*W}")
    print("AWQ SETTING COMPATIBILITY")
    print(f"{'='*W}")

    if result.verdict == "GIBBERISH":
        print(f"""
  duo_scaling=True:   BROKEN  — will produce gibberish
  duo_scaling=False:  RISKY   — avoids worst blow-up but near-zero channels
                                still cause moderate amplification (~3-10x).
                                May produce elevated perplexity.
  duo_scaling="both": BROKEN  — half the grid uses duo=True, inherits the problem

  With scale clamping (max=50):
    duo_scaling=True:   LIKELY OK  — clamping prevents catastrophic blow-up
    duo_scaling=False:  LIKELY OK  — safer combination
""")
    elif result.verdict == "HIGH_PPL":
        print(f"""
  duo_scaling=True:   RISKY   — may amplify near-zero channels
  duo_scaling=False:  OK      — safer choice
  duo_scaling="both": RISKY   — safer to avoid
""")
    else:
        print(f"""
  duo_scaling=True:   OK
  duo_scaling=False:  OK
  duo_scaling="both": OK
""")


# ============================================================================
# Main
# ============================================================================


def run_diagnostic(
    model_path: str,
    group_size: int = 32,
    check_wmean: bool = True,
    wmean_sample_layers: int = 5,
    verbose: bool = False,
) -> DiagnosticResult:
    """Run the full diagnostic and return structured results."""

    config = load_config(model_path)
    weight_map = load_safetensors_index(model_path)

    print(f"Loaded {len(weight_map)} tensor keys")

    # Detect norm type
    norm_type = detect_norm_type(config, weight_map, model_path)
    print(f"Norm type: {norm_type}")

    if norm_type == "unknown":
        print("WARNING: Could not determine norm type. Assuming standard (w) norm.")
        print("  If this is a Gemma or Qwen3.5 model, results may be inaccurate.")
        norm_type = "standard"

    # Detect stored dtype from first norm weight
    norm_keys = [k for k in weight_map if "layernorm.weight" in k or "norm.weight" in k]
    stored_dtype = "unknown"
    if norm_keys:
        sample = load_tensor(model_path, weight_map, norm_keys[0])
        stored_dtype = str(sample.dtype)

    # --- Check 1: Norm weight analysis ---
    print("\nAnalyzing norm weights...")
    norm_results = analyze_norm_weights(model_path, weight_map, norm_type)
    print(f"  Analyzed {len(norm_results)} layers")

    trigger_count = sum(1 for n in norm_results if n.is_trigger)
    print(f"  Trigger layers (gain = 0 exactly): {trigger_count}")
    near_zero_count = sum(1 for n in norm_results if n.gain_lt_001 > 0 and not n.is_trigger)
    print(f"  Near-zero layers (0 < gain < 0.01): {near_zero_count}")

    # --- Check 2: w_mean analysis (sample) ---
    wmean_results: dict[int, WMeanInfo] = {}
    if check_wmean and config.get("num_experts"):
        print(f"\nAnalyzing w_mean (sampling {wmean_sample_layers} layers)...")

        # Sample layers: first, last, and ones with worst norm stats
        sample_indices = set()
        # Always check first and last
        all_layer_indices = sorted(set(n.layer_idx for n in norm_results))
        if all_layer_indices:
            sample_indices.add(all_layer_indices[0])
            sample_indices.add(all_layer_indices[-1])
            sample_indices.add(all_layer_indices[len(all_layer_indices) // 2])

        # Add trigger/near-zero layers
        for n in norm_results:
            if n.is_trigger or n.gain_lt_001 > 0:
                sample_indices.add(n.layer_idx)
                if len(sample_indices) >= wmean_sample_layers:
                    break

        for layer_idx in sorted(sample_indices):
            t0 = time.time()
            wm = analyze_wmean_sample(model_path, weight_map, layer_idx, group_size, config)
            elapsed = time.time() - t0
            if wm:
                wmean_results[layer_idx] = wm
                print(f"  Layer {layer_idx}: w_mean [{wm.w_mean_min:.6f}, {wm.w_mean_max:.4f}] ({elapsed:.1f}s)")

    # --- Assess risk per layer ---
    layer_risks = []
    for norm_info in norm_results:
        wmean_info = wmean_results.get(norm_info.layer_idx)
        risk = assess_layer_risk(norm_info, wmean_info)
        layer_risks.append(risk)

    # --- Compute verdict ---
    verdict, detail, recommendations = compute_verdict(layer_risks, norm_type)

    result = DiagnosticResult(
        model_path=model_path,
        model_type=config.get("model_type", "unknown"),
        hidden_size=config.get("hidden_size", 0),
        num_layers=config.get("num_hidden_layers", 0),
        num_experts=config.get("num_experts"),
        norm_type=norm_type,
        stored_dtype=stored_dtype,
        group_size=group_size,
        verdict=verdict,
        verdict_detail=detail,
        layers=layer_risks,
        recommendations=recommendations,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AWQ Pre-Quantization Safety Diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_path", help="Path to original (unquantized) model")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--no-wmean", action="store_true", help="Skip w_mean analysis (faster)")
    parser.add_argument("--wmean-layers", type=int, default=5, help="Number of layers to sample for w_mean")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    result = run_diagnostic(
        args.model_path,
        group_size=args.group_size,
        check_wmean=not args.no_wmean,
        wmean_sample_layers=args.wmean_layers,
        verbose=args.verbose,
    )

    print_report(result)

    if args.json:
        # Serialize
        def to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [to_dict(v) for v in obj]
            if isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            return obj

        with open(args.json, "w") as f:
            json.dump(to_dict(result), f, indent=2, default=str)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()