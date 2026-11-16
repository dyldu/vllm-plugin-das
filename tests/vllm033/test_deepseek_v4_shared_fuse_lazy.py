"""Contract for DeepseekV4MLP lazy Channel-FP8 fused SiLU+quant enablement.

Importing ``vllm_hcu.models.deepseek_v4`` needs the full vLLM DSV4 stack; this
test mirrors the resolve gate so CI without that wheel still covers the rule.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def _fused_silu_mul_quant_supported(weight: object) -> bool:
    return getattr(weight, "dtype", None) in (
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )


def _resolve(mlp) -> None:
    if mlp._fuse_resolved:
        return
    mlp._fuse_resolved = True
    if not mlp._fuse_requested:
        return
    weight = getattr(mlp.gate_up_proj, "weight", None)
    if not _fused_silu_mul_quant_supported(weight):
        return
    mlp.enable_fuse_silu_mul_quant = True
    mlp.quant_dtype = weight.dtype


def test_fused_silu_mul_quant_supported_dtypes():
    assert _fused_silu_mul_quant_supported(
        SimpleNamespace(dtype=torch.float8_e4m3fn)
    )
    assert not _fused_silu_mul_quant_supported(
        SimpleNamespace(dtype=torch.bfloat16)
    )


def test_resolve_fuse_stays_off_until_fp8_weight():
    mlp = SimpleNamespace(
        _fuse_requested=True,
        _fuse_resolved=False,
        enable_fuse_silu_mul_quant=False,
        quant_dtype=None,
        gate_up_proj=SimpleNamespace(weight=SimpleNamespace(dtype=torch.bfloat16)),
    )
    _resolve(mlp)
    assert mlp.enable_fuse_silu_mul_quant is False

    mlp._fuse_resolved = False
    mlp.gate_up_proj = SimpleNamespace(
        weight=SimpleNamespace(dtype=torch.float8_e4m3fn)
    )
    _resolve(mlp)
    assert mlp.enable_fuse_silu_mul_quant is True
    assert mlp.quant_dtype == torch.float8_e4m3fn
