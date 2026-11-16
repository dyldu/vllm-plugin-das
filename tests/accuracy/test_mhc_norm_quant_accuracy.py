# SPDX-License-Identifier: Apache-2.0
"""Accuracy: fused MHC RMSNorm+quant vs ir.rms_norm + torch per-token quant."""

from __future__ import annotations

import pytest
import torch

from hcu_vllm_das.dsv41_mhc_norm_quant import (
    rmsnorm_per_token_fp8,
    rmsnorm_per_token_int8,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="HCU/HIP required",
)


def _ref_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # Match the fused Triton reduction (fp32 mean-of-squares over H, then
    # round to storage dtype). ir.ops.rms_norm can differ by 1–2 bf16 ULPs.
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * w.float()).to(x.dtype)


def _ref_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Match fused: round already in bf16; amax/448 with min_scale floor.
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = torch.maximum(amax / 448.0, torch.full_like(amax, 1.0 / (448.0 * 512.0)))
    q = (xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


def _ref_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    inv = torch.where(amax == 0, torch.zeros_like(amax), 127.0 / amax)
    q = torch.round(xf * inv).clamp(-127, 127).to(torch.int8)
    return q, scale


@pytest.mark.parametrize("tokens,hidden", [(1, 5120), (8, 5120), (64, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fp8_matches_unfused(tokens, hidden, dtype, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_MHC_NORM_QUANT", "1")
    g = torch.Generator(device="cpu").manual_seed(tokens * hidden)
    x = torch.randn(tokens, hidden, generator=g, dtype=dtype).cuda()
    w = torch.randn(hidden, generator=g, dtype=dtype).cuda()
    eps = 1e-6
    (q, scale, orig), norm = rmsnorm_per_token_fp8(x, w, eps)
    assert orig == dtype
    ref_norm = _ref_rms_norm(x, w, eps)
    torch.testing.assert_close(norm, ref_norm, rtol=0, atol=0)
    rq, rs = _ref_fp8(ref_norm)
    torch.testing.assert_close(scale, rs, rtol=1e-5, atol=1e-6)
    # e4m3 + div_rn vs / can differ on a few rows; bound mismatch ratio.
    deq = q.float() * scale
    rdeq = rq.float() * rs
    rel = (deq - rdeq).abs() / rdeq.abs().clamp_min(1e-3)
    bad = ((deq - rdeq).abs() > 0.05) & (rel > 0.05)
    assert bad.float().mean().item() < 5e-3, bad.float().mean().item()


@pytest.mark.parametrize("tokens,hidden", [(1, 5120), (8, 5120), (32, 1024)])
def test_int8_matches_unfused(tokens, hidden, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_MHC_NORM_QUANT", "1")
    g = torch.Generator(device="cpu").manual_seed(7 + tokens)
    x = torch.randn(tokens, hidden, generator=g, dtype=torch.bfloat16).cuda()
    w = torch.randn(hidden, generator=g, dtype=torch.bfloat16).cuda()
    eps = 1e-6
    (q, scale), norm = rmsnorm_per_token_int8(x, w, eps)
    ref_norm = _ref_rms_norm(x, w, eps)
    torch.testing.assert_close(norm, ref_norm, rtol=0, atol=0)
    rq, rs = _ref_int8(ref_norm)
    torch.testing.assert_close(scale, rs, rtol=1e-5, atol=1e-6)
    # rounding tie-breaks may differ by 1 on a few elems
    mismatch = (q.cpu() != rq.cpu()).sum().item()
    assert mismatch / q.numel() < 1e-3, f"int8 mismatch ratio {mismatch}/{q.numel()}"
