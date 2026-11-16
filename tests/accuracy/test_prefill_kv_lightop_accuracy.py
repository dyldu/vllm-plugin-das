# SPDX-License-Identifier: Apache-2.0
"""Accuracy: LightOp Prefill KV fusions (#596 KV-only) vs torch references."""

from __future__ import annotations

import pytest
import torch

from hcu_vllm_das.prefill_kv_lightop import (
    lightop_pool_pairs_norm_available,
    lightop_rope_fq_store_available,
    ref_pool_pairs_norm,
    try_pool_pairs_norm,
    try_rope_fake_quant_store,
    v4_page_bytes_padded,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="HCU/HIP required",
)


def _round_fp4(x: torch.Tensor) -> torch.Tensor:
    mag = x.abs()
    step = torch.where(mag < 2.0, 0.5, torch.where(mag < 4.0, 1.0, 2.0))
    return torch.round(mag / step) * step * torch.sign(x)


def _fake_quant_compressed_kv(x: torch.Tensor) -> torch.Tensor:
    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 6.0).clamp(min=2**-9, max=448.0)
    scale = scale.to(torch.float8_e4m3fn).float()
    deq = _round_fp4((blocks / scale).clamp(-6.0, 6.0)) * scale
    return deq.flatten(-2).to(x.dtype)


def _rope_tail(x: torch.Tensor, freqs: torch.Tensor, rope_dim: int = 64) -> torch.Tensor:
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tc = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = torch.view_as_complex(freqs.float().unflatten(-1, (-1, 2)).contiguous())
    f = f.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
    rotated = torch.view_as_real(tc * f).flatten(-2).to(x.dtype)
    return torch.cat([head, rotated], dim=-1)


@pytest.mark.parametrize("n", [1, 4, 16])
def test_pool_pairs_norm_matches_ref(n, monkeypatch):
    if not lightop_pool_pairs_norm_available():
        pytest.skip("lightop.dsv41_pool_pairs_norm missing")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_POOL_PAIRS_NORM", "1")
    g = torch.Generator(device="cpu").manual_seed(n * 11)
    prev = torch.randn(n, 512, generator=g, dtype=torch.float32).cuda()
    cur = torch.randn(n, 512, generator=g, dtype=torch.float32).cuda()
    prev_s = torch.randn(n, 512, generator=g, dtype=torch.float32).cuda()
    cur_s = torch.randn(n, 512, generator=g, dtype=torch.float32).cuda()
    weight = (torch.rand(512, generator=g) + 0.5).cuda()
    eps = 1e-6
    fused = try_pool_pairs_norm(prev, cur, prev_s, cur_s, weight, eps)
    assert fused is not None
    ref = ref_pool_pairs_norm(prev, cur, prev_s, cur_s, weight, eps)
    torch.testing.assert_close(fused, ref, rtol=1e-2, atol=1e-2)


def test_pool_pairs_norm_disabled_falls_back(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_POOL_PAIRS_NORM", "0")
    n = 4
    t = torch.randn(n, 512, dtype=torch.float32, device="cuda")
    w = torch.ones(512, device="cuda")
    assert try_pool_pairs_norm(t, t, t, t, w, 1e-6) is None


@pytest.mark.parametrize("n", [1, 8])
def test_rope_fq_store_smoke_and_dummy_slots(n, monkeypatch):
    if not lightop_rope_fq_store_available():
        pytest.skip("lightop.dsv41_rope_fake_quant_store missing")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_ROPE_FQ_STORE", "1")
    page_size = 64
    pages = 2
    # LightOp V4: segregated values+scales, page width padded to 576.
    cache = torch.zeros(
        pages, v4_page_bytes_padded(page_size), dtype=torch.uint8, device="cuda"
    )
    baseline = cache.clone()
    latent = torch.randn(n, 512, dtype=torch.bfloat16, device="cuda")
    ang = torch.rand(n, 32, device="cuda") * 6.28
    freqs = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1).flatten(-2)
    # LightOp rope_fq_store suppresses *negative* indices only (docstring).
    # Slot-0 reservation is the index-K (#474) contract, not this store.
    indices = torch.arange(n, device="cuda", dtype=torch.int64) + 1
    indices[0] = -1
    if n > 1:
        indices[1] = -2
    ok = try_rope_fake_quant_store(latent, freqs, cache, indices, page_size)
    assert ok
    if n == 1:
        assert torch.equal(cache, baseline)
    elif n > 2:
        assert not torch.equal(cache, baseline)


def test_rope_fq_store_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_ROPE_FQ_STORE", "0")
    latent = torch.randn(2, 512, dtype=torch.bfloat16, device="cuda")
    freqs = torch.zeros(2, 64, device="cuda")
    cache = torch.zeros(1, v4_page_bytes_padded(64), dtype=torch.uint8, device="cuda")
    idx = torch.tensor([2, 3], device="cuda")
    assert try_rope_fake_quant_store(latent, freqs, cache, idx, 64) is False


def test_ref_rope_fq_roundtrip_sanity():
    """Pure-torch compressed-KV fake quant + rope stays finite (oracle helper)."""
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    ang = torch.rand(4, 32, device="cuda")
    freqs = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1).flatten(-2)
    out = _fake_quant_compressed_kv(_rope_tail(x, freqs))
    assert torch.isfinite(out.float()).all()
