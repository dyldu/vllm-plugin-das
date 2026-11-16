# SPDX-License-Identifier: Apache-2.0
"""GPU accuracy: compact pack + LightOp selective TopK dequant (#635 slice)."""

from __future__ import annotations

import pytest
import torch

from hcu_vllm_das.compact_topk_dequant_lightop import (
    lightop_compact_topk_available,
    pack_compact_sparse_token_ids,
    try_dequantize_sparse_topk,
    try_pack_sparse_topk_workspace,
    v4_page_bytes_padded,
)
from lightop.kvcache import dequantize_k_cache_paged_hcu_counted

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="HCU/HIP required",
)


def _torch_pack(page_table: torch.Tensor, lengths: torch.Tensor, page_size: int):
    """Reference ragged pack matching FROM_PAGE_TABLE=True."""
    outs = []
    for req in range(page_table.shape[0]):
        n = int(lengths[req].item())
        ids = []
        for t in range(n):
            src = int(page_table[req, t // page_size].item())
            tid = max(src * page_size + (t % page_size), 0)
            ids.append(tid)
        outs.extend(ids if ids else [])
    return torch.tensor(outs, dtype=torch.int32, device=page_table.device)


@pytest.mark.parametrize(
    "num_reqs,page_size,lens",
    [
        (1, 4, [7]),
        (2, 8, [8, 16]),
        (3, 4, [1, 5, 12]),
    ],
)
def test_pack_compact_matches_torch(num_reqs, page_size, lens, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_COMPACT_TOPK_DEQUANT", "1")
    device = "cuda"
    max_blocks = (max(lens) + page_size - 1) // page_size
    g = torch.Generator(device="cpu").manual_seed(num_reqs * 17 + page_size)
    page_table = torch.randint(
        0, 32, (num_reqs, max_blocks), generator=g, dtype=torch.int32
    ).to(device)
    lengths = torch.tensor(lens, dtype=torch.int32, device=device)
    bases = torch.cumsum(lengths, dim=0, dtype=torch.int32) - lengths
    packed = pack_compact_sparse_token_ids(
        page_table, lengths, bases, page_size=page_size, total=int(sum(lens))
    )
    ref = _torch_pack(page_table, lengths, page_size)
    torch.testing.assert_close(packed.cpu(), ref.cpu(), rtol=0, atol=0)


def _finite_v4_cache(num_pages: int, page_size: int) -> torch.Tensor:
    """Padded segregated V4 cache with finite dequant (stable UE8M0 scales)."""
    stride = v4_page_bytes_padded(page_size)
    cache = torch.zeros((num_pages, stride), dtype=torch.uint8, device="cuda")
    cache[:, : page_size * 576] = 64
    cache[:, page_size * 576 : page_size * 576 + page_size * 8] = 120
    return cache


@pytest.mark.skipif(not lightop_compact_topk_available(), reason="LightOp ops missing")
@pytest.mark.parametrize(
    "capacity,topk,queries",
    [
        # Must satisfy full_ids.numel() <= queries * topk (LightOp contract).
        (32, 8, 4),
        (64, 16, 8),
    ],
)
def test_pack_workspace_and_selective_dequant(capacity, topk, queries, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_COMPACT_TOPK_DEQUANT", "1")
    page_size = 64
    flat = torch.arange(capacity, device="cuda", dtype=torch.int32)
    g = torch.Generator(device="cpu").manual_seed(capacity + topk)
    combined = torch.randint(
        0, capacity, (queries, topk), generator=g, dtype=torch.int32
    ).cuda()
    combined[:, -1] = -1
    lens = torch.full((queries,), topk - 1, dtype=torch.int32, device="cuda")
    packed = try_pack_sparse_topk_workspace(
        flat, combined, lens, topk, capacity=capacity
    )
    assert packed is not None
    packed_ids, remapped, count = packed
    assert packed_ids.dtype == torch.int32 and remapped.dtype == torch.int32
    assert count.dtype == torch.int32 and count.numel() == 1

    cache = _finite_v4_cache(
        num_pages=max(2, (capacity + page_size - 1) // page_size), page_size=page_size
    )
    out = torch.zeros((capacity, 1, 512), dtype=torch.bfloat16, device="cuda")
    valid = try_dequantize_sparse_topk(
        cache, flat, combined, lens, out, topk, page_size, fp8_fnuz=False
    )
    assert valid is not None
    assert valid.dtype == torch.int32 and valid.numel() == 1

    out_full = torch.zeros_like(out)
    dequantize_k_cache_paged_hcu_counted(
        cache,
        flat,
        out_full,
        page_size,
        False,
        torch.tensor([capacity], dtype=torch.int32, device="cuda"),
    )
    selected = combined[combined >= 0].unique()
    assert selected.numel() > 0
    torch.testing.assert_close(
        out[selected].float(), out_full[selected].float(), rtol=0, atol=0
    )
    assert torch.isfinite(out[selected].float()).all()
