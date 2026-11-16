# SPDX-License-Identifier: Apache-2.0
"""Compact TopK-union selective KV dequant (SGLang #635 slice, da3925587d).

Takes ONLY:
  * Triton ``_pack_compact_sparse_token_ids_kernel`` (ragged page-table pack)
  * LightOp ``dsv4_pack_sparse_topk_workspace`` / ``dsv4_dequantize_sparse_topk_out``

 deliberately omits mHC/CP fusion and CP in-place reorder from the same PR.

Hook point: SparsePrefill-style flat physical token IDs + combined TopK indices
feeding paged V4/V4.1 KV dequant. vLLM's ``deepseek_v4_attention._forward_prefill``
still uses block-table ``dequantize_and_gather_k_cache``; call
``try_dequantize_sparse_topk`` when a caller already holds the flat-ID contract
(or when a future SparsePrefill port lands). Install records availability only.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from .environment import compact_topk_dequant_enabled

logger = logging.getLogger(__name__)

_pack_op: Callable | None = None
_dequant_op: Callable | None = None
_ops_loaded = False
_installed = False


@triton.jit
def _pack_compact_sparse_token_ids_kernel(
    output_ptr,
    per_req_ids_ptr,
    per_req_stride_0,
    per_req_stride_1,
    compressed_lens_ptr,
    compressed_base_ptr,
    PAGE_SIZE: tl.constexpr,
    FROM_PAGE_TABLE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """Pack request-local extents into one ragged physical-token-ID array."""
    request_idx = tl.program_id(0).to(tl.int64)
    worker_idx = tl.program_id(1).to(tl.int64)
    num_workers = tl.num_programs(1).to(tl.int64)
    gather_len = tl.load(compressed_lens_ptr + request_idx).to(tl.int64)
    output_base = tl.load(compressed_base_ptr + request_idx).to(tl.int64)
    token_offsets = tl.arange(0, BLOCK_TOKENS).to(tl.int64)
    request_base = request_idx * per_req_stride_0

    for first_token in range(
        worker_idx * BLOCK_TOKENS, gather_len, num_workers * BLOCK_TOKENS
    ):
        token_idx = first_token + token_offsets
        valid = token_idx < gather_len
        if FROM_PAGE_TABLE:
            source_idx = token_idx // PAGE_SIZE
        else:
            source_idx = token_idx
        source_id = tl.load(
            per_req_ids_ptr + request_base + source_idx * per_req_stride_1,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        if FROM_PAGE_TABLE:
            token_id = source_id * PAGE_SIZE + token_idx % PAGE_SIZE
        else:
            token_id = source_id
        token_id = tl.maximum(token_id, 0).to(tl.int32)
        tl.store(output_ptr + output_base + token_idx, token_id, mask=valid)


def pack_compact_sparse_token_ids(
    per_req_page_table: torch.Tensor,
    lengths: torch.Tensor,
    bases: torch.Tensor,
    *,
    page_size: int,
    from_page_table: bool = True,
    total: int | None = None,
) -> torch.Tensor:
    """Launch the #635 Triton packer into a flat int32 physical-ID buffer."""
    assert per_req_page_table.ndim == 2 and lengths.ndim == 1 and bases.ndim == 1
    assert lengths.dtype == torch.int32 and bases.dtype == torch.int32
    num_reqs = lengths.shape[0]
    if total is None:
        total = int(lengths.sum().item())
    flat_ids = torch.empty(total, dtype=torch.int32, device=lengths.device)
    if num_reqs == 0 or total == 0:
        return flat_ids
    _pack_compact_sparse_token_ids_kernel[(num_reqs, 32)](
        flat_ids,
        per_req_page_table,
        per_req_page_table.stride(0),
        per_req_page_table.stride(1),
        lengths,
        bases,
        PAGE_SIZE=page_size,
        FROM_PAGE_TABLE=from_page_table,
        BLOCK_TOKENS=256,
    )
    return flat_ids


def _load_ops() -> tuple[Callable | None, Callable | None]:
    global _pack_op, _dequant_op, _ops_loaded
    if not _ops_loaded:
        try:
            from lightop.kvcache import (
                dsv4_dequantize_sparse_topk_out,
                dsv4_pack_sparse_topk_workspace,
            )
        except (ImportError, AttributeError):
            _pack_op = None
            _dequant_op = None
        else:
            _pack_op = dsv4_pack_sparse_topk_workspace
            _dequant_op = dsv4_dequantize_sparse_topk_out
        _ops_loaded = True
    return _pack_op, _dequant_op


def lightop_compact_topk_available() -> bool:
    pack, dequant = _load_ops()
    return callable(pack) and callable(dequant)


def v4_page_bytes_padded(page_size: int) -> int:
    """SGLang / LightOp V4 page width: ``ceil_div(P*584, 576) * 576``."""
    raw = int(page_size) * 584
    return ((raw + 575) // 576) * 576


def as_v4_lightop_cache(
    cache: torch.Tensor, page_size: int
) -> torch.Tensor | None:
    """Return contiguous ``[pages, page_bytes_padded]``, or ``None`` if incompatible."""
    if cache.dtype != torch.uint8 or not cache.is_cuda:
        return None
    want = v4_page_bytes_padded(page_size)
    if cache.ndim == 2 and cache.shape[1] == want and cache.is_contiguous():
        return cache
    return None


def try_pack_sparse_topk_workspace(
    flat_token_ids: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    topk: int,
    *,
    capacity: int | None = None,
    swa_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return LightOp packed IDs / remapped rows / GPU count, or None."""
    if not compact_topk_dequant_enabled():
        return None
    pack, _ = _load_ops()
    if pack is None:
        return None
    if (
        not flat_token_ids.is_cuda
        or flat_token_ids.dtype != torch.int32
        or combined_indices.dtype != torch.int32
        or combined_lens.dtype != torch.int32
    ):
        return None
    return pack(
        flat_token_ids,
        combined_indices,
        combined_lens,
        topk,
        capacity=capacity,
        swa_rows=swa_rows,
    )


def try_dequantize_sparse_topk(
    cache: torch.Tensor,
    flat_token_ids: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    page_size: int,
    *,
    fp8_fnuz: bool = False,
) -> torch.Tensor | None:
    """Selective TopK-union dequant into ``out``; returns GPU valid-count or None.

    LightOp requires ``flat_token_ids.numel() <= combined_indices.shape[0] * topk``
    (full-capacity workspace). Callers that exceed this fall back (``None``).
    """
    if not compact_topk_dequant_enabled():
        return None
    _, dequant = _load_ops()
    if dequant is None:
        return None
    if (
        not flat_token_ids.is_cuda
        or out.dtype != torch.bfloat16
        or flat_token_ids.dtype != torch.int32
    ):
        return None
    if flat_token_ids.numel() > combined_indices.shape[0] * int(topk):
        return None
    cache_v4 = as_v4_lightop_cache(cache, page_size)
    if cache_v4 is None:
        return None
    return dequant(
        cache_v4,
        flat_token_ids,
        combined_indices,
        combined_lens,
        out,
        topk,
        page_size,
        fp8_fnuz=fp8_fnuz,
    )


def install_compact_topk_dequant() -> None:
    """Record availability; attention wiring stays call-site opt-in (ABI differ)."""
    global _installed
    if _installed:
        return
    if not compact_topk_dequant_enabled():
        logger.info("HCU compact TopK KV dequant fusion disabled by env")
        _installed = True
        return
    if not lightop_compact_topk_available():
        logger.warning(
            "HCU compact TopK KV dequant enabled but LightOp pack/dequant ops missing"
        )
        _installed = True
        return
    logger.info(
        "HCU compact TopK KV dequant LightOp ready "
        "(hook: try_dequantize_sparse_topk / pack_compact_sparse_token_ids)"
    )
    _installed = True


def check_compact_topk_dequant() -> None:
    if not _installed:
        raise RuntimeError("Compact TopK dequant fusion was never installed")


__all__ = [
    "_pack_compact_sparse_token_ids_kernel",
    "pack_compact_sparse_token_ids",
    "lightop_compact_topk_available",
    "v4_page_bytes_padded",
    "as_v4_lightop_cache",
    "try_pack_sparse_topk_workspace",
    "try_dequantize_sparse_topk",
    "install_compact_topk_dequant",
    "check_compact_topk_dequant",
]
