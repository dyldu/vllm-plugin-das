# SPDX-License-Identifier: Apache-2.0
"""LightOp full counted KV dequant (#7 measured win vs vLLM gather).

Fair microbench (~3.5×): ``lightop.kvcache.dequantize_k_cache_paged_hcu_counted``
vs vLLM ``dequantize_and_gather_k_cache`` on a flat physical-token-ID contract.

vLLM's production prefill still feeds a block-table gather. This module exposes
the flat-ID LightOp path for callers that already hold that ABI (and for benches).
Install is opt-in via ``VLLM_HCU_USE_LIGHTOP_KV_FULL_DEQUANT`` and does not wrap
the block-table gather by default — converting tables→flat IDs each call would
eat the measured win.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch

from .environment import kv_full_dequant_enabled

logger = logging.getLogger(__name__)

_dequant: Callable | None = None
_ops_loaded = False
_installed = False


def _load_ops() -> None:
    global _dequant, _ops_loaded
    if _ops_loaded:
        return
    try:
        from lightop.kvcache import dequantize_k_cache_paged_hcu_counted
    except (ImportError, AttributeError):
        _dequant = None
    else:
        _dequant = dequantize_k_cache_paged_hcu_counted
    _ops_loaded = True


def lightop_kv_full_dequant_available() -> bool:
    _load_ops()
    return _dequant is not None


def v4_page_bytes_padded(page_size: int) -> int:
    """Padded V4 page byte width: ``ceil_div(P·584, 576) · 576``."""
    raw = int(page_size) * 584
    return ((raw + 575) // 576) * 576


def as_v4_lightop_cache(
    cache: torch.Tensor, page_size: int
) -> torch.Tensor | None:
    """View cache as LightOp ``[pages, page_bytes]`` when layout matches."""
    if cache.ndim == 2:
        return cache.contiguous()
    if cache.ndim == 3 and cache.shape[-1] == 584:
        pages = cache.shape[0]
        return cache.reshape(pages, int(page_size) * 584).contiguous()
    padded = v4_page_bytes_padded(page_size)
    if cache.ndim == 2 and cache.shape[-1] == padded:
        return cache
    return None


def try_dequantize_full_counted(
    cache: torch.Tensor,
    flat_token_ids: torch.Tensor,
    out: torch.Tensor,
    page_size: int,
    *,
    fp8_fnuz: bool = False,
    valid_lens: torch.Tensor | None = None,
) -> bool:
    """Run LightOp full counted dequant when the gate and layout allow.

    ``flat_token_ids``: int32 physical token IDs (or -1).
    ``out``: ``[T, 1, 512]`` BF16 (LightOp ABI) or ``[T, 512]`` (viewed as 1 head).
    """
    if not kv_full_dequant_enabled() or not lightop_kv_full_dequant_available():
        return False
    cache_2d = as_v4_lightop_cache(cache, page_size)
    if cache_2d is None:
        return False
    if out.ndim == 2:
        out_view = out.unsqueeze(1)
    elif out.ndim == 3 and out.shape[1] == 1:
        out_view = out
    else:
        return False
    assert _dequant is not None
    if valid_lens is None:
        valid_lens = torch.tensor(
            [flat_token_ids.numel()], dtype=torch.int32, device=flat_token_ids.device
        )
    _dequant(cache_2d, flat_token_ids, out_view, page_size, fp8_fnuz, valid_lens)
    return True


def install_kv_full_dequant() -> None:
    """Record readiness; call sites opt into ``try_dequantize_full_counted``."""
    global _installed
    if _installed:
        return
    if not kv_full_dequant_enabled():
        logger.info("HCU LightOp full KV dequant disabled by env")
        _installed = True
        return
    if not lightop_kv_full_dequant_available():
        logger.warning(
            "HCU LightOp full KV dequant enabled but "
            "dequantize_k_cache_paged_hcu_counted missing"
        )
        _installed = True
        return
    logger.info(
        "HCU LightOp full KV dequant ready "
        "(hook: try_dequantize_full_counted; block-table gather stays stock)"
    )
    _installed = True


def check_kv_full_dequant() -> None:
    if not _installed:
        raise RuntimeError("LightOp full KV dequant was never installed")


__all__ = [
    "lightop_kv_full_dequant_available",
    "v4_page_bytes_padded",
    "as_v4_lightop_cache",
    "try_dequantize_full_counted",
    "install_kv_full_dequant",
    "check_kv_full_dequant",
]
