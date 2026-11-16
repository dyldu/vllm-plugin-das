# SPDX-License-Identifier: Apache-2.0
"""LightOp Prefill KV fusions from SGLang #596 (KV-only slice).

Ports:
  * ``lightop.dsv41_rope_fake_quant_store`` — RoPE tail + FP4 fake quant + V4 store
  * ``lightop.dsv41_pool_pairs_norm`` — CP pair pooling + RMSNorm

Does **not** port W4A8 MegaMoE or CP OOM gather fixes from the same PR.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from .environment import (
    prefill_pool_pairs_norm_enabled,
    prefill_rope_fq_store_enabled,
)

logger = logging.getLogger(__name__)

_rope_store: Callable | None = None
_pool_norm: Callable | None = None
_ops_loaded = False
_installed = False
_original_insert_cache: Callable | None = None


def _load_ops() -> None:
    global _rope_store, _pool_norm, _ops_loaded
    if _ops_loaded:
        return
    try:
        from lightop import dsv41_pool_pairs_norm, dsv41_rope_fake_quant_store
    except (ImportError, AttributeError):
        _rope_store = None
        _pool_norm = None
    else:
        _rope_store = dsv41_rope_fake_quant_store
        _pool_norm = dsv41_pool_pairs_norm
    _ops_loaded = True


def lightop_prefill_kv_available() -> bool:
    _load_ops()
    return _rope_store is not None and _pool_norm is not None


def lightop_rope_fq_store_available() -> bool:
    _load_ops()
    return _rope_store is not None


def lightop_pool_pairs_norm_available() -> bool:
    _load_ops()
    return _pool_norm is not None


def cos_sin_to_freqs(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int,
    rope_head_dim: int = 64,
) -> torch.Tensor:
    """Gather GPT-J cos/sin rows into interleaved FP32 freqs ``[N, rope_dim]``."""
    half = rope_head_dim // 2
    group_pos = (positions // compress_ratio) * compress_ratio
    rows = cos_sin_cache[group_pos.long()]
    cos = rows[..., :half].float()
    sin = rows[..., half : half * 2].float()
    return torch.stack((cos, sin), dim=-1).flatten(-2).contiguous()


def try_pool_pairs_norm(
    previous: torch.Tensor,
    current: torch.Tensor,
    previous_score: torch.Tensor,
    score: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Return BF16 latent via LightOp, or ``None`` to fall back."""
    if not prefill_pool_pairs_norm_enabled():
        return None
    _load_ops()
    if _pool_norm is None:
        return None
    if (
        not previous.is_cuda
        or previous.dtype != torch.float32
        or current.dtype != torch.float32
        or previous.shape != current.shape
        or previous.shape[-1] != 512
        or previous.stride(-1) != 1
        or current.stride(-1) != 1
        or weight.ndim != 1
        or weight.shape[0] != 512
        or not weight.is_contiguous()
        or weight.dtype not in (torch.float32, torch.bfloat16)
    ):
        return None
    return _pool_norm(previous, current, previous_score, score, weight, eps)


def ref_pool_pairs_norm(
    previous: torch.Tensor,
    current: torch.Tensor,
    previous_score: torch.Tensor,
    score: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Torch reference matching SGLang ``pool_pairs`` + ``finish`` / RMSNorm."""
    kv2 = torch.stack((previous, current), dim=1)
    score2 = torch.stack((previous_score, score), dim=1)
    pooled = (kv2 * score2.softmax(dim=1)).sum(dim=1)
    x = pooled.bfloat16().float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * x).to(torch.bfloat16)


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


def try_rope_fake_quant_store(
    latent: torch.Tensor,
    freqs: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> bool:
    """Write V4 cache via LightOp. Returns True if the fused path ran.

    ``cache`` must be SGLang's padded segregated layout
    ``[pages, v4_page_bytes_padded(page_size)]``. Interleaved
    ``[pages, page_size, 584]`` is rejected (returns False → stock path).
    """
    if not prefill_rope_fq_store_enabled():
        return False
    _load_ops()
    if _rope_store is None:
        return False
    if (
        not latent.is_cuda
        or latent.dtype != torch.bfloat16
        or not latent.is_contiguous()
        or latent.ndim != 2
        or latent.shape[-1] != 512
        or indices.numel() != latent.shape[0]
    ):
        return False
    cache_2d = as_v4_lightop_cache(cache, int(page_size))
    if cache_2d is None:
        return False
    # LightOp only suppresses negative indices; vLLM reserves slot 0 as dummy.
    idx = indices.to(torch.int64).contiguous()
    idx = torch.where(idx > 0, idx, idx.new_full((), -1))
    _rope_store(latent, freqs, cache_2d, idx, int(page_size))
    return True


def _patched_insert_cache(self, latent, positions, rotary_emb) -> None:
    if latent is None:
        assert _original_insert_cache is not None
        return _original_insert_cache(self, latent, positions, rotary_emb)

    if not prefill_rope_fq_store_enabled() or not lightop_rope_fq_store_available():
        assert _original_insert_cache is not None
        return _original_insert_cache(self, latent, positions, rotary_emb)

    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        assert _original_insert_cache is not None
        return _original_insert_cache(self, latent, positions, rotary_emb)

    k_cache_metadata = attn_metadata[self.k_cache_prefix]
    k_cache_layer = self._static_forward_context[self.k_cache_prefix]
    kv_cache = k_cache_layer.kv_cache
    # LightOp requires SGLang's padded segregated V4 page layout. Fall back
    # when the bound cache is the common interleaved [P, S, 584] view.
    if kv_cache.dtype != torch.uint8:
        assert _original_insert_cache is not None
        return _original_insert_cache(self, latent, positions, rotary_emb)

    slot_mapping = k_cache_metadata.slot_mapping
    if kv_cache.ndim == 3:
        page_size = int(kv_cache.shape[1])
    elif kv_cache.ndim == 2:
        # Infer page_size from padded width when possible; else bail.
        page_size = 64
        for cand in (16, 32, 64, 128):
            if kv_cache.shape[1] == v4_page_bytes_padded(cand):
                page_size = cand
                break
        else:
            assert _original_insert_cache is not None
            return _original_insert_cache(self, latent, positions, rotary_emb)
    else:
        assert _original_insert_cache is not None
        return _original_insert_cache(self, latent, positions, rotary_emb)

    if as_v4_lightop_cache(kv_cache, page_size) is None and not (
        kv_cache.ndim == 2 and kv_cache.shape[1] == v4_page_bytes_padded(page_size)
    ):
        assert _original_insert_cache is not None
        return _original_insert_cache(self, latent, positions, rotary_emb)

    freqs = cos_sin_to_freqs(
        rotary_emb.cos_sin_cache,
        positions[: slot_mapping.numel()],
        compress_ratio=int(self.compress_ratio),
        rope_head_dim=int(self.rope_head_dim),
    )
    indices = slot_mapping.to(torch.int64).contiguous()
    if try_rope_fake_quant_store(
        latent[: indices.numel()].contiguous(),
        freqs,
        kv_cache,
        indices,
        page_size,
    ):
        return
    assert _original_insert_cache is not None
    return _original_insert_cache(self, latent, positions, rotary_emb)


def install_prefill_kv_lightop() -> None:
    """Wrap ``DeepseekCompressor.insert_cache`` when the rope-store gate is on."""
    global _original_insert_cache, _installed
    if _installed:
        return
    _installed = True
    if not (
        prefill_rope_fq_store_enabled() or prefill_pool_pairs_norm_enabled()
    ):
        logger.info("HCU Prefill KV LightOp fusions disabled by env")
        return
    if prefill_rope_fq_store_enabled() and not lightop_rope_fq_store_available():
        logger.warning(
            "HCU Prefill KV rope_fq_store enabled but "
            "lightop.dsv41_rope_fake_quant_store missing"
        )
    if prefill_pool_pairs_norm_enabled() and not lightop_pool_pairs_norm_available():
        logger.warning(
            "HCU Prefill KV pool_pairs_norm enabled but "
            "lightop.dsv41_pool_pairs_norm missing"
        )
    if not prefill_rope_fq_store_enabled() or not lightop_rope_fq_store_available():
        return
    try:
        from vllm.models.deepseek_v41.compressor import DeepseekCompressor
    except ImportError:
        logger.warning(
            "DeepseekCompressor unavailable; Prefill KV LightOp store not installed"
        )
        return
    _original_insert_cache = DeepseekCompressor.insert_cache
    DeepseekCompressor.insert_cache = _patched_insert_cache  # type: ignore[method-assign]
    logger.info("HCU Prefill KV LightOp rope_fq_store installed on DeepseekCompressor")


def check_prefill_kv_lightop() -> None:
    if not _installed:
        raise RuntimeError("Prefill KV LightOp fusion was never installed")


__all__ = [
    "lightop_prefill_kv_available",
    "lightop_rope_fq_store_available",
    "lightop_pool_pairs_norm_available",
    "cos_sin_to_freqs",
    "v4_page_bytes_padded",
    "as_v4_lightop_cache",
    "try_pool_pairs_norm",
    "ref_pool_pairs_norm",
    "try_rope_fake_quant_store",
    "install_prefill_kv_lightop",
    "check_prefill_kv_lightop",
]
