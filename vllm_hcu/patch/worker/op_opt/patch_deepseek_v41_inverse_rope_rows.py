# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route V4.1 standalone inverse-RoPE rows through the HCU CUDA suffix op.

AITER sparse MLA leaves inverse RoPE to ``rocm_inverse_rope_rows_``, which
defaults to a Triton GPT-J kernel (~0.87 ms / 43 layers at decode bs=8).
SGLang uses a flat rotary kernel on the same subgraph. The existing HCU
``deepseek_v4_inv_rope`` CUDA kernel already mirrors that warp-per-row layout
for the 64-dim RoPE tail; rebinding here closes gap ④ without changing the
attention epilogue contract (``inverse_rope=False`` in ``_o_proj``).
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm_hcu.platforms import envs as henvs

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.amd.rocm"
SPARSE_MODULE = "vllm.v1.attention.ops.rocm_aiter_mla_sparse"
PATCH_ID = "worker.op_opt.deepseek_v41.inverse_rope_rows"
TARGET_SYMBOL = f"{TARGET_MODULE}.rocm_inverse_rope_rows_"
TARGETS = (
    TARGET_SYMBOL,
    f"{SPARSE_MODULE}.rocm_inverse_rope_rows_",
)
_MARKER = "_vllm_hcu_dsv41_inverse_rope_rows_applied"
_WRAPPER = "_vllm_hcu_dsv41_inverse_rope_rows_wrapper"
logger = init_logger(__name__)


def _register_inverse_rope_fake() -> bool:
    try:
        import vllm_hcu.hcu_ops  # noqa: F401

        getattr(torch.ops.hcu_ops, "deepseek_v4_inv_rope")
    except (ImportError, AttributeError):
        return False

    def deepseek_v4_inv_rope_fake(rope, position_ids, cos_sin_cache):
        return None

    try:
        torch.library.register_fake("hcu_ops::deepseek_v4_inv_rope")(
            deepseek_v4_inv_rope_fake
        )
    except RuntimeError as exc:
        if "already has" not in str(exc) and "already registered" not in str(exc):
            raise
    return True


def _eligible(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> bool:
    return (
        henvs.optional_custom_op_enabled(henvs.VLLM_HCU_INVERSE_ROPE_SUFFIX)
        and torch.version.hip is not None
        and not torch.is_grad_enabled()
        and o.is_cuda
        and o.dtype == torch.bfloat16
        and o.ndim == 3
        and o.shape[0] > 0
        and o.shape[-1] >= rope_head_dim
        and rope_head_dim == 64
        and o.stride(-1) == 1
        and positions.device == o.device
        and positions.ndim == 1
        and positions.shape[0] == o.shape[0]
        and positions.is_contiguous()
        and positions.dtype in (torch.int32, torch.int64)
        and cos_sin_cache.device == o.device
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.shape[1] == rope_head_dim
        and cos_sin_cache.stride(-1) == 1
    )


def _wrap(original):
    @functools.wraps(original)
    def inverse_rope_rows_(o, positions, cos_sin_cache, rope_head_dim):
        if o.shape[0] == 0:
            return
        if not _eligible(o, positions, cos_sin_cache, rope_head_dim):
            return original(o, positions, cos_sin_cache, rope_head_dim)
        # Suffix-only CUDA op: NoPE lanes are already bf16 and pass through.
        torch.ops.hcu_ops.deepseek_v4_inv_rope(
            o[..., -rope_head_dim:], positions, cos_sin_cache
        )

    setattr(inverse_rope_rows_, _WRAPPER, True)
    return inverse_rope_rows_


def _install(owner: ModuleType, symbol: str) -> bool:
    original = require_callable(owner, "rocm_inverse_rope_rows_", symbol)
    if getattr(original, _WRAPPER, False):
        return False
    if getattr(owner, _MARKER, False) and not getattr(original, _WRAPPER, False):
        raise PatchCompatibilityError(f"stale patch for {symbol}")
    require_exact_signature(
        original,
        symbol,
        positional=("o", "positions", "cos_sin_cache", "rope_head_dim"),
    )
    owner.rocm_inverse_rope_rows_ = _wrap(original)
    setattr(owner, _MARKER, True)
    return True


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    _register_inverse_rope_fake()
    changed = _install(rocm, TARGET_SYMBOL)
    try:
        sparse = load_exact_module(SPARSE_MODULE, None)
    except PatchCompatibilityError:
        sparse = None
    if sparse is not None:
        changed = _install(sparse, f"{SPARSE_MODULE}.rocm_inverse_rope_rows_") or changed
    if changed:
        logger.info_once(
            "DeepSeek V4.1 routes inverse-RoPE rows through the HCU CUDA suffix op"
        )
    return changed


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "SPARSE_MODULE",
    "apply",
    "apply_to_module",
    "_eligible",
]
