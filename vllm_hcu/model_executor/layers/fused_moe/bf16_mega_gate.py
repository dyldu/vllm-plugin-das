# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""SGLang-align BF16 mega-gate (GEMM + sqrtsoftplus top-k) via LightOp.

Mirrors ``sglang.srt.layers.moe.topk.forward_hcu_mega_gate`` / ``hcu_bf16_mega_gate``.
Not covered by dyldu PR#1–5 (prefill-kv / inv-rope / mhc-boundary / mhc-norm-quant /
compact-topk); MoE-phase only.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _load_bf16_mega_gate_hcu() -> Callable[..., Any] | None:
    try:
        from lightop.moe import bf16_mega_gate_hcu
    except ImportError:
        return None
    if not callable(bf16_mega_gate_hcu):
        return None
    try:
        import inspect

        params = inspect.signature(bf16_mega_gate_hcu).parameters
    except (TypeError, ValueError):
        return None
    # Same capability probe as SGLang: older wheels lack logits / scale knobs.
    if not {"return_logits", "renormalize", "apply_scale"}.issubset(params):
        return None
    return bf16_mega_gate_hcu


def is_bf16_mega_gate_available() -> bool:
    return _load_bf16_mega_gate_hcu() is not None


def _inputs_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mask: torch.Tensor,
    top_k: int,
) -> bool:
    if not (
        x.is_cuda
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.ndim == 2
        and weight.shape[1] == x.shape[1]
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.is_contiguous()
        and bias is not None
        and bias.shape == (weight.shape[0],)
        and bias.dtype == torch.float32
        and bias.device == x.device
        and bias.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and bias.data_ptr() % 32 == 0
        and mask.shape == (x.shape[0],)
        and mask.dtype == torch.bool
        and mask.device == x.device
        and mask.is_contiguous()
    ):
        return False
    m, hidden = x.shape
    experts = weight.shape[0]
    if not (
        hidden > 0
        and hidden % 256 == 0
        and 0 < experts <= 512
        and experts % 4 == 0
        and 0 < top_k <= min(experts, 32)
        and max(m * hidden, hidden * experts, m * experts) <= 2**31 - 1
    ):
        return False
    return True


def can_use_bf16_mega_gate(
    hidden_states: torch.Tensor,
    gate: Any,
    router: Any,
    *,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
) -> bool:
    """Return whether fused mega-gate can replace gate GEMM + top-k."""
    del input_ids  # hash path only; we reject hash via tid2eid / table
    if hash_indices_table is not None or getattr(gate, "tid2eid", None) is not None:
        return False
    if getattr(router, "scoring_func", None) != "sqrtsoftplus":
        return False
    if bool(getattr(router, "use_grouped_topk", False)):
        return False
    top_k = int(getattr(router, "top_k", 0) or 0)
    if top_k <= 0:
        return False
    scale = float(getattr(router, "routed_scaling_factor", 1.0))
    if not math.isfinite(scale):
        return False
    weight = getattr(gate, "weight", None)
    bias = getattr(gate, "e_score_correction_bias", None)
    if weight is None or bias is None:
        return False
    if isinstance(bias, torch.nn.Parameter):
        bias = bias.data
    # All-valid mask for dense decode / cudagraph tokens (SGL prepares per-batch).
    mask = torch.ones(
        (hidden_states.shape[0],),
        dtype=torch.bool,
        device=hidden_states.device,
    )
    return _inputs_supported(hidden_states, weight, bias, mask, top_k)


def try_run_bf16_mega_gate(
    hidden_states: torch.Tensor,
    gate: Any,
    router: Any,
    *,
    indices_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run LightOp mega-gate.

    Returns ``(topk_weights, topk_ids, router_logits)`` or ``None`` to fall back.
    """
    op = _load_bf16_mega_gate_hcu()
    if op is None:
        return None
    weight = gate.weight
    bias = gate.e_score_correction_bias
    if isinstance(bias, torch.nn.Parameter):
        bias = bias.data
    top_k = int(router.top_k)
    renormalize = bool(getattr(router, "renormalize", True))
    scale = float(getattr(router, "routed_scaling_factor", 1.0))
    # Scale inside LightOp only when the router owns a non-1 factor (same as
    # fused_moe_gate / SGL apply_routed_scaling_factor_on_output). Avoids
    # double-scaling when MoERunner also multiplies the fused output.
    apply_scale = abs(scale - 1.0) > 1e-12
    mask = torch.ones(
        (hidden_states.shape[0],),
        dtype=torch.bool,
        device=hidden_states.device,
    )
    if not _inputs_supported(hidden_states, weight, bias, mask, top_k):
        return None
    try:
        # DeepGEMM / LightOp public order: ids, weights[, logits]
        ids, weights, logits = op(
            hidden_states,
            weight,
            num_topk=top_k,
            use_shared_as_routed=False,
            num_shared_experts=0,
            routed_scaling_factor=scale,
            ep_rank=0,
            scoring_func="sqrtsoftplus",
            mask=mask,
            bias=bias,
            return_logits=True,
            renormalize=renormalize,
            apply_scale=apply_scale,
        )
    except Exception as exc:  # noqa: BLE001 — fall back to stock gate path
        logger.warning_once(
            "bf16_mega_gate_hcu failed (%s); falling back to gate GEMM + top-k.",
            exc,
        )
        return None
    if indices_dtype is not None and ids.dtype != indices_dtype:
        ids = ids.to(dtype=indices_dtype)
    logger.warning_once("Using LightOp bf16_mega_gate_hcu MoE routing (SGL-align).")
    return weights, ids, logits


__all__ = [
    "can_use_bf16_mega_gate",
    "is_bf16_mega_gate_available",
    "try_run_bf16_mega_gate",
]
