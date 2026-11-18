# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""SGLang-align BF16 mega-gate (GEMM + sqrtsoftplus top-k) via LightOp.

Mirrors ``sglang.srt.layers.moe.topk.forward_hcu_mega_gate`` / ``hcu_bf16_mega_gate``.
Not covered by dyldu PR#1–5 (prefill-kv / inv-rope / mhc-boundary / mhc-norm-quant /
compact-topk); MoE-phase only.
"""

from __future__ import annotations

import math
import functools
import weakref
from typing import Any, Callable

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    _get_padding_mask,
)
from vllm_hcu.platforms import envs as henvs

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


def _invalidate_on_weight_load(gate: torch.nn.Module, weight: torch.Tensor) -> None:
    # vLLM's loaders use param.data.copy_(), which does not bump _version.
    # Invalidate at that existing boundary as well as checking storage/version.
    previous = getattr(gate, "_hcu_mega_gate_loader_weight", lambda: None)
    if previous() is weight:
        return
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    original = getattr(weight, "weight_loader", default_weight_loader)
    gate_ref = weakref.ref(gate)

    @functools.wraps(original)
    def load_and_invalidate(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            owner = gate_ref()
            if owner is not None:
                owner._hcu_mega_gate_weight_key = None

    weight.weight_loader = load_and_invalidate
    gate._hcu_mega_gate_loader_weight = weakref.ref(weight)


def _gate_weight_nt(gate: Any, hidden: int) -> torch.Tensor | None:
    """Present LightOp's [experts, hidden] layout without copying each step.

    HCU's unquantized linear loader stores NN weights [hidden, experts].
    Keep the gate's original parameter for the fallback and cache the small
    transposed copy after loading. Replacement or ordinary in-place updates
    invalidate the copy before the next call.
    """
    weight = getattr(gate, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return None
    if weight.shape[1] == hidden:
        return weight
    if not henvs.VLLM_USE_NN or weight.shape[0] != hidden:
        return None
    if not (isinstance(gate, torch.nn.Module)
            and weight.dtype == torch.bfloat16 and weight.is_contiguous()):
        return None
    # Inference tensors have no version counter. Their storage identity still
    # detects replacement; model parameters normally carry a version counter.
    try:
        version = weight._version
    except RuntimeError:
        version = None
    key = (id(weight), weight.data_ptr(), version, tuple(weight.shape), weight.device)
    if getattr(gate, "_hcu_mega_gate_weight_key", None) != key:
        packed = weight.detach().T.contiguous()
        gate.register_buffer("_hcu_mega_gate_weight_nt", packed, persistent=False)
        _invalidate_on_weight_load(gate, weight)
        gate._hcu_mega_gate_weight_key = key
    return gate._hcu_mega_gate_weight_nt


def _inputs_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mask: torch.Tensor | None,
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
        and (mask is None or (
            mask.shape == (x.shape[0],)
            and mask.dtype == torch.bool
            and mask.device == x.device
            and mask.is_contiguous()
        ))
    ):
        return False
    m, hidden = x.shape
    experts = weight.shape[0]
    if not (
        m > 0
        and hidden > 0
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
    if getattr(gate, "_vllm_hcu_deepseek_v4_gate", False):
        # That older loader path explicitly rounds router logits to BF16.
        return False
    if hash_indices_table is not None or getattr(gate, "tid2eid", None) is not None:
        return False
    if getattr(router, "scoring_func", None) != "sqrtsoftplus":
        return False
    if bool(getattr(router, "use_grouped_topk", False)):
        return False
    # Preselected routes bypass BaseRouter.select_experts. Keep features with
    # additional selection or postprocessing on that original path.
    if any(getattr(router, name, None) is not None for name in (
        "eplb_state", "capture_fn", "_hash_indices_table",
        "num_expert_group",
    )) or getattr(router, "num_fused_shared_experts", 0):
        return False
    top_k = int(getattr(router, "top_k", 0) or 0)
    if top_k <= 0:
        return False
    scale = float(getattr(router, "routed_scaling_factor", 1.0))
    if not math.isfinite(scale):
        return False
    weight = _gate_weight_nt(gate, hidden_states.shape[-1])
    bias = getattr(gate, "e_score_correction_bias", None)
    if weight is None or bias is None:
        return False
    if isinstance(bias, torch.nn.Parameter):
        bias = bias.data
    return (_inputs_supported(hidden_states, weight, bias, None, top_k)
            and _vision_inputs_supported(hidden_states, weight, router, input_ids))


def _vision_inputs_supported(x, weight, router, input_ids):
    image_bias = getattr(router, "bias_vl", None)
    if image_bias is None:
        return True
    # The correction reuses the native DSV4 six-expert router. Keep other
    # shapes and routing features on their original path.
    return (
        isinstance(image_bias, torch.Tensor)
        and image_bias.dtype == torch.float32
        and image_bias.shape == (weight.shape[0],)
        and image_bias.device == x.device and image_bias.is_contiguous()
        and weight.shape[0] in (256, 384) and router.top_k == 6
        and bool(getattr(router, "renormalize", True))
        and isinstance(getattr(router, "image_sentinel_lo", 0), int)
        and isinstance(input_ids, torch.Tensor)
        and input_ids.device == x.device and input_ids.is_contiguous()
        and input_ids.dtype in (torch.int32, torch.int64)
        and input_ids.ndim == 1 and input_ids.numel() >= x.shape[0]
    )


def try_run_bf16_mega_gate(
    hidden_states: torch.Tensor,
    gate: Any,
    router: Any,
    *,
    indices_dtype: torch.dtype | None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run LightOp mega-gate.

    Returns ``(topk_weights, topk_ids, router_logits)`` or ``None`` to fall back.
    """
    op = _load_bf16_mega_gate_hcu()
    if op is None:
        return None
    weight = _gate_weight_nt(gate, hidden_states.shape[-1])
    if weight is None:
        return None
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
    # The stock router marks CUDA Graph padding with -1 IDs / zero weights.
    # LightOp takes the inverse convention (True means a valid token) and
    # produces the same sentinels. None avoids an all-ones launch when unpadded.
    is_padding = _get_padding_mask(hidden_states.shape[0])
    mask = None if is_padding is None else torch.logical_not(is_padding)
    if not _inputs_supported(hidden_states, weight, bias, mask, top_k):
        return None
    if not _vision_inputs_supported(hidden_states, weight, router, input_ids):
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
        image_bias = getattr(router, "bias_vl", None)
        sentinel = getattr(router, "image_sentinel_lo", 0)
        if image_bias is not None and sentinel > 0:
            from hcu_vllm_das.moe.vision_gate import correct_image_routes

            correct_image_routes(logits, bias, is_padding, weights, ids, scale,
                                 input_ids, image_bias, sentinel)
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
