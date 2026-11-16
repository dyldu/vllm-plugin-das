# SPDX-License-Identifier: Apache-2.0
"""Fuse MHC layer RMSNorm with per-token FP8/INT8 quant (SGLang #648 slice).

Local ``hcu_mhc_boundary`` already completes Sinkhorn inside the boundary launch.
This module ports the remaining #648 win: BF16 RMSNorm → per-token quant in one
kernel so ReplicatedLinear / shared-expert paths can reuse the operand without
rewriting the BF16 norm to HBM and reading it back.

API mirrors ``rmsnorm_per_token_{fp8,int8}_with_sinkhorn``'s quant half:
  FP8  → (q e4m3, scale[M,1] fp32, orig_dtype), norm bf16
  INT8 → (q int8, scale[M,1] fp32), norm bf16
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .environment import mhc_norm_quant_enabled


@triton.jit
def _rmsnorm_per_token_quant_kernel(
    x_ptr,
    w_ptr,
    norm_ptr,
    q_ptr,
    scale_ptr,
    stride_x,
    stride_norm,
    stride_q,
    eps,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    Q_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
    INT8: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    y = x * tl.rsqrt(var + eps) * w
    # Match ir.ops.rms_norm / #648: round to storage dtype before amax.
    y_store = y.to(norm_ptr.dtype.element_ty)
    tl.store(norm_ptr + row * stride_norm + cols, y_store, mask=mask)
    yf = y_store.to(tl.float32)
    amax = tl.max(tl.abs(yf), axis=0)
    if INT8:
        scale = amax / Q_MAX
        inv = tl.where(amax == 0.0, 0.0, Q_MAX / amax)
        scaled = yf * inv
        qv = tl.minimum(tl.maximum(scaled, -Q_MAX), Q_MAX)
        qi = (qv + tl.where(qv >= 0, 0.5, -0.5)).to(tl.int32).to(tl.int8)
        tl.store(q_ptr + row * stride_q + cols, qi, mask=mask)
        tl.store(scale_ptr + row, scale)
    else:
        scale = tl.maximum(amax / Q_MAX, MIN_SCALE)
        scaled = tl.div_rn(yf, scale)
        qv = tl.minimum(tl.maximum(scaled, -Q_MAX), Q_MAX)
        tl.store(
            q_ptr + row * stride_q + cols,
            qv.to(q_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(scale_ptr + row, scale)


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    q_dtype: torch.dtype,
    q_max: float,
    min_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2 and weight.ndim == 1 and weight.shape[0] == x.shape[1]
    assert x.is_cuda and weight.device == x.device
    x = x.contiguous()
    weight = weight.contiguous()
    m, h = x.shape
    norm = torch.empty_like(x)
    q = torch.empty((m, h), device=x.device, dtype=q_dtype)
    scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    if m == 0:
        return norm, q, scale
    block = triton.next_power_of_2(h)
    _rmsnorm_per_token_quant_kernel[(m,)](
        x,
        weight,
        norm,
        q,
        scale,
        x.stride(0),
        norm.stride(0),
        q.stride(0),
        eps,
        H=h,
        BLOCK=block,
        Q_MAX=q_max,
        MIN_SCALE=min_scale,
        INT8=q_dtype == torch.int8,
        num_warps=8 if block >= 2048 else 4,
    )
    return norm, q, scale


def rmsnorm_per_token_fp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.dtype], torch.Tensor]:
    """Return ``((q, scale, x.dtype), norm)`` — #648 FP8 operand layout."""
    norm, q, scale = _launch(
        x,
        weight,
        eps,
        q_dtype=torch.float8_e4m3fn,
        q_max=448.0,
        min_scale=1.0 / (448.0 * 512.0),
    )
    return (q, scale, x.dtype), norm


def rmsnorm_per_token_int8(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return ``((q, scale), norm)`` — #648 INT8 operand layout."""
    norm, q, scale = _launch(
        x, weight, eps, q_dtype=torch.int8, q_max=127.0, min_scale=0.0
    )
    return (q, scale), norm


def apply_mhc_norm_quant(
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    *,
    int8: bool = False,
):
    """Drop-in for ``_apply_mhc_norm`` that also returns a quant operand or None."""
    if norm_weight is None:
        return layer_input, None
    if not mhc_norm_quant_enabled() or not layer_input.is_cuda:
        from vllm.model_executor.layers import mhc as native

        return native._apply_mhc_norm(layer_input, norm_weight, norm_eps), None
    if int8:
        operand, norm = rmsnorm_per_token_int8(layer_input, norm_weight, norm_eps)
    else:
        operand, norm = rmsnorm_per_token_fp8(layer_input, norm_weight, norm_eps)
    return norm, operand


__all__ = [
    "rmsnorm_per_token_fp8",
    "rmsnorm_per_token_int8",
    "apply_mhc_norm_quant",
]
