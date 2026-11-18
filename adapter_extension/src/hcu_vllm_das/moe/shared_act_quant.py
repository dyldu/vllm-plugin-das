# SPDX-License-Identifier: Apache-2.0
"""Shared-expert activation/FP8 fusion preserving the stock BF16 boundaries."""
from __future__ import annotations

import weakref

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    expose_input_quant_key,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym
from vllm.triton_utils import tl, triton
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFP8ScaledMMLinearKernel

logger = init_logger(__name__)

class TokenFP8TritonLinearKernel(TritonFP8ScaledMMLinearKernel):
    """Advertise the base apply_weights' existing token-FP8 input contract."""

    def input_quant_key(self):
        return kFp8DynamicTokenSym


def enable_shared_prequant(down):
    kernel = getattr(getattr(down, "scheme", None), "fp8_linear", None)
    if (type(kernel) is TritonFP8ScaledMMLinearKernel
            and kernel.config.activation_quant_key == kFp8DynamicTokenSym
            and kernel.fp8_dtype == torch.float8_e4m3fn
            and not getattr(down, "requires_unquantized_input", False)):
        # Same weights, quantizer, GEMM, output shape and dtype. The superclass
        # already consumes QuantizedActivation; this unpadded dynamic-token
        # specialization supplies the missing capability declaration.
        down.scheme.fp8_linear = TokenFP8TritonLinearKernel(
            kernel.config, kernel.layer_param_names)
        expose_input_quant_key(down, down.scheme.fp8_linear)


@triton.jit
def _silu_clamp_bf16_quant(X, Q, S, N: tl.constexpr, LIMIT: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    gate = tl.load(X + row * (2 * N) + col, col < N, other=0).to(tl.float32)
    up = tl.load(X + row * (2 * N) + N + col, col < N, other=0).to(tl.float32)
    gate = tl.minimum(gate, LIMIT)
    up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    # Stock computes (up + beta), including beta=+0: -0 + +0 becomes +0.
    # Preserve that sign rule explicitly, since a compiler may fold up + 0.
    up = tl.where(up == 0.0, 0.0, up)
    # _C.silu_and_mul_with_clamp rounds the activation, then the product.
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    value = (activated * up).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.where(col < N, tl.abs(value), 0), 0)
    scale = tl.maximum(tl.div_rn(amax, 448.0), 1.0 / (448.0 * 512.0))
    # Stock scaled_fp8_conversion<false> divides by scale. Multiplying by a
    # rounded reciprocal changes FP8 tie decisions for some BF16 rows.
    quant = tl.minimum(tl.maximum(tl.div_rn(value, scale), -448.0), 448.0)
    tl.store(Q + row * N + col, quant, col < N)
    tl.store(S + row, scale)


@triton.jit
def _shared_bf16_value(gate, up, LIMIT: tl.constexpr):
    gate = tl.minimum(gate.to(tl.float32), LIMIT)
    up = tl.minimum(tl.maximum(up.to(tl.float32), -LIMIT), LIMIT)
    up = tl.where(up == 0.0, 0.0, up)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    return (activated * up).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _bounded_bf16_quant_div(value, scale):
    # After the two BF16 boundaries with limit=10, |value| <= amax <= 100.
    # Correct the reciprocal product with its division residual. Exhaustive
    # BF16 value/amax enumeration verifies the resulting E4M3 codes against
    # div_rn; an uncorrected reciprocal changes FP8 tie decisions.
    reciprocal = 1.0 / scale
    first = value * reciprocal
    result = tl.fma(tl.fma(-first, scale, value), reciprocal, first)
    return tl.where(value == 0.0, value, result)


@triton.jit
def _silu_clamp_bf16_quant_2304(X, Q, S, LIMIT: tl.constexpr):
    # Model width is 2304. A 4096-column tile spends almost half its work on
    # padding; reduce the 2048 main columns and 256 tail separately instead.
    row = tl.program_id(0)
    col = tl.arange(0, 2048)
    tail = 2048 + tl.arange(0, 256)
    value = _shared_bf16_value(tl.load(X + row * 4608 + col),
                              tl.load(X + row * 4608 + 2304 + col), LIMIT)
    last = _shared_bf16_value(tl.load(X + row * 4608 + tail),
                             tl.load(X + row * 4608 + 2304 + tail), LIMIT)
    amax = tl.maximum(tl.max(tl.abs(value), 0), tl.max(tl.abs(last), 0))
    scale = tl.maximum(tl.div_rn(amax, 448.0), 1.0 / (448.0 * 512.0))
    if LIMIT == 10.0:
        divided = _bounded_bf16_quant_div(value, scale)
        divided_tail = _bounded_bf16_quant_div(last, scale)
    else:
        divided = tl.div_rn(value, scale)
        divided_tail = tl.div_rn(last, scale)
    quant = tl.minimum(tl.maximum(divided, -448.0), 448.0)
    quant_tail = tl.minimum(tl.maximum(divided_tail, -448.0), 448.0)
    tl.store(Q + row * 2304 + col, quant)
    tl.store(Q + row * 2304 + tail, quant_tail)
    tl.store(S + row, scale)


def silu_clamp_bf16_quant(x: torch.Tensor, limit: float):
    if (not x.is_cuda or x.ndim != 2 or x.dtype != torch.bfloat16
            or not x.is_contiguous()):
        raise ValueError("shared activation fusion requires contiguous BF16 [M,2N]")
    m, two_n = x.shape
    if two_n <= 0 or two_n % 2 or two_n > 16384:
        raise ValueError("unsupported shared activation width")
    n = two_n // 2
    q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m:
        if n == 2304:
            _silu_clamp_bf16_quant_2304[(m,)](
                x, q, s, limit, num_warps=4, enable_fp_fusion=False)
        else:
            _silu_clamp_bf16_quant[(m,)](
                x, q, s, n, limit, triton.next_power_of_2(n),
                num_warps=4, enable_fp_fusion=False)
    return q, s


class QuantizedSharedActivation(torch.nn.Module):
    """Use the consumer's public QuantizedActivation contract after loading."""

    def __init__(self, original, down_proj):
        super().__init__()
        self.original = original
        self._down = weakref.ref(down_proj)

    def forward(self, x):
        down = self._down()
        key = get_input_quant_key(down) if down is not None else None
        if (key != kFp8DynamicTokenSym or not x.is_cuda or x.ndim != 2
                or x.dtype != torch.bfloat16 or not x.is_contiguous()
                or x.shape[-1] <= 0 or x.shape[-1] % 2 or x.shape[-1] > 16384):
            return self.original(x)
        q, scale = silu_clamp_bf16_quant(x, self.original.swiglu_limit)
        return QuantizedActivation(q, scale, x.dtype,
                                   torch.Size((x.shape[0], x.shape[1] // 2)), key)


def install_shared_activation(shared):
    """Install only the owned Channel-FP8 shared MLP's default clamped SiLU."""
    if shared is None or type(shared).__name__ != "DeepseekV4MLP":
        logger.info_once("Channel-FP8 shared activation unchanged: shared=%s",
                         type(shared).__name__)
        return
    from hcu_adapter.channel_fp8_empty_linear import HcuDSV41ChannelFP8LinearMethod
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    down, act = shared.down_proj, shared.act_fn
    logger.info_once("Channel-FP8 shared MLP: method=%s, activation=%s",
                     type(down.quant_method).__name__, type(act).__name__)
    if (isinstance(down.quant_method, HcuDSV41ChannelFP8LinearMethod)
            and type(act) is SiluAndMulWithClamp and act.alpha == 1 and act.beta == 0):
        enable_shared_prequant(down)
        shared.act_fn = QuantizedSharedActivation(act, down)
        logger.info_once("Channel-FP8 shared activation consumer: %s, quant_key=%s",
                         type(down.scheme.fp8_linear).__name__, get_input_quant_key(down))
