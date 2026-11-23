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
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFP8ScaledMMLinearKernel

logger = init_logger(__name__)

class TokenFP8TritonLinearKernel(TritonFP8ScaledMMLinearKernel):
    """Advertise the base apply_weights' existing token-FP8 input contract."""

    def input_quant_key(self):
        return kFp8DynamicTokenSym


def enable_shared_prequant(down):
    kernel = getattr(getattr(down, "scheme", None), "fp8_linear", None)
    if type(kernel) is TokenFP8TritonLinearKernel:
        return True
    if (type(kernel) is TritonFP8ScaledMMLinearKernel
            and kernel.config.activation_quant_key == kFp8DynamicTokenSym
            and kernel.fp8_dtype == torch.float8_e4m3fn
            and _quant_contract(kernel) is not None
            and not getattr(down, "requires_unquantized_input", False)):
        # Same weights, quantizer, GEMM, output shape and dtype. The superclass
        # already consumes QuantizedActivation; this unpadded dynamic-token
        # specialization supplies the missing capability declaration.
        down.scheme.fp8_linear = TokenFP8TritonLinearKernel(
            kernel.config, kernel.layer_param_names)
        expose_input_quant_key(down, down.scheme.fp8_linear)
        return True
    return False


def _quant_contract(kernel):
    quantizer = kernel.quant_fp8
    if (quantizer.use_aiter or quantizer.is_group_quant
            or quantizer.num_token_padding is not None):
        return None
    method = type(quantizer).forward_cuda
    backend = getattr(method, "_hcu_token_fp8_backend", None)
    if backend == "lightop":
        return ("lightop", method._hcu_token_fp8_graph_enabled)
    # Accept only the untouched upstream implementation. Unknown replacements
    # retain their original activation and quantization paths.
    if (method.__code__.co_filename.endswith("/input_quant_fp8.py")
            and method.__name__ == "forward_cuda"):
        return ("upstream", False)
    return None


def silu_clamp_bf16_quant(x, limit, *, lightop=False):
    from ..adapter_api import adapter_module

    return adapter_module("dsv41_shared_act_quant").silu_clamp_bf16_quant(
        x, limit, lightop=lightop)


class QuantizedSharedActivation(torch.nn.Module):
    """Use the consumer's public QuantizedActivation contract after loading."""

    def __init__(self, original, down_proj):
        super().__init__()
        self.original = original
        self._down = weakref.ref(down_proj)
        self._quant_contract = _quant_contract(down_proj.scheme.fp8_linear)
        from ..adapter_api import adapter_module

        self._quantize = adapter_module("dsv41_shared_act_quant").silu_clamp_bf16_quant

    def forward(self, x):
        down = self._down()
        key = get_input_quant_key(down) if down is not None else None
        if (key != kFp8DynamicTokenSym or not x.is_cuda or x.ndim != 2
                or x.dtype != torch.bfloat16 or not x.is_contiguous()
                or x.shape[-1] <= 0 or x.shape[-1] % 2 or x.shape[-1] > 16384):
            return self.original(x)
        if self._quant_contract is None:
            return self.original(x)
        backend, graph_enabled = self._quant_contract
        use_lightop = backend == "lightop" and not torch.compiler.is_compiling()
        if use_lightop and not graph_enabled and torch.cuda.is_current_stream_capturing():
            use_lightop = False
        if use_lightop and x.shape[-1] % 32:
            return self.original(x)
        q, scale = self._quantize(
            x, self.original.swiglu_limit, lightop=use_lightop)
        return QuantizedActivation(q, scale, x.dtype,
                                   torch.Size((x.shape[0], x.shape[1] // 2)), key)


def install_shared_activation(shared):
    """Install only the owned Channel-FP8 shared MLP's default clamped SiLU."""
    if shared is None or type(shared).__name__ != "DeepseekV4MLP":
        logger.info_once("Channel-FP8 shared activation unchanged: shared=%s",
                         type(shared).__name__)
        return False
    from ..adapter_api import adapter_module
    HcuDSV41ChannelFP8LinearMethod = adapter_module("channel_fp8_empty_linear").HcuDSV41ChannelFP8LinearMethod
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    down, act = shared.down_proj, shared.act_fn
    if isinstance(act, QuantizedSharedActivation):
        return True
    logger.info_once("Channel-FP8 shared MLP: method=%s, activation=%s",
                     type(down.quant_method).__name__, type(act).__name__)
    if (isinstance(down.quant_method, HcuDSV41ChannelFP8LinearMethod)
            and type(act) is SiluAndMulWithClamp and act.alpha == 1 and act.beta == 0):
        from ..adapter_api import adapter_module_name, optional_adapter_module

        if optional_adapter_module(adapter_module_name("dsv41_shared_act_quant")) is None:
            return False
        if not enable_shared_prequant(down):
            return False
        shared.act_fn = QuantizedSharedActivation(act, down)
        logger.info_once("Channel-FP8 shared activation consumer: %s, quant_key=%s",
                         type(down.scheme.fp8_linear).__name__, get_input_quant_key(down))
        return True
    return False


def bind_shared_expert_fp8(config, moe):
    """Use the existing model-construction seam; unsupported cases stay native."""
    import os

    if os.getenv("VLLM_HCU_USE_CUSTOM_OPS", "1").lower() in ("0", "false", "off"):
        return False
    parallel = config.parallel_config
    if (parallel.tensor_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or not torch.cuda.is_available()):
        return False
    arch = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()),
                   "gcnArchName", "").split(":", 1)[0]
    if arch != "gfx938":
        return False
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP

    shared = moe.shared_experts
    if (type(shared) is not DeepseekV4MLP or moe.use_mega_moe
            or moe.fuse_heterogeneous_shared_expert
            or getattr(shared.forward, "__func__", None) is not DeepseekV4MLP.forward):
        return False
    return install_shared_activation(shared)


def bind_dspark_shared_experts_fp8(config, model):
    """Bind each native draft MLP without changing draft routing or transport."""
    count = sum(bind_shared_expert_fp8(config, layer.ffn)
                for layer in model.layers)
    if count:
        logger.info("Channel-FP8 DSpark shared activation: %d/%d layers",
                    count, len(model.layers))
    return count
