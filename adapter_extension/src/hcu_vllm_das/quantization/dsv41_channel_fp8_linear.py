# SPDX-License-Identifier: Apache-2.0
"""Skip only local empty-row computation for the distinct DSV4.1 FP8 method.

CT still owns weight creation/finalization and every nonempty computation. The
outer Linear forward still runs, including its distributed collectives. Native
QuantizedActivation replaces the historical prequantized-input wrapper.
"""
from __future__ import annotations

from math import prod

import torch
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)


@register_weight_loader_v2_supported_method
class HcuDSV41ChannelFP8LinearMethod(CompressedTensorsLinearMethod):
    """Keep CT's fused weight-loader ABI; specialize only empty local inputs."""

    def apply(self, layer, x: torch.Tensor | QuantizedActivation, bias=None):
        shape = x.orig_shape if isinstance(x, QuantizedActivation) else x.shape
        if len(shape) < 2 or prod(shape[:-1]) != 0:
            return super().apply(layer, x, bias)

        # Native CT channel finalization defines the physical (K, N) layout.
        # Missing state is a lifecycle error; never guess an output dimension.
        if (
            not hasattr(layer, "processed_weight_dims")
            or layer.processed_weight_dims != (0, 1)
        ):
            raise RuntimeError("empty Channel-FP8 requires processed (K, N) weights")
        kernel = layer.scheme.fp8_linear
        weight = layer.weight
        if (
            weight.ndim != 2
            or weight.shape[0] <= 0
            or weight.shape[1] <= 0
            or shape[-1] != weight.shape[0]
        ):
            raise ValueError("empty Channel-FP8 input K must match positive (K, N) weights")
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
            is_weak_contiguous,
        )

        if not is_weak_contiguous(weight):
            raise ValueError("empty Channel-FP8 weight layout must be weak-contiguous")
        weight_scale = layer.weight_scale
        if (
            not isinstance(weight_scale, torch.Tensor)
            or weight_scale.device != weight.device
            or not weight_scale.is_floating_point()
            or tuple(weight_scale.shape)
            not in ((), (1,), (1, 1), (weight.shape[1],), (weight.shape[1], 1))
        ):
            raise ValueError("empty Channel-FP8 weight scale must be scalar or per-channel")
        qa = as_quantized_activation(x, kernel.input_quant_key())
        data = x if qa is None else qa.data
        if data.device != weight.device:
            raise ValueError("empty Channel-FP8 input and weight device must match")
        if qa is not None:
            if data.ndim != 2 or tuple(data.shape) != (0, weight.shape[0]):
                raise ValueError("empty Channel-FP8 quantized data must have shape (0, K)")
            if data.dtype != kernel.fp8_dtype:
                raise ValueError("empty Channel-FP8 quantized dtype must match the kernel")
            if (
                qa.scale.device != data.device
                or qa.scale.dtype != torch.float32
                or tuple(qa.scale.shape) not in ((), (1,), (0,), (1, 1), (0, 1))
            ):
                raise ValueError("empty Channel-FP8 scale must be scalar or per-token FP32")
        elif not data.is_floating_point():
            raise ValueError("empty Channel-FP8 input dtype must be floating point")

        original_dtype = data.dtype if qa is None else qa.orig_dtype
        out_dtype = kernel.config.out_dtype
        if out_dtype is None:
            out_dtype = original_dtype
        if bias is not None and (
            bias.ndim != 1
            or bias.shape[0] != weight.shape[1]
            or bias.device != data.device
            or not bias.is_floating_point()
        ):
            raise ValueError("empty Channel-FP8 bias must match output width/device and float")
        return torch.empty((*shape[:-1], weight.shape[1]), dtype=out_dtype, device=data.device)
