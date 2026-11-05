# SPDX-License-Identifier: Apache-2.0
"""Local empty rows for the owned DSV4.1 CT W8A8 INT8 method.

CT owns construction, loading, finalization and every nonempty computation.
Only the audited native ROCm INT8 [K,N] layouts receive an empty fast path.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
    is_weak_contiguous,
)


@dataclass(frozen=True)
class _FinalizedINT8:
    scheme: object
    kernel: object
    weight: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    data_ptr: int
    device: torch.device
    dtype: torch.dtype


def _metadata_tensor(value, weight, shapes, dtype, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.device != weight.device
        or value.dtype != dtype
        or tuple(value.shape) not in shapes
        or not value.is_contiguous()
    ):
        raise ValueError(f"empty CT INT8 invalid {name} shape/dtype/device/layout")


@register_weight_loader_v2_supported_method
class HcuDSV41Int8LinearMethod(CompressedTensorsLinearMethod):
    """Retain native CT's constructor/loader ABI; specialize local M=0."""

    def process_weights_after_loading(self, layer):
        # Native INT8 transpose is not idempotent, including when K == N.
        # Invalidate first: a failed or repeated finalize cannot enable M=0.
        if hasattr(layer, "_hcu_int8_empty_state"):
            layer._hcu_int8_empty_state = "failed"
            layer._hcu_int8_empty_witness = None
            raise RuntimeError("CT INT8 postprocess cannot be repeated; reconstruct the layer")
        layer._hcu_int8_empty_state = "failed"
        super().process_weights_after_loading(layer)
        scheme = layer.scheme
        kernel = scheme.kernel
        if type(scheme) is CompressedTensorsW8A8Int8 and type(kernel) in (
            TritonInt8ScaledMMLinearKernel,
            AiterInt8ScaledMMLinearKernel,
        ):
            weight = layer.weight
            layer._hcu_int8_empty_witness = _FinalizedINT8(
                scheme, kernel, weight, tuple(weight.shape), tuple(weight.stride()),
                weight.data_ptr(), weight.device, weight.dtype,
            )
            layer._hcu_int8_empty_state = "ready"
        else:
            # Other backends retain native postprocess and nonempty apply.
            layer._hcu_int8_empty_state = "unsupported"

    def apply(self, layer, x, bias=None):
        if isinstance(x, QuantizedActivation):
            raise TypeError("native CT INT8 does not support QuantizedActivation")
        if not isinstance(x, torch.Tensor):
            return super().apply(layer, x, bias)
        if x.numel() != 0:
            return super().apply(layer, x, bias)
        shape = x.shape
        if len(shape) < 2 or shape[-1] <= 0:
            raise ValueError("CT INT8 input requires rank >=2 and positive K")
        if prod(shape[:-1]) != 0:
            return super().apply(layer, x, bias)

        witness = getattr(layer, "_hcu_int8_empty_witness", None)
        if getattr(layer, "_hcu_int8_empty_state", None) != "ready" or witness is None:
            raise RuntimeError("empty CT INT8 requires successful audited native postprocess")
        scheme, weight = layer.scheme, layer.weight
        kernel = scheme.kernel
        if (
            witness.scheme is not scheme
            or witness.kernel is not kernel
            or witness.weight is not weight
            or tuple(weight.shape) != witness.shape
            or tuple(weight.stride()) != witness.stride
            or weight.device != witness.device
            or weight.dtype != witness.dtype
            or weight.data_ptr() != witness.data_ptr
        ):
            raise RuntimeError("empty CT INT8 postprocess witness no longer matches the layer")
        if (
            weight.ndim != 2
            or weight.dtype != torch.int8
            or min(weight.shape) <= 0
            or weight.shape[0] != shape[-1]
            or not is_weak_contiguous(weight)
        ):
            raise ValueError("empty CT INT8 requires int8 positive weak-contiguous (K,N) weights")
        k, n = weight.shape
        if (
            k != layer.input_size_per_partition
            or n != sum(layer.output_partition_sizes)
            or n != layer.output_size_per_partition
        ):
            raise ValueError("empty CT INT8 finalized weight dimensions must match local partitions")
        if x.device != weight.device or x.dtype not in (
            torch.float16, torch.bfloat16, torch.float32
        ):
            raise ValueError("empty CT INT8 input must have supported floating dtype and weight device")
        config = kernel.config
        if (
            config.is_static_input_scheme != scheme.is_static_input_scheme
            or config.input_symmetric != scheme.input_symmetric
            or config.is_channelwise != (scheme.strategy == "channel")
        ):
            raise ValueError("empty CT INT8 kernel config must match the native scheme")
        per_channel = config.is_channelwise or len(layer.logical_widths) > 1
        weight_shapes = ((n,), (n, 1)) if per_channel else ((), (1,), (1, 1))
        _metadata_tensor(layer.weight_scale, weight, weight_shapes, torch.float32, "weight scale")
        input_scale = getattr(layer, "input_scale", None)
        input_zp = getattr(layer, "input_zero_point", None)
        azp_adj = getattr(layer, "azp_adj", None)
        if config.is_static_input_scheme:
            _metadata_tensor(input_scale, weight, ((), (1,), (1, 1)), torch.float32, "input scale")
        elif input_scale is not None:
            raise ValueError("empty CT INT8 dynamic input scale must be None")
        if config.input_symmetric:
            if input_zp is not None or azp_adj is not None:
                raise ValueError("empty CT INT8 symmetric zero point and AZP adjustment must be None")
        else:
            _metadata_tensor(azp_adj, weight, ((1, n),), torch.int32, "AZP adjustment")
            if config.is_static_input_scheme:
                _metadata_tensor(input_zp, weight, ((), (1,), (1, 1)), torch.int32, "input zero point")
            elif input_zp is not None:
                raise ValueError("empty CT INT8 dynamic input zero point must be None")
        if type(kernel) is AiterInt8ScaledMMLinearKernel and (
            not config.input_symmetric
            or config.is_static_input_scheme
            or not per_channel
            or k % 16
            or n % 16
            or x.dtype not in (torch.float16, torch.bfloat16)
        ):
            raise ValueError("empty CT INT8 AITER requires its native per-token/channel aligned FP16/BF16 contract")
        if bias is not None:
            _metadata_tensor(bias, weight, ((n,),), x.dtype, "bias")
        return torch.empty((*shape[:-1], n), dtype=x.dtype, device=x.device)
