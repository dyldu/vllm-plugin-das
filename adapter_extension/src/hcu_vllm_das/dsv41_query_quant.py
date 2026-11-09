"""Fuse Q/KV RMSNorm with per-token Q FP8 for the selected quantizer.

Q rounds to the activation dtype before quantization. Both projections share
that FP8 operand; KV keeps the upstream BF16 RMSNorm contract.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_rmsnorm_fp8_kernel(
    query_ptr, kv_ptr, query_weight_ptr, kv_weight_ptr,
    query_out_ptr, scale_ptr, kv_out_ptr,
    query_stride, kv_stride, epsilon,
    QUERY_SIZE: tl.constexpr, KV_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    LIGHTOP_QUANT: tl.constexpr,
):
    token = tl.program_id(0)
    task = tl.program_id(1)
    columns = tl.arange(0, BLOCK_SIZE)
    if task == 0:
        width = QUERY_SIZE
        input_ptr = query_ptr + token * query_stride
        weight_ptr = query_weight_ptr
    else:
        width = KV_SIZE
        input_ptr = kv_ptr + token * kv_stride
        weight_ptr = kv_weight_ptr
    valid = columns < width
    weight = tl.load(weight_ptr + columns, valid, other=0).to(tl.float32)
    values = tl.load(input_ptr + columns, valid, other=0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / width
    normalized = values * tl.rsqrt(variance + epsilon) * weight
    rounded = normalized.to(query_ptr.dtype.element_ty).to(tl.float32)
    if task == 0:
        absmax = tl.max(tl.abs(rounded), axis=0)
        scale = tl.div_rn(absmax, 448.0)
        if LIGHTOP_QUANT:
            inverse = tl.where(absmax == 0.0, 0.0, tl.div_rn(448.0, absmax))
            scaled = rounded * inverse
        else:
            scale = tl.maximum(scale, 1.0 / (448.0 * 512.0))
            scaled = tl.div_rn(rounded, scale)
        quantized = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
        tl.store(query_out_ptr + token * QUERY_SIZE + columns,
                 quantized.to(query_out_ptr.dtype.element_ty), valid)
        tl.store(scale_ptr + token, scale)
    else:
        tl.store(kv_out_ptr + token * KV_SIZE + columns,
                 rounded.to(kv_out_ptr.dtype.element_ty), valid)


def rmsnorm_quant(
    query: torch.Tensor, kv: torch.Tensor, query_weight: torch.Tensor,
    kv_weight: torch.Tensor, epsilon: float, lightop_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output, scale, kv_output = rmsnorm_quant_fake(
        query, kv, query_weight, kv_weight, epsilon, lightop_quant,
    )
    if query.shape[0]:
        block = triton.next_power_of_2(max(query.shape[1], kv.shape[1]))
        _qkv_rmsnorm_fp8_kernel[(query.shape[0], 2)](
            query, kv, query_weight, kv_weight, output, scale, kv_output,
            query.stride(0), kv.stride(0), epsilon,
            QUERY_SIZE=query.shape[1], KV_SIZE=kv.shape[1], BLOCK_SIZE=block,
            LIGHTOP_QUANT=lightop_quant,
            num_warps=8 if block >= 2048 else 4,
        )
    return output, scale, kv_output


def rmsnorm_quant_fake(
    query: torch.Tensor, kv: torch.Tensor, query_weight: torch.Tensor,
    kv_weight: torch.Tensor, epsilon: float, lightop_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(query.shape, device=query.device, dtype=torch.float8_e4m3fn),
        torch.empty((query.shape[0], 1), device=query.device, dtype=torch.float32),
        torch.empty(kv.shape, device=kv.device, dtype=kv.dtype),
    )


_REGISTERED = False


def register_op():
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="hcu_dsv41_query_rmsnorm_quant", op_func=rmsnorm_quant,
        mutates_args=[], fake_impl=rmsnorm_quant_fake,
        dispatch_key="CompositeExplicitAutograd",
    )
    _REGISTERED = True
