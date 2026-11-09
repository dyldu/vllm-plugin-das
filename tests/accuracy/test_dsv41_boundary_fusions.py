"""Numerical, dynamic-input and graph checks for DSV4.1 boundary fusions.

Adapted for the 033 layout: boundary/query live in ``hcu_vllm_das``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is None,
        reason="requires HCU",
    ),
]


def _boundary_mod():
    from hcu_vllm_das.dsv41_mhc_boundary import boundary

    return boundary


def _query_mod():
    from hcu_vllm_das.dsv41_query_quant import rmsnorm_quant

    return rmsnorm_quant


def make_boundary_inputs(tokens):
    torch.manual_seed(42)
    options = {"device": "cuda", "dtype": torch.bfloat16}
    return (
        torch.randn(tokens, 4, 5120, **options),
        torch.randn(tokens, 5120, **options),
        torch.rand(tokens, 4, 1, device="cuda"),
        torch.softmax(torch.rand(tokens, 4, 4, device="cuda"), dim=-1),
        torch.rand(tokens, 4, device="cuda"),
        torch.randn(24, 20480, device="cuda") * 0.005,
        torch.tensor([0.1, 0.2, 0.3], device="cuda"),
        torch.randn(24, device="cuda") * 0.1,
    )


def boundary_reference(inputs, scalar_args):
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch
    from vllm.model_executor.layers.mhc import mhc_kernels

    residual, sublayer, post, comb, pre, weight, scale, base = inputs
    updated = mhc_kernels.mhc_post_torch(sublayer, residual, post, comb)
    return (updated, *mhc_pre_delayed_torch(
        updated, weight, scale, base, *scalar_args, pre_mix=pre,
    ))


def assert_boundary_close(actual, expected):
    for index, (output, reference) in enumerate(zip(actual, expected)):
        if index == 0:
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
        elif index == 3:
            error = (output.float() - reference.float()).square().mean().sqrt()
            rms = reference.float().square().mean().sqrt().clamp_min(1e-5)
            assert error / rms < 2e-5
        else:
            torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("tokens", [1, 8, 48, 64])
@pytest.mark.parametrize("pre_eps,sinkhorn_eps,post_mult,iters", [
    (1e-6, 1e-6, 2.0, 20), (1e-4, 1e-5, 1.5, 3),
])
def test_mhc_boundary_uses_previous_mix(tokens, pre_eps, sinkhorn_eps, post_mult, iters):
    boundary = _boundary_mod()
    inputs = make_boundary_inputs(tokens)
    scalar_args = (1e-6, pre_eps, sinkhorn_eps, post_mult, iters)
    actual = boundary(*inputs, *scalar_args)
    assert_boundary_close(actual, boundary_reference(inputs, scalar_args))


@pytest.mark.parametrize("lightop_quant", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 8, 48, 64])
@pytest.mark.parametrize("magnitude", [0.0, 1e-7, 1.0, 100.0])
def test_query_quant_matches_separate_bf16_norm_and_fp8(tokens, magnitude, lightop_quant):
    from vllm.models.common.ops import fused_q_kv_rmsnorm

    rmsnorm_quant = _query_mod()
    torch.manual_seed(20261105)
    packed = torch.randn(tokens, 1792, device="cuda", dtype=torch.bfloat16) * magnitude
    query, kv = packed.split((1280, 512), dim=-1)
    query_weight = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    original = packed.clone()
    actual, scale, kv_output = rmsnorm_quant(
        query, kv, query_weight, kv_weight, 1e-6, lightop_quant,
    )
    query_norm, expected_kv = fused_q_kv_rmsnorm(
        query, kv, query_weight, kv_weight, 1e-6,
    )
    if tokens:
        if lightop_quant:
            from lightop.quant import per_token_quant_fp8

            expected, expected_scale = per_token_quant_fp8(
                query_norm, dtype=torch.float8_e4m3fn,
            )
        else:
            from vllm._custom_ops import scaled_fp8_quant

            expected, expected_scale = scaled_fp8_quant(
                query_norm, use_per_token_if_dynamic=True,
            )
        torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)
    torch.testing.assert_close(kv_output, expected_kv, atol=0, rtol=0)
    torch.testing.assert_close(packed, original, atol=0, rtol=0)


def test_query_eligible_rejects_aiter_block_scaled(monkeypatch):
    """Regression guard for the HumanEval 0.9451 path (group-quant override)."""
    from types import SimpleNamespace

    from hcu_vllm_das import v41_query_quant_activation as activation

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_DSV41_QUERY_RMSNORM_FP8", "1")
    packed = torch.empty(8, 1792, dtype=torch.bfloat16)
    attention = SimpleNamespace(
        _wq_b_uses_aiter_block_scaled=True,
        q_lora_rank=1280,
        head_dim=512,
    )
    assert activation.eligible(attention, packed) is False


def test_prepare_decode_indices_module_imports():
    from vllm_hcu.v1.attention.ops.deepseek_v41_decode_indices import (
        prepare_decode_indices,
    )

    assert callable(prepare_decode_indices)
