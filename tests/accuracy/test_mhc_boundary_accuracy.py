# SPDX-License-Identifier: Apache-2.0
"""Accuracy: fused MHC boundary vs unfused Post+collapse+Sinkhorn.

Residual uses SGLang-style bf16 ULP gate (<=1 ulp, excluding near-zero).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is None,
        reason="requires HCU/HIP",
    ),
]


def _unfused_boundary_torch(
    residual: torch.Tensor,
    sublayer_out: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    pre_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Unfused Post → collapse(previous pre) → project+Sinkhorn(next coeffs)."""
    try:
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
    except ImportError:  # pragma: no cover - older layouts
        from vllm.model_executor.layers.mhc import mhc_kernels

        mhc_post_torch = mhc_kernels.mhc_post_torch

    updated = mhc_post_torch(sublayer_out, residual, post_layer_mix, comb_res_mix)
    hc_mult = updated.shape[-2]
    hidden = updated.shape[-1]
    flat = updated.view(-1, hc_mult, hidden)
    tokens = flat.shape[0]
    pre = pre_mix.reshape(tokens, hc_mult).to(torch.float32)
    layer_input = torch.sum(pre.unsqueeze(-1) * flat.to(torch.float32), dim=1).to(
        updated.dtype
    )

    x = flat.reshape(tokens, hc_mult * hidden).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden) + rms_eps)

    next_pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + (
        hc_pre_eps
    )
    post = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )
    comb = mixes[:, 2 * hc_mult :].view(tokens, hc_mult, hc_mult) * hc_scale[2] + (
        hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    )
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    outer = residual.shape[:-2]
    return (
        updated,
        post.view(*outer, hc_mult, 1),
        comb.view(*outer, hc_mult, hc_mult),
        layer_input.view(*outer, hidden),
        next_pre.view(*outer, hc_mult),
    )


def _make_inputs(tokens: int, seed: int = 42):
    torch.manual_seed(seed)
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


def _ulp_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


def assert_boundary_close(actual, expected):
    for index, (output, reference) in enumerate(zip(actual, expected)):
        if index == 0:
            # residual: <=1 bf16 ULP excl near-zero (SGLang test_hc_boundary_hip)
            ulp = _ulp_bf16(output, reference)
            abs_err = (output.float() - reference.float()).abs()
            near_zero = abs_err <= 2.0**-18
            bad = (ulp > 1) & ~near_zero
            assert int(bad.sum().item()) == 0, (
                f"residual ulp>1 count={int(bad.sum())} max_ulp={int(ulp.max())}"
            )
        elif index == 3:
            error = (output.float() - reference.float()).square().mean().sqrt()
            rms = reference.float().square().mean().sqrt().clamp_min(1e-5)
            assert error / rms < 2e-5
        else:
            torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("tokens", [1, 8, 48, 64])
@pytest.mark.parametrize(
    "pre_eps,sinkhorn_eps,post_mult,iters",
    [
        (1e-6, 1e-6, 2.0, 20),
        (1e-4, 1e-5, 1.5, 3),
    ],
)
def test_mhc_boundary_matches_unfused(
    tokens, pre_eps, sinkhorn_eps, post_mult, iters
):
    from hcu_vllm_das.dsv41_mhc_boundary import boundary

    inputs = _make_inputs(tokens)
    scalar_args = (1e-6, pre_eps, sinkhorn_eps, post_mult, iters)
    actual = boundary(*inputs, *scalar_args)
    expected = _unfused_boundary_torch(*inputs, *scalar_args)
    assert_boundary_close(actual, expected)


def test_mhc_boundary_empty_tokens():
    from hcu_vllm_das.dsv41_mhc_boundary import boundary

    inputs = _make_inputs(0)
    scalar_args = (1e-6, 1e-6, 1e-6, 2.0, 3)
    actual = boundary(*inputs, *scalar_args)
    assert actual[0].shape == (0, 4, 5120)
    assert actual[3].shape == (0, 5120)
