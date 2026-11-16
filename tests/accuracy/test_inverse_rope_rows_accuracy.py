# SPDX-License-Identifier: Apache-2.0
"""Accuracy: HCU CUDA inverse-RoPE suffix vs GPT-J torch / stock Triton rows_."""

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


def _resolve_op():
    try:
        import vllm_hcu.hcu_ops  # noqa: F401

        if hasattr(torch.ops.hcu_ops, "deepseek_v4_inv_rope"):
            return torch.ops.hcu_ops.deepseek_v4_inv_rope
    except Exception:
        pass
    # Fall back to the same JIT path the microbench uses.
    from tests.accuracy.bench_inverse_rope_rows_perf import _resolve_op as bench_resolve

    op, _source = bench_resolve()
    return op


@pytest.fixture(scope="module")
def inv_rope_op():
    return _resolve_op()


def _gptj_inv_rope_torch(x, positions, cos_sin_cache, rope_dim):
    half = rope_dim // 2
    nope = x.shape[-1] - rope_dim
    dtype = x.dtype
    xf = x.float()
    cache = cos_sin_cache.index_select(0, positions.long())
    cos = cache[:, :half].float().view(positions.shape[0], *([1] * (x.dim() - 2)), half)
    sin = cache[:, half : 2 * half].float().view(
        positions.shape[0], *([1] * (x.dim() - 2)), half
    )
    rope = xf[..., nope:]
    even, odd = rope[..., 0::2], rope[..., 1::2]
    out = xf.clone()
    out[..., nope:] = torch.stack(
        (even * cos + odd * sin, odd * cos - even * sin), -1
    ).flatten(-2)
    return out.to(dtype)


def _make_inputs(tokens: int, heads: int = 2, hidden: int = 512, rope_dim: int = 64):
    torch.manual_seed(20261116 + tokens)
    o = torch.randn(tokens, heads, hidden, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    cache = torch.randn(
        max(tokens * 2, 128), rope_dim, device="cuda", dtype=torch.float32
    )
    return o, positions, cache, rope_dim


@pytest.mark.parametrize("tokens", [1, 8, 32, 64])
@pytest.mark.parametrize("heads", [1, 2])
def test_cuda_suffix_matches_gptj_torch(tokens, heads, inv_rope_op):
    o, positions, cache, rope_dim = _make_inputs(tokens, heads=heads)
    ref = _gptj_inv_rope_torch(o, positions, cache, rope_dim)
    got = o.clone()
    inv_rope_op(got[..., -rope_dim:], positions, cache)
    torch.testing.assert_close(got[..., :-rope_dim], o[..., :-rope_dim], atol=0, rtol=0)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [1, 8, 32])
def test_rows_wrapper_matches_stock_triton(tokens, inv_rope_op):
    from vllm.models.deepseek_v41.amd import rocm as upstream

    o, positions, cache, rope_dim = _make_inputs(tokens, heads=2)
    stock = o.clone()
    upstream.rocm_inverse_rope_rows_(stock, positions, cache, rope_dim)
    cuda = o.clone()
    inv_rope_op(cuda[..., -rope_dim:], positions, cache)
    torch.testing.assert_close(cuda, stock, atol=0, rtol=0)
