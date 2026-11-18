# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DeepSeek-V4.1 attention-input RMSNorm + FP8 quant path.

Background
----------
The SGLang reference trace (dp0_sgl_moe.pt.trace.json.gz) contains 430 launches
of ``rms_norm_fp8_quant_<c10::BFloat16, unsigned char, float, 16, 128, true, ...>``
immediately followed by an FP8 block-scale GEMM (``Cijk_*F8BS_MT256x64x128*``).
That is the attention input norm fused with its per-token FP8 quantization.

The vLLM baseline trace instead shows the two steps separately:
``rmsnorm_forward_kernel_fast`` (1565 launches) followed by
``dynamic_per_token_scaled_fp8_quant_kernel_strided`` (3490 launches), with 418
adjacent rms_norm -> quant pairs.

These tests cover the two building blocks that a fused implementation needs:
  * ``lightop.rms_norm_per_token_fp8_quant``  -- RMSNorm + per-token FP8 quant
  * ``lightop.quant.per_token_group_quant_fp8`` -- the group-128 quant used by
    the block-scale GEMM path

and check that both are numerically sound and faster than the split chain.

Run with:
    python3 -m pytest tests/vllm033/test_rms_norm_fp8_quant.py -v
"""

import os

import pytest
import torch

torch.manual_seed(0)

HIDDEN = 5120
EPS = 1e-6
FP8_MAX = 448.0


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no GPU available")


@pytest.fixture(scope="module")
def lightop():
    _require_gpu()
    try:
        import lightop
        # lightop.quant is a submodule that is not auto-imported by the
        # package __init__, so pull it in explicitly.
        import lightop.quant  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"lightop unavailable: {exc}")
    return lightop


def _bf16_rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm computed in fp32 then rounded to bf16."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * inv * w.float()).to(torch.bfloat16)


def _per_token_fp8_ref(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token dynamic FP8 (e4m3) quantization.

    scale = max(amax, 1e-10) / 448 ; q = clamp(t / scale, -448, 448)
    """
    amax = t.float().abs().amax(-1, keepdim=True).clamp(min=1e-10)
    scale = amax / FP8_MAX
    q = (t.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def _group_fp8_ref(t: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference group-wise FP8 quantization along the last dim."""
    n_group = t.shape[-1] // group
    tg = t.reshape(*t.shape[:-1], n_group, group).float()
    amax = tg.abs().amax(-1, keepdim=True).clamp(min=1e-10)
    scale = amax / FP8_MAX
    q = (tg / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(t.shape), scale.reshape(*t.shape[:-1], n_group)


class TestPerTokenFp8Quant:
    """lightop.rms_norm_per_token_fp8_quant: RMSNorm fused with FP8 quant."""

    @pytest.mark.parametrize("tokens", [1, 8, 64, 512])
    def test_matches_split_reference(self, lightop, tokens):
        """Fused kernel must agree with rmsnorm-then-quant within FP8 noise."""
        x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)

        normed = _bf16_rmsnorm_ref(x, w, EPS)
        q_ref, s_ref = _per_token_fp8_ref(normed)

        q, s = lightop.rms_norm_per_token_fp8_quant(x, w, EPS)

        assert q.shape == x.shape
        assert q.dtype == torch.float8_e4m3fn
        assert s.shape == (tokens, 1)
        assert s.dtype == torch.float32

        # Scale is computed from the same amax, so it must match closely.
        scale_rel = (
            (s.float() - s_ref.float()).abs().max()
            / s_ref.float().abs().max().clamp(min=1e-9)
        )
        assert scale_rel < 1e-5, f"scale mismatch rel={scale_rel.item():.3e}"

        # FP8 codes may differ by one ulp where the value sits on a rounding
        # boundary, so compare dequantized values with a relative tolerance.
        dq = q.float() * s.float()
        dq_ref = q_ref.float() * s_ref.float()
        denom = dq_ref.abs().amax().clamp(min=1e-6)
        err = (dq - dq_ref).abs().max() / denom
        assert err < 5e-2, f"dequant mismatch rel={err.item():.3e}"

    def test_is_faster_than_split_chain(self, lightop):
        """Fusing must beat rmsnorm + separate quant."""
        x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)

        def split():
            n = torch.nn.functional.rms_norm(
                x.float(), (HIDDEN,), w.float(), EPS
            ).to(torch.bfloat16)
            return _per_token_fp8_ref(n)

        def fused():
            return lightop.rms_norm_per_token_fp8_quant(x, w, EPS)

        # warmup
        for _ in range(5):
            split()
            fused()
        torch.cuda.synchronize()

        import time

        def bench(fn, iters=50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e6

        t_split = bench(split)
        t_fused = bench(fused)
        assert t_fused < t_split, (
            f"fused {t_fused:.1f}us not faster than split {t_split:.1f}us"
        )


class TestGroupFp8Quant:
    """lightop.quant.per_token_group_quant_fp8: group-128 FP8 quant."""

    @pytest.mark.parametrize("group", [128])
    def test_matches_group_reference(self, lightop, group):
        x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16)
        q_ref, s_ref = _group_fp8_ref(x, group)

        q, s = lightop.quant.per_token_group_quant_fp8(x, group)

        assert q.shape == x.shape
        assert q.dtype == torch.float8_e4m3fn
        assert s.shape == (64, HIDDEN // group)

        scale_rel = (
            (s.float() - s_ref.float()).abs().max()
            / s_ref.float().abs().max().clamp(min=1e-9)
        )
        assert scale_rel < 1e-5, f"scale mismatch rel={scale_rel.item():.3e}"

        dq = q.float() * s.float().repeat_interleave(group, dim=-1)
        dq_ref = q_ref.float() * s_ref.float().repeat_interleave(group, dim=-1)
        denom = dq_ref.abs().amax().clamp(min=1e-6)
        err = (dq - dq_ref).abs().max() / denom
        assert err < 5e-2, f"dequant mismatch rel={err.item():.3e}"

    def test_group_quant_is_faster_than_per_token_for_wide_rows(self, lightop):
        """Sanity: group quant should not be slower than a naive torch path."""
        x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16)

        import time

        def torch_group():
            return _group_fp8_ref(x, 128)

        def lightop_group():
            return lightop.quant.per_token_group_quant_fp8(x, 128)

        for _ in range(5):
            torch_group()
            lightop_group()
        torch.cuda.synchronize()

        def bench(fn, iters=30):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e6

        t_lightop = bench(lightop_group)
        t_torch = bench(torch_group)
        assert t_lightop < t_torch, (
            f"lightop group quant {t_lightop:.1f}us not faster than torch {t_torch:.1f}us"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
