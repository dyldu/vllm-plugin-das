# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Accuracy + performance contract tests for the SGLang-align MoE fusions.

Each operator is checked against the *reference semantics* used by both
SGLang and vLLM, and must additionally be measurably faster than the
unfused path. A fusion is only allowed to be committed when both hold.

Golden reference for the routing op
-----------------------------------
SGLang's ``moe_fused_gate`` (and vLLM's port of it in
``vllm/model_executor/layers/fused_moe/router/dsv4_topk.py``) define the
DeepSeek-V4 "noaux_tc" routing as::

    w_e      = sqrt(softplus(logit_e))       # bias-free score
    score_e  = w_e + correction_bias_e       # bias steers SELECTION only
    picked   = top-k(score_e)
    weight_e = w_e of the picked experts     # unbiased weights
    out      = weight_e * rsf / sum(weight_e)

Note that the bias shifts *which* experts are selected but does **not**
change the weight of a selected expert.

Model constants are taken from
``DeepSeek-V4.1-Flash-Channel-FP8/config.json``:
``n_routed_experts=384``, ``num_experts_per_tok=6``,
``routed_scaling_factor=1.5``, ``hidden_size=5120``.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

NUM_EXPERTS = 384
TOP_K = 6
ROUTED_SCALING_FACTOR = 1.5
HIDDEN = 5120

# fp8 e4m3fn has 3 mantissa bits -> worst-case relative rounding error
# of a correctly-rounded value is 2^-4 = 6.25 %.
FP8_E4M3_REL_BOUND = 6.3e-2
FP8_MAX = 448.0


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HCU device required")


def _cuda_time_us(fn: Callable[[], object], warmup: int = 20, iters: int = 200) -> float:
    """Median wall time of ``fn`` in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def _golden_scores(logits: torch.Tensor, correction_bias: torch.Tensor) -> torch.Tensor:
    """The bias-corrected selection score of every expert."""
    lg = logits.float()
    weights = torch.sqrt(
        torch.where(lg > 20.0, lg, torch.log1p(torch.exp(lg.clamp(max=20.0))))
    )
    return weights + correction_bias.float().reshape(1, -1)


def _golden_dsv4_topk(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch port of vLLM's ``dsv4_topk`` Triton kernel."""
    lg = logits.float()
    # weights = sqrt(softplus(logits)) with the >20 overflow guard
    weights = torch.sqrt(
        torch.where(lg > 20.0, lg, torch.log1p(torch.exp(lg.clamp(max=20.0))))
    )
    score = weights + correction_bias.float().reshape(1, -1)

    ids = torch.empty((lg.shape[0], top_k), dtype=torch.int64, device=lg.device)
    out_w = torch.empty((lg.shape[0], top_k), dtype=torch.float32, device=lg.device)
    for r in range(lg.shape[0]):
        # stable sort => ties broken by the lowest expert id, matching
        # the reference kernel's ``tl.min(candidate)`` tie-break.
        sel = torch.argsort(score[r], descending=True, stable=True)[:top_k]
        ids[r] = sel
        w = weights[r, sel]
        s = w.sum()
        out_w[r] = w * routed_scaling_factor / torch.where(s > 0, s, torch.ones_like(s))
    return out_w, ids


# ---------------------------------------------------------------------------
# bf16_mega_gate_hcu
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128, 512])
def test_bf16_mega_gate_matches_reference(num_tokens: int) -> None:
    """Fused BF16 mega-gate must reproduce the reference routing exactly."""
    _require_cuda()
    from lightop.moe import bf16_mega_gate_hcu

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(num_tokens, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
    w = (
        torch.randn(NUM_EXPERTS, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
        * 0.02
    )
    b = torch.randn(NUM_EXPERTS, generator=g, dtype=torch.float32).to(device)
    mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    with torch.no_grad():
        ids, wts, logits = bf16_mega_gate_hcu(
            x,
            w,
            num_topk=TOP_K,
            use_shared_as_routed=False,
            num_shared_experts=0,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            ep_rank=0,
            scoring_func="sqrtsoftplus",
            mask=mask,
            bias=b,
            return_logits=True,
            renormalize=True,
            apply_scale=True,
        )

    ref_logits = torch.nn.functional.linear(x.float(), w.float())
    ref_w, ref_ids = _golden_dsv4_topk(
        ref_logits, b, TOP_K, ROUTED_SCALING_FACTOR
    )

    # ---- (a) The ROUTING RULE, checked on the kernel's own logits ----
    # BF16 inputs accumulate into FP32, without rounding logits to BF16.
    # The independent FP32 GEMM can use a different addition order. To test
    # the routing *semantics* (not the GEMM), replay the reference rule on
    # the logits the kernel itself produced. This isolates the question
    # "does the kernel implement the SGLang rule?" from "is the GEMM
    # bit-identical?", which the gate GEMM cannot be.
    kernel_w, kernel_ids = _golden_dsv4_topk(
        logits.float(), b, TOP_K, ROUTED_SCALING_FACTOR
    )
    for r in range(num_tokens):
        got = ids[r].tolist()
        exp = kernel_ids[r].tolist()
        assert set(got) == set(exp), (
            f"row {r}: kernel ids {got} != reference ids {exp} "
            f"(routing rule mismatch)"
        )
        gmap = {int(e): float(v) for e, v in zip(got, wts[r].tolist())}
        for j, e in enumerate(exp):
            want = float(kernel_w[r][j])
            got_v = gmap[int(e)]
            assert abs(got_v - want) <= 1e-6 + 1e-5 * abs(want), (
                f"row {r}: weight for expert {e}: {got_v} != {want}"
            )

    # ---- (b) The SELECTION, checked against the independent torch GEMM ----
    # Explain a changed boundary only by the measured FP32 score error,
    # rather than an arbitrary relative near-tie allowance.
    score_all = _golden_scores(ref_logits, b)
    score_error = (_golden_scores(logits, b) - score_all).abs().amax(1)
    for r in range(num_tokens):
        got = set(ids[r].tolist())
        exp = set(ref_ids[r].tolist())
        if got == exp:
            continue
        srt = torch.sort(score_all[r], descending=True).values
        margin = abs(srt[TOP_K - 1].item() - srt[TOP_K].item())
        measured_bound = 2 * score_error[r].item() + 1e-6
        assert margin <= measured_bound, (
            f"row {r}: ids {sorted(got)} != {sorted(exp)} and the top-k "
            f"boundary margin {margin:.3e} exceeds measured bound {measured_bound:.3e}"
        )

    # the returned logits are the *unbiased* router GEMM output
    torch.testing.assert_close(logits, ref_logits, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("num_tokens", [8, 128])
def test_bf16_mega_gate_is_faster_than_unfused(num_tokens: int) -> None:
    """The fused gate must beat the split gate-GEMM + top-k path."""
    _require_cuda()
    from lightop.moe import bf16_mega_gate_hcu

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(num_tokens, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
    w = (
        torch.randn(NUM_EXPERTS, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
        * 0.02
    )
    b = torch.randn(NUM_EXPERTS, generator=g, dtype=torch.float32).to(device)
    mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    def run_fused():
        return bf16_mega_gate_hcu(
            x,
            w,
            num_topk=TOP_K,
            use_shared_as_routed=False,
            num_shared_experts=0,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            ep_rank=0,
            scoring_func="sqrtsoftplus",
            mask=mask,
            bias=b,
            return_logits=True,
            renormalize=True,
            apply_scale=True,
        )

    def run_unfused():
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias

        lg = torch.mm(x, w.T, out_dtype=torch.float32)
        return fused_topk_bias(x, lg, "sqrtsoftplus", b, TOP_K, True,
                               indices_type=torch.int32,
                               routed_scaling_factor=ROUTED_SCALING_FACTOR)

    t_fused = _cuda_time_us(run_fused)
    t_unfused = _cuda_time_us(run_unfused)
    assert t_fused < t_unfused, (
        f"fused mega-gate slower: {t_fused:.1f}us vs unfused {t_unfused:.1f}us"
    )


# ---------------------------------------------------------------------------
# rms_norm_per_token_fp8_quant
# ---------------------------------------------------------------------------
def _rms_norm_fp8_quant_ref(
    x: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: RMSNorm rounded to BF16, then symmetric per-token FP8 quant."""
    xf = x.float()
    rms = xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    norm = (xf * rms * w.float()).to(torch.bfloat16).float()
    scale = norm.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    q = (norm / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
def test_rms_norm_fp8_quant_matches_reference(num_tokens: int) -> None:
    """Fused RMSNorm+FP8 quant must stay within the FP8 e4m3 rounding bound."""
    _require_cuda()
    lightop = pytest.importorskip("lightop")
    rms_norm_per_token_fp8_quant = getattr(
        lightop, "rms_norm_per_token_fp8_quant", None
    )
    if rms_norm_per_token_fp8_quant is None:
        pytest.skip("lightop.rms_norm_per_token_fp8_quant unavailable")

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(num_tokens, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
    w = (torch.randn(HIDDEN, generator=g, dtype=torch.bfloat16).to(device).abs() + 0.5)

    with torch.no_grad():
        out_fp8, out_scale = rms_norm_per_token_fp8_quant(
            x, w, epsilon=1e-6, fp8type=0, residual=None, update_input=False
        )

    ref_q, ref_scale = _rms_norm_fp8_quant_ref(x, w, 1e-6)

    # per-token scales must agree
    torch.testing.assert_close(
        out_scale.float().reshape(-1),
        ref_scale.reshape(-1),
        rtol=2e-2,
        atol=1e-6,
    )

    deq = out_fp8.float() * out_scale.float().reshape(-1, 1)
    exact = (x.float()
             * x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
             * w.float()).to(torch.bfloat16).float()

    # relative error, restricted to the FP8 normal range (below that the
    # relative error is dominated by the subnormal tail and is meaningless)
    sc = out_scale.float().reshape(-1, 1)
    normal = exact.abs() > (FP8_MAX * 0.0625) * sc
    if normal.any():
        rel = ((deq - exact).abs() / exact.abs().clamp_min(1e-30))[normal].max().item()
        assert rel <= FP8_E4M3_REL_BOUND, f"relative error {rel} exceeds e4m3 bound"

    # absolute error must stay within one FP8 ulp of the row maximum
    ulp = sc * (FP8_MAX / 8.0)
    err_ulp = ((deq - exact).abs() / ulp.clamp_min(1e-30)).max().item()
    assert err_ulp <= 1.05, f"error {err_ulp} ulp exceeds 1 ulp"

    # the kernel must not be less accurate than a plain torch implementation
    ref_deq = ref_q.float() * ref_scale.reshape(-1, 1)
    assert (deq - exact).abs().max().item() <= (
        ref_deq - exact
    ).abs().max().item() * 1.05 + 1e-9

    # any bit difference from the torch reference must be a single code step
    kernel_code = out_fp8.view(torch.uint8).int()
    ref_code = ref_q.view(torch.uint8).int()
    assert (kernel_code - ref_code).abs().max().item() <= 1


@pytest.mark.parametrize("num_tokens", [8, 128])
def test_rms_norm_fp8_quant_is_faster_than_unfused(num_tokens: int) -> None:
    """The fused op must beat the split RMSNorm + quant path."""
    _require_cuda()
    lightop = pytest.importorskip("lightop")
    rms_norm_per_token_fp8_quant = getattr(
        lightop, "rms_norm_per_token_fp8_quant", None
    )
    if rms_norm_per_token_fp8_quant is None:
        pytest.skip("lightop.rms_norm_per_token_fp8_quant unavailable")

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(num_tokens, HIDDEN, generator=g, dtype=torch.bfloat16).to(device)
    w = (torch.randn(HIDDEN, generator=g, dtype=torch.bfloat16).to(device).abs() + 0.5)

    def run_fused():
        return rms_norm_per_token_fp8_quant(
            x, w, epsilon=1e-6, fp8type=0, residual=None, update_input=False
        )

    def run_unfused():
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        o = (xf * rms * w.float()).to(torch.bfloat16)
        s = o.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        return (o.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s

    t_fused = _cuda_time_us(run_fused)
    t_unfused = _cuda_time_us(run_unfused)
    assert t_fused < t_unfused, (
        f"fused rms_norm+fp8 quant slower: {t_fused:.1f}us vs {t_unfused:.1f}us"
    )
