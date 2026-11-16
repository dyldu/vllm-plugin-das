#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MHC boundary fused vs unfused: ULP accuracy + CUDA-Event p50 perf.

Fused: ``hcu_vllm_das.dsv41_mhc_boundary.boundary`` (Triton RTNE + opaque FMA/fadd).
Unfused: stock ``mhc_post_torch`` + previous-pre collapse + next mix/Sinkhorn.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

HC, H = 4, 5120


def _mhc_post_torch():
    try:
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

        return mhc_post_torch
    except ImportError:
        from vllm.model_executor.layers.mhc import mhc_kernels

        return mhc_kernels.mhc_post_torch


def unfused_boundary_torch(
    residual,
    sublayer_out,
    post_layer_mix,
    comb_res_mix,
    pre_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    mhc_post_torch = _mhc_post_torch()
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


def _make_inputs(tokens: int, seed: int, device: str = "cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    residual = torch.randn(tokens, HC, H, generator=g, dtype=torch.bfloat16).to(device)
    sublayer = torch.randn(tokens, H, generator=g, dtype=torch.bfloat16).to(device)
    pre = torch.rand(tokens, HC, generator=g).to(device)
    post = torch.rand(tokens, HC, 1, generator=g).to(device) * 2.0
    comb = torch.softmax(torch.randn(tokens, HC, HC, generator=g), dim=-1).to(device)
    fn = (torch.randn((2 + HC) * HC, HC * H, generator=g) * 0.005).to(device)
    scale = torch.tensor([0.1, 0.2, 0.3], device=device)
    base = (torch.randn((2 + HC) * HC, generator=g) * 0.1).to(device)
    return residual, sublayer, post, comb, pre, fn, scale, base


def _ulp_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


def _pct(xs, p):
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    return ys[f] if f == c else ys[f] + (ys[c] - ys[f]) * (k - f)


def _time(fn, warmup, iters):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(float(s.elapsed_time(e)))
    return out


@dataclass
class AccRow:
    tokens: int
    residual_ulp_bad: int
    residual_max_ulp: int
    residual_pass: bool
    layer_input_rel_rms: float
    layer_input_pass: bool
    coeffs_pass: bool
    pass_: bool


@dataclass
class PerfRow:
    tokens: int
    fused_p50_ms: float
    unfused_p50_ms: float
    speedup_p50: float


def _check_acc(fused, unfused) -> AccRow:
    names = ("residual", "post", "comb", "layer_input", "next_pre")
    residual_pass = True
    ulp_bad = 0
    max_ulp = 0
    li_rel = 0.0
    li_pass = True
    coeffs_pass = True
    for i, name in enumerate(names):
        got, ref = fused[i], unfused[i]
        if i == 0:
            ulp = _ulp_bf16(got, ref)
            abs_err = (got.float() - ref.float()).abs()
            near_zero = abs_err <= 2.0**-18
            bad = (ulp > 1) & ~near_zero
            ulp_bad = int(bad.sum().item())
            max_ulp = int(ulp.max().item())
            residual_pass = ulp_bad == 0
        elif i == 3:
            rms = ref.float().square().mean().sqrt().clamp_min(1e-5)
            li_rel = float(
                ((got.float() - ref.float()).square().mean().sqrt() / rms).item()
            )
            li_pass = li_rel < 2e-5
        else:
            coeffs_pass &= bool(
                torch.allclose(got.float(), ref.float(), atol=2e-5, rtol=2e-5)
            )
    ok = residual_pass and li_pass and coeffs_pass
    return AccRow(
        tokens=fused[0].shape[0],
        residual_ulp_bad=ulp_bad,
        residual_max_ulp=max_ulp,
        residual_pass=residual_pass,
        layer_input_rel_rms=li_rel,
        layer_input_pass=li_pass,
        coeffs_pass=coeffs_pass,
        pass_=ok,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 33, 48, 64])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--json-out", default="")
    p.add_argument("--skip-perf", action="store_true")
    args = p.parse_args(argv)

    if not torch.cuda.is_available() or torch.version.hip is None:
        print("requires HCU/HIP CUDA device", file=sys.stderr)
        return 2

    from hcu_vllm_das.dsv41_mhc_boundary import boundary

    print(
        f"device={torch.cuda.get_device_name(0)} hip={torch.version.hip} "
        f"torch={torch.__version__}"
    )
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")

    scalar_sets = [
        (1e-6, 1e-6, 1e-6, 2.0, 20),
        (1e-6, 1e-4, 1e-5, 1.5, 3),
    ]
    acc_rows = []
    all_ok = True
    for scalars in scalar_sets:
        for tokens in args.tokens:
            inputs = _make_inputs(tokens, args.seed + tokens)
            fused = boundary(
                *[t.clone() if torch.is_tensor(t) else t for t in inputs], *scalars
            )
            unfused = unfused_boundary_torch(
                *[t.clone() if torch.is_tensor(t) else t for t in inputs], *scalars
            )
            row = _check_acc(fused, unfused)
            acc_rows.append({**asdict(row), "scalars": list(scalars)})
            mark = "PASS" if row.pass_ else "FAIL"
            print(
                f"acc tokens={tokens} scalars={scalars} residual_ulp_bad="
                f"{row.residual_ulp_bad} max_ulp={row.residual_max_ulp} "
                f"li_rel={row.layer_input_rel_rms:.3e} {mark}"
            )
            all_ok &= row.pass_

    perf_rows = []
    if not args.skip_perf:
        scalars = scalar_sets[0]
        for tokens in args.tokens:
            if tokens == 0:
                continue
            inputs = _make_inputs(tokens, args.seed + tokens)

            def fused_fn(ins=inputs, sc=scalars):
                return boundary(*ins, *sc)

            def unfused_fn(ins=inputs, sc=scalars):
                return unfused_boundary_torch(*ins, *sc)

            # warm compile once outside timer
            fused_fn()
            unfused_fn()
            f = _time(fused_fn, args.warmup, args.iters)
            u = _time(unfused_fn, args.warmup, args.iters)
            fp, up = _pct(f, 0.5), _pct(u, 0.5)
            sp = up / fp if fp else float("nan")
            perf_rows.append(
                asdict(PerfRow(tokens, fp, up, sp))
            )
            print(
                f"perf tokens={tokens}: fused p50={fp:.4f}ms "
                f"unfused={up:.4f}ms speedup={sp:.3f}x"
            )

    payload = {
        "device": torch.cuda.get_device_name(0),
        "hip": torch.version.hip,
        "torch": torch.__version__,
        "accuracy": acc_rows,
        "perf": perf_rows,
        "result": "PASS" if all_ok else "FAIL",
    }
    text = json.dumps(payload, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
        print("JSON ->", args.json_out)
    else:
        print(text)

    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
