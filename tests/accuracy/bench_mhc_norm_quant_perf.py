#!/usr/bin/env python3
"""Fused MHC RMSNorm+quant vs unfused rms_norm + torch quant (HCU)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class Compare:
    name: str
    tokens: int
    hidden: int
    fused_p50_ms: float
    unfused_p50_ms: float
    speedup_p50: float
    verdict: str


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


def _verdict(sp, tol=0.03):
    if sp > 1 + tol:
        return "faster"
    if sp < 1 - tol:
        return "slower"
    return "flat"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 256])
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--mode", choices=["fp8", "int8"], default="fp8")
    p.add_argument("--json-out", default="")
    args = p.parse_args(argv)
    if not torch.cuda.is_available() or torch.version.hip is None:
        return 2
    os.environ.setdefault("VLLM_HCU_USE_CUSTOM_OPS", "1")
    os.environ.setdefault("VLLM_HCU_USE_MHC_NORM_QUANT", "1")

    from hcu_vllm_das.dsv41_mhc_norm_quant import (
        rmsnorm_per_token_fp8,
        rmsnorm_per_token_int8,
    )
    from vllm import ir

    print(f"device={torch.cuda.get_device_name(0)} mode={args.mode} H={args.hidden}")
    compares = []
    for t in args.tokens:
        x = torch.randn(t, args.hidden, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(args.hidden, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6
        fused_fn = (
            (lambda: rmsnorm_per_token_fp8(x, w, eps))
            if args.mode == "fp8"
            else (lambda: rmsnorm_per_token_int8(x, w, eps))
        )

        def unfused():
            n = ir.ops.rms_norm(x, w, eps)
            xf = n.float()
            amax = xf.abs().amax(-1, keepdim=True)
            if args.mode == "fp8":
                scale = torch.maximum(amax / 448.0, torch.full_like(amax, 1.0 / (448.0 * 512.0)))
                q = (xf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            else:
                scale = amax / 127.0
                inv = torch.where(amax == 0, torch.zeros_like(amax), 127.0 / amax)
                q = torch.round(xf * inv).clamp(-127, 127).to(torch.int8)
            return q, scale, n

        f = _time(fused_fn, args.warmup, args.iters)
        u = _time(unfused, args.warmup, args.iters)
        fp, up = _pct(f, 0.5), _pct(u, 0.5)
        sp = up / fp if fp else float("nan")
        c = Compare("mhc_norm_quant_" + args.mode, t, args.hidden, fp, up, sp, _verdict(sp))
        compares.append(c)
        print(
            f"  T={t}: fused p50={fp:.4f}ms unfused={up:.4f}ms {sp:.3f}x ({c.verdict})"
        )

    print(
        f"\n{'op':22s} {'T':>4s} {'fused':>10s} {'unfused':>10s} {'speedup':>8s} {'verdict':>8s}"
    )
    print("-" * 70)
    for c in compares:
        print(
            f"{c.name:22s} {c.tokens:4d} {c.fused_p50_ms:10.4f} "
            f"{c.unfused_p50_ms:10.4f} {c.speedup_p50:8.3f}x {c.verdict:>8s}"
        )
    payload = {"compares": [asdict(c) for c in compares], "device": torch.cuda.get_device_name(0)}
    text = json.dumps(payload, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
        print("JSON ->", args.json_out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
