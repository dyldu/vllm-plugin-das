#!/usr/bin/env python3
"""LightOp Prefill KV microbench (#596 KV-only) vs torch references (HCU)."""

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
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--json-out", default="")
    args = p.parse_args(argv)
    if not torch.cuda.is_available() or torch.version.hip is None:
        return 2
    os.environ.setdefault("VLLM_HCU_USE_CUSTOM_OPS", "1")
    os.environ.setdefault("VLLM_HCU_USE_LIGHTOP_POOL_PAIRS_NORM", "1")
    os.environ.setdefault("VLLM_HCU_USE_LIGHTOP_ROPE_FQ_STORE", "1")

    from hcu_vllm_das.prefill_kv_lightop import (
        lightop_pool_pairs_norm_available,
        lightop_rope_fq_store_available,
        ref_pool_pairs_norm,
        try_pool_pairs_norm,
        try_rope_fake_quant_store,
    )

    print(f"device={torch.cuda.get_device_name(0)}")
    rows: list[Compare] = []

    if lightop_pool_pairs_norm_available():
        for n in args.tokens:
            g = torch.Generator(device="cpu").manual_seed(n)
            prev = torch.randn(n, 512, generator=g, dtype=torch.float32).cuda()
            cur = torch.randn(n, 512, generator=g, dtype=torch.float32).cuda()
            ps = torch.randn(n, 512, generator=g, dtype=torch.float32).cuda()
            cs = torch.randn(n, 512, generator=g, dtype=torch.float32).cuda()
            w = (torch.rand(512, generator=g) + 0.5).cuda()
            eps = 1e-6

            def fused():
                out = try_pool_pairs_norm(prev, cur, ps, cs, w, eps)
                assert out is not None

            def unfused():
                ref_pool_pairs_norm(prev, cur, ps, cs, w, eps)

            fs = _time(fused, args.warmup, args.iters)
            us = _time(unfused, args.warmup, args.iters)
            fp50, up50 = _pct(fs, 0.5), _pct(us, 0.5)
            sp = up50 / fp50 if fp50 > 0 else float("nan")
            row = Compare(
                name="pool_pairs_norm",
                tokens=n,
                fused_p50_ms=fp50,
                unfused_p50_ms=up50,
                speedup_p50=sp,
                verdict=_verdict(sp),
            )
            rows.append(row)
            print(
                f"pool_pairs_norm N={n}: LightOp p50={fp50:.4f} ms  "
                f"torch p50={up50:.4f} ms  {sp:.2f}× ({row.verdict})"
            )
    else:
        print("SKIP pool_pairs_norm: LightOp op missing")

    if lightop_rope_fq_store_available():
        from hcu_vllm_das.prefill_kv_lightop import v4_page_bytes_padded

        for n in args.tokens:
            page_size = 64
            pages = max(2, (n + page_size - 1) // page_size + 1)
            page_bytes = v4_page_bytes_padded(page_size)
            cache_f = torch.zeros(
                pages, page_bytes, dtype=torch.uint8, device="cuda"
            )
            cache_u = torch.zeros_like(cache_f)
            latent = torch.randn(n, 512, dtype=torch.bfloat16, device="cuda")
            ang = torch.rand(n, 32, device="cuda") * 6.28
            freqs = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1).flatten(-2)
            indices = torch.arange(1, n + 1, device="cuda", dtype=torch.int64)

            def fused():
                cache_f.zero_()
                assert try_rope_fake_quant_store(
                    latent, freqs, cache_f, indices, page_size
                )

            def unfused():
                from test_prefill_kv_lightop_accuracy import (
                    _fake_quant_compressed_kv,
                    _rope_tail,
                )

                cache_u.copy_(cache_f)
                _ = _fake_quant_compressed_kv(_rope_tail(latent, freqs))

            fs = _time(fused, args.warmup, args.iters)
            us = _time(unfused, args.warmup, args.iters)
            fp50, up50 = _pct(fs, 0.5), _pct(us, 0.5)
            sp = up50 / fp50 if fp50 > 0 else float("nan")
            row = Compare(
                name="rope_fq_store",
                tokens=n,
                fused_p50_ms=fp50,
                unfused_p50_ms=up50,
                speedup_p50=sp,
                verdict=_verdict(sp),
            )
            rows.append(row)
            print(
                f"rope_fq_store N={n}: LightOp p50={fp50:.4f} ms  "
                f"torch p50={up50:.4f} ms  {sp:.2f}× ({row.verdict})"
            )
    else:
        print("SKIP rope_fq_store: LightOp op missing")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([asdict(r) for r in rows], indent=2) + "\n"
        )
    return 0 if rows else 3


if __name__ == "__main__":
    sys.exit(main())
