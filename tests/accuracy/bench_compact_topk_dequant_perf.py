#!/usr/bin/env python3
"""LightOp selective TopK dequant vs full-capacity counted dequant (p50)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


def _percentile(xs, p):
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _time(fn, *, warmup=10, iters=50):
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        samples.append(float(starter.elapsed_time(ender)))
    return samples


def main():
    os.environ.setdefault("VLLM_HCU_USE_CUSTOM_OPS", "1")
    os.environ.setdefault("VLLM_HCU_USE_LIGHTOP_COMPACT_TOPK_DEQUANT", "1")
    from hcu_vllm_das.compact_topk_dequant_lightop import (
        lightop_compact_topk_available,
        try_dequantize_sparse_topk,
        v4_page_bytes_padded,
    )
    from lightop.kvcache import dequantize_k_cache_paged_hcu_counted

    if not lightop_compact_topk_available():
        print("lightop missing compact topk ops", file=sys.stderr)
        return 2

    page_size = 64
    stride = v4_page_bytes_padded(page_size)
    # capacity <= queries * topk (LightOp full-capacity workspace contract)
    rows = [
        (32, 8, 4),
        (256, 32, 8),
        (1024, 64, 16),
    ]
    results = []
    for capacity, topk, queries in rows:
        flat = torch.arange(capacity, device="cuda", dtype=torch.int32)
        combined = torch.randint(
            0, capacity, (queries, topk), device="cuda", dtype=torch.int32
        )
        lens = torch.full((queries,), topk, dtype=torch.int32, device="cuda")
        num_pages = max(4, (capacity + page_size - 1) // page_size + 1)
        cache = torch.zeros((num_pages, stride), device="cuda", dtype=torch.uint8)
        cache[:, : page_size * 576] = 64
        cache[:, page_size * 576 : page_size * 576 + page_size * 8] = 120
        out_sel = torch.empty((capacity, 1, 512), dtype=torch.bfloat16, device="cuda")
        out_full = torch.empty((capacity, 1, 512), dtype=torch.bfloat16, device="cuda")
        valid_count = torch.tensor([capacity], dtype=torch.int32, device="cuda")

        def fused():
            try_dequantize_sparse_topk(
                cache, flat, combined, lens, out_sel, topk, page_size
            )

        def unfused():
            dequantize_k_cache_paged_hcu_counted(
                cache.view(torch.uint8),
                flat,
                out_full,
                page_size,
                False,
                valid_count,
            )

        # smoke
        fused()
        unfused()
        fs = _time(fused)
        us = _time(unfused)
        fp50, up50 = _percentile(fs, 0.5), _percentile(us, 0.5)
        sp = up50 / fp50 if fp50 > 0 else float("nan")
        print(
            f"capacity={capacity} topk={topk} q={queries} "
            f"selective_p50={fp50:.4f}ms full_p50={up50:.4f}ms speedup={sp:.2f}x"
        )
        results.append(
            {
                "capacity": capacity,
                "topk": topk,
                "queries": queries,
                "selective_p50_ms": fp50,
                "full_p50_ms": up50,
                "speedup_p50": sp,
            }
        )

    out_path = Path(os.environ.get("BENCH_JSON_OUT", "/tmp/compact_topk_dequant_perf.json"))
    out_path.write_text(json.dumps(results, indent=2))
    print("wrote", out_path)
    # Also try worktree copy when writable.
    wt = Path(__file__).with_name("compact_topk_dequant_perf.json")
    try:
        wt.write_text(out_path.read_text())
        print("copied", wt)
    except OSError as exc:
        print(f"worktree copy skipped ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
