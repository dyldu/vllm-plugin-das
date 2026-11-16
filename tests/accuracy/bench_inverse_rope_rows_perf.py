#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HCU CUDA inverse-RoPE suffix vs stock Triton ``rocm_inverse_rope_rows_``.

Uses installed ``torch.ops.hcu_ops.deepseek_v4_inv_rope`` when available;
otherwise JIT-compiles ``vllm_hcu/csrc/fused_deepseek_v4_inv_rope_kernel.cu``
into a temporary ``hcu_ops_bench`` namespace for local microbench.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


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


def _resolve_op():
    try:
        import vllm_hcu.hcu_ops  # noqa: F401

        if hasattr(torch.ops.hcu_ops, "deepseek_v4_inv_rope"):
            return torch.ops.hcu_ops.deepseek_v4_inv_rope, "hcu_ops"
    except Exception:
        pass

    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    cu = root / "vllm_hcu/csrc/fused_deepseek_v4_inv_rope_kernel.cu"
    bind = Path("/tmp/hcu_inv_rope_bind.cpp")
    bind.write_text(
        """
#include <torch/extension.h>
void deepseek_v4_inv_rope(torch::Tensor& rope, torch::Tensor const& positions,
                          torch::Tensor const& cos_sin_cache);
TORCH_LIBRARY(hcu_ops_bench, m) {
  m.def("deepseek_v4_inv_rope(Tensor! rope, Tensor position_ids, Tensor cos_sin_cache) -> ()");
}
TORCH_LIBRARY_IMPL(hcu_ops_bench, CUDA, m) {
  m.impl("deepseek_v4_inv_rope", &deepseek_v4_inv_rope);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
"""
    )
    load(
        name="hcu_inv_rope_bench",
        sources=[str(bind), str(cu)],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return torch.ops.hcu_ops_bench.deepseek_v4_inv_rope, "jit"


def _pct(xs, p):
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    return ys[f] if f == c else ys[f] + (ys[c] - ys[f]) * (k - f)


def _time(fn, warmup, iters):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--json-out", default="")
    args = p.parse_args(argv)

    if not torch.cuda.is_available() or torch.version.hip is None:
        print("requires HCU/HIP", file=sys.stderr)
        return 2

    op, source = _resolve_op()
    from vllm.models.deepseek_v41.amd import rocm as upstream

    print(
        f"device={torch.cuda.get_device_name(0)} hip={torch.version.hip} "
        f"op_source={source}"
    )

    acc = []
    for tokens in [t for t in args.tokens if t > 0][:4]:
        for heads in (1, args.heads) if args.heads != 1 else (1,):
            torch.manual_seed(42 + tokens + heads)
            o = torch.randn(
                tokens, heads, 512, device="cuda", dtype=torch.bfloat16
            )
            pos = torch.arange(tokens, device="cuda", dtype=torch.int64)
            cache = torch.randn(
                max(128, tokens * 2), 64, device="cuda", dtype=torch.float32
            )
            ref = _gptj_inv_rope_torch(o, pos, cache, 64)
            cuda = o.clone()
            op(cuda[..., -64:], pos, cache)
            stock = o.clone()
            upstream.rocm_inverse_rope_rows_(stock, pos, cache, 64)
            max_torch = float((cuda.float() - ref.float()).abs().max())
            max_stock = float((cuda.float() - stock.float()).abs().max())
            ok = max_torch < 2e-2 and max_stock < 2e-2
            acc.append(
                {
                    "tokens": tokens,
                    "heads": heads,
                    "max_abs_vs_torch": max_torch,
                    "max_abs_vs_stock": max_stock,
                    "pass": ok,
                }
            )
            print(
                f"acc T={tokens} H={heads} vs_torch={max_torch:.3e} "
                f"vs_stock={max_stock:.3e} pass={ok}"
            )

    perf = []
    for tokens in args.tokens:
        torch.manual_seed(7 + tokens)
        o = torch.randn(
            tokens, args.heads, 512, device="cuda", dtype=torch.bfloat16
        )
        pos = torch.arange(tokens, device="cuda", dtype=torch.int64)
        cache = torch.randn(
            max(256, tokens * 2), 64, device="cuda", dtype=torch.float32
        )

        def cuda_fn(o=o, pos=pos, cache=cache):
            x = o.clone()
            op(x[..., -64:], pos, cache)
            return x

        def stock_fn(o=o, pos=pos, cache=cache):
            x = o.clone()
            upstream.rocm_inverse_rope_rows_(x, pos, cache, 64)
            return x

        cuda_fn()
        stock_fn()
        c = _time(cuda_fn, args.warmup, args.iters)
        s = _time(stock_fn, args.warmup, args.iters)
        cp, sp = _pct(c, 0.5), _pct(s, 0.5)
        speed = sp / cp if cp else float("nan")
        perf.append(
            {
                "tokens": tokens,
                "heads": args.heads,
                "cuda_p50_ms": cp,
                "stock_triton_p50_ms": sp,
                "speedup_p50": speed,
            }
        )
        print(
            f"perf T={tokens}: cuda={cp:.4f}ms stock={sp:.4f}ms speedup={speed:.3f}x"
        )

    payload = {
        "device": torch.cuda.get_device_name(0),
        "hip": torch.version.hip,
        "torch": torch.__version__,
        "op_source": source,
        "accuracy": acc,
        "perf": perf,
        "result": "PASS" if all(r["pass"] for r in acc) else "FAIL",
    }
    text = json.dumps(payload, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
        print("JSON ->", args.json_out)
    else:
        print(text)
    print("RESULT", payload["result"])
    return 0 if payload["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
