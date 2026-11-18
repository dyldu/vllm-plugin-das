#!/usr/bin/env python3
"""Isolate Q RoPE and the unchanged native KV branch; no model route is changed."""
import argparse
import json

import torch

torch.cuda.set_device(0)
from vllm import _custom_ops  # register the installed native operators
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph


@triton.jit
def q_rope(Q, Pos, CS, Out, MODE: tl.constexpr, RTZ: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1) * 4 + tl.arange(0, 4)[:, None]
    col = tl.arange(0, 512)[None, :]
    raw = tl.load(Q + token * 32768 + head * 512 + col).to(tl.float32)
    peer = tl.load(Q + token * 32768 + head * 512 + (col ^ 1), col >= 448, other=0).to(tl.float32)
    pos = tl.load(Pos + token)
    idx = (col - 448) // 2
    c = tl.load(CS + pos * 64 + idx, col >= 448, other=1.)
    s = tl.load(CS + pos * 64 + 32 + idx, col >= 448, other=0.)
    even = col % 2 == 0
    if MODE == 0:
        value = tl.where(even, tl.fma(raw, c, -peer * s), tl.fma(peer, s, raw * c))
    elif MODE == 1:
        value = tl.where(even, tl.fma(-peer, s, raw * c), tl.fma(raw, c, peer * s))
    elif MODE == 3:
        value = tl.where(even, tl.fma(raw, c, -peer * s), tl.fma(raw, c, peer * s))
    elif MODE == 4:
        value = tl.where(even, tl.fma(-peer, s, raw * c), tl.fma(peer, s, raw * c))
    elif MODE == 7 or MODE == 8:
        ordinary = tl.where(even, tl.fma(raw, c, -peer * s), tl.fma(raw, c, peer * s))
        unfused = raw * c + tl.where(even, -peer * s, peer * s)
        if MODE == 7:
            value = tl.where(col % 16 == 0, unfused, ordinary)
        else:
            value = tl.where(col % 16 < 2, unfused, ordinary)
    else:
        value = raw * c + tl.where(even, -peer * s, peer * s)
    value = tl.where(col >= 448, value, raw)
    if RTZ:
        value = (value.to(tl.int32, bitcast=True) >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
    tl.store(Out + token * 32768 + head * 512 + col, value)


def native(q, kv, cache, slots, pos, cs):
    return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, cache.view(cache.shape[0], -1), slots, pos, cs,
        64, 1e-6, 64, False, True, True, False)


def split(q, kv, cache, slots, pos, cs, mode, rtz):
    if mode == 6:
        out = native(q, kv, cache, slots[:0], pos, cs)
    else:
        out = torch.empty_like(q)
        q_rope[(q.shape[0], 16)](q, pos, cs, out, mode, rtz, num_warps=4, enable_fp_fusion=False)
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(kv, cache, slots, pos, cs, 64, None, True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", type=int, nargs="+", default=[7, 8])
    parser.add_argument("--seed", type=int, default=194231)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[0., .001, .25, 1., 8.])
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--include-rtz", action="store_true")
    parser.add_argument("--debug-mismatches", action="store_true")
    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    angle = torch.randn(8192, 32, device="cuda")
    cs = torch.cat((angle.cos(), angle.sin()), dim=-1)
    rows = []
    rtz_modes = (False, True) if opts.include_rtz else (False,)
    for tokens in (1, 8, 40, 48, 64):
        pos = torch.randint(0, 8192, (tokens,), device="cuda", dtype=torch.int64)
        slots = torch.randperm(256, device="cuda", dtype=torch.int64)[:tokens].clone()
        slots[::7] = -1
        for mag in opts.magnitudes:
            q = (torch.randn(tokens, 64, 512, device="cuda") * mag).bfloat16()
            kv = (torch.randn(tokens, 512, device="cuda") * mag).bfloat16()
            original = torch.full((4, 64, 528), 165, device="cuda", dtype=torch.uint8)
            cache_ref = original.clone()
            expected = native(q, kv, cache_ref, slots, pos, cs)
            for mode in opts.modes:
                for rtz in rtz_modes:
                    cache = original.clone()
                    actual = split(q, kv, cache, slots, pos, cs, mode, rtz)
                    row = dict(tokens=tokens, magnitude=mag, mode=mode, rtz=rtz,
                        q_different_bf16=(actual.view(torch.int16) != expected.view(torch.int16)).sum().item(),
                        q_max_abs=(actual.float() - expected.float()).abs().max().item(),
                        cache_different_bytes=(cache != cache_ref).sum().item(),
                        finite=torch.isfinite(actual).all().item())
                    rows.append(row)
                    if opts.debug_mismatches and row['q_different_bf16']:
                        for t, h, c in (actual.view(torch.int16) != expected.view(torch.int16)).nonzero()[:4].tolist():
                            i = (c - 448) // 2
                            print(json.dumps(dict(case=row, token=t, head=h, column=c,
                                pair=q[t,h,c//2*2:c//2*2+2].float().tolist(),
                                cos_sin=[cs[pos[t],i].item(),cs[pos[t],i+32].item()],
                                actual=actual[t,h,c].item(),reference=expected[t,h,c].item())), flush=True)
    assert len(rows) == 5 * len(opts.magnitudes) * len(opts.modes) * len(rtz_modes)
    admitted = [(m, r) for m in opts.modes for r in rtz_modes if all(
        x['q_different_bf16'] == 0 and x['cache_different_bytes'] == 0 and x['finite']
        for x in rows if x['mode'] == m and x['rtz'] == r)]
    timings = []
    for tokens in (() if opts.no_bench else (1, 8, 40, 48, 64)):
        q = torch.randn(tokens, 64, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
        slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
        pos = slots + 4000
        cache = torch.zeros(4, 64, 528, device="cuda", dtype=torch.uint8)
        timings.append(dict(tokens=tokens, variant="native", us=do_bench_cudagraph(
            lambda: native(q, kv, cache, slots, pos, cs), rep=50) * 1000))
        for mode, rtz in admitted:
            timings.append(dict(tokens=tokens, variant="split", mode=mode, rtz=rtz,
                us=do_bench_cudagraph(lambda: split(q, kv, cache, slots, pos, cs, mode, rtz), rep=50) * 1000))
    result = dict(precision=rows, admitted_modes=admitted, timing=timings)
    with open(opts.output, 'w') as handle: json.dump(result, handle, indent=2)
    print(json.dumps(dict(admitted_modes=admitted, timing=timings)), flush=True)


if __name__ == '__main__':
    main()
