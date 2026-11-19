#!/usr/bin/env python3
"""Actual owned attention path versus installed native Q/KV, including graphs."""
import argparse
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

torch.cuda.set_device(0)
from vllm import _custom_ops  # noqa: F401: register installed native operators
from hcu_vllm_das import fp4_indexer, qkv_split
from triton.testing import do_bench_cudagraph


def native(q, kv, cache, slots, pos, cs):
    return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, cache.view(cache.shape[0], -1), slots, pos, cs,
        64, 1e-6, cache.shape[1], False, False, True, False)


def owner(cache, cs, slots):
    layer = object.__new__(fp4_indexer.Fp4Attention)
    torch.nn.Module.__init__(layer)
    layer.use_pcp = False
    layer.kv_mxfp8 = False
    assert not layer.accepts_unnormed_unroped_query
    layer.n_local_heads = layer.padded_heads = 64
    layer.swa_cache_layer = NS(prefix='swa', kv_cache=cache)
    layer.rotary_emb = NS(cos_sin_cache=cs)
    layer.eps = 1e-6
    metadata = {'swa': NS(block_size=cache.shape[1], slot_mapping=slots)}
    return lambda q, kv, pos: layer._fused_qnorm_rope_kv_insert(q, kv, pos, metadata)


def check(actual, expected, cache, reference):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
        'Q mismatch', (actual.float() - expected.float()).abs().max().item())
    assert torch.equal(cache, reference), 'KV cache byte mismatch'
    assert torch.isfinite(actual).all().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--block-size', type=int, choices=(32, 64), default=64)
    parser.add_argument('--page-padding', type=int, default=64)
    parser.add_argument('--seed', type=int, default=194241)
    parser.add_argument('--validate-native-sha256', help='Probe a named new native build without admitting it in serving')
    opts = parser.parse_args()
    native_sha = hashlib.sha256(Path(find_spec('vllm._C_stable_libtorch').origin).read_bytes()).hexdigest()
    validation_only = bool(opts.validate_native_sha256)
    if validation_only:
        assert native_sha == opts.validate_native_sha256, 'probe native fingerprint differs'
        # This process contains isolated differential tests, not a server.
        # Production keeps its old fingerprint until all native consumers pass.
        qkv_split._NATIVE_BUILD_MATCHES = True
    assert qkv_split._NATIVE_BUILD_MATCHES, 'native build is outside verified contract'
    torch.manual_seed(opts.seed)
    angle = torch.randn(8192, 32, device='cuda')
    cs = torch.cat((angle.cos(), angle.sin()), dim=1)
    cs_saved = cs.clone()
    eager, replays, timings = [], [], []
    buckets = (1, 4, 8, 16, 32, 40, 48, 64)
    for tokens in buckets:
        q = torch.empty(tokens, 64, 512, device='cuda', dtype=torch.bfloat16)
        kv = torch.empty(tokens, 512, device='cuda', dtype=torch.bfloat16)
        pos = torch.randint(0, 8192, (tokens,), device='cuda', dtype=torch.int64)
        for length in sorted({0, tokens // 2, tokens}):
            capacity = 4 * opts.block_size
            slots = torch.randperm(capacity, device='cuda', dtype=torch.int64)[:length].clone()
            width = opts.block_size * 584
            backing = torch.full((4, width + opts.page_padding), 165, device='cuda', dtype=torch.uint8)
            cache = backing[:, :width].view(4, opts.block_size, 584)
            padding = backing[:, width:].clone()
            run = owner(cache, cs, slots)
            for magnitude in (0., .001, .25, 1., 8., 100.):
                q.normal_().mul_(magnitude)
                kv.normal_().mul_(magnitude)
                for negative_slots in (False, True):
                    slots.copy_(torch.randperm(capacity, device='cuda')[:length])
                    if negative_slots:
                        slots.fill_(-1)
                    cache.random_(0, 256)
                    initial = cache.clone()
                    reference = initial.clone()
                    saved = [x.clone() for x in (q, kv, slots, pos)]
                    expected = native(q, kv, reference, slots, pos, cs)
                    for variant in ('helper', 'owned'):
                        cache.copy_(initial)
                        actual = (qkv_split.split_qkv(q, kv, cache, slots, pos, cs)
                                  if variant == 'helper' else run(q, kv, pos))
                        check(actual, expected, cache, reference)
                        assert torch.equal(backing[:, width:], padding), 'page padding changed'
                        assert all(torch.equal(x.view(torch.uint8), y.view(torch.uint8))
                                   for x, y in zip((q, kv, slots, pos), saved))
                    eager.append(dict(tokens=tokens, slots=length, magnitude=magnitude,
                                      all_negative_slots=negative_slots, variants=2))
            # Cancellation, signed zero and identity/quarter-turn angles.
            for angle_case in (0, 1):
                cs[:, :32] = 1. if angle_case == 0 else 0.
                cs[:, 32:] = 0. if angle_case == 0 else 1.
                q.fill_(1.)
                q[..., 1::4] = -0.
                q[..., ::4] = 0.
                kv.copy_(q[:, 0])
                cache.random_(0, 256)
                reference = cache.clone()
                expected = native(q, kv, reference, slots, pos, cs)
                actual = run(q, kv, pos)
                check(actual, expected, cache, reference)
                eager.append(dict(tokens=tokens, slots=length, crafted_angle=angle_case))
            cs.copy_(cs_saved)
            # Capture the real owned method, then change every device payload.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run(q, kv, pos)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run(q, kv, pos)
            for replay in range(5):
                q.normal_().mul_(2 ** replay)
                kv.normal_().mul_(2 ** replay)
                pos.random_(0, 8192)
                cs.copy_(cs_saved.roll(replay + 1, dims=0))
                slots.copy_(torch.randperm(capacity, device='cuda')[:length])
                slots[::3] = -1
                cache.random_(0, 256)
                reference = cache.clone()
                expected = native(q, kv, reference, slots, pos, cs)
                graph.replay()
                check(captured, expected, cache, reference)
                assert torch.equal(backing[:, width:], padding), 'page padding changed on replay'
                replays.append(dict(tokens=tokens, slots=length, replay=replay))
            del graph
            cs.copy_(cs_saved)
        slots = torch.arange(tokens, device='cuda')
        pos = slots + 4096
        run = owner(cache, cs, slots)
        for variant, function in [('native', lambda: native(q, kv, cache, slots, pos, cs)),
                                  ('owned', lambda: run(q, kv, pos))]:
            timings.append(dict(tokens=tokens, variant=variant,
                                us=do_bench_cudagraph(function, rep=100) * 1000))
    assert len(eager) == 23 * 14 and len(replays) == 23 * 5
    result = dict(accepted=True, block_size=opts.block_size, page_padding=opts.page_padding,
                  native_sha256=native_sha, validation_only=validation_only,
                  seed=opts.seed, eager=eager, replays=replays, timings=timings,
                  gate='Q BF16 bytes and entire KV cache bytes exact; input immutable')
    with open(opts.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(accepted=True, eager=len(eager), replays=len(replays), timings=timings)))


if __name__ == '__main__':
    main()
