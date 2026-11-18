#!/usr/bin/env python3
"""Compare actual stock codes/scales and graph cost, including changing counts."""
import json
import argparse
import torch

torch.cuda.set_device(0)
from vllm import _custom_ops as ops
from hcu_vllm_das.moe.masked_token_quant import masked_token_quant
from triton.testing import do_bench_cudagraph

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
torch.manual_seed(190319)
shape = (16, 512, 5120)
x = torch.empty(shape, dtype=torch.bfloat16, device='cuda')
counts = torch.zeros(16, dtype=torch.int32, device='cuda')
patterns = {'empty': [0]*16, 'balanced8': [8]*16,
            'balanced64': [64]*16, 'uneven': list(range(16)),
            'hot': [320]+[0]*15, 'full': [512]*16}
results = []
checks = 0
for name, nums in patterns.items():
    counts.copy_(torch.tensor(nums, dtype=torch.int32, device='cuda'))
    mask = torch.arange(512, device='cuda')[None, :] < counts[:, None]
    for magnitude in (0., .000001, .001, .25, 1., 8., 100., 10000.):
        x.normal_().mul_(magnitude)
        refq, refs = ops.scaled_fp8_quant(x.view(-1,5120), use_per_token_if_dynamic=True)
        refq, refs = refq.view(shape), refs.view(shape[:2])
        for lanes in (4, 8, 16, 32, 64):
            q, s = masked_token_quant(x, counts, lanes=lanes)
            assert torch.equal(q.view(torch.uint8)[mask], refq.view(torch.uint8)[mask]), (name,magnitude,lanes,'codes')
            assert torch.equal(s[mask], refs[mask]), (name,magnitude,lanes,'scales')
            checks += 1
    row = {'pattern': name, 'active_rows': sum(nums)}
    row['stock_us'] = do_bench_cudagraph(lambda: ops.scaled_fp8_quant(x.view(-1,5120),use_per_token_if_dynamic=True),rep=30)*1000
    row['masked_us'] = {}
    for lanes in (4, 8, 16, 32, 64):
        row['masked_us'][lanes] = do_bench_cudagraph(lambda: masked_token_quant(x, counts, lanes=lanes),rep=30)*1000
    results.append(row)
    print(json.dumps(row),flush=True)
# Counts are device data, not frozen capture metadata. Reuse one graph while
# changing both payload and active rows; compare all newly active locations.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    q, s = masked_token_quant(x, counts)
for name, nums in patterns.items():
    counts.copy_(torch.tensor(nums, dtype=torch.int32, device='cuda'))
    x.normal_()
    graph.replay()
    rq, rs = ops.scaled_fp8_quant(x.view(-1,5120),use_per_token_if_dynamic=True)
    mask = torch.arange(512,device='cuda')[None,:] < counts[:,None]
    assert torch.equal(q.view(torch.uint8)[mask],rq.view(shape).view(torch.uint8)[mask]), name
    assert torch.equal(s[mask],rs.view(shape[:2])[mask]), name
    checks += 1
print('EXACT_PASS',checks,flush=True)
# Include every finite BF16 bit pattern (subnormals, signed zero, max finite),
# distributed through active rows at several independently chosen row maxima.
bits = torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
bits = bits[torch.isfinite(bits)]
for maximum in (1e-6, .01, 1., 100., 1e10, float(torch.finfo(torch.bfloat16).max)):
    vals = bits[bits.abs().float() <= maximum]
    counts.fill_(16)
    x.zero_()
    x[:, :16].copy_(vals.repeat((16*16*5120+vals.numel()-1)//vals.numel())[:16*16*5120].view(16,16,5120))
    x[:, :16, -1] = maximum
    rq, rs = ops.scaled_fp8_quant(x.view(-1,5120),use_per_token_if_dynamic=True)
    q, s = masked_token_quant(x,counts)
    assert torch.equal(q[:,:16].view(torch.uint8),rq.view(shape)[:,:16].view(torch.uint8)), maximum
    assert torch.equal(s[:,:16],rs.view(shape[:2])[:,:16]), maximum
    checks += 1
print('ALL_FINITE_BF16_PASS',checks,flush=True)
with open(args.output,'w') as f: json.dump(dict(checks=checks,codes_scales_exact=True,results=results),f,indent=2)
