#!/usr/bin/env python3
"""GPU differential checks for configurable draft rows and draft positions."""
import argparse
import json
from pathlib import Path

import torch
from hcu_vllm_das.dspark_ll_quant import quantize_receiver
from hcu_vllm_das.dspark_greedy_pointwise import greedy
from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8


def exact(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def main(output):
    torch.manual_seed(20261120)
    report = {'passed': False, 'receiver': [], 'greedy': []}
    for capacity in (128, 256, 320, 384, 512):
        rows, experts, hidden = capacity * 8, 16, 5120
        x = torch.randn((experts, rows, hidden), device='cuda', dtype=torch.bfloat16)
        counts = torch.zeros(experts, device='cuda', dtype=torch.int32)
        quantize_receiver(x, counts)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, scales = quantize_receiver(x, counts)
        for count in (0, 1, 17, 64, rows):
            values = torch.tensor([min(rows, count + e % 3) if count else 0
                for e in range(experts)], device='cuda', dtype=torch.int32)
            counts.copy_(values)
            x.normal_()
            graph.replay()
            active = torch.arange(rows, device='cuda')[None, :] < counts[:, None]
            if active.any():
                expected = per_token_quant_int8(x[active])
                exact(expected[0], q[active]); exact(expected[1], scales[active])
            report['receiver'].append({'capacity': capacity, 'count': count})
        del graph, q, scales, x
        torch.cuda.empty_cache()
    vocab, rank = 129280, 256
    w1 = torch.randn((vocab, rank), device='cuda', dtype=torch.bfloat16) / 8
    w2 = torch.randn_like(w1) / 8
    for requests, steps in ((1,1), (17,3), (32,5), (64,5), (64,7)):
        base = torch.randn((requests, steps, vocab), device='cuda', dtype=torch.bfloat16)
        anchors = torch.randint(vocab, (requests,), device='cuda')
        def native():
            prev = anchors
            result = []
            for pos in range(steps):
                bias = torch.nn.functional.linear(torch.nn.functional.embedding(prev, w1), w2)
                prev = (base[:, pos] + bias).argmax(-1)
                result.append(prev)
            return torch.stack(result, 1)
        actual = greedy(w1, w2, base, anchors)
        exact(native(), actual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = greedy(w1, w2, base, anchors)
        for _ in range(3):
            base.normal_()
            anchors.random_(vocab)
            graph.replay()
            exact(native(), captured)
        report['greedy'].append({'requests': requests, 'steps': steps})
    torch.cuda.synchronize()
    report['passed'] = True
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
