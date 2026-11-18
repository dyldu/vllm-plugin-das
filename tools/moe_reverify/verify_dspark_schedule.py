#!/usr/bin/env python3
"""Compare the actual private draft forward with the installed native body.

The three-layer fixtures isolate the changed schedules. Serving HE32 and paired
TPOT/profile are separate required gates; these fixtures do not replace them.
"""
import argparse
import json

import torch

torch.cuda.set_device(0)
torch.set_float32_matmul_precision('highest')
from vllm import _custom_ops  # noqa: F401
from hcu_vllm_das import dspark_schedule as owned
from triton.testing import do_bench_cudagraph


def exact(a, b):
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
        'byte mismatch', (a.float() - b.float()).abs().max().item())


def capture(function):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = function()
    return graph, output


class ForwardLayer(torch.nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.x = torch.empty(tokens, 5120, device='cuda', dtype=torch.bfloat16)
        self.residual = torch.empty(tokens, 4, 5120, device='cuda', dtype=torch.bfloat16)
        self.post = torch.empty(tokens, 4, 1, device='cuda')
        self.comb = torch.empty(tokens, 4, 4, device='cuda')
        self.pre = torch.empty(tokens, 4, device='cuda')

    def change(self, magnitude):
        self.x.normal_().mul_(magnitude)
        self.residual.normal_().mul_(magnitude)
        self.post.uniform_(0., 1.)
        self.comb.uniform_(-1., 1.)
        self.pre.uniform_(0., 1.)

    def forward(self, *unused):
        return self.x, self.residual, self.post, self.comb, self.pre


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(194319)
    buckets = (1, 4, 8, 16, 32, 40, 48, 64)
    forward_cases, timings = [], []
    # Exercise the actual cloned forward including unchanged hc_collapse.
    for tokens in buckets:
        model = object.__new__(owned.DraftModel)
        torch.nn.Module.__init__(model)
        model.hc_mult = 4
        model.use_sequence_parallel = False
        model.layers = torch.nn.ModuleList([ForwardLayer(tokens) for _ in range(3)])
        ids = torch.zeros(tokens, device='cuda', dtype=torch.int64)
        pos = torch.arange(tokens, device='cuda')
        inputs = torch.randn(tokens, 5120, device='cuda', dtype=torch.bfloat16)
        for layer in model.layers:
            layer.change(1.)
        run = lambda: model(ids, pos, inputs)
        native_run = lambda: owned.native.DSparkDeepseekV4Model.forward(model, ids, pos, inputs)
        graph, output = capture(run)
        for magnitude in (0., .001, .25, 1., 8., 100.):
            inputs.normal_().mul_(magnitude)
            for layer in model.layers:
                layer.change(magnitude)
            saved = [t.clone() for layer in model.layers for t in
                     (layer.x, layer.residual, layer.post, layer.comb, layer.pre)]
            expected = native_run()
            exact(run(), expected)
            graph.replay(); exact(output, expected)
            assert torch.isfinite(output).all().item()
            for value, before in zip([t for layer in model.layers for t in
                    (layer.x, layer.residual, layer.post, layer.comb, layer.pre)], saved):
                exact(value, before)
            forward_cases.append(dict(tokens=tokens, magnitude=magnitude, eager=True, graph=True))
        del graph
        for name, function in (('native_forward', native_run), ('private_forward', run)):
            timings.append(dict(tokens=tokens, variant=name,
                                us=do_bench_cudagraph(function, rep=100) * 1000))
    result = dict(accepted=True, forward=forward_cases, timings=timings,
                  gate='actual private/native forward BF16 byte-exact under changed-input graphs; inputs immutable',
                  context_precompute='unchanged native method; KV-only shortcut rejected on one-byte FP8 counterexample')
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in result.items()}))


if __name__ == '__main__':
    main()
