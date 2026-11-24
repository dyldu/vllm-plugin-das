# SPDX-License-Identifier: Apache-2.0
"""Native add/argmax semantics, including ties, NaNs and padded vocab tiles."""
import pytest
import torch

from hcu_vllm_das.dspark_greedy_pointwise import step


@pytest.mark.parametrize('rows,vocab', [(1, 17), (7, 2049), (65, 4097)])
@pytest.mark.parametrize('case', ['random', 'tie', 'nan', 'infinity', 'rounding'])
@torch.inference_mode()
def test_native_add_argmax_embedding(rows, vocab, case):
    torch.manual_seed(9328)
    base = torch.randn((rows, vocab), device='cuda', dtype=torch.bfloat16)
    bias = torch.randn_like(base)
    if case == 'tie':
        base.fill_(-10)
        bias.zero_()
        base[:, 0] = base[:, -1] = 2
    elif case == 'nan':
        base[:, 1] = base[:, -1] = float('nan')
    elif case == 'infinity':
        base[:, 1] = base[:, -1] = float('inf')
    elif case == 'rounding':
        base.fill_(1)
        bias.zero_()
        # Both additions round to 1 in BF16. FP32 argmax would pick the last.
        bias[:, 0] = 0.0001
        bias[:, -1] = 0.001
    table = torch.randn((vocab, 256), device='cuda', dtype=torch.bfloat16)
    tokens = torch.full((rows + 1, 3), -7, device='cuda', dtype=torch.int64)
    embedding = torch.empty((rows, 256), device='cuda', dtype=torch.bfloat16)
    blocks = (vocab + 2047) // 2048
    maxima = torch.empty((rows, blocks), device='cuda', dtype=torch.float32)
    indices = torch.empty((rows, blocks), device='cuda', dtype=torch.int32)

    def run():
        step(base, bias, table, tokens[:rows, 1], embedding, maxima, indices)

    def check():
        expected = (base + bias).argmax(-1)
        torch.testing.assert_close(tokens[:rows, 1], expected, rtol=0, atol=0)
        torch.testing.assert_close(embedding, table[expected], rtol=0, atol=0)
        assert bool((tokens[:, 0] == -7).all())
        assert bool((tokens[:, 2] == -7).all())
        assert tokens[-1, 1].item() == -7

    run()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for _ in range(3):
        base.copy_(base.roll(1, dims=1))
        bias.copy_(bias.roll(1, dims=1))
        table.normal_()
        graph.replay()
        check()
