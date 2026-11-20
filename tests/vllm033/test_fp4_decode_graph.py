# SPDX-License-Identifier: Apache-2.0
"""Real LightOp FP4 reader graph replay, including changed device metadata."""
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import fp4_indexer_lightop as reader
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata


@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('rows,context_len,topk', [
    (2, 128, 4), (8, 32768, 1024), (48, 4096, 1024), (64, 32768, 1024),
])
def test_decode_graph_reads_changed_query_cache_and_metadata(
        monkeypatch, ratio, rows, context_len, topk):
    if not torch.cuda.is_available():
        pytest.skip('HCU required')
    torch.cuda.set_device(0)
    torch.manual_seed(79)
    page_records = 64 // ratio
    pages_per_row = (context_len + page_records - 1) // page_records
    pages = rows * pages_per_row
    cache = torch.empty((pages, page_records, 68), dtype=torch.uint8, device='cuda')
    # The ABI stores page-local payload/scale planes, not 68-byte records.
    # Randomizing record-shaped slices corrupts UE8M0 scales (including 255/NaN).
    cache_bytes = cache.view(pages, page_records * 68)
    payload = cache_bytes[:, :page_records * 64]
    payload.random_(0, 256)
    cache_bytes[:, page_records * 64:] = 127
    table = torch.arange(pages, device='cuda', dtype=torch.int32).reshape(rows, pages_per_row)
    lengths = torch.full((rows,), 37, device='cuda', dtype=torch.int32)
    query = reader.Fp4Query(
        torch.randint(256, (rows, 32, 64), device='cuda', dtype=torch.uint8),
        torch.full((rows, 32), 0x7F7F7F7F, device='cuda', dtype=torch.int32),
        torch.randn(rows, 32, 128, device='cuda', dtype=torch.bfloat16))
    weights = torch.rand(rows, 32, device='cuda') / 32
    hidden = torch.empty(rows, 128, device='cuda', dtype=torch.bfloat16)
    layer = NS(k_cache=NS(prefix='indexer', kv_cache=cache),
               topk_indices_buffer=torch.empty(rows, topk, device='cuda', dtype=torch.int32),
               topk_tokens=topk, candidate_blocks=None, candidate_block_size=0,
               candidate_write=False, max_model_len=context_len, compress_ratio=ratio)
    meta = DeepseekV32IndexerMetadata(
        seq_lens=lengths, max_seq_len=context_len,
        slot_mapping=torch.zeros(rows, device='cuda', dtype=torch.int64),
        num_decodes=rows, num_decode_tokens=rows, num_prefills=0, num_prefill_tokens=0,
        prefill=None, block_table=table,
        decode=NS(block_table=table, seq_lens=lengths[:, None],
                  decode_lens=torch.ones(rows, device='cuda', dtype=torch.int32),
                  requires_padding=False))
    context = NS(attn_metadata={'indexer': meta}, cudagraph_runtime_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(reader, 'get_forward_context', lambda: context)
    provider = reader.load_providers()
    scored = []

    def decode(*args, **kwargs):
        logits = provider.decode(*args, **kwargs)
        scored.append(logits)
        return logits

    monkeypatch.setattr(reader, 'providers', lambda: reader.Providers(
        provider.gather, provider.prefill, decode, provider.native, provider.plan))
    cfg = NS(cache_config=NS(block_size=64), speculative_config=None,
             scheduler_config=NS(max_num_seqs=rows, max_num_batched_tokens=rows))
    reader.reserve_decode_staging(cfg, context_len, ratio, hidden.device)

    def run():
        return reader.execute(layer, hidden, query, weights)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    captured_logits = scored[-1]
    for pair in ((context_len, context_len - 1), (52, 19), (17, 57), (0, 7), (4, 0)):
        values = [pair[row % 2] for row in range(rows)]
        lengths.copy_(torch.tensor(values, device='cuda', dtype=torch.int32))
        table.copy_(table.flip(0))
        query[0].random_(0, 256)
        weights.uniform_(0, 1 / 32)
        payload.random_(0, 256)
        graph.replay()
        replayed = captured.clone()
        replayed_logits = captured_logits.clone()
        expected = run().clone()
        torch.cuda.synchronize()
        # Top-k may permute equal-score IDs between eager and graph launches.
        # Require byte-exact native logits and selected scores instead, while
        # retaining exact padding, valid counts and unique in-range indices.
        valid_scores = torch.arange(context_len, device='cuda')[None, :] < lengths[:, None]
        assert torch.equal(replayed_logits[valid_scores].view(torch.uint8),
                           scored[-1][valid_scores].view(torch.uint8))
        assert torch.equal(replayed < 0, expected < 0)
        replay_scores = replayed_logits.gather(1, replayed.clamp_min(0).long())
        expected_scores = scored[-1].gather(1, expected.clamp_min(0).long())
        valid_output = replayed >= 0
        assert torch.equal(replay_scores[valid_output].view(torch.uint8),
                           expected_scores[valid_output].view(torch.uint8))
        for row, length in enumerate(values):
            for output in (replayed, expected):
                valid = output[row][output[row] >= 0]
                assert valid.numel() == min(length, layer.topk_tokens)
                assert (valid < length).all()
                assert torch.unique(valid).numel() == valid.numel()


def test_full_graph_refuses_prefill_metadata(monkeypatch):
    cache = torch.empty(2, 64, 68, dtype=torch.uint8)
    layer = NS(k_cache=NS(prefix='indexer', kv_cache=cache))
    context = NS(cudagraph_runtime_mode=CUDAGraphMode.FULL,
                 attn_metadata={'indexer': NS(num_prefill_tokens=1, prefill=object())})
    monkeypatch.setattr(reader, 'get_forward_context', lambda: context)
    with pytest.raises(ValueError, match='decode-only'):
        reader.execute(layer, None, None, None)
