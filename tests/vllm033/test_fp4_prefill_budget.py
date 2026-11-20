# SPDX-License-Identifier: Apache-2.0
"""Bound real score allocations and release them between query/KV chunks."""
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from hcu_vllm_das import fp4_indexer_lightop as reader


@pytest.mark.parametrize('extent,budget', [
    (0, 16), (1, 17), (131071, 512 * 1024**2),
    (131073, 512 * 1024**2), (1048576, 512 * 1024**2),
])
def test_plan_respects_aligned_bytes_and_runtime_extent(extent, budget):
    width, rows = reader.prefill_logits_plan(NS(max_seqlen_k=extent), 1048576, budget)
    assert width >= extent and width % 4 == 0 and width - extent <= 4
    assert rows * width * 4 <= budget < (rows + 1) * width * 4


@pytest.mark.parametrize('extent,budget', [(1048577, 512 * 1024**2), (-1, 16), (9, 4), (4, 0)])
def test_invalid_plan_rejects_before_allocation(extent, budget):
    with pytest.raises(ValueError, match='FP4 prefill'):
        reader.prefill_logits_plan(NS(max_seqlen_k=extent), 1048576, budget)


@pytest.mark.parametrize('write', [False, True])
def test_chunk_query_slices_preserve_metadata_candidates_and_lifetimes(write):
    rows, tokens, k = 7, 9, 4
    cache = torch.zeros((1, 64, 68), dtype=torch.uint8)
    query = NS(bf16=torch.arange((rows + 2) * 4).reshape(rows + 2, 1, 4).bfloat16())
    weights = torch.arange(rows + 2).float().reshape(-1, 1)
    output = torch.full((rows + 3, k), -77, dtype=torch.int32)
    bank = torch.arange((rows + 3) * 2, dtype=torch.int32).reshape(-1, 2)
    layer = NS(max_model_len=1048576, candidate_blocks=bank,
               candidate_block_size=8, candidate_write=write)
    starts = torch.tensor([0, 0, 0, 3, 3, 3, 3], dtype=torch.int32)
    ends = torch.tensor([1, 2, 3, 7, 8, 9, 9], dtype=torch.int32)
    chunk = NS(token_start=2, token_end=rows + 2, total_seq_lens=tokens,
        max_seqlen_k=6, token_to_seq=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 3, 9], dtype=torch.int32),
        block_table=torch.tensor([[0], [0]], dtype=torch.int32),
        cu_seqlen_ks=starts, cu_seqlen_ke=ends)
    logits_refs, kv_refs, recorded, selected = [], [], [], []

    def gather(flat, slots, records):
        assert all(ref() is None for ref in kv_refs)
        values = torch.zeros((tokens, 64), dtype=torch.int8)
        scales = torch.zeros(tokens, dtype=torch.int32)
        kv_refs.extend((weakref.ref(values), weakref.ref(scales)))
        assert slots.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5]
        return values, scales

    def prefill(q, kv, w, ks, ke, clean, width, candidates, lens):
        assert all(ref() is None for ref in logits_refs)
        recorded.append((q[0].clone(), w.clone(), ks.clone(), ke.clone(), width))
        assert width == 8 and clean is False
        scores = torch.arange(width).float().expand(q[0].shape[0], -1).clone()
        scores.masked_fill_(torch.arange(width)[None, :] >= (ke-ks)[:, None], -float('inf'))
        logits_refs.append(weakref.ref(scores))
        return scores

    def select(logits, lo, hi, out, candidates, block_size, writing):
        selected.append(candidates.clone())
        assert block_size == 8 and writing == write
        values, indices = logits.topk(k, dim=-1)
        out.copy_(indices.int().masked_fill(values == -float('inf'), -1))

    providers = NS(gather=gather, prefill=prefill)
    for _ in range(2):
        reader._prefill_chunk(layer, chunk, cache, cache.reshape(1, -1), query,
                              weights, output, providers, select, 2 * 8 * 4)
        assert all(ref() is None for ref in (*logits_refs, *kv_refs))
    assert [r[0].shape[0] for r in recorded] == [2, 2, 2, 1] * 2
    torch.testing.assert_close(torch.cat([r[0] for r in recorded[:4]]), query.bf16[2:9])
    torch.testing.assert_close(torch.cat([r[1] for r in recorded[:4]]), weights[2:9])
    assert torch.equal(torch.cat([r[2] for r in recorded[:4]]), starts)
    assert torch.equal(torch.cat([r[3] for r in recorded[:4]]), ends)
    assert torch.equal(torch.cat(selected[:4]), bank[2:9])
    assert output[:2].eq(-77).all() and output[9:].eq(-77).all()
    for row, count in enumerate((ends-starts).tolist()):
        expected = set(range(max(0, count-k), count))
        assert set(output[row+2][output[row+2] >= 0].tolist()) == expected
