"""Native FP4 prefill byte oracle and incremental allocator-peak measurement.

Run on an idle HCU, with the plugin/adapter source overlays on PYTHONPATH.
The logits budget is per tensor, not a limit on total device allocations.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from hcu_vllm_das import fp4_indexer_lightop as reader
from hcu_vllm_das.v41_indexer import _select


def peak_start():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def peak_end(before):
    torch.cuda.synchronize()
    return dict(initial_allocated_bytes=before,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                incremental_peak_bytes=torch.cuda.max_memory_allocated()-before,
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def run(output):
    torch.cuda.set_device(0)
    torch.manual_seed(79)
    native = reader.load_providers()
    rows, topk, maximum, records = 2048, 512, 1048576, 32
    extents = (131073, 65539)
    total = sum(extents)
    pages_per_request = [(n+records-1)//records for n in extents]
    pages = sum(pages_per_request)
    cache = torch.empty((pages, records, 68), dtype=torch.uint8, device='cuda')
    flat = cache.view(pages, records*68)
    flat[:, :records*64].random_(0, 256)
    flat[:, records*64:] = 127
    table = torch.zeros((2, max(pages_per_request)), dtype=torch.int32, device='cuda')
    table[0, :pages_per_request[0]] = torch.arange(pages_per_request[0], device='cuda', dtype=torch.int32)
    table[1, :pages_per_request[1]] = torch.arange(pages_per_request[0], pages, device='cuda', dtype=torch.int32)
    q = torch.randn(rows, 32, 128, device='cuda', dtype=torch.bfloat16)
    weights = torch.rand(rows, 32, device='cuda')/32
    starts = torch.cat((torch.zeros(rows//2, dtype=torch.int32, device='cuda'),
                        torch.full((rows//2,), extents[0], dtype=torch.int32, device='cuda')))
    lengths = torch.cat([torch.arange(n-rows//2+1, n+1, dtype=torch.int32, device='cuda')
                         for n in extents])
    lengths[:4] = torch.tensor([0, 3, 15, 511], dtype=torch.int32, device='cuda')
    ends = starts+lengths
    chunk = NS(token_start=0, token_end=rows, total_seq_lens=total,
               max_seqlen_k=max(extents), block_table=table,
               cu_seq_lens=torch.tensor([0, extents[0], total], dtype=torch.int32, device='cuda'),
               token_to_seq=torch.cat([torch.full((n,), i, dtype=torch.int32, device='cuda')
                                       for i, n in enumerate(extents)]),
               cu_seqlen_ks=starts, cu_seqlen_ke=ends)
    layer = NS(max_model_len=maximum, candidate_blocks=None,
               candidate_block_size=0, candidate_write=False)
    query = NS(bf16=q)
    out = torch.empty((rows, topk), dtype=torch.int32, device='cuda')
    zero = torch.zeros_like(lengths)
    # Initialize scorer/selection before measuring the large allocation.
    slots = torch.arange(64, dtype=torch.int64, device='cuda')
    warm_k, warm_sf = native.gather(flat, slots, records)
    warm = native.prefill((q[:2], None), (warm_k, warm_sf), weights[:2],
                         zero[:2], torch.full_like(zero[:2], 64), False, 64, None, None)
    _select(warm, zero[:2], torch.full_like(zero[:2], 64), out[:2], None, 0, False)
    del warm, warm_k, warm_sf, slots
    before = peak_start()
    positions = torch.arange(total, dtype=torch.int64, device='cuda')
    request_ids = chunk.token_to_seq.long()
    local = positions-chunk.cu_seq_lens[request_ids]
    slots = (table[request_ids, local//records].long()*records+local%records).contiguous()
    k, sf = native.gather(flat, slots, records)
    old = native.prefill((q, None), (k, sf), weights, starts, ends, False, maximum, None, None)
    assert old.shape == (rows, maximum) and old.dtype == torch.float32
    _select(old, zero, lengths, out, None, 0, False)
    old_peak = peak_end(before)
    width, _ = reader.prefill_logits_plan(chunk, maximum, 512*1024**2)
    oracle = old[:, :width].clone()
    reference_ids = out.clone()
    reference_scores = oracle.gather(1, reference_ids.clamp_min(0).long())
    valid = reference_ids >= 0
    del old, k, sf, positions, request_ids, local, slots
    reports = []
    for mib in (64, 512):
        budget = mib*1024**2
        calls, offset = [], [0]

        def checked(*args):
            logits = native.prefill(*args)
            size = logits.numel()*logits.element_size()
            assert size <= budget and logits.shape[1] == width
            lo, hi = offset[0], offset[0]+logits.shape[0]
            assert torch.equal(logits.view(torch.uint8), oracle[lo:hi].view(torch.uint8))
            offset[0] = hi
            calls.append(dict(shape=list(logits.shape), logits_bytes=size))
            return logits

        providers = NS(gather=native.gather, prefill=checked)
        # Prove the wide causal-prefix contract reaches native TopK instead
        # of silently passing through the old generic torch.topk fallback.
        from hcu_adapter.kernels import indexer_topk_lightop
        selected_natively = []
        original_select = indexer_topk_lightop.select_topk

        def checked_select(logits, row_starts, row_ends, count, *, valid_lengths=None):
            assert valid_lengths is not None
            result = original_select(logits, row_starts, row_ends, count,
                                     valid_lengths=valid_lengths)
            assert result is not None, 'Wide FP4 prefill lost native selection'
            selected_natively.append(logits.shape[0])
            return result

        indexer_topk_lightop.select_topk = checked_select
        try:
            reader._prefill_chunk(layer, chunk, cache, flat, query, weights, out,
                                  providers, _select, budget)
        finally:
            indexer_topk_lightop.select_topk = original_select
        assert offset[0] == rows and torch.equal(out < 0, ~valid)
        assert sum(selected_natively) == rows
        selected_scores = oracle.gather(1, out.clamp_min(0).long())
        assert torch.equal(selected_scores[valid].view(torch.uint8),
                           reference_scores[valid].view(torch.uint8))
        # Sort IDs independently to detect duplicates/in-range violations.
        ordered = out.masked_fill(out < 0, maximum).sort(dim=-1).values
        live_pairs = ordered[:, 1:] < maximum
        assert not ((ordered[:, 1:] == ordered[:, :-1]) & live_pairs).any().item()
        assert torch.equal((out >= 0).sum(1), lengths.clamp_max(topk).long())
        assert ((out < lengths[:, None]) | (out < 0)).all().item()
        for row, length in enumerate((0, 3, 15, 511)):
            assert torch.equal(out[row, :length], torch.arange(length, dtype=torch.int32, device='cuda'))
        del selected_scores, ordered, live_pairs
        before = peak_start()
        reader._prefill_chunk(layer, chunk, cache, flat, query, weights, out,
                              native, _select, budget)
        measured = peak_end(before)
        reports.append(dict(logits_budget_bytes=budget, calls=calls,
                            valid_logits_bytes_exact=True, selected_score_ranks_exact=True,
                            padding_unique_ids_short_order_exact=True,
                            wide_prefill_native_selection_verified=True, **measured))
    result = dict(passed=True, device=torch.cuda.get_device_name(),
                  rows=rows, request_kv_extents=list(extents), max_model_len=maximum,
                  old_logits_bytes=rows*maximum*4, old_operation_peak=old_peak,
                  candidate_measurements=reports,
                  reader_sha256=hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest(),
                  measurement='PyTorch allocator peak; persistent inputs and oracle excluded from incremental peak; not whole-device peak')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
