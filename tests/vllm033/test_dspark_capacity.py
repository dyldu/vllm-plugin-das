# SPDX-License-Identifier: Apache-2.0
"""DSpark request slots and target/draft row buffers have distinct bounds."""
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import dspark_graph as graph
from hcu_vllm_das import dspark_agrs as agrs


def config(requests, budget, drafts, anchor=True):
    return NS(scheduler_config=NS(max_num_seqs=requests,
        max_num_batched_tokens=budget, max_num_scheduled_tokens=budget),
        speculative_config=NS(num_speculative_tokens=drafts,
            draft_model_config=NS(hf_config=NS(sample_from_anchor=anchor))),
        compilation_config=object(), parallel_config=object())


@pytest.mark.parametrize('requests,budget,drafts,anchor', [
    (16, 64, 3, True), (32, 256, 5, True), (64, 256, 5, True),
    (96, 256, 5, True), (17, 64, 4, False), (64, 384, 5, False)])
def test_private_buffers_cover_short_prefills_and_preserve_target(
        requests, budget, drafts, anchor):
    cfg = config(requests, budget, drafts, anchor)
    query = drafts if anchor else drafts + 1
    capacity = graph.token_capacity(cfg)
    assert agrs.token_capacity(cfg) == capacity
    assert capacity.query_rows == max(budget, requests * query)
    assert capacity.verify_rows <= budget
    local = graph.draft_buffer_config(cfg)
    assert local.scheduler_config.max_num_batched_tokens >= requests * query
    assert local.scheduler_config.max_num_scheduled_tokens == budget
    assert cfg.scheduler_config.max_num_batched_tokens == budget
    assert cfg.scheduler_config.max_num_seqs == requests
    assert local.compilation_config is cfg.compilation_config
    assert local.parallel_config is cfg.parallel_config
    assert graph.draft_buffer_config(local) is local


@pytest.mark.parametrize('requests,budget,drafts', [(65,64,5), (1,0,5), (0,256,5), (1,5,0)])
def test_rejects_invalid_native_bounds(requests, budget, drafts):
    with pytest.raises(ValueError):
        graph.token_capacity(config(requests, budget, drafts))


def test_actual_batch_guard_runs_before_input_writes():
    calls = []
    class Speculator:
        _speculator_name = 'DSpark'
        max_num_reqs, max_num_tokens, num_query_per_req = 64, 320, 5
        _hcu_target_token_budget = 256
        def propose(self, input_batch, dp_sync_state=None, dummy_run=False):
            calls.append(input_batch)
            return 'prepared'
    graph._patch_draft_batch(Speculator)
    graph._patch_draft_batch(Speculator)
    worker = Speculator()
    # A batch of 64 one-token prefills fits the target budget but needs 320
    # draft rows. It must be supported, rather than only testing full decodes.
    assert worker.propose(NS(num_reqs=64, num_tokens=64)) == 'prepared'
    for batch in (NS(num_reqs=65, num_tokens=65), NS(num_reqs=64, num_tokens=257)):
        with pytest.raises(ValueError):
            worker.propose(batch)
    worker.max_num_tokens = 319
    with pytest.raises(ValueError, match='before input preparation'):
        worker.propose(NS(num_reqs=64, num_tokens=64))
    assert len(calls) == 1
    assert worker.propose(NS(num_reqs=64, num_tokens=64),
        dp_sync_state=NS(num_reqs=8), dummy_run=True) == 'prepared'
    worker._speculator_name = 'DFlash'
    assert worker.propose(NS(num_reqs=100, num_tokens=1000)) == 'prepared'


def test_constructor_and_slot_mappings_expand_before_capture(monkeypatch):
    monkeypatch.setattr(graph, 'native_decode_dspark', lambda _: True)
    monkeypatch.setattr(agrs, '_owned_quant_kind', lambda _: 'ct_w4a8')
    calls = []
    class Speculator:
        def __init__(self, vllm_config, device):
            self.vllm_config = vllm_config
            self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self.input_ids = torch.zeros(self.max_num_tokens)
        def set_attn(self, model_state, kv_cache_config, block_tables,
                     target_input_buffers, target_attn_groups):
            calls.append((block_tables, target_input_buffers))
    graph._patch_draft_buffers(Speculator)
    graph._patch_draft_buffers(Speculator)
    cfg = config(64, 256, 5)
    worker = Speculator(cfg, 'cpu')
    assert worker.input_ids.numel() == 320
    assert worker._hcu_target_token_budget == 256
    assert cfg.scheduler_config.max_num_batched_tokens == 256
    tables = NS(slot_mappings=torch.zeros((3, 256), dtype=torch.int64))
    target_inputs = object()
    worker.set_attn(None, None, tables, target_inputs, None)
    assert tables.slot_mappings.shape == (3, 320)
    assert calls == [(tables, target_inputs)]
    previous = tables.slot_mappings
    worker.set_attn(None, None, tables, target_inputs, None)
    assert tables.slot_mappings is previous
