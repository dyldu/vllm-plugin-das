# SPDX-License-Identifier: Apache-2.0
"""Startup accounting must cover the selected reader without phantom scratch."""
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import fp4_indexer_lightop as reader
from hcu_adapter.kernels import fp4_staging_triton as staging


def config(slots=64, budget=256, drafts=5):
    return NS(cache_config=NS(block_size=64),
              scheduler_config=NS(max_num_seqs=slots, max_num_batched_tokens=budget),
              speculative_config=NS(num_speculative_tokens=drafts) if drafts else None)


@pytest.mark.parametrize('slots,budget,drafts,expected', [
    (64, 256, 5, 256), (64, 384, 5, 384), (64, 511, 5, 384),
    (63, 150, 3, 150), (17, 33, 0, 17),
])
def test_decode_capacity_obeys_runtime_token_budget(slots, budget, drafts, expected):
    assert reader.decode_rows(config(slots, budget, drafts)) == expected


@pytest.mark.parametrize('fused', [True, False])
def test_one_million_context_profile_reserves_only_selected_route(monkeypatch, fused):
    """Record real reservation requests without allocating tens of GiB."""
    allocations, pools = [], []
    monkeypatch.setattr(staging, 'supports_staging_device', lambda device: fused)
    monkeypatch.setattr(reader, '_reserved_pools', {})
    monkeypatch.setattr(reader, '_reserved_fused', set())

    def pool(plan, device):
        pools.append(plan)
        return NS(device=torch.device(device))

    def allocate(size, **kwargs):
        allocations.append(size)
        return object()

    monkeypatch.setattr(reader, '_staging_pool', pool)
    monkeypatch.setattr(reader.torch, 'empty', allocate)
    reader.reserve_decode_staging(config(), 1048576 // 2, 2, 'cuda:0')
    assert len(pools) == 1 and len(allocations) == 1
    assert pools[0]['shape'] == (256, 8192, 4352)
    if fused:
        # A table of 256 rows x 8192 canonical pages is 8 MiB. The fused
        # copier has no gathered payload or per-token PyTorch index tensors.
        assert allocations[0] == 8 * 1024**2
        assert (torch.device('cuda:0'), 32) in reader._reserved_fused
    else:
        # The actual gather/repack fallback still needs its transient buffers.
        assert allocations[0] > 20 * 1024**3
        assert not reader._reserved_fused


def test_native_64_record_pages_need_no_staging_reservations(monkeypatch):
    monkeypatch.setattr(staging, 'supports_staging_device', lambda device: True)
    monkeypatch.setattr(reader, '_staging_pool', lambda *args: pytest.fail('zero-copy'))
    assert reader.reserve_decode_staging(config(), 1048576, 1, 'cuda:0') is None
