# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU contract tests for SGL-align bf16_mega_gate wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import bf16_mega_gate as mega


def test_can_use_rejects_hash_table():
    x = torch.randn(8, 5120, dtype=torch.bfloat16)
    gate = SimpleNamespace(
        weight=torch.randn(384, 5120, dtype=torch.bfloat16),
        e_score_correction_bias=torch.zeros(384, dtype=torch.float32),
        tid2eid=None,
    )
    router = SimpleNamespace(
        scoring_func="sqrtsoftplus",
        top_k=6,
        renormalize=True,
        routed_scaling_factor=1.0,
        use_grouped_topk=False,
    )
    assert (
        mega.can_use_bf16_mega_gate(
            x,
            gate,
            router,
            hash_indices_table=torch.zeros(2, 6, dtype=torch.int32),
            input_ids=None,
        )
        is False
    )


def test_try_run_calls_lightop_and_returns_weights_ids_logits():
    if not torch.cuda.is_available():
        pytest.skip("cuda required for contiguous device contract")

    m, h, e, k = 4, 5120, 384, 6
    x = torch.randn(m, h, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(e, h, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(e, dtype=torch.float32, device="cuda")
    gate = SimpleNamespace(weight=weight, e_score_correction_bias=bias, tid2eid=None)
    router = SimpleNamespace(
        scoring_func="sqrtsoftplus",
        top_k=k,
        renormalize=True,
        routed_scaling_factor=1.0,
        use_grouped_topk=False,
    )

    fake_ids = torch.zeros(m, k, dtype=torch.int32, device="cuda")
    fake_w = torch.ones(m, k, dtype=torch.float32, device="cuda")
    fake_logits = torch.zeros(m, e, dtype=torch.float32, device="cuda")
    calls = {}

    def fake_op(*args, **kwargs):
        calls["kwargs"] = kwargs
        return fake_ids, fake_w, fake_logits

    with patch.object(mega, "_load_bf16_mega_gate_hcu", return_value=fake_op):
        out = mega.try_run_bf16_mega_gate(
            x, gate, router, indices_dtype=torch.int32
        )
    assert out is not None
    w, ids, logits = out
    assert w is fake_w and ids is fake_ids and logits is fake_logits
    assert calls["kwargs"]["return_logits"] is True
    assert calls["kwargs"]["scoring_func"] == "sqrtsoftplus"
    assert calls["kwargs"]["num_topk"] == k
    assert calls["kwargs"]["apply_scale"] is False


def test_loader_rejects_incomplete_signature():
    def bare(x, weight, num_topk=1):
        return None

    class FakeMoe:
        bf16_mega_gate_hcu = staticmethod(bare)

    import sys

    with patch.dict(sys.modules, {"lightop.moe": FakeMoe()}):
        # Force re-import path by calling loader (imports lightop.moe)
        assert mega._load_bf16_mega_gate_hcu() is None
