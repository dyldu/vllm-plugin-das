# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU contract tests for SGL-align bf16_mega_gate wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import bf16_mega_gate as mega


def test_nn_weight_cache_preserves_original_and_invalidates(monkeypatch):
    monkeypatch.setattr(mega.henvs, "VLLM_USE_NN", True)
    gate = torch.nn.Module()
    gate.weight = torch.nn.Parameter(torch.randn(256, 16, dtype=torch.bfloat16))
    parameter = gate.weight
    packed = mega._gate_weight_nt(gate, 256)
    torch.testing.assert_close(packed, parameter.T, rtol=0, atol=0)
    assert packed.is_contiguous()
    assert gate.weight is parameter
    assert mega._gate_weight_nt(gate, 256) is packed
    assert "_hcu_mega_gate_weight_nt" not in gate.state_dict()
    with torch.no_grad():
        parameter.add_(1)
    updated = mega._gate_weight_nt(gate, 256)
    assert updated is not packed
    torch.testing.assert_close(updated, parameter.T, rtol=0, atol=0)
    gate.weight = torch.nn.Parameter(parameter.detach().clone())
    assert mega._gate_weight_nt(gate, 256) is not updated


def test_nn_weight_cache_rejects_unreviewed_layout(monkeypatch):
    gate = SimpleNamespace(weight=torch.zeros(256, 16, dtype=torch.bfloat16))
    monkeypatch.setattr(mega.henvs, "VLLM_USE_NN", False)
    assert mega._gate_weight_nt(gate, 256) is None
    monkeypatch.setattr(mega.henvs, "VLLM_USE_NN", True)
    gate.weight = torch.zeros(256, 16, dtype=torch.float32)
    assert mega._gate_weight_nt(gate, 256) is None


def test_nn_loader_data_copy_invalidates_cached_weight(monkeypatch):
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    monkeypatch.setattr(mega.henvs, "VLLM_USE_NN", True)
    gate = torch.nn.Module()
    gate.weight = torch.nn.Parameter(torch.zeros(256, 16, dtype=torch.bfloat16))
    gate.weight.weight_loader = default_weight_loader
    packed = mega._gate_weight_nt(gate, 256)
    loader, version = gate.weight.weight_loader, gate.weight._version
    loader(gate.weight, torch.ones_like(gate.weight))
    assert gate.weight._version == version  # the real loader bypasses it
    updated = mega._gate_weight_nt(gate, 256)
    assert updated is not packed
    assert gate.weight.weight_loader is loader  # do not chain hooks each step
    torch.testing.assert_close(updated, gate.weight.T, rtol=0, atol=0)


@pytest.mark.parametrize("padded", [False, True])
def test_real_lightop_accepts_hcu_nn_weight(monkeypatch, padded):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    from lightop.moe import bf16_mega_gate_hcu

    monkeypatch.setattr(mega.henvs, "VLLM_USE_NN", True)
    torch.manual_seed(42)
    x = torch.randn(8, 5120, dtype=torch.bfloat16, device="cuda")
    gate = torch.nn.Module()
    gate.weight = torch.nn.Parameter(torch.randn(5120, 384, dtype=torch.bfloat16, device="cuda") * 0.02)
    gate.e_score_correction_bias = torch.zeros(384, dtype=torch.float32, device="cuda")
    gate.tid2eid = None
    router = SimpleNamespace(scoring_func="sqrtsoftplus", top_k=6,
                             renormalize=True, routed_scaling_factor=1.5,
                             use_grouped_topk=False)
    is_padding = torch.tensor([False] * 5 + [True] * 3, device="cuda") if padded else None
    monkeypatch.setattr(mega, "_get_padding_mask", lambda m: is_padding)
    assert mega.can_use_bf16_mega_gate(x, gate, router, hash_indices_table=None, input_ids=None)
    gate._vllm_hcu_deepseek_v4_gate = True
    assert not mega.can_use_bf16_mega_gate(x, gate, router, hash_indices_table=None, input_ids=None)
    gate._vllm_hcu_deepseek_v4_gate = False
    actual = mega.try_run_bf16_mega_gate(x, gate, router, indices_dtype=torch.int32)
    ids, weights, logits = bf16_mega_gate_hcu(
        x, gate.weight.T.contiguous(), num_topk=6, bias=gate.e_score_correction_bias,
        use_shared_as_routed=False, num_shared_experts=0, ep_rank=0,
        mask=None if is_padding is None else ~is_padding, scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5, renormalize=True, apply_scale=True, return_logits=True,
    )
    assert actual is not None
    torch.testing.assert_close(actual[0], weights, rtol=0, atol=0)
    torch.testing.assert_close(actual[1].long(), ids.long(), rtol=0, atol=0)
    torch.testing.assert_close(actual[2], logits, rtol=0, atol=0)
    if padded:
        assert (actual[1][5:] == -1).all()
        assert (actual[0][5:] == 0).all()


@pytest.mark.parametrize("feature,value", [
    ("eplb_state", object()), ("capture_fn", lambda _: None),
    ("bias_vl", object()), ("_hash_indices_table", object()),
    ("num_expert_group", 1), ("num_fused_shared_experts", 1),
])
def test_rejects_routing_features_requiring_select_experts(feature, value):
    if not torch.cuda.is_available():
        pytest.skip("HCU device required to test rejection on otherwise valid input")
    x = torch.empty(1, 256, device="cuda", dtype=torch.bfloat16)
    gate = SimpleNamespace(weight=torch.empty(16, 256, device="cuda", dtype=x.dtype),
                           e_score_correction_bias=torch.zeros(16, device="cuda"),
                           tid2eid=None)
    router = SimpleNamespace(scoring_func="sqrtsoftplus", top_k=6)
    assert mega.can_use_bf16_mega_gate(x, gate, router,
                                    hash_indices_table=None, input_ids=None)
    setattr(router, feature, value)
    assert not mega.can_use_bf16_mega_gate(
        x, gate, router,
        hash_indices_table=None, input_ids=None,
    )


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
