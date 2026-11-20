# SPDX-License-Identifier: Apache-2.0
"""Owned norm weight ABI, call lifetime, native fallback and exceptions."""
from types import SimpleNamespace

import pytest
import torch
from hcu_adapter import rms_int8_scope as scope
from vllm import ir
from vllm.config import VllmConfig, set_current_vllm_config

from hcu_vllm_das.rms_int8 import PairedAttentionRMSNorm, install_owned_decoder
from hcu_vllm_das.rms_norm import HcuRMSNorm


@pytest.fixture
def context():
    # CPU tests use the public native Torch provider; GPU evidence separately
    # fixes serving's vllm_c priority and compares BF16/INT8/FP32 bytes.
    with set_current_vllm_config(VllmConfig()), ir.ops.rms_norm.set_priority(['native']), torch.inference_mode():
        yield


def original():
    norm = HcuRMSNorm(5120, 1.e-20, dtype=torch.bfloat16)
    norm.weight.data.copy_(torch.linspace(.5, 1.5, 5120).bfloat16())
    return norm


def forbidden(*args):
    raise AssertionError('unsupported call reached the fused kernel')


def test_weight_identity_checkpoint_key_and_native_no_scope_fallback(context):
    source = original()
    candidate = PairedAttentionRMSNorm(source, forbidden)
    assert candidate.weight is source.weight
    assert list(candidate.state_dict()) == list(source.state_dict()) == ['weight']
    assert candidate.weight.dummy_weight_value == 1.
    x = torch.randn((8, 5120), dtype=torch.bfloat16)
    assert torch.equal(candidate(x), source(x))
    replacement = torch.linspace(.1, .3, 5120).bfloat16()
    candidate.load_state_dict({'weight': replacement})
    assert torch.equal(candidate.weight, replacement)
    assert candidate.weight is source.weight


def test_cpu_input_keeps_native_inside_owned_scope(context):
    source = original()
    candidate = PairedAttentionRMSNorm(source, forbidden)
    for rows in (0, 1, 8, 64, 65):
        x = torch.randn((rows, 5120), dtype=torch.bfloat16)
        expected = source(x)
        with scope.attention_norm_scope():
            actual = candidate(x)
            assert scope.lookup(actual) is None
            assert torch.equal(actual, expected)
    assert not scope.active()


def test_publication_requires_the_scope_and_restores_it_after_exception(context, monkeypatch):
    source = original()
    x = torch.randn((8, 5120), dtype=torch.bfloat16)
    y = source(x)
    q, s = torch.zeros_like(x, dtype=torch.int8), torch.zeros((8, 1))
    calls = []

    def fused(value, weight, epsilon):
        calls.append((value, weight, epsilon))
        return y, q, s

    candidate = PairedAttentionRMSNorm(source, fused)
    monkeypatch.setattr(scope, 'supported', lambda *args: True)
    monkeypatch.setattr(ir.ops.rms_norm, 'get_priority', lambda: ['vllm_c', 'native'])
    # Without a scope, native forwarding still applies. A different provider
    # is used only for this fallback reference, since vllm_c is GPU-only.
    assert scope.lookup(y) is None
    with pytest.raises(RuntimeError, match='downstream failed'):
        with scope.attention_norm_scope():
            assert candidate(x) is y
            assert scope.lookup(y) == (q, s)
            assert scope.lookup(x) is None
            raise RuntimeError('downstream failed')
    assert not scope.active() and scope.lookup(y) is None
    assert len(calls) == 1 and calls[0][0] is x and calls[0][1] is source.weight
    assert calls[0][2] == 1.e-20


def test_cpu_model_construction_does_not_compile_a_gpu_kernel(context):
    decoder = SimpleNamespace(attn_norm=original(), attn=SimpleNamespace(fused_wqa_wkv=None))
    assert not install_owned_decoder(decoder)
