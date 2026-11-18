# SPDX-License-Identifier: Apache-2.0
"""Private DSpark construction preserves native symbols and loader ownership."""
import inspect

import pytest
import torch
from hcu_vllm_das import dspark_schedule as schedule
from hcu_vllm_das.models.dsv41_dspark import HcuDSparkDeepseekV41ForCausalLM
from hcu_adapter.dsv41_draft_weight_alias import HcuDSV41DraftWeightAliasModel


def test_native_namespaces_remain_unchanged():
    decoder = schedule.native_model.DeepseekV4DecoderLayer.__init__
    assert schedule.DraftDecoder.__init__.__globals__ is not decoder.__globals__
    assert schedule.DraftDecoder.__init__.__globals__['_select_dsv4_attn_cls'](None) is schedule.DraftAttention
    assert decoder.__globals__['_select_dsv4_attn_cls'] is schedule.native_model._select_dsv4_attn_cls
    assert schedule.DraftModel.__init__.__globals__['DeepseekV4DecoderLayer'] is schedule.DraftDecoder
    assert schedule.native.DSparkDeepseekV4Model.__init__.__globals__['DeepseekV4DecoderLayer'] is schedule.native_model.DeepseekV4DecoderLayer
    assert schedule.DraftModel.forward.__globals__['mhc_post_torch'] is schedule._final_post
    assert schedule.native.DSparkDeepseekV4Model.forward.__globals__['mhc_post_torch'] is schedule.native.mhc_post_torch
    assert (schedule.DraftModel.precompute_and_store_context_kv
            is schedule.native.DSparkDeepseekV4Model.precompute_and_store_context_kv)


def test_loader_and_parameter_enumeration_remain_inherited():
    assert HcuDSparkDeepseekV41ForCausalLM.load_weights is HcuDSV41DraftWeightAliasModel.load_weights
    assert HcuDSparkDeepseekV41ForCausalLM.named_parameters is HcuDSV41DraftWeightAliasModel.named_parameters


@pytest.mark.parametrize('private,original', [
    (schedule.DraftDecoder.__init__, schedule.native_model.DeepseekV4DecoderLayer.__init__),
    (schedule.DraftModel.__init__, schedule.native.DSparkDeepseekV4Model.__init__),
    (schedule.DraftModel.forward, schedule.native.DSparkDeepseekV4Model.forward),
    (schedule.DraftModel.precompute_and_store_context_kv,
     schedule.native.DSparkDeepseekV4Model.precompute_and_store_context_kv),
    (schedule._OUTER_INIT, schedule.native.DSparkDeepseekV4ForCausalLM.__init__)])
def test_native_method_abi_is_preserved(private, original):
    assert inspect.signature(private) == inspect.signature(original)


def test_changed_native_method_is_rejected_before_allocation(monkeypatch):
    monkeypatch.setattr(schedule.native.DSparkDeepseekV4Model, 'forward', lambda self: None)
    with pytest.raises(RuntimeError, match='identity changed'):
        schedule.initialize_owned(object(), vllm_config=None, prefix='')


def test_cpu_final_post_retains_torch():
    x = torch.randn(2, 5120).bfloat16()
    residual = torch.randn(2, 4, 5120).bfloat16()
    post = torch.randn(2, 4, 1)
    comb = torch.randn(2, 4, 4)
    actual = schedule._final_post(x, residual, post, comb)
    expected = schedule.native.mhc_post_torch(x, residual, post, comb)
    assert torch.equal(actual, expected)
