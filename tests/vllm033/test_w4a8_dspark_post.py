# SPDX-License-Identifier: Apache-2.0
"""Default Post is scoped to the verified model and keeps native loading intact."""
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import dspark_post
from hcu_vllm_das.quantization.ct_w4a8 import HcuDSV41CTW4A8Config
from vllm.models.deepseek_v41.amd import dspark as native


def config():
    return NS(additional_config={dspark_post.KEY: True},
        quant_config=object.__new__(HcuDSV41CTW4A8Config),
        model_config=NS(hf_config=NS(model_type='deepseek_v41', hidden_size=5120, hc_mult=4)),
        speculative_config=NS(method='dspark', num_speculative_tokens=5),
        parallel_config=NS(tensor_parallel_size=1, prefill_context_parallel_size=1,
            pipeline_parallel_size=1, data_parallel_size=8,
            enable_expert_parallel=True, all2all_backend='deepep_low_latency'))


@pytest.mark.parametrize('extra', [None, {}, {dspark_post.KEY: False}])
def test_default_does_not_resolve_a_private_native_namespace(extra):
    assert not dspark_post.requested(NS(additional_config=extra))


@pytest.mark.parametrize('extra', [None, {}, {'unrelated_option': True}])
def test_verified_model_uses_post_without_an_extra_flag(extra):
    cfg = config(); cfg.additional_config = extra
    before = vars(cfg.model_config.hf_config).copy()
    assert dspark_post.requested(cfg)
    assert cfg.additional_config is extra
    assert vars(cfg.model_config.hf_config) == before


def test_explicit_false_keeps_the_original_draft_schedule():
    cfg = config(); cfg.additional_config = {dspark_post.KEY: False}
    assert not dspark_post.requested(cfg)


@pytest.mark.parametrize('change', ['hidden', 'hc', 'n', 'quant', 'dp', 'tp'])
def test_unsupported_default_retains_native_path(change):
    cfg = config(); cfg.additional_config = {}
    if change == 'hidden': cfg.model_config.hf_config.hidden_size = 4096
    if change == 'hc': cfg.model_config.hf_config.hc_mult = 8
    if change == 'n': cfg.speculative_config.num_speculative_tokens = 3
    if change == 'quant': cfg.quant_config = object()
    if change == 'dp': cfg.parallel_config.data_parallel_size = 1
    if change == 'tp': cfg.parallel_config.tensor_parallel_size = 2
    assert not dspark_post.requested(cfg)


@pytest.mark.parametrize('value', [0, 1, 'true', 'false'])
def test_flag_requires_a_boolean(value):
    cfg = config(); cfg.additional_config[dspark_post.KEY] = value
    with pytest.raises(ValueError, match='boolean'):
        dspark_post.requested(cfg)


def test_verified_w4a8_shape_and_scope():
    cfg = config()
    assert dspark_post.requested(cfg)
    cfg.model_config.hf_config.hidden_size = 4096
    with pytest.raises(ValueError, match='H5120'):
        dspark_post.requested(cfg)
    cfg = config(); cfg.quant_config = object()
    with pytest.raises(ValueError, match='W4A8'):
        dspark_post.requested(cfg)


def test_private_body_keeps_native_globals_and_unchanged_loading():
    original = native.DSparkDeepseekV4Model.forward
    original_post = original.__globals__['mhc_post_torch']
    model, constructor, anchors = dspark_post.implementation()
    assert native.DSparkDeepseekV4Model.forward is original
    assert original.__globals__['mhc_post_torch'] is original_post
    assert constructor.__globals__['DSparkDeepseekV4Model'] is model
    assert model.__init__ is native.DSparkDeepseekV4Model.__init__
    assert all(getattr(owner, name) is function for owner, name, function in anchors)
    x = torch.randn(2, 5120, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 5120, dtype=torch.bfloat16)
    post = torch.randn(2, 4, 1); comb = torch.randn(2, 4, 4)
    expected = original_post(x, residual, post, comb)
    actual = model.forward.__globals__['mhc_post_torch'](x, residual, post, comb)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))


def test_changed_native_source_is_rejected():
    with pytest.raises(RuntimeError, match='source changed'):
        dspark_post._clone(native.DSparkDeepseekV4Model, 'forward', 'unknown', {})


@pytest.mark.parametrize('anchor', ['forward', 'mhc_post_torch'])
def test_changed_native_identity_is_rejected_before_construction(monkeypatch, anchor):
    dspark_post.implementation()
    owner = native.DSparkDeepseekV4Model if anchor == 'forward' else native
    monkeypatch.setattr(owner, anchor, lambda *args: None)
    with pytest.raises(RuntimeError, match='identity changed'):
        dspark_post.initialize_owned(object(), vllm_config=config(), prefix='draft')
