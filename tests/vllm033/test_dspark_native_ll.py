# SPDX-License-Identifier: Apache-2.0
"""Native draft LL selection must retain the native config and manager path."""
import copy
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import dspark_graph
from hcu_vllm_das.dspark_backend import KEY, native_draft_ll_requested
from hcu_vllm_das.models.dsv41_dspark import HcuDSparkDeepseekV41ForCausalLM
from hcu_adapter.dsv41_draft_weight_alias import HcuDSV41DraftWeightAliasModel
from hcu_vllm_das.quantization.dsv41_channel_fp8 import HcuDSV41ChannelFP8Config


def config():
    hf = NS(model_type='deepseek_v41', sample_from_anchor=True)
    return NS(
        additional_config={KEY: 'deepep_low_latency'},
        quant_config=object.__new__(HcuDSV41ChannelFP8Config),
        model_config=NS(hf_config=hf, enforce_eager=True),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
            data_parallel_size=8, enable_expert_parallel=True,
            all2all_backend='deepep_low_latency', enable_eplb=False,
            enable_elastic_ep=False, enable_dbo=False),
        kernel_config=NS(moe_backend='deep_gemm', monokernel=False),
        speculative_config=NS(method='dspark', num_speculative_tokens=5,
            draft_model_config=NS(hf_config=hf, enforce_eager=True)),
        scheduler_config=NS(max_num_seqs=8, max_num_batched_tokens=64),
        use_v2_model_runner=True, lora_config=None)


@pytest.mark.parametrize('extra', [None, {}, {KEY: 'allgather_reducescatter'}])
def test_default_keeps_agrs(extra):
    cfg = config(); cfg.additional_config = extra
    assert not native_draft_ll_requested(cfg)


def test_explicit_native_selection_does_not_mutate_target():
    cfg = config(); parallel = copy.copy(cfg.parallel_config)
    assert native_draft_ll_requested(cfg)
    assert vars(cfg.parallel_config) == vars(parallel)
    assert cfg.additional_config == {KEY: 'deepep_low_latency'}


@pytest.mark.parametrize('change', ['unknown', 'dp1', 'tp2', 'ht', 'quant', 'capacity', 'eplb'])
def test_native_selection_rejects_unsupported_contract(change):
    cfg = config()
    if change == 'unknown': cfg.additional_config[KEY] = 'other'
    if change == 'dp1': cfg.parallel_config.data_parallel_size = 1
    if change == 'tp2': cfg.parallel_config.tensor_parallel_size = 2
    if change == 'ht': cfg.parallel_config.all2all_backend = 'deepep_high_throughput'
    if change == 'quant': cfg.quant_config = object()
    if change == 'capacity': cfg.scheduler_config.max_num_batched_tokens = 39
    if change == 'eplb': cfg.parallel_config.enable_eplb = True
    with pytest.raises(ValueError): native_draft_ll_requested(cfg)


def test_loader_native_path_preserves_config_without_agrs_scope():
    cfg = config(); target = object(); result = object(); seen = []
    def original(model, actual):
        seen.append((model, actual)); return result
    assert dspark_graph._load_with_scope(original, target, cfg) is result
    assert seen == [(target, cfg)]
    assert dspark_graph._ACTIVE_SCOPE.get() is None


def test_native_constructor_receives_original_config(monkeypatch):
    cfg = config(); seen = []
    def constructor(self, *, vllm_config, prefix=''):
        torch.nn.Module.__init__(self); seen.append(vllm_config)
    monkeypatch.setattr(HcuDSV41DraftWeightAliasModel, '__init__', constructor)
    HcuDSparkDeepseekV41ForCausalLM(vllm_config=cfg)
    assert seen == [cfg]
