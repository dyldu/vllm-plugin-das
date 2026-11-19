# SPDX-License-Identifier: Apache-2.0
"""W4A8 LL policy preserves fixed capacity, native resources and default AG/RS."""
import copy
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import dspark_graph
from hcu_vllm_das.dspark_backend import KEY, native_draft_ll_requested
from hcu_vllm_das.models.dsv41_dspark import HcuDSparkDeepseekV41ForCausalLM
from hcu_adapter.dsv41_draft_weight_alias import HcuDSV41DraftWeightAliasModel
from hcu_vllm_das.quantization.ct_w4a8 import HcuDSV41CTW4A8Config


def config():
    hf = NS(model_type='deepseek_v41', sample_from_anchor=True,
            hidden_size=5120, n_routed_experts=384, num_experts_per_tok=6,
            dspark_n_routed_experts=128, dspark_num_experts_per_tok=3)
    return NS(
        additional_config={KEY: 'deepep_low_latency'},
        quant_config=object.__new__(HcuDSV41CTW4A8Config),
        model_config=NS(hf_config=hf, enforce_eager=True),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
            data_parallel_size=8, enable_expert_parallel=True,
            all2all_backend='deepep_low_latency', enable_eplb=False,
            enable_elastic_ep=False, enable_dbo=False),
        kernel_config=NS(moe_backend='deep_gemm', monokernel=False),
        speculative_config=NS(method='dspark', num_speculative_tokens=5,
            draft_model_config=NS(hf_config=hf, enforce_eager=True)),
        scheduler_config=NS(max_num_seqs=32, max_num_batched_tokens=256),
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


@pytest.mark.parametrize('change', ['unknown', 'dp1', 'tp2', 'ht', 'quant',
                                    'capacity', 'eplb', 'hidden', 'draft_topk'])
def test_native_selection_rejects_unsupported_contract(change):
    cfg = config()
    if change == 'unknown': cfg.additional_config[KEY] = 'other'
    if change == 'dp1': cfg.parallel_config.data_parallel_size = 1
    if change == 'tp2': cfg.parallel_config.tensor_parallel_size = 2
    if change == 'ht': cfg.parallel_config.all2all_backend = 'deepep_high_throughput'
    if change == 'quant': cfg.quant_config = object()
    if change == 'capacity': cfg.scheduler_config.max_num_batched_tokens = 39
    if change == 'eplb': cfg.parallel_config.enable_eplb = True
    if change == 'hidden': cfg.model_config.hf_config.hidden_size = 4096
    if change == 'draft_topk': cfg.model_config.hf_config.dspark_num_experts_per_tok = 6
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


def ipc_manager(monkeypatch):
    from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager
    from hcu_vllm_das.dspark_ll import SharedIPCManager

    for key in ('ROCSHMEM_BACKEND', 'DEEPEP_ENABLE_LL_LAYERED_OPT',
                'DEEPEP_ENABLE_LL_LAYERED_DEVELOP', 'ROCSHMEM_IPC_MNVL',
                'DEEP_EP_NORMAL_MNVL'):
        monkeypatch.delenv(key, raising=False)
    native = object.__new__(DeepEPLLAll2AllManager)
    native.internode = False
    native.world_size = native.dp_world_size = 8
    native.support_fault_tolerance = False
    return native, SharedIPCManager


def test_ipc_layouts_retain_one_native_allocation_and_destructor(monkeypatch):
    native, wrapper = ipc_manager(monkeypatch)
    seen = []
    cache = {}
    def get_handle(kwargs):
        seen.append(kwargs.copy())
        return cache.setdefault(tuple(sorted(kwargs.items())), object())
    native.get_handle = get_handle
    manager = wrapper(native)
    handles = []
    for e in (384, 128, 384, 128):
        handles.append(manager.get_handle(dict(max_num_tokens_per_dp_rank=256,
            token_hidden_size=5120, num_ep_ranks=8,
            num_global_experts=e, num_local_experts=e // 8)))
    assert len(cache) == 1 and all(h is handles[0] for h in handles)
    assert all(k['num_local_experts'] == 48 and k['num_global_experts'] == 384 for k in seen)
    assert 'destroy' not in vars(manager) and manager.native is native


@pytest.mark.parametrize('change', ['internode', 'dp', 'ft', 'gda', 'layered', 'mnvl'])
def test_ipc_resource_policy_refuses_other_transports(monkeypatch, change):
    native, wrapper = ipc_manager(monkeypatch)
    if change == 'internode': native.internode = True
    if change == 'dp': native.dp_world_size = 4
    if change == 'ft': native.support_fault_tolerance = True
    if change == 'gda': monkeypatch.setenv('ROCSHMEM_BACKEND', 'gda')
    if change == 'layered':
        import deep_ep
        monkeypatch.setattr(deep_ep.Buffer, 'is_ll_layered_opt_enabled', lambda n: True)
    if change == 'mnvl': monkeypatch.setenv('ROCSHMEM_IPC_MNVL', '1')
    with pytest.raises(ValueError): wrapper(native)


def test_ipc_admits_baseline_inert_layered_request(monkeypatch):
    native, wrapper = ipc_manager(monkeypatch)
    monkeypatch.setenv('DEEPEP_ENABLE_LL_LAYERED_OPT', '1')
    assert wrapper(native).native is native


@pytest.mark.parametrize('key,value', [('max_num_tokens_per_dp_rank', 64),
    ('token_hidden_size', 4096), ('num_ep_ranks', 4),
    ('num_global_experts', 256), ('num_local_experts', 16)])
def test_ipc_resource_policy_refuses_unvalidated_geometry(monkeypatch, key, value):
    native, wrapper = ipc_manager(monkeypatch)
    native.get_handle = lambda _: pytest.fail('must reject before allocation')
    kwargs = dict(max_num_tokens_per_dp_rank=256, token_hidden_size=5120,
        num_ep_ranks=8, num_global_experts=384, num_local_experts=48)
    kwargs[key] = value
    with pytest.raises(ValueError): wrapper(native).get_handle(kwargs)


@pytest.mark.parametrize('reset', [False, True])
def test_captured_dispatch_records_clean_without_python_layout_cache(monkeypatch, reset):
    from hcu_vllm_das.moe import ct_w4a8_prepare as mod

    calls = []
    handle = object()

    class Buffer:
        def clean_low_latency_buffer(self, *args, **kwargs):
            calls.append(('clean', args, kwargs))

        def low_latency_dispatch(self, x, ids, weights, capacity, experts, **kwargs):
            calls.append(('dispatch', experts, kwargs))
            # Meta tensors test the real shape/dtype admission with the full
            # frozen extent, without allocating a fake 500 MB CPU workspace.
            e, t = experts // 8, capacity * 8
            q = torch.empty((e, t, 5120), dtype=torch.int8, device='meta')
            s = torch.empty((e, t, 1), dtype=torch.float32, device='meta')
            counts = torch.empty((e,), dtype=torch.int32, device='meta')
            return (q, s), counts, handle, None, lambda: calls.append(('hook',))

    monkeypatch.setattr(mod, 'dbo_current_ubatch_id', lambda: 0)
    prepare = mod.HcuCTLLPrepareAndFinalize(Buffer(), 256, 8,
                                           reset_before_dispatch=reset)
    quant = NS(quant_dtype=torch.int8, per_act_token_quant=True, block_shape=None)
    for experts, topk in ((384, 6), (128, 3), (384, 6)):
        hook, receive = prepare.prepare_async(
            torch.empty((48, 5120), dtype=torch.bfloat16, device='meta'),
            torch.empty((48, topk), dtype=torch.float32, device='meta'),
            torch.empty((48, topk), dtype=torch.int32, device='meta'),
            experts, None, False, quant)
        hook()
        q, scales, metadata, ids, weights = receive()
        assert q.dtype == torch.int8 and scales.dtype == torch.float32
        assert metadata.expert_num_tokens.shape == (experts // 8,)
        assert prepare.handles[0] is handle and ids is weights is None
    expected = ['clean', 'dispatch', 'hook'] if reset else ['dispatch', 'hook']
    assert [c[0] for c in calls] == expected * 3
    if reset:
        assert [c[1:] for c in calls if c[0] == 'clean'] == [
            ((256, 5120, e), dict(quant_group_size=0, num_topk=k))
            for e, k in ((384, 6), (128, 3), (384, 6))]


def test_draft_receiver_quantizes_only_after_transport_hook(monkeypatch):
    from hcu_vllm_das.moe import ct_w4a8_prepare as mod

    calls = []
    raw = torch.empty((16, 2048, 5120), dtype=torch.bfloat16, device='meta')
    counts = torch.empty((16,), dtype=torch.int32, device='meta')
    handle = object()

    class Buffer:
        def low_latency_dispatch(self, *args, **kwargs):
            assert kwargs['quant_type'] == 0
            calls.append('dispatch')
            return raw, counts, handle, None, lambda: calls.append('hook')

    def quantize(x, actual_counts):
        assert calls == ['dispatch', 'hook'] and x is raw and actual_counts is counts
        calls.append('native_quantize')
        return x.to(torch.int8), torch.empty((16, 2048, 1), dtype=torch.float32, device='meta')

    monkeypatch.setitem(sys.modules, 'hcu_vllm_das.dspark_ll_quant', NS(quantize_receiver=quantize))
    monkeypatch.setattr(mod, 'dbo_current_ubatch_id', lambda: 0)
    prepare = mod.HcuCTLLPrepareAndFinalize(Buffer(), 256, 8, preserve_agrs_quantization=True)
    quant = NS(quant_dtype=torch.int8, per_act_token_quant=True, block_shape=None)
    hook, receive = prepare.prepare_async(
        torch.empty((48, 5120), dtype=torch.bfloat16, device='meta'),
        torch.empty((48, 3), dtype=torch.float32, device='meta'),
        torch.empty((48, 3), dtype=torch.int32, device='meta'), 128, None, False, quant)
    assert calls == ['dispatch']
    hook()
    q, scale, metadata, ids, weights = receive()
    assert q.dtype == torch.int8 and scale.dtype == torch.float32
    assert metadata.expert_num_tokens is counts and ids is weights is None
    assert calls == ['dispatch', 'hook', 'native_quantize'] and prepare.handles[0] is handle


