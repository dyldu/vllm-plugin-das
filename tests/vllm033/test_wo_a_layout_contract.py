# SPDX-License-Identifier: Apache-2.0
"""Admission/fallback contracts; numerical WO_A+INT8 evidence uses real GPUs."""
from types import MethodType, SimpleNamespace as NS

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from hcu_vllm_das import fp4_indexer, wo_a_layout as layout
from hcu_vllm_das.quantization.ct_w4a8 import HcuDSV41CTW4A8Config


def config():
    return NS(quant_config=object.__new__(HcuDSV41CTW4A8Config),
        model_config=NS(hf_config=NS(model_type='deepseek_v41', hidden_size=5120)),
        parallel_config=NS(tensor_parallel_size=1, prefill_context_parallel_size=1,
            pipeline_parallel_size=1, data_parallel_size=8, enable_expert_parallel=True))


def test_unknown_installation_keeps_native(monkeypatch):
    monkeypatch.setattr(layout, 'version', lambda _: 'unknown')
    assert not layout._source_matches()
    monkeypatch.setattr(layout, '_SOURCE_MATCHES', False)
    assert not layout.configured(config())


@pytest.mark.parametrize('name', ['ROCBLAS_TENSILE_LIBPATH','ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH',
                                'HIPBLASLT_TENSILE_LIBPATH','HIPBLASLT_TUNING_OVERRIDE_FILE'])
def test_unverified_gemm_tuning_retains_native(monkeypatch, name):
    monkeypatch.setenv(name, '/unknown-tuning')
    assert not layout._source_matches()


def test_known_nn_cache_can_install_after_plugin_import(monkeypatch):
    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_rocm_wo_a_layout as patch
    monkeypatch.setattr(layout.native, '_get_cached_wo_a_bf16', layout._CACHE)
    monkeypatch.setattr(layout.native, patch._MARKER, False, raising=False)
    monkeypatch.setattr(layout.native, '_vllm_hcu_original_get_cached_wo_a_bf16', None, raising=False)
    assert layout._cache_contract(layout.native._get_cached_wo_a_bf16) == (True, False)
    assert patch.apply()
    assert layout._cache_contract(layout.native._get_cached_wo_a_bf16) == (True, True)


@pytest.mark.parametrize('explicit_text_property', [False, True])
def test_actual_multimodal_config_uses_text_hidden_size(monkeypatch, explicit_text_property):
    monkeypatch.setattr(layout, '_SOURCE_MATCHES', True)
    cfg = config()
    text = NS(model_type='deepseek_v41_text', hidden_size=5120)
    cfg.model_config.hf_config = NS(model_type='deepseek_v41', text_config=text)
    if explicit_text_property:
        cfg.model_config.hf_text_config = text
    assert layout.configured(cfg)
    text.hidden_size = 4096
    assert not layout.configured(cfg)


@pytest.mark.parametrize('field,value', [('hidden_size',4096), ('model_type','other'),
    ('tensor_parallel_size',2), ('prefill_context_parallel_size',2)])
def test_unverified_model_and_parallelism_keep_native(monkeypatch, field, value):
    monkeypatch.setattr(layout, '_SOURCE_MATCHES', True)
    cfg = config()
    assert layout.configured(cfg)
    owner = cfg.model_config.hf_config if field in ('hidden_size','model_type') else cfg.parallel_config
    setattr(owner, field, value)
    assert not layout.configured(cfg)


@pytest.fixture
def fake_case(monkeypatch):
    mode = FakeTensorMode()
    def tensor(shape, dtype=torch.bfloat16, device='cuda:0'):
        return FakeTensor(mode, torch.empty(shape, dtype=dtype, device='meta'), torch.device(device))
    owner = NS(n_local_heads=64, n_local_groups=8, o_lora_rank=1024,
               _wo_b_scale=None, wo_a=NS(weight=tensor((8192,4096))))
    owner._wo_b_after_wo_a = MethodType(layout._CONSUMER, owner)
    monkeypatch.setattr(layout, '_SOURCE_MATCHES', True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda _: NS(gcnArchName='gfx938', multi_processor_count=64))
    with torch.no_grad():
        yield owner, tensor((48,64,512)), tensor


def test_supported_route_excludes_gradients_and_autocast(fake_case, monkeypatch):
    owner, x, _ = fake_case
    assert layout.supported(owner, x)
    with torch.enable_grad():
        assert not layout.supported(owner, x)
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda _: True)
    assert not layout.supported(owner, x)


@pytest.mark.parametrize('case', ['dtype','rows','heads','hidden','device','strides',
    'weight','cached_weight','consumer','native_identity','scale','hardware'])
def test_changed_contract_refuses_out_bmm(fake_case, monkeypatch, case):
    owner, x, tensor = fake_case
    assert layout.supported(owner, x)
    if case == 'dtype': x = tensor((48,64,512), torch.float32)
    if case == 'rows': x = tensor((0,64,512))
    if case == 'heads': x = tensor((48,32,512))
    if case == 'hidden': x = tensor((48,64,256))
    if case == 'device': x = tensor((48,64,512), device='cpu')
    if case == 'strides': x = tensor((64,48,512)).transpose(0,1)
    if case == 'weight': owner.wo_a.weight = tensor((8192,4096), torch.float32)
    if case == 'cached_weight': owner.wo_a._dsv4_wo_a_bf16 = tensor((8,512,8192))
    if case == 'consumer': owner._wo_b_after_wo_a = lambda _: None
    if case == 'native_identity': monkeypatch.setattr(layout.native, '_get_cached_wo_a_bf16', lambda *a: None)
    if case == 'scale': owner._wo_b_scale = tensor((5120,), torch.float32)
    if case == 'hardware':
        monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: NS(gcnArchName='gfx950', multi_processor_count=64))
    assert not layout.supported(owner, x)
    with pytest.raises(ValueError, match='verified W4A8 contract'):
        layout.project(owner, x)


def test_nn_weights_require_verified_native_layout_wrapper(fake_case, monkeypatch):
    owner, x, tensor = fake_case
    owner.wo_a.weight = tensor((4096,8192))
    monkeypatch.setattr(layout, '_cache_contract', lambda _: (True, False))
    assert not layout.supported(owner, x)
    monkeypatch.setattr(layout, '_cache_contract', lambda _: (True, True))
    assert layout.supported(owner, x)


@pytest.mark.parametrize('enabled', [False, True])
def test_owned_o_projection_retains_native_fallback(monkeypatch, enabled):
    owner = object.__new__(fp4_indexer.Fp4Attention)
    torch.nn.Module.__init__(owner)
    owner._hcu_wo_a_layout = enabled
    sentinel = object()
    monkeypatch.setattr(layout, 'supported', lambda *a: False)
    monkeypatch.setattr(layout.rocm.DeepseekV41ROCMAiterMLAAttention, '_o_proj', lambda *a: sentinel)
    monkeypatch.setattr(layout, 'project', lambda *a: pytest.fail('unsupported projection executed'))
    assert owner._o_proj(None, None) is sentinel


def test_compile_factor_tracks_provider_admission(monkeypatch):
    from hcu_vllm_das import v41_indexer_activation as activation
    assert any(path.name == 'wo_a_layout.py' for path in activation._digest_paths())
    monkeypatch.setattr(layout, '_SOURCE_MATCHES', True)
    verified = activation._compute_digest()
    monkeypatch.setattr(layout, '_SOURCE_MATCHES', False)
    assert verified != activation._compute_digest()


@pytest.mark.parametrize('rows', [1, 7, 17, 32, 65, 96, 128, 256])
def test_row_count_is_runtime_shape(fake_case, rows):
    owner, _, tensor = fake_case
    assert layout.supported(owner, tensor((rows, 64, 512)))


@pytest.mark.parametrize('field,value', [('pipeline_parallel_size', 2),
    ('data_parallel_size', 1), ('enable_expert_parallel', False)])
def test_local_projection_does_not_depend_on_unrelated_parallelism(monkeypatch, field, value):
    monkeypatch.setattr(layout, '_SOURCE_MATCHES', True)
    cfg = config()
    setattr(cfg.parallel_config, field, value)
    assert layout.configured(cfg)
