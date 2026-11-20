# SPDX-License-Identifier: Apache-2.0
"""Unknown native binaries and unsupported attention layouts retain Core."""
from types import SimpleNamespace as NS

import pytest
import torch
from hcu_vllm_das import qkv_split
from hcu_vllm_das import fp4_indexer
from vllm.models.deepseek_v41 import attention


@pytest.mark.parametrize('spec', [None, NS(origin=None), NS(origin=__file__)])
def test_native_binary_identity_rejects_unknown(monkeypatch, spec):
    monkeypatch.setattr(qkv_split, 'find_spec', lambda _: spec)
    assert not qkv_split._native_build_matches()


def test_cpu_tensors_never_enter_gpu_kernel(monkeypatch):
    monkeypatch.setattr(qkv_split, '_NATIVE_BUILD_MATCHES', True)
    args = (torch.empty(1, 64, 512, dtype=torch.bfloat16),
            torch.empty(1, 512, dtype=torch.bfloat16), torch.empty(1, 64, 584, dtype=torch.uint8),
            torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, 64))
    assert not qkv_split.supported(*args)
    with pytest.raises(ValueError, match='verified'):
        qkv_split.split_qkv(*args)


def test_compile_factor_changes_when_native_build_is_not_verified(monkeypatch):
    from hcu_vllm_das import v41_indexer_activation as activation
    assert any(path.name == 'qkv_split.py' for path in activation._digest_paths())
    monkeypatch.setattr(qkv_split, '_NATIVE_BUILD_MATCHES', True)
    verified = activation._compute_digest()
    monkeypatch.setattr(qkv_split, '_NATIVE_BUILD_MATCHES', False)
    assert activation._compute_digest() != verified


@pytest.mark.parametrize('mode', ['pcp', 'interleaved', 'bf16_cache', 'other_heads', 'unknown_binary', 'mxfp8'])
def test_owned_path_falls_back_to_native(monkeypatch, mode):
    layer = object.__new__(fp4_indexer.Fp4Attention)
    torch.nn.Module.__init__(layer)
    layer.use_pcp = mode == 'pcp'
    layer.kv_mxfp8 = mode == 'mxfp8'
    monkeypatch.setattr(fp4_indexer.Fp4Attention, 'accepts_unnormed_unroped_query',
                        property(lambda self: mode == 'interleaved'))
    layer.n_local_heads = layer.padded_heads = 32 if mode == 'other_heads' else 64
    layer.swa_cache_layer = NS(prefix='swa', kv_cache=torch.empty(
        1, 64, 584, dtype=torch.bfloat16 if mode == 'bf16_cache' else torch.uint8))
    layer.rotary_emb = NS(cos_sin_cache=None)
    monkeypatch.setattr(qkv_split, '_NATIVE_BUILD_MATCHES', False)
    sentinel = object()
    monkeypatch.setattr(attention.DeepseekV4Attention, '_fused_qnorm_rope_kv_insert',
                        lambda *args: sentinel)
    assert layer._fused_qnorm_rope_kv_insert(
        None, None, None, {'swa': NS(block_size=64, slot_mapping=None)}) is sentinel


@pytest.mark.parametrize('rows', [1, 17, 65, 96, 128, 256])
def test_native_layout_accepts_dynamic_rows(monkeypatch, rows):
    from torch._subclasses.fake_tensor import FakeTensorMode
    monkeypatch.setattr(qkv_split, '_NATIVE_BUILD_MATCHES', True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: NS(gcnArchName='gfx938'))
    with FakeTensorMode():
        args = (torch.empty((rows, 64, 512), device='cuda', dtype=torch.bfloat16),
                torch.empty((rows, 512), device='cuda', dtype=torch.bfloat16),
                torch.empty((5, 64, 584), device='cuda', dtype=torch.uint8),
                torch.empty(rows, device='cuda', dtype=torch.int64),
                torch.empty(rows, device='cuda', dtype=torch.int64),
                torch.empty((1024, 64), device='cuda'))
        assert qkv_split.supported(*args)
        assert not qkv_split.supported(args[0][:0], *args[1:])
