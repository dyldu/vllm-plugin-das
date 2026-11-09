# SPDX-License-Identifier: Apache-2.0
"""Opt-in DSV4.1 Q/KV RMSNorm + per-token Q FP8 fusion for Channel-FP8.

Hooks DeepseekV4Attention / ROCm subclass `_split_qkv_and_norm` so both the
main Attention and indexer `wq_b` share one QuantizedActivation (kFp8DynamicTokenSym).
`_wq_b_proj` already forwards QuantizedActivation into the linear; no second patch.
"""
from __future__ import annotations

import functools
import os
from threading import RLock

import torch
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

from .dsv41_activation import _require_release
from .environment import _flag, custom_ops_enabled

_FLAG = 'VLLM_HCU_DSV41_QUERY_RMSNORM_FP8'
_LIGHTOP_FLAG = 'VLLM_HCU_USE_LIGHTOP_PER_TOKEN_QUANT_FP8'
_FACTOR = 'VLLM_HCU_DAS_V41_QUERY_RMSNORM_FP8'
_MARKER = '_hcu_das_query_rmsnorm_fp8_applied'
_WRAPPER = '_hcu_das_query_rmsnorm_fp8_wrapper'

_state = None
_failure = None
_publishing = False
_lock = RLock()


def _fork():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = 'fork inherited incomplete V41 query RMSNorm/FP8 publication'


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_fork)


def _is_hcu():
    from vllm.platforms import current_platform
    return current_platform.is_rocm() and current_platform.device_name == 'hcu'


def query_quant_enabled():
    return custom_ops_enabled() and _flag(_FLAG, default=True)


def policy_factor():
    return (
        query_quant_enabled() if _state is None else _state['enabled'],
        _flag(_LIGHTOP_FLAG, default=False),
    )


def _fatal(message):
    global _failure
    if _failure is None:
        _failure = message
    raise ExtensionCommitError(_failure)


def _projection_supported(linear):
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
    )

    if getattr(linear, 'requires_unquantized_input', False):
        return False
    if getattr(linear, 'input_scale_ub', None) is not None:
        return False
    weight = getattr(linear, 'weight', None)
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.float8_e4m3fn:
        return False
    return get_input_quant_key(linear) == kFp8DynamicTokenSym


def eligible(attention, packed):
    if not query_quant_enabled():
        return False
    if not (packed.is_cuda and packed.dtype == torch.bfloat16
            and packed.ndim == 2 and packed.is_contiguous()
            and 0 < packed.shape[0] <= 64
            and attention.q_lora_rank == 1280 and attention.head_dim == 512):
        return False
    if not _projection_supported(attention.wq_b):
        return False
    if attention.indexer is not None and not _projection_supported(attention.indexer.wq_b):
        return False
    return all(
        norm.weight.dtype == packed.dtype and norm.weight.is_contiguous()
        and norm.weight.device == packed.device and norm.weight.shape == (width,)
        for norm, width in ((attention.q_norm, 1280), (attention.kv_norm, 512))
    )


def _wrap_split(original):
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
    )

    @functools.wraps(original)
    def split(self, qr_kv):
        if not eligible(self, qr_kv):
            return original(self, qr_kv)
        query, kv = qr_kv.split((self.q_lora_rank, self.head_dim), dim=-1)
        data, scale, kv_out = torch.ops.vllm.hcu_dsv41_query_rmsnorm_quant(
            query, kv, self.q_norm.weight, self.kv_norm.weight, self.eps,
            _flag(_LIGHTOP_FLAG, default=False),
        )
        return (
            QuantizedActivation(data, scale, query.dtype, query.shape,
                                kFp8DynamicTokenSym),
            None,
            kv_out,
        )

    setattr(split, _WRAPPER, True)
    return split


def _install_hooks():
    from importlib import import_module

    from .dsv41_query_quant import register_op

    register_op()
    attention_mod = import_module('vllm.models.deepseek_v41.attention')
    targets = [attention_mod.DeepseekV4Attention]
    try:
        rocm = import_module('vllm.models.deepseek_v41.amd.rocm')
        targets.append(rocm.DeepseekV41ROCMAiterMLAAttention)
    except Exception:
        pass
    installed = []
    for cls in targets:
        original = getattr(cls, '_split_qkv_and_norm')
        if getattr(original, _WRAPPER, False) or getattr(cls, _MARKER, False):
            installed.append((cls, original, getattr(cls, '_split_qkv_and_norm')))
            continue
        wrapper = _wrap_split(original)
        setattr(cls, '_hcu_das_original_split_qkv_and_norm', original)
        cls._split_qkv_and_norm = wrapper
        setattr(cls, _MARKER, True)
        installed.append((cls, original, wrapper))
    return installed


def check_health():
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _state is None:
            if not _is_hcu():
                return
            raise RuntimeError('V41 query RMSNorm/FP8 fusion is not active')
        from vllm import envs
        try:
            valid = (
                query_quant_enabled() == _state['enabled']
                and envs.environment_variables.get(_FACTOR) is policy_factor
                and all(
                    getattr(cls, '_split_qkv_and_norm') is wrapper
                    for cls, _, wrapper in _state['owners']
                )
            )
            if not valid:
                _fatal('V41 query RMSNorm/FP8 ownership/policy changed; restart process')
        except ExtensionCommitError:
            raise
        except BaseException as error:
            try:
                _fatal('V41 query RMSNorm/FP8 public health is unavailable')
            except ExtensionCommitError as refusal:
                raise refusal from error


def activate_extension(context):
    global _state, _publishing
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publishing:
            _fatal('recursive V41 query RMSNorm/FP8 activation')
        if _state is not None:
            try:
                _require_release(context.vllm_version)
                check_health()
            except BaseException as error:
                try:
                    _fatal('V41 query RMSNorm/FP8 contract changed after publication')
                except ExtensionCommitError as refusal:
                    raise refusal from error
            return
        _require_release(context.vllm_version)
        if not _is_hcu():
            return
        selected = query_quant_enabled()
        _publishing = True
        try:
            from vllm import envs
            factor = envs.environment_variables.get(_FACTOR)
            if factor is not None and factor is not policy_factor:
                raise RuntimeError('V41 query RMSNorm/FP8 compile factor has a foreign owner')
            owners = _install_hooks() if selected else []
            envs.environment_variables[_FACTOR] = policy_factor
            _state = dict(enabled=selected, owners=owners)
            check_health()
        except BaseException as error:
            try:
                _fatal('V41 query RMSNorm/FP8 preparation failed; restart process')
            except ExtensionCommitError as refusal:
                raise refusal from error
        finally:
            _publishing = False


def extension_spec():
    return ExtensionSpec(
        name='das-hcu-v41-query-quant',
        api_version=1,
        priority=115,
        capabilities=frozenset({'hcu-dsv41-query-rmsnorm-fp8'}),
        activate=activate_extension,
        check_health=check_health,
    )
