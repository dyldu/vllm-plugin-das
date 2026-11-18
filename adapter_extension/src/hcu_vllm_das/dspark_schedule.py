# SPDX-License-Identifier: Apache-2.0
"""Private native DSpark namespaces for exact Q/KV and final Post schedules."""
from __future__ import annotations

import hashlib
import inspect
from types import FunctionType

from hcu_adapter.kernels import mhc_post_5120
from vllm.models.deepseek_v41.amd import dspark as native
from vllm.models.deepseek_v41.amd import model as native_model
from vllm.models.deepseek_v41.amd import rocm

from .qkv_split import try_attention_split


_REFERENCES = []


def _clone(owner, name, expected, substitutions):
    original = getattr(owner, name)
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != expected:
        raise RuntimeError('Owned DSpark native source changed: ' + name)
    function = inspect.unwrap(original)
    if not substitutions or any(key not in function.__globals__ for key in substitutions):
        raise RuntimeError('Owned DSpark native namespace anchor missing: ' + name)
    namespace = dict(function.__globals__)
    namespace.update(substitutions)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    clone.__qualname__ = function.__qualname__
    _REFERENCES.append((owner, name, original))
    return clone


class DraftAttention(rocm.DeepseekV41ROCMAiterMLAAttention):
    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        result = try_attention_split(self, q, kv, positions, attn_metadata)
        if result is not None:
            return result
        return super()._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)


_DECODER_INIT = _clone(native_model.DeepseekV4DecoderLayer, '__init__',
    '919c98099d1ba3e443ea5a5c5569e6bd407caf52c40945fb792e9b4e85a13a86',
    {'_select_dsv4_attn_cls': lambda cfg: DraftAttention})


class DraftDecoder(native_model.DeepseekV4DecoderLayer):
    __init__ = _DECODER_INIT


_MODEL_INIT = _clone(native.DSparkDeepseekV4Model, '__init__',
    '43de47a75df71afd16a474bb831a477de34ce3d9a77a6d8680cd9b1b9e6e13f2',
    {'DeepseekV4DecoderLayer': DraftDecoder})


def _final_post(x, residual, post, comb):
    if mhc_post_5120.supported(x, residual, post, comb):
        return mhc_post_5120.post_5120(x, residual, post, comb)
    return native.mhc_post_torch(x, residual, post, comb)


_FORWARD = _clone(native.DSparkDeepseekV4Model, 'forward',
    '4fae143677e054b267a5861db30277df79b51608cca8c5b4540220e63746fd01',
    {'mhc_post_torch': _final_post})


class DraftModel(native.DSparkDeepseekV4Model):
    __init__ = _MODEL_INIT
    forward = _FORWARD


_OUTER_INIT = _clone(native.DSparkDeepseekV4ForCausalLM, '__init__',
    '8f63625682361086e3df60598866076e0aefb657a200c03fbc7725b07ddb299f',
    {'DSparkDeepseekV4Model': DraftModel})


def initialize_owned(model, *, vllm_config, prefix):
    if any(getattr(owner, name) is not original for owner, name, original in _REFERENCES):
        raise RuntimeError('Owned DSpark native implementation identity changed')
    _OUTER_INIT(model, vllm_config=vllm_config, prefix=prefix)
