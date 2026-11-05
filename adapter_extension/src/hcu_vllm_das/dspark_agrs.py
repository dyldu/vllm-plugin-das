# SPDX-License-Identifier: Apache-2.0
"""Instance-owned V41 CT draft AG-RS; no target/group-global replacements.

Native DSpark constructs from a shallow private config. Native AG-RS uses the
actual draft forward DP sizes; reviewed native PF functions resolve an owned
manager only within their private instance namespaces.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
from types import FunctionType, MethodType
import torch
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group, get_pcp_group
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import MoEPrepareAndFinalizeNaiveDPEPModular
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

_BACKEND = 'allgather_reducescatter'
_OWNER = object()


@dataclass(frozen=True)
class TokenCapacity:
    query_per_request: int
    query_rows: int
    verify_rows: int
    budget: int


def token_capacity(cfg):
    spec, scheduler = cfg.speculative_config, cfg.scheduler_config
    n, reqs, budget = spec.num_speculative_tokens, scheduler.max_num_seqs, scheduler.max_num_batched_tokens
    if any(type(x) is not int or x <= 0 for x in (n, reqs, budget)):
        raise ValueError('DSpark requires positive integer n/requests/token budget')
    anchor = getattr(spec.draft_model_config.hf_config, 'sample_from_anchor', True)
    if type(anchor) is not bool:
        raise ValueError('DSpark sample_from_anchor must be bool')
    query = n if anchor else n + 1
    result = TokenCapacity(query, reqs * query, reqs * (n + 1), budget)
    if max(result.query_rows, result.verify_rows) > budget:
        raise ValueError(f'DSpark capacity exceeds original budget: query={result.query_rows}, verify={result.verify_rows}, budget={budget}')
    return result


def is_owned_draft(cfg):
    return getattr(cfg, '_hcu_dspark_agrs_owner', None) is _OWNER


def draft_config(cfg):
    """Copy only construction policy, retaining native live registration tables."""
    from .quantization.ct_w4a8 import HcuDSV41CTW4A8Config
    p, spec = cfg.parallel_config, cfg.speculative_config
    if not isinstance(cfg.quant_config, HcuDSV41CTW4A8Config):
        raise ValueError('owned draft AG-RS requires the HCU CT W4A8 config')
    if spec is None or spec.method != 'dspark' or spec.draft_model_config.hf_config.model_type not in ('deepseek_v41', 'deepseek_v41_text'):
        raise ValueError('owned draft AG-RS requires native V41 DSpark')
    if any(getattr(p, name) != 1 for name in ('tensor_parallel_size', 'pipeline_parallel_size', 'prefill_context_parallel_size', 'decode_context_parallel_size')):
        raise ValueError('DSpark AG-RS requires TP1/PP1/PCP1/DCP1')
    if type(p.data_parallel_size) is not int or p.data_parallel_size <= 1 or not p.enable_expert_parallel:
        raise ValueError('DSpark AG-RS requires actual DP+EP membership')
    if (cfg.lora_config is not None or p.enable_eplb or p.enable_elastic_ep or p.enable_dbo
            or cfg.kernel_config.monokernel):
        raise ValueError('DSpark AG-RS does not support LoRA/EPLB/ElasticEP/DBO/monokernel')
    if not cfg.use_v2_model_runner or not cfg.model_config.enforce_eager or not spec.draft_model_config.enforce_eager:
        raise ValueError('DSpark AG-RS requires eager MRV2 target and draft; Graph not validated')
    if cfg.kernel_config.moe_backend != 'deep_gemm':
        raise ValueError('DSpark CT W4A8 requires the owned DeepGEMM expert path')
    capacity = token_capacity(cfg)
    result = copy.copy(cfg)
    result.parallel_config = copy.copy(p)
    result.parallel_config.all2all_backend = _BACKEND
    result._hcu_dspark_agrs_owner = _OWNER
    result._hcu_dspark_capacity = capacity
    return result


class HcuDSparkAgRsManager(AgRsAll2AllManager):
    """Native variable AG-RS over borrowed, ordered DP/EP groups; no ownership."""
    def __init__(self, capacity):
        self._dp, self._ep, self._tp, self._pcp = get_dp_group(), get_ep_group(), get_tp_group(), get_pcp_group()
        self._capacity = capacity
        self._ranks = tuple(self._dp.ranks)
        self.check_health()
        super().__init__(self._ep.cpu_group)
        if self.world_size != self._dp.world_size or self.rank != self._dp.rank_in_group or not self.use_ep:
            raise RuntimeError('DSpark AG-RS manager EP/DP coordinates disagree')

    def check_health(self):
        if (get_dp_group() is not self._dp or get_ep_group() is not self._ep
                or get_tp_group() is not self._tp or get_pcp_group() is not self._pcp
                or self._tp.world_size != 1 or self._pcp.world_size != 1
                or tuple(self._dp.ranks) != self._ranks or tuple(self._ep.ranks) != self._ranks
                or self._dp.world_size != len(self._ranks) or self._ep.world_size != len(self._ranks)
                or self._dp.rank_in_group != self._ep.rank_in_group):
            raise RuntimeError('DSpark AG-RS borrowed group identity/order changed')

    def _get_comm_group(self, is_sequence_parallel):
        self.check_health()
        if is_sequence_parallel:
            raise ValueError('DSpark AG-RS requires non-SP local query rows')
        return self._dp

    def _get_sizes(self, num_local_tokens, comm_group):
        sizes = super()._get_sizes(num_local_tokens, comm_group)
        if len(sizes) != self.world_size or any(type(n) is not int or n < 0 or n > self._capacity for n in sizes):
            raise ValueError('DSpark current query DP sizes exceed checked local capacity')
        return sizes

    def destroy(self):
        raise RuntimeError('DSpark AG-RS does not own its borrowed process groups')


_REVIEWED = {
    'prepare': 'f4069809c8fe534dd02304aad504d2bf795da5da1823fed424cb8acf89bdd8ca',
    'finalize': '9cddd21280de8991d002d51941ab1e12337949867752631737e8d9b2d99b2bde',
    '_allocate_combine_input': 'f01d608b957578f2c95c64c82a2c08890b8d94a15c1d3e08a676617fe6d6b1c2',
}


def _bind(instance, name, manager):
    function = getattr(MoEPrepareAndFinalizeNaiveDPEPModular, name)
    actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    if actual != _REVIEWED[name]:
        raise RuntimeError(f'DSpark native modular PF source changed: {name}: {actual}')
    namespace = dict(function.__globals__, get_ep_group=lambda: manager)
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__, clone.__annotations__ = function.__kwdefaults__, function.__annotations__
    return MethodType(clone, instance)


class HcuCTAgRsPrepareAndFinalize(MoEPrepareAndFinalizeNaiveDPEPModular):
    def __init__(self, manager):
        super().__init__(is_sequence_parallel=False, num_dispatchers=manager.world_size)
        self.manager = manager
        self._native_prepare = _bind(self, 'prepare', manager)
        self.finalize = _bind(self, 'finalize', manager)
        self._allocate_combine_input = _bind(self, '_allocate_combine_input', manager)

    def prepare(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                apply_router_weight_on_input, quant_config, defer_input_quant=False):
        self.manager.check_health()
        if (defer_input_quant or quant_config.quant_dtype != torch.int8
                or not quant_config.per_act_token_quant or quant_config.block_shape is not None):
            raise ValueError('DSpark CT AG-RS requires owned dynamic token INT8')
        if (a1.ndim != 2 or a1.dtype != torch.bfloat16 or a1.shape[0] > self.manager._capacity
                or topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2
                or topk_ids.shape[0] != a1.shape[0] or topk_ids.dtype != torch.int64
                or topk_weights.dtype != torch.float32):
            raise ValueError('DSpark AG-RS local input/routes violate the checked wire contract')
        # Gather BF16 first so even a zero-query rank reaches native collectives;
        # INT8 token quantization is then independent of unequal DP row counts.
        x, _, meta, ids, weights = self._native_prepare(a1, topk_weights, topk_ids,
            num_experts, expert_map, apply_router_weight_on_input, quant_config, defer_input_quant=True)
        if not x.shape[0]:
            q, scales = x.to(torch.int8), x.new_empty((0, 1), dtype=torch.float32)
        else:
            q, scales = moe_kernel_quantize_input(x, None, quant_dtype=torch.int8,
                per_act_token_quant=True, block_shape=None)
        if scales.ndim == 1:
            scales = scales.unsqueeze(-1)
        if q.dtype != torch.int8 or scales.dtype != torch.float32 or scales.device != q.device or scales.shape != (q.shape[0], 1):
            raise RuntimeError('DSpark AG-RS did not produce per-token INT8 scales')
        return q, scales, meta, ids, weights


def make_draft_prepare(moe, cfg):
    if not is_owned_draft(cfg):
        raise ValueError('AG-RS CT preparation requires owned DSpark construction')
    if token_capacity(cfg) != cfg._hcu_dspark_capacity:
        raise RuntimeError('DSpark token capacity changed after construction')
    manager = HcuDSparkAgRsManager(moe.max_num_tokens)
    actual = moe.moe_parallel_config
    if actual.ep_size != manager.world_size or actual.dp_size != manager.dp_world_size or actual.ep_rank != manager.rank:
        raise RuntimeError('DSpark MoE membership differs from borrowed AG-RS groups')
    return HcuCTAgRsPrepareAndFinalize(manager)
