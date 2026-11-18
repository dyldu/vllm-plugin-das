# SPDX-License-Identifier: Apache-2.0
"""Bounded single-node IPC LL resource policy for the explicit draft experiment.

The installed HCU wheel invalidates target dispatch after constructing a second
RocSHMEM Buffer. Keep one native target-sized allocation and its native cache
entry/destructor. IPC layout switching additionally requires an explicit clean
before dispatch (also recorded inside CUDA graphs). This policy is deliberately
unavailable for GDA/internode transports, where expert/QP contracts differ.
"""
from __future__ import annotations

import os
import weakref


_MODEL_ENTRY_OWNERS = weakref.WeakKeyDictionary()


def install_model_entry_reset(model, config, num_experts):
    """Record one native IPC clean at each complete target/draft model entry.

    Full graph capture records the hook's GPU clean. Piecewise/eager calls run
    it before invoking their compiled pieces. It does not infer the last layout
    from Python state, and does not alter calls inside another model's graph.
    Unknown model geometries retain the conservative per-dispatch clean.
    """
    from .dspark_backend import native_draft_ll_requested
    if not native_draft_ll_requested(config):
        return False
    mc = config.model_config
    hf = getattr(mc, 'hf_text_config', None)
    if hf is None:
        hf = getattr(mc.hf_config, 'text_config', mc.hf_config)
    names = ('hidden_size', 'n_routed_experts', 'num_experts_per_tok',
             'dspark_n_routed_experts', 'dspark_num_experts_per_tok',
             'num_hidden_layers')
    if tuple(getattr(hf, n, None) for n in names) != (5120, 384, 6, 128, 3, 40):
        return False
    if num_experts not in (128, 384):
        raise ValueError('Unsupported model entry IPC expert count')
    from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
    manager = get_ep_all2all_manager()
    policy = SharedIPCManager(manager)
    owners = _MODEL_ENTRY_OWNERS.setdefault(manager, {})
    previous = owners.get(num_experts)
    if previous is not None and previous() is not None:
        raise ValueError('Duplicate live model entry IPC reset owner')
    buffer = None
    kwargs = dict(max_num_tokens_per_dp_rank=64, token_hidden_size=5120,
                  num_ep_ranks=8, num_global_experts=num_experts,
                  num_local_experts=num_experts // 8)

    def clean_at_entry(_module, _args):
        nonlocal buffer
        if buffer is None:
            buffer = policy.get_handle(kwargs)
        buffer.clean_low_latency_buffer(64, 5120, num_experts,
                                        quant_group_size=0,
                                        num_topk=6 if num_experts == 384 else 3)

    model.register_forward_pre_hook(clean_at_entry)
    owners[num_experts] = weakref.ref(model)
    return True


class SharedIPCManager:
    """Factory argument only; never replaces the EP group's native manager."""

    def __init__(self, manager):
        import deep_ep
        from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager

        if (not isinstance(manager, DeepEPLLAll2AllManager)
                or manager.internode or manager.world_size != 8
                or manager.dp_world_size != 8 or manager.support_fault_tolerance):
            raise ValueError('DSpark shared LL requires single-node DP8/EP8 native LL')
        if os.environ.get('ROCSHMEM_BACKEND', 'ipc') != 'ipc':
            raise ValueError('DSpark shared LL is validated only with ROCSHMEM IPC')
        # The supplied scripts request layered OPT, but this wheel resolves
        # both layered selectors to False for <=8 ranks. Check the effective
        # native decision, preserving the baseline environment unchanged.
        for name in ('is_ll_layered_opt_enabled', 'is_ll_layered_develop_enabled'):
            select = getattr(deep_ep.Buffer, name, None)
            if not callable(select) or select(8):
                raise ValueError('DSpark shared LL requires native legacy IPC selection')
        for key in ('ROCSHMEM_IPC_MNVL', 'DEEP_EP_NORMAL_MNVL'):
            if os.environ.get(key, '0').strip().lower() not in ('0', 'false', 'off', ''):
                raise ValueError('DSpark shared LL does not support ' + key)
        if os.environ.get('DEEPEP_TEST_FORCE_ROUTE_MODE', '').strip():
            raise ValueError('DSpark shared LL does not support forced test routes')
        self.native = manager
        self.world_size = manager.world_size
        self.dp_world_size = manager.dp_world_size

    def has_model_entry_reset(self, num_experts):
        owner = _MODEL_ENTRY_OWNERS.get(self.native, {}).get(num_experts)
        return owner is not None and owner() is not None

    def get_handle(self, kwargs):
        expected = dict(max_num_tokens_per_dp_rank=64, token_hidden_size=5120,
                        num_ep_ranks=8, num_global_experts=384, num_local_experts=48)
        e = kwargs.get('num_global_experts')
        actual = dict(expected, num_global_experts=e,
                      num_local_experts=e // 8 if type(e) is int else None)
        if e not in (128, 384) or kwargs != actual:
            raise ValueError('DSpark shared LL geometry is outside the validated contract')
        # Preserve the native 384-expert allocation/QP setting, cache key and
        # single destruction. Never add an alias entry for the smaller layout.
        return self.native.get_handle(expected)

