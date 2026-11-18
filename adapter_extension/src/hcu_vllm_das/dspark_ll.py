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

