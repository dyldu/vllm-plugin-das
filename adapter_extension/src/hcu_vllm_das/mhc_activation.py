# SPDX-License-Identifier: Apache-2.0
"""Independent framework-free discovery and public OOT mHC publication."""
from __future__ import annotations

import inspect
import os
from threading import RLock

from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec

from .dsv41_activation import _require_release

_state = None
_failure = None
_publishing = False
_lock = RLock()
_FACTOR = 'VLLM_HCU_DAS_MHC_POLICY'


def _fork():
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = 'fork inherited incomplete mHC publication'


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_fork)


def _is_hcu():
    from vllm.platforms import current_platform
    return current_platform.is_rocm() and current_platform.device_name == 'hcu'


def policy_factor():
    from .mhc_runtime import read_policy
    p = _state['policy'] if _state is not None else read_policy()
    return p.master, p.mhc, p.pre, p.post, p.tilelang, p.fused


def _fatal(reason):
    global _failure
    if _failure is None:
        _failure = reason
    raise ExtensionCommitError(_failure)


def _method(function):
    return getattr(function, '__func__', function)


def check_health():
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _state is None:
            if not _is_hcu():
                return
            raise RuntimeError('HCU mHC compatibility is not active')
        from vllm import envs
        from vllm.model_executor import custom_op
        from . import mhc_runtime as runtime
        try:
            if (runtime.read_policy() != _state['policy']
                    or runtime.policy != _state['policy']):
                _fatal('HCU mHC policy changed after activation; restart worker')
            valid = (
                custom_op.op_registry_oot is _state['registry']
                and _method(custom_op.CustomOp.register_oot) is _state['register']
                and envs.environment_variables.get(_FACTOR) is policy_factor
                and all(custom_op.op_registry_oot.get(base.__name__) is impl
                        for base, impl in _state['owners'])
                and tuple(base.forward_hip for base, _ in _state['owners']) == _state['native']
            )
            if not valid:
                _fatal('HCU mHC public operator registration changed owner')
        except ExtensionCommitError:
            raise
        except BaseException as error:
            try:
                _fatal('HCU mHC registration health cannot be verified')
            except ExtensionCommitError as refusal:
                raise refusal from error


def activate_extension(context):
    global _state, _publishing
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publishing:
            _fatal('recursive HCU mHC activation')
        if _state is not None:
            try:
                _require_release(context.vllm_version)
                check_health()
            except BaseException as error:
                try:
                    _fatal('HCU mHC contract changed after publication')
                except ExtensionCommitError as refusal:
                    raise refusal from error
            return
        _require_release(context.vllm_version)
        if not _is_hcu():
            return
        from . import mhc_runtime as runtime
        selected = runtime.read_policy()
        if selected.tilelang or selected.fused:
            raise ValueError('HCU mHC requires TILELANG=False and FUSED=False')
        _publishing = True
        try:
            from vllm import envs
            from vllm.model_executor import custom_op
            from .mhc_ops import PAIRS
        except BaseException as error:
            try:
                _fatal('HCU mHC native preparation failed; restart process')
            except ExtensionCommitError as refusal:
                raise refusal from error
        finally:
            _publishing = False
        registry = custom_op.op_registry_oot
        for base, impl in PAIRS:
            if registry.get(base.__name__) not in (None, impl):
                raise RuntimeError(f'foreign HCU mHC operator owner: {base.__name__}')
            if inspect.signature(base.forward_hip) != inspect.signature(impl.forward_hip):
                # Annotations are intentionally ignored; parameter names, kinds
                # and defaults are the actual instance-call contract.
                left = inspect.signature(base.forward_hip).parameters
                right = inspect.signature(impl.forward_hip).parameters
                if tuple(left) != tuple(right) or any(
                    left[n].kind != right[n].kind or left[n].default != right[n].default
                    for n in left
                ):
                    raise RuntimeError(f'mHC native signature changed: {base.__name__}')
        factor = envs.environment_variables.get(_FACTOR)
        if factor is not None and factor is not policy_factor:
            raise RuntimeError('HCU mHC compile factor has a foreign owner')
        _publishing = True
        try:
            # Native/provider preparation can publish operators during import;
            # failures from here must never be reported as a clean fallback.
            runtime.configure(selected)
            runtime.register_ops()
            envs.environment_variables[_FACTOR] = policy_factor
            if selected.post_enabled and runtime.provider.mhc_post is not None:
                os.environ.setdefault('AITER_MHC_POST_KERNEL', 'v3')
            for base, impl in PAIRS:
                if base.__name__ not in registry:
                    base.register_oot()(impl)
            _state = dict(policy=selected, registry=registry, owners=PAIRS,
                          native=tuple(base.forward_hip for base, _ in PAIRS),
                          register=_method(custom_op.CustomOp.register_oot))
            check_health()
        except BaseException as error:
            try:
                _fatal('HCU mHC preparation/publication failed; restart process')
            except ExtensionCommitError as refusal:
                raise refusal from error
        finally:
            _publishing = False


def extension_spec():
    return ExtensionSpec(name='das-hcu-mhc', api_version=1, priority=110,
                         capabilities=frozenset({'hcu-mhc-hip-routing'}),
                         activate=activate_extension, check_health=check_health)
