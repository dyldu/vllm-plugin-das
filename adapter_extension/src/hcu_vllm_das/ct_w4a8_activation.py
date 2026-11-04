# SPDX-License-Identifier: Apache-2.0
"""Independent public registration for the distinct CT W4A8 adapter."""
from __future__ import annotations
from importlib import import_module
import inspect
import os
from threading import RLock
from hcu_adapter.extensions import ExtensionCommitError, ExtensionSpec
from hcu_vllm_das.dsv41_activation import _require_release

NAME='hcu_dsv41_ct_w4a8'
_lock=RLock()
_publication=None
_failure=None
_publishing=False


def _fork():
    global _lock, _failure
    _lock=RLock()
    if _publishing and _failure is None: _failure='fork inherited incomplete CT W4A8 registration'


if hasattr(os,'register_at_fork'): os.register_at_fork(after_in_child=_fork)


def _fatal(message):
    global _failure
    if _failure is None: _failure=message
    raise ExtensionCommitError(_failure)


def check_health():
    with _lock:
        if _failure is not None: raise ExtensionCommitError(_failure)
        if _publication is None: raise RuntimeError('CT W4A8 registration is not active')
        quant,cls,register,resolve=_publication
        try:
            owned=(quant.register_quantization_config is register and quant.get_quantization_config is resolve
                   and NAME in quant.QUANTIZATION_METHODS and resolve(NAME) is cls)
        except BaseException as error:
            try: _fatal('CT W4A8 public selector is unavailable')
            except ExtensionCommitError as refusal: raise refusal from error
        if not owned: _fatal('CT W4A8 public registration changed owner')


def activate_extension(context):
    global _publication,_publishing
    with _lock:
        if _failure is not None: raise ExtensionCommitError(_failure)
        if _publishing: _fatal('recursive CT W4A8 activation')
        if _publication is not None:
            try: _require_release(context.vllm_version); check_health()
            except BaseException as error:
                try: _fatal('CT W4A8 contract changed after publication')
                except ExtensionCommitError as refusal: raise refusal from error
            return
        _require_release(context.vllm_version)
        _publishing=True
        try:
            try:
                quant=import_module('vllm.model_executor.layers.quantization')
                cls=import_module('hcu_vllm_das.quantization.ct_w4a8').HcuDSV41CTW4A8Config
            except BaseException as error:
                try: _fatal('CT W4A8 native preparation failed')
                except ExtensionCommitError as refusal: raise refusal from error
            register,resolve=quant.register_quantization_config,quant.get_quantization_config
            if any(tuple(inspect.signature(fn).parameters)!=('quantization',) for fn in (register,resolve)):
                raise RuntimeError('CT W4A8 public quantization registration ABI changed')
            if NAME in quant.QUANTIZATION_METHODS:
                if resolve(NAME) is not cls: raise RuntimeError('CT W4A8 name has a foreign owner')
            else:
                try: register(NAME)(cls)
                except BaseException as error:
                    try: _fatal('CT W4A8 publication failed; restart the process')
                    except ExtensionCommitError as refusal: raise refusal from error
            _publication=(quant,cls,register,resolve)
            check_health()
        finally:
            _publishing=False


def extension_spec():
    return ExtensionSpec(name='das-hcu-ct-w4a8',api_version=1,priority=105,
        capabilities=frozenset({'dsv41-ct-w4a8'}),activate=activate_extension,check_health=check_health)
