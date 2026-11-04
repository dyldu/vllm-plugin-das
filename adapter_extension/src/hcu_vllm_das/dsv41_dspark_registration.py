# SPDX-License-Identifier: Apache-2.0
"""Public model-registry handoff for the HCU V41 draft's scale-name adapter.

Only registration metadata is read during preparation. Publication uses the
public register_model API and a lazy path; Core classes and caches are untouched.
Like the public quantization registry, this assumes serialized plugin startup.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect

ARCHITECTURE = "DSparkV41DraftModel"
_NATIVE = ("vllm.models.deepseek_v41", "DSparkDeepseekV4ForCausalLM")
_PLUGIN = ("hcu_vllm_das.models.dsv41_dspark", "HcuDSparkDeepseekV41ForCausalLM")


def _key(entry):
    return getattr(entry, "module_name", None), getattr(entry, "class_name", None)


def _function(callable_):
    return getattr(callable_, "__func__", callable_)


@dataclass(frozen=True)
class DraftRegistration:
    registry: object
    entry: object
    register_function: object


def prepare_registration() -> DraftRegistration:
    from vllm.model_executor.models import ModelRegistry

    register = ModelRegistry.register_model
    if tuple(inspect.signature(register).parameters) != ("model_arch", "model_cls"):
        raise RuntimeError("DSV4.1 public model registration signature changed")
    entry = ModelRegistry.models.get(ARCHITECTURE)
    if _key(entry) not in (_NATIVE, _PLUGIN):
        raise RuntimeError("DSV4.1 draft model entry has a missing or foreign owner")
    return DraftRegistration(ModelRegistry, entry, _function(register))


def publish_registration(plan: DraftRegistration) -> DraftRegistration:
    registry = plan.registry
    if (
        registry.models.get(ARCHITECTURE) is not plan.entry
        or _function(registry.register_model) is not plan.register_function
    ):
        raise RuntimeError("DSV4.1 draft model owner changed after preparation")
    if _key(plan.entry) == _NATIVE:
        registry.register_model(ARCHITECTURE, ":".join(_PLUGIN))
    entry = registry.models.get(ARCHITECTURE)
    if _key(entry) != _PLUGIN:
        raise RuntimeError("DSV4.1 public model registration did not retain its owner")
    return DraftRegistration(registry, entry, plan.register_function)


def check_registration(state: DraftRegistration) -> None:
    if (
        state.registry.models.get(ARCHITECTURE) is not state.entry
        or _function(state.registry.register_model) is not state.register_function
        or _key(state.entry) != _PLUGIN
    ):
        raise RuntimeError("DSV4.1 public draft model registration changed owner")
