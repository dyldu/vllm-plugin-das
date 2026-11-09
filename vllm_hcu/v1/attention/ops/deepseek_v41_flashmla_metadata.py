# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Own fresh decode leases and per-build fixed-width FlashMLA schedules.

Dynamic-length standalone callers retain isolated plans. Framework-owned
fixed-width schedules may share buffers only within one metadata build and
one native shape/stream; negative indices mask the actual per-layer lengths.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from vllm_hcu.platforms import envs as henvs

SCHEDULERS_ATTR = "_vllm_hcu_flashmla_decode_schedulers"


def decode_context_supported(config) -> bool:
    parallel = config.parallel_config
    return (
        parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and decode_query_limit(config) > 0
    )


def decode_query_limit(config) -> int:
    spec = config.speculative_config
    if spec is None:
        return 1
    count = getattr(spec, "num_speculative_tokens", None)
    if (
        not henvs.dsv41_flashmla_dspark_enabled()
        or getattr(spec, "method", None) != "dspark"
        or type(count) is not int or count < 1
    ):
        return 0
    # Match AttentionMetadataBuilder's spec-as-decode threshold. Actual rows
    # still use explicit per-token indices; no causal mask is synthesized here.
    return 1 + (2 if getattr(spec, "parallel_drafting", False) else 1) * count


@dataclass(frozen=True)
class DecodeBuildPolicy:
    query_limit: int
    allow_graph: bool
    speculative: bool


_BUILD_POLICIES: ContextVar[dict | None] = ContextVar(
    "dsv41_decode_build_policies", default=None,
)
_BUILD_SCHEDULERS: ContextVar[dict | None] = ContextVar(
    "dsv41_decode_build_schedulers", default=None,
)


@contextmanager
def decode_metadata_build_scope():
    """Resolve decode policy once per config and attention-metadata pass."""
    token = _BUILD_POLICIES.set({})
    scheduler_token = _BUILD_SCHEDULERS.set({})
    try:
        yield
    finally:
        _BUILD_SCHEDULERS.reset(scheduler_token)
        _BUILD_POLICIES.reset(token)


def decode_scheduler_pool(config) -> dict:
    pools = _BUILD_SCHEDULERS.get()
    if pools is None:
        return {}
    key = id(config)
    if key not in pools:
        pools[key] = (config, {})
    return pools[key][1]


def _resolve_decode_build_policy(config) -> DecodeBuildPolicy | None:
    if not henvs.dsv41_flashmla_decode_enabled():
        return None
    parallel = config.parallel_config
    if (
        parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
    ):
        return None
    query_limit = decode_query_limit(config)
    if query_limit < 1:
        return None
    return DecodeBuildPolicy(
        query_limit=query_limit,
        allow_graph=henvs.dsv41_flashmla_decode_graph_enabled(),
        speculative=config.speculative_config is not None,
    )


def decode_build_policy(config) -> DecodeBuildPolicy | None:
    policies = _BUILD_POLICIES.get()
    if policies is None:
        return _resolve_decode_build_policy(config)
    key = id(config)
    if key not in policies:
        policies[key] = (config, _resolve_decode_build_policy(config))
    return policies[key][1]


class DecodeScheduler:
    """Keep dynamic plans isolated; share only explicitly fixed-width plans."""

    def __init__(
        self, factory: Callable, *, allow_graph: bool = False,
        query_limit: int = 1, speculative: bool = False,
        shared_plans: dict | None = None,
    ):
        self._factory = factory
        self._pending = factory()[0]
        self.allow_graph = allow_graph
        self.query_limit = query_limit
        self.speculative = speculative
        self.shared_plans = shared_plans

    def take(self, key=None):
        if key is not None and self.shared_plans is not None:
            if key in self.shared_plans:
                return self.shared_plans[key]
        plan = self._pending
        if plan is None:
            plan = self._factory()[0]
        # Consume before launch: a failed native initialization must not leave
        # a partially initialized plan available to a subsequent invocation.
        self._pending = None
        if key is not None and self.shared_plans is not None:
            self.shared_plans[key] = plan
        return plan

    def discard(self, key, plan):
        if self.shared_plans is not None and self.shared_plans.get(key) is plan:
            del self.shared_plans[key]
