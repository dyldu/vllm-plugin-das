# SPDX-License-Identifier: Apache-2.0
"""Contract: env gate + availability for compact TopK dequant."""

from __future__ import annotations

import pytest

from hcu_vllm_das import compact_topk_dequant_lightop as mod
from hcu_vllm_das.environment import compact_topk_dequant_enabled


def test_env_default_requires_master(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    monkeypatch.delenv("VLLM_HCU_USE_LIGHTOP_COMPACT_TOPK_DEQUANT", raising=False)
    assert compact_topk_dequant_enabled() is False


def test_env_default_off_with_master(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.delenv("VLLM_HCU_USE_LIGHTOP_COMPACT_TOPK_DEQUANT", raising=False)
    assert compact_topk_dequant_enabled() is False


def test_child_on(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_COMPACT_TOPK_DEQUANT", "1")
    assert compact_topk_dequant_enabled() is True


def test_install_idempotent(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_COMPACT_TOPK_DEQUANT", "1")
    mod._installed = False
    mod.install_compact_topk_dequant()
    mod.install_compact_topk_dequant()
    mod.check_compact_topk_dequant()
