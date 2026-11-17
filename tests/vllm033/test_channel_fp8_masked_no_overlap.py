# SPDX-License-Identifier: Apache-2.0
"""Channel-FP8 masked GEMM defaults enable_overlap=False (SGL decode arity)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

from hcu_vllm_das.moe import channel_fp8_runtime as runtime


def test_deepgemm_masked_defaults_enable_overlap_false(monkeypatch):
    seen = {}

    def fake_masked(*args, **kwargs):
        seen["kwargs"] = dict(kwargs)
        return "ok"

    monkeypatch.setitem(
        sys.modules,
        "deepgemm",
        SimpleNamespace(m_grouped_fp8_gemm_nt_masked=fake_masked),
    )
    assert runtime.deepgemm_masked("a", "b", "d", "m", 16) == "ok"
    assert seen["kwargs"]["enable_overlap"] is False

    assert (
        runtime.deepgemm_masked("a", "b", "d", "m", 16, enable_overlap=True) == "ok"
    )
    assert seen["kwargs"]["enable_overlap"] is True
