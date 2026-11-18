# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for Channel-FP8 LL expected_m stage alignment."""
from __future__ import annotations

import pytest

from hcu_vllm_das.moe.channel_fp8_runtime import (
    ChannelFP8RuntimeError,
    estimate_ll_expected_m,
)


def test_estimate_ll_expected_m_matches_sglang_then_round16():
    # SGLang raw: (64 * 8 * 6 + 384) // 384 == 9; round-up-16 -> 16.
    assert (
        estimate_ll_expected_m(
            local_num_tokens=64,
            num_dispatchers=8,
            topk=6,
            global_num_experts=384,
            capacity=128,
        )
        == 16
    )


def test_estimate_ll_expected_m_clamps_to_capacity():
    assert (
        estimate_ll_expected_m(
            local_num_tokens=64,
            num_dispatchers=8,
            topk=6,
            global_num_experts=384,
            capacity=8,
        )
        == 8
    )


def test_estimate_ll_expected_m_empty_capacity():
    assert (
        estimate_ll_expected_m(
            local_num_tokens=0,
            num_dispatchers=8,
            topk=6,
            global_num_experts=384,
            capacity=0,
        )
        == 0
    )


def test_estimate_ll_expected_m_rejects_bad_inputs():
    with pytest.raises(ChannelFP8RuntimeError):
        estimate_ll_expected_m(
            local_num_tokens=1,
            num_dispatchers=0,
            topk=6,
            global_num_experts=384,
            capacity=16,
        )
