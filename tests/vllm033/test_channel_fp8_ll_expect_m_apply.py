# SPDX-License-Identifier: Apache-2.0
"""Channel-FP8 LL apply must forward expect_m into LightOp (SGLang stage align)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from hcu_vllm_das.moe import channel_fp8_runtime as runtime


def test_ll_apply_forwards_expect_m_to_activation_and_gemms():
    e, capacity, k, topk = 2, 16, 4, 6
    experts = SimpleNamespace(
        max_num_tokens=16,
        num_dispatchers=8,
        w1_scale=torch.ones(e, 8, dtype=torch.float32),
        w2_scale=torch.ones(e, k, dtype=torch.float32),
        quant_config=SimpleNamespace(gemm1_clamp_limit=None),
        moe_problem_size=lambda hidden_states, w1, w2, topk_ids: (
            hidden_states.shape[0],
            hidden_states.shape[1],
            w1.shape[1],
            hidden_states.shape[-1],
            topk_ids.shape[1],
        ),
    )
    hidden = torch.zeros(e, capacity, k, dtype=torch.float8_e4m3fn)
    scale = torch.ones(e, capacity, dtype=torch.float32)
    w1 = torch.zeros(e, 8, k, dtype=torch.float8_e4m3fn)
    w2 = torch.zeros(e, k, 4, dtype=torch.float8_e4m3fn)
    topk_ids = torch.zeros(8, topk, dtype=torch.int32)
    topk_weights = torch.ones(8, topk)
    counts = torch.tensor([2, 1], dtype=torch.int32)
    meta = SimpleNamespace(expert_num_tokens=counts)
    out = torch.zeros(e, capacity, k, dtype=torch.bfloat16)
    workspace2 = torch.empty(e * capacity * 8, dtype=torch.bfloat16)

    gemm_expected = []
    act_kwargs = {}

    def fake_masked(a, b, d, masked_m, expected_m, *args, **kwargs):
        gemm_expected.append(expected_m)
        d.zero_()
        return d

    def fake_act(gateup, *, fp8type, tokens_per_expert, expect_m=-1, limit=None):
        act_kwargs["expect_m"] = expect_m
        q = torch.zeros(
            gateup.shape[0],
            gateup.shape[1],
            gateup.shape[-1] // 2,
            dtype=torch.float8_e4m3fn,
        )
        s = torch.ones(gateup.shape[0], gateup.shape[1], dtype=torch.float32)
        return q, s

    with (
        patch.object(runtime, "deepgemm_masked", fake_masked),
        patch.object(runtime, "fuse_silu_mul_fp8_quant_ep", fake_act),
        patch.object(
            runtime,
            "_resize_cache",
            lambda buf, shape: torch.empty(shape, dtype=torch.bfloat16),
        ),
        patch.object(runtime, "_token_scale", lambda scale, shape: scale.view(shape)),
    ):
        runtime.ChannelFP8MaskedExperts.apply(
            experts,
            out,
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            MoEActivation.SILU,
            384,
            None,
            scale,
            None,
            torch.empty(0),
            workspace2,
            meta,
            False,
        )

    # (8 * 8 * 6 + 384) // 384 == 9 -> round16 == 16, clamp capacity 16.
    assert gemm_expected == [16, 16]
    assert act_kwargs["expect_m"] == 16
