# SPDX-License-Identifier: Apache-2.0
"""Stock token-FP8 conversion for active rows of the bounded IPC draft layout."""
from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _masked_token_quant(X, Counts, Q, S, CAPACITY: tl.constexpr,
                        HIDDEN: tl.constexpr, LANES: tl.constexpr,
                        BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    lane = tl.program_id(1)
    count = tl.minimum(tl.load(Counts + expert), CAPACITY)
    col = tl.arange(0, BLOCK)
    for row in range(lane, count, LANES):
        offset = (expert * CAPACITY + row) * HIDDEN + col
        value = tl.load(X + offset, col < HIDDEN, other=0).to(tl.float32)
        amax = tl.max(tl.abs(value), 0)
        # Match vLLM's dynamic_per_token_scaled_fp8_quant_kernel_strided:
        # FP32 scale, its minimum, round-to-nearest division, E4M3 conversion.
        scale = tl.maximum(tl.div_rn(amax, 448.0), 1.0 / (448.0 * 512.0))
        quant = tl.minimum(tl.maximum(tl.div_rn(value, scale), -448.0), 448.0)
        tl.store(Q + offset, quant, col < HIDDEN)
        tl.store(S + expert * CAPACITY + row, scale)


def masked_token_quant(x: torch.Tensor, counts: torch.Tensor, *, lanes=64):
    """Leave inactive rows unspecified, as required by masked expert consumers."""
    if (not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 3
            or not x.is_contiguous() or x.shape != (16, 512, 5120)
            or counts.device != x.device or counts.dtype != torch.int32
            or counts.shape != (16,) or not counts.is_contiguous()
            or lanes not in (4, 8, 16, 32, 64)):
        raise ValueError('masked token quant requires the owned draft IPC layout')
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[:2], dtype=torch.float32, device=x.device)
    _masked_token_quant[(16, lanes)](
        x, counts, q, scales, 512, 5120, lanes, 8192,
        num_warps=4, enable_fp_fusion=False)
    return q, scales
