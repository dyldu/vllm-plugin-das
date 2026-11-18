# SPDX-License-Identifier: Apache-2.0
"""Preserve image-sentinel routing after the text-only mega-gate.

Selection follows vLLM's router/dsv4_topk.py. That module only defines its
Triton kernel on CUDA; HCU exposes ROCm semantics, so keep this small portable
device implementation here rather than depending on its conditional symbol.
"""
from vllm.triton_utils import tl, triton


@triton.jit
def _correct_image_routes(LOGITS, BIAS, PADDING, WEIGHTS, IDS, SCALE, TOKENS,
                          IMAGE_BIAS, SENTINEL: tl.constexpr, E: tl.constexpr,
                          BLOCK: tl.constexpr, HAS_PADDING: tl.constexpr):
    row = tl.program_id(0)
    token = tl.load(TOKENS + row).to(tl.int64)
    image = (token >= SENTINEL) & (token < SENTINEL + 5)
    if HAS_PADDING:
        image = image & ~tl.load(PADDING + row)
    # Text rows retain the mega-gate output without loading logits or bias.
    # Only image rows replace the correction bias. Match the upstream router's
    # stable expert-ID tie break and bias-free normalized output weights.
    if image:
        col = tl.arange(0, BLOCK)
        logits = tl.load(LOGITS + row * E + col, col < E, other=0).to(tl.float32)
        bias = tl.load(IMAGE_BIAS + col, col < E, other=0).to(tl.float32)
        raw = tl.sqrt(tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits))))
        score = tl.where(col < E, raw + bias, -float("inf"))
        score = tl.where(score == score, score, -1e30)
        slots = tl.arange(0, 8)
        values = tl.zeros([8], tl.float32)
        experts = tl.zeros([8], tl.int32)
        for slot in tl.static_range(6):
            best = tl.max(score, 0)
            expert = tl.min(tl.where(score == best, col, E), 0).to(tl.int32)
            value = tl.sum(tl.where(col == expert, raw, 0.0), 0)
            values = tl.where(slots == slot, value, values)
            experts = tl.where(slots == slot, expert, experts)
            score = tl.where(col == expert, -float("inf"), score)
        total = tl.sum(values, 0)
        values *= SCALE / tl.where(total > 0, total, 1.0)
        tl.store(WEIGHTS + row * 6 + slots, values, slots < 6)
        tl.store(IDS + row * 6 + slots, experts, slots < 6)


def correct_image_routes(logits, bias, padding, weights, ids, scale,
                         input_ids, image_bias, sentinel):
    m, experts = logits.shape
    _correct_image_routes[(m,)](
        logits, bias, padding, weights, ids, scale, input_ids, image_bias,
        sentinel, experts, triton.next_power_of_2(experts), padding is not None,
        num_warps=1,
    )
