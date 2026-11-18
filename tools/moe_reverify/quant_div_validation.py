#!/usr/bin/env python3
"""Exhaust the production division's bounded BF16-to-E4M3 input domain."""
import argparse
import json

import torch

torch.cuda.set_device(0)
from vllm.triton_utils import tl, triton
from hcu_vllm_das.moe.shared_act_quant import _bounded_bf16_quant_div


@triton.jit
def check_div(counts, BLOCK: tl.constexpr):
    bits = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    max_bits = tl.program_id(1)
    value = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    amax = max_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    scale = tl.maximum(tl.div_rn(amax, 448.0), 1.0 / (448.0 * 512.0))
    reference = tl.div_rn(value, scale)
    actual = _bounded_bf16_quant_div(value, scale)
    reference = tl.minimum(tl.maximum(reference, -448.0), 448.0)
    actual = tl.minimum(tl.maximum(actual, -448.0), 448.0)
    ref_code = reference.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    code = actual.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    valid = tl.abs(value) <= amax
    mismatches = tl.sum((valid & (ref_code != code)).to(tl.int32), 0)
    tl.atomic_add(counts + max_bits, mismatches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # At limit=10 both clamped factors bound the BF16 product to [-100,100].
    # Include every nonnegative BF16 amax through 100 and both signs of every
    # BF16 value whose magnitude is no greater than that row's amax.
    max_bits = 0x42C8
    counts = torch.zeros(max_bits + 1, device="cuda", dtype=torch.int32)
    check_div[(16, max_bits + 1)](
        counts, 4096, num_warps=4, enable_fp_fusion=False)
    torch.cuda.synchronize()
    bad = counts.nonzero().flatten().cpu().tolist()
    result = dict(
        device=torch.cuda.get_device_name(0),
        amax_bitpatterns=max_bits + 1,
        value_bitpatterns=65536,
        tested_pairs_before_valid_mask=(max_bits + 1) * 65536,
        theoretical_valid_bf16_pairs=(max_bits + 1) * (max_bits + 2),
        different_codes=counts.sum().item(),
        bad_amax_count=len(bad),
        first_bad_amax_bits=bad[:20],
        exact=not bad,
    )
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)
    return int(bool(bad))


if __name__ == "__main__":
    raise SystemExit(main())
