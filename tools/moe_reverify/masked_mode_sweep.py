#!/usr/bin/env python3
"""Validate and time the installed pair32 modes at the actual MoE dimensions.

Run only on idle GPUs. This does not change the serving kernel selection.
"""
import argparse
import json

import torch

torch.cuda.set_device(0)
from deepgemm import m_grouped_fp8_gemm_nt_masked, marlin_fp8_masked_weight
from lightop.activation import fuse_silu_mul_fp8_quant_ep

from op_validation import diff_quant, timed


def sweep(capacity):
    torch.manual_seed(42)
    e, k, two_n = 48, 5120, 4608
    a = (torch.randn(e, capacity, k, device="cuda") * 16).to(torch.float8_e4m3fn)
    sa = torch.full((e, capacity), 0.02, device="cuda")
    w1 = marlin_fp8_masked_weight(
        (torch.randn(e, two_n, k, device="cuda") * 16).to(torch.float8_e4m3fn))
    w2 = marlin_fp8_masked_weight(
        (torch.randn(e, k, two_n // 2, device="cuda") * 16).to(torch.float8_e4m3fn))
    s1 = torch.full((e, two_n), 0.002, device="cuda")
    s2 = torch.full((e, k), 0.002, device="cuda")
    counts = torch.empty(e, device="cuda", dtype=torch.int32)
    cases = {
        "stress": torch.tensor([0, 1, 7, 16, 32, 63, 64, 8] * 6, device="cuda"),
        "decode_mean6": torch.poisson(torch.full((e,), 6.0, device="cuda")),
        "all_zero": torch.zeros(e, device="cuda"),
    }
    gate = torch.empty(e, capacity, two_n, device="cuda", dtype=torch.bfloat16)
    down = torch.empty(e, capacity, k, device="cuda", dtype=torch.bfloat16)

    def chain(mode):
        config = None if mode == "default" else {"MODE": mode}
        m_grouped_fp8_gemm_nt_masked(
            (a, sa), (w1, s1), gate, counts, 16, config=config, enable_overlap=False)
        q, sq = fuse_silu_mul_fp8_quant_ep(
            gate, fp8type=0, tokens_per_expert=counts, expect_m=16, limit=10.0)
        m_grouped_fp8_gemm_nt_masked(
            (q, sq.reshape(e, capacity)), (w2, s2), down, counts, 16,
            config=config, enable_overlap=False)
        return q, sq

    results = []
    for label, values in cases.items():
        counts.copy_(values.int())
        active = torch.arange(capacity, device="cuda")[None, :] < counts[:, None]
        ref_q, ref_sq = chain("default")
        ref_q, ref_sq = ref_q.clone(), ref_sq.clone()
        ref_gate, ref_down = gate.clone(), down.clone()
        for mode in ("default", 1000, 1001, 1002):
            q, sq = chain(mode)
            if bool(active.any()):
                error = {
                    "activation": diff_quant((q, sq), (ref_q, ref_sq), active),
                    "gate_max_abs": (gate[active].float() - ref_gate[active].float()).abs().max().item(),
                    "output_max_abs": (down[active].float() - ref_down[active].float()).abs().max().item(),
                    "output_relative_l2": ((down[active].float() - ref_down[active].float()).norm()
                                           / ref_down[active].float().norm().clamp_min(1e-12)).item(),
                    "finite": bool(torch.isfinite(down[active]).all()),
                }
            else:
                error = {"active_rows": 0}
            row = {"case": label, "mode": mode, "counts": counts.tolist(),
                   "error": error, "timing": timed(lambda: chain(mode))}
            results.append(row)
            print(json.dumps(row), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--capacity", type=int, default=512)
    args = parser.parse_args()
    result = {"device": torch.cuda.get_device_properties(0).__repr__(),
              "capacity": args.capacity, "results": sweep(args.capacity)}
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
