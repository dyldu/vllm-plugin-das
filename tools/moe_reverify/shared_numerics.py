#!/usr/bin/env python3
"""Compare both candidate fusions with the real serving activation/quant ABI."""
import argparse
import json

import torch
torch.cuda.set_device(0)
import torch.nn.functional as F
from vllm import _custom_ops as ops
from lightop.activation import fuse_silu_mul_fp8_quant
from lightop.quant import per_token_quant_fp8
from hcu_vllm_das.moe.shared_act_quant import silu_clamp_bf16_quant
from op_validation import diff_quant, timed

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--timing", action="store_true")
args = parser.parse_args()
torch.cuda.set_device(0)
torch.manual_seed(42)
rows = []
for m in (1, 8, 48, 64, 128, 256):
    for magnitude in (0.0, 0.001, 0.25, 1.0, 8.0):
        x = (torch.randn(m, 4608, device="cuda") * magnitude).bfloat16()
        # Include clamp boundary values and both signs in every nonzero test.
        if magnitude:
            x[:, :6] = torch.tensor([-20, -10, -0.01, 0, 10, 20], device="cuda")
            x[:, 2304:2310] = x[:, :6]

        def activation():
            y = torch.empty((m, 2304), dtype=torch.bfloat16, device="cuda")
            torch.ops._C.silu_and_mul_with_clamp(y, x, 10.0, 1.0, 0.0)
            return y

        def stock():
            return ops.scaled_fp8_quant(activation(), use_per_token_if_dynamic=True)

        def fixed():
            return silu_clamp_bf16_quant(x, 10.0)

        reference = stock()
        y = activation()
        gate, up = x.float().chunk(2, -1)
        formula = (F.silu(gate.clamp(max=10)).bfloat16().float()
                   * up.clamp(-10, 10)).bfloat16()
        row = dict(M=m, magnitude=magnitude,
                   stock_activation_formula_max_abs=(y.float()-formula.float()).abs().max().item(),
                   lightop_quant_vs_stock=diff_quant(per_token_quant_fp8(y, dtype=torch.float8_e4m3fn), reference),
                   lightop_fused_vs_stock=diff_quant(fuse_silu_mul_fp8_quant(x, fp8type=0, limit=10), reference),
                   fixed_vs_stock=diff_quant(fixed(), reference))
        if args.timing and magnitude == 1:
            row["stock_timing"] = timed(stock)
            row["fixed_timing"] = timed(fixed)
        rows.append(row)
        print(json.dumps(row), flush=True)
with open(args.output, "w") as handle:
    json.dump(rows, handle, indent=2)
