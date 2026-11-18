#!/usr/bin/env python3
"""Idle-device validation/timing of the serving mega-gate plus image routing."""
import argparse
import json
from types import SimpleNamespace

import torch

torch.cuda.set_device(0)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
from vllm_hcu.model_executor.layers.fused_moe.bf16_mega_gate import try_run_bf16_mega_gate
from op_validation import timed


def validate():
    torch.manual_seed(42)
    results = []
    for m in (1, 8, 48, 64):
        x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
        gate = torch.nn.Module()
        gate.weight = torch.nn.Parameter(torch.randn(384, 5120, device="cuda").bfloat16() * 0.02)
        gate.e_score_correction_bias = torch.randn(384, device="cuda") * 0.1
        router = SimpleNamespace(top_k=6, renormalize=True, scoring_func="sqrtsoftplus",
                                 routed_scaling_factor=1.5, image_sentinel_lo=128000,
                                 bias_vl=torch.randn(384, device="cuda"))
        input_ids = torch.zeros(m, device="cuda", dtype=torch.int64)
        for label in ("text", "mixed"):
            if label == "mixed":
                input_ids[::2] = 128002

            def fused():
                return try_run_bf16_mega_gate(x, gate, router, indices_dtype=torch.int32,
                                             input_ids=input_ids)

            def split():
                logits = torch.mm(x, gate.weight.T, out_dtype=torch.float32)
                weights, ids = fused_topk_bias(
                    x, logits, "sqrtsoftplus", gate.e_score_correction_bias, 6, True,
                    indices_type=torch.int32, input_tokens=input_ids,
                    routed_scaling_factor=1.5, bias_vl=router.bias_vl,
                    image_sentinel_lo=router.image_sentinel_lo)
                return weights, ids, logits

            actual, expected = fused(), split()
            assert actual is not None
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
            row = {"M": m, "case": label,
                   "logits_max_abs": (actual[2] - expected[2]).abs().max().item(),
                   "weights_max_abs": (actual[0] - expected[0]).abs().max().item(),
                   "routing_rows_changed": (actual[1] != expected[1]).any(1).sum().item(),
                   "split": timed(split), "candidate": timed(fused)}
            results.append(row)
            print(json.dumps(row), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate()
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
