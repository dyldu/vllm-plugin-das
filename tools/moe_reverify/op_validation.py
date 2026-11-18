#!/usr/bin/env python3
"""GPU numerical and timing evidence for the actual PR 6 provider calls."""
import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def timed(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    wall = []
    for _ in range(30):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - start) * 1e6)
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            fn()
    capture_stream.synchronize()
    with torch.cuda.graph(graph, stream=capture_stream):
        result = fn()
    graph.replay()
    torch.cuda.synchronize()
    elapsed = []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        begin.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed.append(begin.elapsed_time(end) * 10)  # ms / 100 * 1000
    del result
    # A single-node replay loop also measures CPU submission bubbles. Unroll
    # calls inside one graph to resolve differences between small GPU kernels.
    from triton.testing import do_bench_cudagraph
    unrolled = 1000 * do_bench_cudagraph(fn, rep=20, return_mode="median")
    return {"wall_median_us": statistics.median(wall),
            "graph_gpu_median_us": statistics.median(elapsed),
            "unrolled_graph_gpu_median_us": unrolled}


def diff_quant(got, reference, mask=None):
    q, s = got
    rq, rs = reference
    x = q.float() * s.reshape(*q.shape[:-1], 1)
    y = rq.float() * rs.reshape(*rq.shape[:-1], 1)
    code_delta = q.view(torch.uint8).int() - rq.view(torch.uint8).int()
    if mask is not None:
        x, y, code_delta = x[mask], y[mask], code_delta[mask]
        s, rs = s.reshape(mask.shape)[mask], rs.reshape(mask.shape)[mask]
    error = (x - y).abs()
    return {"max_abs": error.max().item(),
            "relative_l2": (torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y).clamp_min(1e-30)).item(),
            "different_codes": (code_delta != 0).sum().item(),
            "max_code_delta": code_delta.abs().max().item(),
            "max_scale_abs": (s - rs).abs().max().item(),
            "finite": bool(torch.isfinite(x).all())}


def shared():
    from hcu_vllm_das.moe.shared_act_quant import silu_clamp_bf16_quant
    from vllm import _custom_ops as ops

    rows = []
    for m in (1, 8, 48, 64, 128, 256):
        for magnitude in (0.25, 1.0, 8.0):
            x = (torch.randn(m, 4608, device="cuda") * magnitude).bfloat16()

            def split():
                y = torch.empty((m,2304),device=x.device,dtype=x.dtype)
                torch.ops._C.silu_and_mul_with_clamp(y,x,10.,1.,0.)
                return ops.scaled_fp8_quant(y,use_per_token_if_dynamic=True)

            def fused():
                return silu_clamp_bf16_quant(x,10.0)

            reference = split()
            candidate = fused()
            row = {"M": m, "K": 4608, "magnitude": magnitude,
                   "accuracy": diff_quant(candidate, reference)}
            if magnitude == 1.0:
                row["split"] = timed(split)
                row["fused"] = timed(fused)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = fused()
                graph.replay()
                torch.cuda.synchronize()
                row["graph_replay"] = diff_quant(graph_output, candidate)
            rows.append(row)
            print("shared", json.dumps(row), flush=True)
    return rows


def gate():
    from lightop.moe import bf16_mega_gate_hcu
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias

    rows = []
    for m in (1, 8, 48, 64, 128):
        x = torch.randn(m, 5120, dtype=torch.bfloat16, device="cuda")
        w = (torch.randn(384, 5120, device="cuda") * 0.02).bfloat16()
        bias = torch.randn(384, device="cuda")
        mask = torch.ones(m, dtype=torch.bool, device="cuda")

        def fused():
            ids, weights, logits = bf16_mega_gate_hcu(
                x, w, num_topk=6, use_shared_as_routed=False, num_shared_experts=0,
                ep_rank=0, bias=bias, mask=mask, scoring_func="sqrtsoftplus",
                routed_scaling_factor=1.5, renormalize=True, apply_scale=True, return_logits=True)
            return weights, ids, logits

        def split():
            logits = torch.mm(x, w.T, out_dtype=torch.float32)
            weights, ids = fused_topk_bias(x, logits, "sqrtsoftplus", bias, 6,
                                          True, indices_type=torch.int32,
                                          routed_scaling_factor=1.5)
            return weights, ids, logits

        actual_w, actual_ids, actual_logits = fused()
        ref_w, ref_ids, ref_logits = split()
        # Independent accumulation oracle: BF16 inputs, FP32 output, no BF16
        # cast of logits before routing (GateLinear.out_dtype is FP32).
        oracle_logits = F.linear(x.float(), w.float())
        raw = F.softplus(oracle_logits).sqrt()
        selected = torch.argsort(raw + bias, dim=-1, descending=True, stable=True)[:, :6]
        expected_w = raw.gather(1, selected)
        expected_w = expected_w / expected_w.sum(1, keepdim=True) * 1.5
        dense_actual = torch.zeros(m, 384, device="cuda").scatter_(1, actual_ids.long(), actual_w)
        dense_ref = torch.zeros_like(dense_actual).scatter_(1, ref_ids.long(), ref_w)
        dense_oracle = torch.zeros_like(dense_actual).scatter_(1, selected, expected_w)
        row = {"M": m, "hidden": 5120, "experts": 384,
               "logits_max_abs_vs_oracle": (actual_logits - oracle_logits).abs().max().item(),
               "logits_max_abs_vs_baseline": (actual_logits - ref_logits).abs().max().item(),
               "routing_rows_changed_vs_baseline": (actual_ids.sort(1).values != ref_ids.sort(1).values).any(1).sum().item(),
               "weights_max_abs_vs_baseline": (dense_actual - dense_ref).abs().max().item(),
               "weights_max_abs_vs_oracle": (dense_actual - dense_oracle).abs().max().item(),
               "split": timed(split), "fused": timed(fused)}
        print("gate", json.dumps(row), flush=True)
        rows.append(row)
    return rows


def masked(capacity=512):
    from deepgemm import m_grouped_fp8_gemm_nt_masked, marlin_fp8_masked_weight
    from lightop.activation import fuse_silu_mul_fp8_quant_ep

    e, k, two_n = 48, 5120, 4608
    a = (torch.randn(e, capacity, k, device="cuda") * 16).to(torch.float8_e4m3fn)
    a_scale = torch.full((e, capacity), 0.02, device="cuda")
    w1 = marlin_fp8_masked_weight((torch.randn(e, two_n, k, device="cuda") * 16).to(torch.float8_e4m3fn))
    w2 = marlin_fp8_masked_weight((torch.randn(e, k, two_n // 2, device="cuda") * 16).to(torch.float8_e4m3fn))
    s1 = torch.full((e, two_n), 0.002, device="cuda")
    s2 = torch.full((e, k), 0.002, device="cuda")
    counts = torch.tensor(([0, 1, 7, 16, 32, 63, 64, 8] * 6), device="cuda", dtype=torch.int32)
    active = torch.arange(capacity, device="cuda")[None, :] < counts[:, None]
    gates = [torch.full((e, capacity, two_n), float("nan"), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    downs = [torch.full((e, capacity, k), float("nan"), device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    def chain(index, hint, pass_hint):
        m_grouped_fp8_gemm_nt_masked((a, a_scale), (w1, s1), gates[index], counts, hint, enable_overlap=False)
        kwargs = {"expect_m": hint} if pass_hint else {}
        q, scale = fuse_silu_mul_fp8_quant_ep(gates[index], fp8type=0, tokens_per_expert=counts, limit=10.0, **kwargs)
        m_grouped_fp8_gemm_nt_masked((q, scale.reshape(e, capacity)), (w2, s2), downs[index], counts, hint, enable_overlap=False)
        return q, scale

    old = chain(0, 7, False)
    new = chain(1, 16, True)
    torch.cuda.synchronize()
    result = {"E": e, "capacity": capacity, "K": k, "two_N": two_n,
              "counts": counts.tolist(), "activation": diff_quant(new, old, active),
              "gemm1_max_abs": (gates[0][active].float() - gates[1][active].float()).abs().max().item(),
              "gemm2_max_abs": (downs[0][active].float() - downs[1][active].float()).abs().max().item(),
              "finite": bool(torch.isfinite(downs[1][active]).all()),
              "legacy": timed(lambda: chain(0, 7, False)),
              "candidate": timed(lambda: chain(1, 16, True))}
    # The stress case above covers counts beyond the hint. Time decode's
    # approximate mean of six assignments/expert separately from that case.
    counts.copy_(torch.poisson(torch.full((e,), 6.0, device="cuda")).int())
    result["decode_counts"] = counts.tolist()
    result["decode_legacy"] = timed(lambda: chain(0, 7, False))
    result["decode_candidate"] = timed(lambda: chain(1, 16, True))
    print("masked", json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("shared", "masked", "gate", "all"), default="all")
    parser.add_argument("--masked-capacity", type=int, default=512)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    results = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    try:
        if args.stage in ("shared", "all"):
            results["shared"] = shared()
        if args.stage in ("masked", "all"):
            results["masked"] = masked(args.masked_capacity)
        if args.stage in ("gate", "all"):
            results["gate"] = gate()
    finally:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
