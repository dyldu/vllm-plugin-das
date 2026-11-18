#!/usr/bin/env python3
"""Count GPU stages only; retain phase, stream and launch geometry evidence."""
import argparse
import collections
import gzip
import json
import statistics


def stage(name):
    families = (
        ("mla_output_mask_copy", ("_mask_copy_native_mla",)),
        ("mla_selectors", ("_compact_selectors",)),
        ("q_rope_split", ("_q_rope_native_512",)),
        ("native_kv_insert", (
            "fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<c10::BFloat16, 0,",
            "fusedDeepseekV4KVRopeInsertKernel")),
        ("native_qkv_with_norm", (
            "fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<c10::BFloat16, 64, true,",)),
        ("native_qkv_fused", ("fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel",)),
        ("mhc_post_5120", ("_mhc_post_5120",)),
        ("ipc_clean", ("clean_low_latency",)),
        ("masked_token_quant", ("_masked_token_quant",)),
        ("grouped_masked", ("grouped_w8a8_fp8_nt_masked",)),
        ("grouped_contiguous", ("grouped_w8a8_fp8_nt_contiguous", "m_grouped_fp8_gemm_nt_contiguous")),
        ("dispatch_ll", ("internode_ll::dispatch",)),
        ("combine_ll", ("internode_ll::combine",)),
        ("gate_mega", ("mega_gate_kernel",)),
        ("gate_vision", ("_correct_image_routes",)),
        ("gate_topk", ("topkGatingSoftplusSqrt", "topk_softplus_sqrt", "moe_fused_gate")),
        ("expert_act_masked", ("act_and_mul_fp8_quant_ep_masked",)),
        ("expert_act_limit", ("act_and_mul_fp8_quant_kernel_opt1_ep_with_limit",)),
        ("expert_act_contiguous", ("act_and_mul_fp8_quant_kernel_opt1_new_with_limit",)),
        ("shared_act_quant", ("_silu_clamp_bf16_quant", "act_and_mul_fp8_quant_kernel_opt1_one_with_limit")),
        ("token_quant", ("dynamic_per_token_scaled_fp8_quant",)),
        ("act_other", ("act_and_mul", "silu_and_mul")),
        ("collective", ("nccl", "rccl")),
        ("norm_quant", ("rms_norm_fp8_quant",)),
    )
    for family, symbols in families:
        if any(symbol in name for symbol in symbols):
            return family
    if "Cijk_" in name and "_SB_" in name:
        return "fp32_sb"
    return None


def stats(values):
    return {"count": len(values), "sum_us": sum(values),
            "median_us": statistics.median(values),
            "mean_us": statistics.mean(values), "max_us": max(values)}


def report(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    if not kernels:
        raise ValueError("No GPU kernel events: " + str(path))
    spans = [e for e in events if e.get("cat") == "gpu_user_annotation"
             and e.get("name", "").startswith("step[") and e.get("dur", 0) > 0]
    attribution = "SGLang GPU step annotations"
    if not spans:
        # GPUWorker.execute_model annotates the target forward. sample_tokens
        # (including DSpark propose) follows it outside this range. Use the
        # main annotation stream, selected by total target duration; auxiliary
        # stream ranges can extend across the following target/draft boundary.
        target = [e for e in events if e.get("cat") == "gpu_user_annotation"
                  and e.get("name", "").startswith("execute_context_")
                  and e.get("dur", 0) > 0]
        if target:
            durations = collections.Counter()
            for e in target:
                durations[e["tid"]] += e["dur"]
            main = durations.most_common(1)[0][0]
            spans = [dict(e, name="target:" + e["name"]) for e in target
                     if e["tid"] == main]
            spans += [dict(e, name="draft_and_sampling") for e in events
                      if e.get("cat") == "gpu_user_annotation"
                      and e.get("name", "").startswith("ProfilerStep#")
                      and e.get("tid") == main and e.get("dur", 0) > 0]
            attribution = "vLLM main-stream execute_model GPU annotation; trailing ProfilerStep covers draft and sampling"
    buckets = collections.defaultdict(list)
    all_kernels = collections.defaultdict(list)
    geometry = collections.Counter()
    for event in kernels:
        candidates = [s for s in spans if s["ts"] <= event["ts"] < s["ts"] + s["dur"]]
        phase = min(candidates, key=lambda s: s["dur"])["name"] if candidates else "unattributed"
        all_kernels[(phase, event["name"])].append(event["dur"])
        family = stage(event["name"])
        if family is None:
            continue
        buckets[(phase, family)].append(event["dur"])
        args = event.get("args", {})
        geometry[(phase, family, str(args.get("grid")), str(args.get("block")))] += 1
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    busy = 0
    start, end = intervals[0]
    for left, right in intervals[1:]:
        if left > end:
            busy += end - start
            start, end = left, right
        else:
            end = max(end, right)
    busy += end - start
    return {
        "path": str(path), "deviceProperties": trace.get("deviceProperties"),
        "distributedInfo": trace.get("distributedInfo"), "gpu_kernel_count": len(kernels),
        "phase_attribution": attribution,
        "kernel_sum_us": sum(e["dur"] for e in kernels), "gpu_busy_union_us": busy,
        "gpu_wall_us": max(right for _, right in intervals) - intervals[0][0],
        "phase_spans": [{k: e[k] for k in ("name", "ts", "dur", "tid")} for e in spans],
        "stage_stats": [{"phase": p, "stage": s, **stats(v)} for (p, s), v in sorted(buckets.items())],
        # Keep unclassified compute visible when a newly optimized stage
        # exposes the next bottleneck. These are kernel sums, not wall time.
        "top_kernels": [
            {"phase": phase, "name": name, **stats(values)}
            for phase in sorted({p for p, _ in all_kernels})
            for (_, name), values in sorted(
                ((key, values) for key, values in all_kernels.items() if key[0] == phase),
                key=lambda item: sum(item[1]), reverse=True)[:25]
        ],
        "launch_geometry": [{"phase": p, "stage": s, "grid": g, "block": b, "count": n}
                            for (p, s, g, b), n in sorted(geometry.items())],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = [report(path) for path in args.traces]
    with open(args.output, "w") as handle:
        json.dump(data, handle, indent=2)
    for item in data:
        print(item["path"])
        print("phase | stage | count | mean us | median us")
        for row in item["stage_stats"]:
            print("{phase} | {stage} | {count} | {mean_us:.2f} | {median_us:.2f}".format(**row))
