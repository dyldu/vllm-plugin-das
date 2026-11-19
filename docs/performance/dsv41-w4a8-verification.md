# W4A8 Decode verification — frozen before baseline measurement

Model: `/models/DeepSeek-V4.1-Flash-Channel-INT4-w4a8`, config SHA256
`ae297e9513323a65d120a3cb05a0f8d3d7b41ddaf358c048016d9e7924619244`.
Baseline plugin: release branch `release_dsv41-flash_20261114_v0330`,
`a5b5806d4c49aeba9a0fbcac3f9bdd6b49ce5384` (clean independent worktree).
Original Adapter: `35c397aed8c3ec31542bd11a0ea971e304b90fcf` (preserved worktree).
At the user's request, the current Adapter baseline is HYGON-AI/main
`ef63bbbd73a42812db9de1683e6720239c42fd68` in a separate clean worktree.
Keep these two source baselines and their measurements distinct.
The three pre-existing local edits on the former main branch were stashed,
backed up and excluded from this release baseline. See BRANCH_SWITCH.json.
Historical FP8 measurements and previous W4A8 runs on other source snapshots
are reference records and do not establish this baseline.

## Fixed launch and execution order

All screenshot arguments are frozen in CONSTRAINTS.json: TP1/DP8/EP,
`deep_gemm`, `deepep_low_latency`, utilization0.85, block64, max-seqs32,
max-model-len32768, max-batched-tokens256, capture sizes1/4/8/16/32/40/48/64.
Do not shrink capacity, context, concurrency or change checkpoint to avoid OOM.
The performance reference script's utilization0.75, dated checkpoint path and
additional capture bucket2 are replaced ONLY by the user's explicit new values.
The exact amended scripts, complete argv, source hashes and runtime environment
are saved with each run. Rebaseline if any frozen input changes.

Baseline uses PIECEWISE, target LL and existing draft AG/RS. First establish
baseline accuracy and unprofiled performance plus separate profiles; then
MoE operator changes, operator accuracy/performance, HE32 and matched formal
performance; then graph/gap work; finally DSpark transport/DeepGEMM/OOM work.
FULL_AND_PIECEWISE is a later, explicitly recorded experimental variable.
Keep all failed launches/tests and fallback paths. Known no-kernel startup
failures may be retried twice under the identical configuration.

nmz20/nmz22: the original Nov19 12:30 CST window was superseded by the
user's authorization to continue until explicitly stopped. RESOURCE_AUTHORIZATION.json
records this extension; frozen model, launch arguments and SLA are unchanged.
nmz18: no GPU work before Nov19 13:00; authorized until Nov20 08:00 CST.
Record hardware, container image and native providers on each node. Compare
candidate and baseline on the SAME node; another node's result is not a matched
repeat even when the GPU product name matches.

## Workload and TPOT acceptance rules

Formal workload derives from prof-w4.sh:64 requests, concurrency64, input4096,
output128, seed0, temperature0, ignore EOS,8 warmup requests. Two formal passes
run with profiling disabled; a third pass captures all eight worker ranks.
Preserve original synthetic DSpark acceptance vector
[0.6,0.36,0.216,0.1296,0.07776], n=5, and zero-filled DecodeBenchConnector KV.
Record actual prompt/output token counts; nominal lengths alone are insufficient.
This isolates Decode performance; it is not a real-KV or production latency claim.
Accuracy uses real prefill/KV and real rejection sampling, separately.

Before a candidate runs, freeze absolute TPOT ceilings from two successful
baseline repeats in SLA_FROZEN.json. Each repeat must have64 completed requests,
zero errors, expected output lengths and workflow/client exit0. Let B1/B2 be
baseline median TPOT values and P1/P2 baseline p99 TPOT values:

* Every matched candidate median <=1.02*Bi; mean of its two medians <=
  1.02*mean(B1,B2). No prior accepted intermediate-stage ceiling is dropped.
* Every matched candidate p99 <=1.05*Pi.
* A claimed end-to-end improvement requires >=2% lower mean median TPOT,
  supported by both repeats and the stage trace; smaller changes remain
  inconclusive. Retest baseline/candidate in alternating order when noise or
  repeat spread could account for the apparent gain. No significance claim
  from two repeats.
* A locally faster operator is not an end-to-end performance acceptance.

The relative rules above are fixed now. Numeric ceilings must be computed
from the NEW W4A8 baseline, not inherited from the old FP8 protocol. Do not tune
thresholds after a candidate failure.

## Correctness gates

Run the release baseline on all164 HumanEval tasks. The latest user-approved
minimum full-set score is95% (at least156/164); the user accepts95-97%.
Scores above97% also pass. This supersedes the earlier96% (158/164) request.
Retain item-level changes and the strict first32 candidate check below. Audit unique task IDs and
prompt hashes, item-level execution results, API errors, missing outputs,
stop reasons/truncation. Retain all failures; do not demand164/164 to declare a
baseline evaluation complete. Earlier155/164 belongs to a different source snapshot.
For each candidate: first32 identical task/prompt entries, batch16, seed42,
temperature0, top_p0.95, max_tokens16384, enable_thinking=false, stream=true.
Require no lost baseline passes and >=baseline first32 score, zero API errors,
missing outputs or truncation. If the new baseline is32/32, candidate requires32/32.
Final admitted composition also reruns full164, must reach the user-approved
minimum156/164 with no missing/error/truncated output, and reports every
changed task outcome against the recorded baseline.

Before full-model evaluation, test each changed operator against the actual
pre-change implementation AND an independent oracle. Cover native dtype,
shape/stride/layout, zero rows, empty/skewed experts, ties/rounding boundaries,
clamp/scale floors and changed-metadata graph replay as applicable.
Pure fusion/scheduling/reuse changes require exact active output bytes, routing
IDs and quantized INT8 values/scales, including native-consumer outputs; inputs
and padding must remain unchanged unless the original contract explicitly owns
that storage. Do not weaken to aggregate closeness to hide a one-code quantization
or routing error. Any arithmetic/GEMM algorithm candidate requiring a different
numerical contract must document and freeze an independent, scale-aware bound
BEFORE measuring it; it cannot enter the exact-equivalence path by assertion.
Benchmark matched old/new operator calls outside the profiler, with warmup,
quantiles and representative distributed expert counts; preserve all raw cases.

## SGLang attribution and comparison limits

Reference trace:
`/public/home/duyunlong/laibao/hyq/1793613146.8607268-TP-0-DP-0-EP-0.trace.json.gz`.
Map phases by semantics and producer/consumer work, not identical kernel spelling.
Keep target VERIFY and draft separate. Report per-rank load/skew, communication
wait, kernel sum, busy union and phase wall time separately. A matched operator
should reach <=1.10x reference duration at established comparable work; a missing
or unmatched stage never passes by omission. The whole target/draft step remains
part of the objective after MoE optimization.

The old reference hardware was reported as BW1100/144GiB; nmz18 was
BW1101/95.98GiB. Reinspect new-node hardware. Reference checkpoint identity,
actual tensor shapes, context lengths, routing and startup parameters remain
unproven by the previous reports; incoming user SGLang launch scripts are evidence
only after tying them to the trace. Their dated/modified checkpoint path,
64 redundant experts and static placement must not be silently treated as the
same vLLM workload. Do not restate the old15–16% observed trace duration difference
as a controlled performance gap. Preserve raw traces and distinguish observed
reference timing from a matched comparison.

## Review and publication

Create a NEW W4A8 MR from release_dsv41-flash_20261114_v0330. Do not append
W4A8 changes to the old FP8 PR6. Candidate worktree:
`/public2/home/duyunlong/vllm-hcu033-wt-w4a8-moe-align-20261119`, branch
`feat/dsv41-w4a8-moe-align-20261119`. Keep the clean release baseline separate.
Use separate worktrees for each additional candidate and Adapter change. Review
PR6 and history as candidates; old FP8 kernels are not W4A8 acceptance evidence. Publish
only W4A8 changes that pass the relevant operator, accuracy, formal-performance
and trace-dispatch gates. Provide complete reviewable patches, source SHAs,
exact reproduction commands, raw results and remaining limitations. Adapter
changes may be staged on a dyldu branch. Never overwrite the user's baseline
checkout or silently apply the saved local edits to the release baseline.

## Measured W4A8 masked MoE change

Code commit `ea82e732eb2f586a4d257877739cec476329c237` replaces the
workspace-capacity scheduling hint (2048) with a token-derived, rounded hint
for the target masked W4A8 GEMMs and clamp/INT8 quantizer. Native device expert
counts still control every valid row; allocation sizes remain unchanged.
The activation condition is restricted to the measured gfx938/64-CU device,
capacity256/dispatchers8, E48/H5120/FFN2304, global384 experts/topk6 and clamp10.
Other configurations retain the existing hint. Draft AG/RS is unchanged.

Matched nmz22 results use Adapter main ef63bbb, the same native providers,
all frozen launch arguments and PIECEWISE. Both workflows exited0; each formal
repeat completed64/64 requests, zero failures, actual4096 input/128 output.

| TPOT | Release baseline | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Repeat1 median | 70.4067ms | 66.2890ms | 5.85% |
| Repeat2 median | 70.0225ms | 65.5945ms | 6.32% |
| Mean of medians | 70.2146ms | 65.9417ms | 6.09% |
| Repeat1 p99 | 81.4512ms | 75.9197ms | 6.79% |
| Repeat2 p99 | 82.6422ms | 77.0615ms | 6.75% |

Both repeats pass the frozen median/p99 SLA. These are two repeats of the
DecodeBench zero-KV/synthetic-acceptance workload, with no statistical
significance or real-KV production latency claim.

Native before/after probes passed on both nmz20 and nmz22: exact active BF16
outputs, INT8 codes/scales and following native GEMM outputs; independent
signed-INT4 CPU-dot oracle; empty/skewed experts, poison padding and changed
input/count/scale graph replays. Example nmz22 graph/event operator medians
(microseconds, capacity2048 to hint16 for decode or hint32 for verify):

| Operator | Decode before/after | Verify before/after |
| --- | ---: | ---: |
| Gate/up W4A8 GEMM | 745.055 / 338.312 | 755.319 / 359.808 |
| Down W4A8 GEMM | 374.616 / 169.672 | 383.040 / 191.824 |
| Clamp and INT8 quant | 375.112 / 17.760 | 411.600 / 58.568 |

These standalone workloads are exact operator comparisons, not per-layer
end-to-end stage durations. The checked-in native benchmark is
`tests/vllm033/benchmark_ct_w4a8_hints.py`; run in the documented container with
`HCU_ADAPTER_HOST=none VLLM_PLUGINS= HIP_VISIBLE_DEVICES=0 python3
 tests/vllm033/benchmark_ct_w4a8_hints.py --output report.json` on an idle device.

Baseline full164:158/164 (96.34146%), first32:32/32, zero request errors,
missing outputs or truncation. Candidate HE32:32/32, identical prompts,
zero errors/missing/truncation and no lost baseline passes. Candidate full164
completed with156/164 (95.12195%), first32:32/32, zero request errors,
missing outputs or truncation, workflow exit0. This meets the user's95%
minimum, but loses the baseline passes on HumanEval/99 and130 with no new
passes. Both are generated-code differences, not missing responses or grader
timeouts; their cause is not established. The full run has identical prompts,
serving argv, Adapter commit, native Python hashes and package versions.
Keep the MR draft while these differences are investigated; the score gate
and sampled exact operator probes do not prove whole-model equivalence.
The additional same-source nmz20 full164 run is diagnostic and is not the
nmz22 baseline. Its slow-progress root cause remains unproven.

All eight candidate gzip traces passed CRC verification. DP0 shows800/800
target masked GEMMs changed from the native128-row template to16-row template;
400 masked clamp/quant calls remain present. Trace durations support stage
attribution but do not replace the unprofiled TPOT pair or controlled native
probe. Physical launch grids are absent in the exported trace args.

Workspace evidence root: `run/dsv4.1/033/w4a8_reverify_20261119/`.

- Baseline full164: `nmz22/upstream_main_accuracy/20261119_114057/`;
  `nmz22/upstream_accuracy_gate.json`.
- Baseline performance/profile:
  `nmz22/upstream_main_performance/20261119_115114_prof_w4_HN7tRU/`;
  `nmz22/SLA_FROZEN.json`.
- Native probes: `nmz20/moe_hint_operator_probe.json` and
  `nmz22/moe_hint_operator_probe.json`.
- Candidate HE32: `nmz22/combined_hint_he32/20261119_120611/`;
  `nmz22/combined_hint_he32_summary.json`.
- Candidate full164: `nmz22/combined_hint_full164/20261119_124734/`;
  `nmz22/combined_hint_full164.gate.json` and
  `CANDIDATE_FULL164_COMPARISON.json`.
- Candidate performance/profile:
  `nmz22/combined_hint_performance/20261119_121259_prof_w4_mQ9eGl/`;
  `nmz22/combined_hint_performance_comparison.json`.
- Trace audit: `nmz22/combined_hint_trace_dispatch_gate.json`;
  `nmz22/upstream_baseline_dp0_moe_trace.json` and
  `nmz22/combined_hint_dp0_moe_trace.json`.

Every raw run retains its complete script, argv, source provenance, results and
logs. Prior interrupted runs and FP8 results remain separate. The profiler's
existing CUDA-time text-table diagnostic is disabled in both accepted runs;
trace recording is unchanged. The graph extension below is a separate measured
step on top of this operator patch. Draft transport changes remain outstanding.

## Measured FULL_AND_PIECEWISE extension

The plugin now admits the narrowly validated native V4.1 TP1/DP8/EP8 DSpark
FP4 decode path into FULL_AND_PIECEWISE capture. Prefill remains excluded.
Because the native FP4 reader leaves the logits tail unwritten when
`clean_logits=False`, the decode path masks columns outside each device-side
sequence length to negative infinity before top-k selection. General FP8
host-metadata capture policy is unchanged.

This requires the optional, default-deny FP4 owner predicate in Adapter
commit `3bf4b68362387f8e33b816645a8917609bfd58fd`:
https://github.com/dyldu/adapter-das/pull/1 . The Adapter does not import the
plugin. Tested plugin commit is `f9af645a85040f26427e6f5f222632d3f92d9322`;
the published series cherry-picks the same runtime and test files onto the
MoE MR branch. Documentation is the only difference from that tested snapshot.

All screenshot parameters and the frozen SLA remain unchanged; the explicit
graph mode and paired plugin/Adapter changes are the experimental variables.
Two unprofiled formal repeats on nmz22 completed64/64 requests each, zero
failures, actual4096 input/128 output, workflow0:

| TPOT | Release PIECEWISE | MoE PIECEWISE | MoE + FULL graphs |
| --- | ---: | ---: | ---: |
| Repeat1 median | 70.4067ms | 66.2890ms | 52.7124ms |
| Repeat2 median | 70.0225ms | 65.5945ms | 52.3160ms |
| Mean of medians | 70.2146ms | 65.9417ms | 52.5142ms |
| Repeat1 p99 | 81.4512ms | 75.9197ms | 59.5065ms |
| Repeat2 p99 | 82.6422ms | 77.0615ms | 60.6634ms |

The mean of medians falls25.21% from release and20.36% from the MoE-only
candidate. Both repeats pass the frozen median/p99 ceilings and the
pre-recorded2% incremental improvement gate. The zero-KV/synthetic-acceptance
and two-repeat limitations stated above still apply.

Validation:66 CPU checks passed (two GPU cases skipped there); the separate
native GPU graph probe passed9/9 without skips. GPU cases cover compression
ratios1/2, rows2/8/48/64, contexts128/4096/32768 and top-k4/1024, changed query,
cache payload, weights, lengths and block tables across replays, exact eager
comparison and zero/short/full contexts. The ninth check rejects FULL prefill.
The composed real-KV HumanEval first32 run passed32/32 with identical prompts,
no lost baseline passes, no API errors, missing outputs or truncation and
workflow0. The exact graph composition subsequently completed real-KV full164
on nmz18: **158/164 (96.34%)**, first32 32/32, no request errors, missing
outputs or truncation, workflow0. This passes the user minimum156/164.
The nmz18 accelerator has approximately96GiB VRAM; the earlier nmz22
performance evidence above belongs to the144GiB device class. Do not combine
them into a same-node performance comparison. Task-level differences alone
do not prove whole-model numerical equivalence.

All eight graph traces passed gzip CRC and retain SHA256 values. DP0 contains
800 target masked GEMMs; each80-GEMM target span maps to exactly one native
`hipGraphLaunch`. Total graph launches fell450 to20 and manual kernel launches
29840 to520 versus the MoE PIECEWISE capture. Median observed target-phase
wall/busy/gap times fell179.406/154.618/24.423ms to
120.112/117.370/2.721ms. Draft-plus-sampling wall/gap fell72.297/14.029ms to
9.649/0.206ms. These are profiler observations, separate from formal TPOT.

Evidence under the workspace root:

- Frozen experiment: `GRAPH_EXPERIMENT.json`.
- Native GPU checks: `nmz22/fp4_graph_native_probe.log` and `.json`.
- Real-KV HE32: `nmz22/fullgraph_he32/20261119_130618/`;
  `nmz22/fullgraph_he32_summary.json`.
- Formal/profile: `nmz22/fullgraph_performance/20261119_131324_prof_w4_UtGOwm/`;
  `nmz22/fullgraph_performance_audit.json` and
  `nmz22/fullgraph_performance_comparison.json`.
- Actual FULL execution and phase gaps: `nmz22/fullgraph_trace_audit.json`.
- Graph full164: `nmz18/fullgraph_he164/20261119_142616/`;
  `nmz18/fullgraph_he164.gate.json` (158/164, workflow0).

Keep both MRs draft while the two extra MoE-only task failures remain
unexplained. The final graph composition now passes its full164 gate; tasks99
and130 pass in that run, which does not establish the cause of their earlier
MoE-only failures. A diagnostic-only real-input old/new
MoE comparison lives in a separate worktree and is excluded from this patch
and all performance measurements. The nmz20 retry failed before inference
after a driver/device fault; it is not an accuracy result or evidence that
the previous nmz20 stall has the same cause.

Latest SGLang phase reference supplied by the user:
`/public/home/duyunlong/laibao/fake_128k_c56-1795054024.3325562-TP-0-DP-0-EP-0-DECODE.trace.json.gz`.
It contains ten VERIFY and ten DRAFT GPU spans, each labeled bs=7, with
masked W8A8 FP8 GEMMs and low-latency DeepEP dispatch/combine in both phases.
It supersedes the earlier phase reference. vLLM target annotations are
generation_8(48), with INT4 W4A8 target GEMMs. Latest SGLang device metadata
matches nmz22's BW1100/64-CU/approximately144GiB device class, but checkpoint,
quantization, workload and effective context equivalence remain unestablished.
These traces therefore support
stage/scheduling diagnosis, not a controlled cross-framework percentage.
Per-span data, kernel identities and SHA256 are recorded in
`LATEST_SGLANG_PHASE_COMPARISON.json`, with a compact table in the matching
Markdown file under the workspace evidence root.


## Reuse composition under verification

The subsequent candidate preserves this complete MoE scheduling and FULL graph
series, then layers the earlier exact H5120 mHC Post and integer MLA selector
kernels onto Adapter3bf4b68 in a separate worktree. CPU77 checks pass. The Post
probe passes147 eager cases and50 changed-input graph replays with exact BF16
bytes, including rows192/256; rows40 isolated graph timing is162.9607us Torch
versus6.47193us fused. These isolated timings are not an end-to-end claim.
The selector native-consumer fixture initially passed block32 to a block64
cache; that test-only ABI error is repaired and the failed artifact retained.
The new Post/selector runtime is excluded from this MR until its operator,
HE32, full164, same-node paired performance and actual trace-dispatch gates
complete. See POST_SELECTORS_REUSE_EXPERIMENT.json and its probe amendment.
