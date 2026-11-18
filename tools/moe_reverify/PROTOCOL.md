# Channel-FP8 decode revalidation, 2026-11-18

Hardware: nmz18, all eight HCU devices. Container: duyl-033-1112,
image sha256:48e8098630ce9ebde50079e81c2974448ef8b141e120f6dd5b4ac0c5fbd59b54.
Actual device: BW1101, 95.98 GiB each. SGLang trace deviceProperties identify
BW1100, 154602045440 bytes each. Cross-trace stage comparisons therefore also
have a hardware limitation; the paired baseline/candidate uses this same nmz18.
Baseline plugin: b5d88fb (parent of PR 6's first commit).
Initial candidate: 9bf0250 (local PR 6 branch, including unpublished mega-gate/tests).
Adapter: 35c397a, separate worktree. No edits to the user's source checkouts.

The shell harnesses are copies of the supplied acc-w4.sh / prof-w4.sh with
explicit FP8 weights, worktree paths and an output root. The actual model is DeepSeek-V4.1-Flash-Channel-FP8. All valid runs use the
verified local copy /tmp/moe-reverify-model/DeepSeek-V4.1-Flash-Channel-FP8
(773542091218 bytes), rather than reading the original NFS checkpoint per run.

## Fixed performance gates

TP=1, DP=EP=8, DeepEP low latency target, DSpark n=5, PIECEWISE graph,
FP4 indexer, gpu_memory_utilization=0.90, max sequences per rank=8,
max batched tokens=64, max model length=8192, block size=64.
Disable redundant JIT warmup; graph capture still warms the serving kernels.
The original 256-token launch OOMed during target LL graph capture before any
benchmark. The current eight devices expose 95.98 GiB each (not the historical
144 GiB setup). Reduce the dispatch capacity for BOTH paired runs, retaining
the 64-request workload and graph capture sizes. This amendment preceded all
successful performance or accuracy measurements; the failed launch is retained.
The first capacity-64 attempt was refused by the existing DSpark guard:
32 sequences * 6 verify tokens > 64. Before any successful measurement, set
max sequences per rank to 8 for BOTH runs (48 verify / 40 draft query rows).
The benchmark still submits 64 concurrent requests to 8 DP ranks.
The cap64/maxseq8 run completed graph-memory estimation but exposed -1.83 GiB
KV budget at utilization0.85. Before any successful measurement, raise utilization
to0.90 for BOTH runs; this adds4.80 GiB to the planning budget, without changing
the model, requests, SLA, accuracy gates or token capacities.
An intermediate0.93 launch was rejected before weights loaded: distributed
initialization already reserves7.3-8.1 GiB/device, leaving87.92-88.67 GiB
free, less than the89.27 GiB request. Utilization0.90 requests86.38 GiB and
fits this initialization floor. No unrelated GPU job was found or terminated.
64 requests, concurrency 64, input 4096, output 128, temperature 0,
ignore EOS, seed 0 (bench default), eight warmup requests.
DecodeBenchConnector has zero-filled fake KV; it is only a performance workload.
DSpark synthetic acceptance rates=[0.6,0.6,0.6,0.6,0.6].

Measure formal TPOT without profiling. Capture traces in a separate run.
Candidate median TPOT must be <= baseline median * 1.02, and <=284.9 ms.
Repeat baseline/candidate under identical configuration before accepting a gain.
Baseline valid repeats (formal_1/formal_2): median TPOT 37.64855795 /
36.38672786 ms; median of repeat medians 37.01764291 ms. Before any candidate
measurement, make aggregation explicit: repeat-median aggregate <=37.75799576 ms,
and each matched repeat must also meet its baseline*1.02 bound
(38.40152911 /37.11446241 ms). Report both repeats and all failures.
The historical 279.31 ms run used FULL_AND_PIECEWISE, an FP8 indexer and
geometric synthetic rates, so it is historical context, not the paired baseline.

Map target/draft stages before comparing SGLang timings. The SGLang reference is
/public/home/duyunlong/laibao/hyq/1793613146.8607268-TP-0-DP-0-EP-0.trace.json.gz.
Goal: analogous MoE stages <=1.10 times the reference at comparable dimensions;
report incomparable shapes/windows explicitly. Missing stages never count as a pass.
Kernel sums, GPU busy union and wall time are separate quantities.

## Fixed correctness gates

Operator tests compare candidate against the actual pre-change path and an
independent formula. Check active expert rows, empty experts, clamp boundaries,
quantization scales/layout, ties and graph replay as applicable.
Preserve BF16 rounding before FP8 quantization. Pure scheduling/hint changes
must be bitwise equal on active rows. For arithmetic fusions require max_abs
<=0.01 for real dequantized/output values at model-representative magnitudes;
also report relative error and FP8 code/scale differences. Routing IDs must
match unless an independently measured score near-tie explains the difference.
Do not relax thresholds after a failure; retain fallback for unsupported inputs.

HumanEval: first 32 entries, temperature 0, top_p=0.95, seed 42, batch 16,
max_tokens=16384, enable_thinking=false, streaming, real prefill, real DSpark
acceptance (no synthetic acceptance and no DecodeBench connector).
Accuracy launches use max_model_len=32768 so prompt+16384 completion fits;
other shared controls remain the same. Baseline passed32/32, so the frozen
candidate requirement is now32/32 (original rule: at least29/32 AND no reduction).
Record item-level outcomes and truncation/errors. A failed/skipped evaluation
does not establish correctness. Kernel, serving accuracy and performance gates
all precede cudagraph gap work; DSpark backend/OOM work follows graph work.
