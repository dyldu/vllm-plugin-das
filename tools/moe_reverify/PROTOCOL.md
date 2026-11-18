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

## DSpark transport experiment

`DSPARK_DRAFT_BACKEND=deepep_low_latency` selects the explicit additional-config
key `hcu_dsv41_dspark_draft_backend`; the default remains AG/RS. The admitted
single-node IPC experiment retains the native target-sized Buffer and cleans
each dispatch layout inside graphs. Draft BF16 dispatch uses stock token-FP8
quantization at the receiver: vendor LL quantization failed the frozen operator
threshold against the existing AG/RS quantizer. Target quantization is unchanged.
The tested geometry is H5120, target384/draft128 experts, cap64, DP8/EP8; GDA,
internode, fault tolerance and effective layered/MNVL routes are refused.

Before DSpark formal measurements, retain the original SLA and additionally
freeze a 2% ceiling against accepted FULL + AG/RS: matched repeats
33.05649256564706 / 32.73739875246603 ms, aggregate32.89694565905654 ms.
HE32 still requires32/32 with identical prompts/tasks and zero errors/truncations.
Use two unprofiled formal repeats before independent eight-rank profiling.

## Post experiment

The artifact `mhc_post_preflight_sla.json` freezes an additional2% gate against
accepted model-entry cleanup before Post serving measurement: matched
32.512422146964965 /32.19653568146237ms, aggregate32.35447891421367ms. Original
and DSpark gates remain required. Actual Post operator and graph outputs must
be byte-exact with Torch for the tested shapes/magnitudes, followed by HE32
and two formal repeats plus independent profiling. See POST_RESULTS.md.

## Q/KV split experiment

`qkv_split_preflight_sla.json` freezes another 2% ceiling against accepted Post
before serving measurement: matched 28.74440217966317 / 27.642779170176176 ms,
aggregate 28.193590674919673 ms. All previous SLA gates remain. Require exact
Q BF16 bytes, full KV cache bytes and untouched input/page padding, including
changed-device-metadata graph replays. HE32 retains its existing gate. See
QKV_RESULTS.md for the actual 584-byte page contract and the aborted initial
528-byte-layout workflow.

## Sparse MLA selectors (2026-11-19)

The additional preflight SLA is in `mla_selectors_preflight_sla.json`: matched ceilings27.728960/26.693019ms and aggregate27.210990ms, fixed before measurement against accepted target Q/KV. Integer ids/counts and native consumer BF16 bytes must be exact under changing-metadata graphs; original precision/HE32 and all earlier serving gates remain required. See MLA_RESULTS.md for accepted results.

## Native MLA I/O admission, frozen 06:31 CST Nov19

Measured plugin ec93021 and adapter03ef92e. Prior accepted selector TPOT
22.870670023/22.569727734ms, aggregate22.720198878ms; new 2% ceilings
23.328083424/23.021122288ms and aggregate23.174602856ms. All original,
DSpark, Post and Q/KV gates retain force. Pure I/O changes require complete
native BF16 bytes and dynamic Graph replay exact, inputs immutable, HE32
32/32 identical tasks/prompts with zero errors/truncations. Final source
passes45CPU,160mask/1,152consumer GPU cases. HE32 passes; both formal repeats,
separate profile and workflow exit0,8traces prove400target/30draft new kernels.
Accepted formal medians22.223018363/21.874368957ms, mean22.048693660ms.
Held standalone DSpark schedule207817d is excluded. CRLF/EOF-only cleanup
after measurement preserves full AST. See MLA_IO_RESULTS.md and three JSONs.

## Draft + mask/RoPE composition, frozen06:59 CST Nov19

Measured plugin2820632/adapteraf8d174. Baseline accepted MLA I/O repeat medians
22.223018363/21.874368957ms; matched2% ceilings22.667478730/22.311856336ms,
aggregate22.489667533ms. Pure schedule precision remains byte-exact including
changing Graph replay, with HE32 32/32 identical tasks/prompts,zero errors/trunc.
All prior SLA gates remain mandatory.153CPU checks and isolated operator
probes pass; HE32 passes; formal21.906416365/21.540173914ms, mean21.723295139ms,
all exits0. Eight traces prove target400/draft30 fused mask/RoPE calls and
exact draft Qsplit/KV/final Post selection. Earlier standalone draft207817d
formal failure is retained; this composition includes additional optimizations.

## Fresh native MLA plan admission, frozen 07:31:54 CST Nov19

Pluginf83ff54 plus adapter9bee65c retain all earlier gates. The accepted
composition21.906416365/21.540173914ms is the additional control: matched
ceilings22.344544692/21.970977393ms and aggregate22.157761042ms,2% each.
Current native counts must generate fresh plan contents on every Graph replay;
active descriptors/splits and complete BF16 native consumer bytes must match,
with inputs immutable.832direct plus1,152complete bridge cases and49CPU
checks pass before serving. HE32 remains32/32, same prompts/tasks and
zero errors/truncations. Separate unprofiled formal repeats and eight-rank
fresh-plan selection evidence are mandatory. See MLA_PLAN_RESULTS.md.
