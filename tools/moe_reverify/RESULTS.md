# Channel-FP8 MoE validation on nmz18

Validated production commit: 8e5ae71. Container duyl-033-1112, 8 x BW1101
95.98 GiB; adapter 35c397a. Baseline b5d88fb. The SGLang reference ran on
BW1100, so its stage ratios have a hardware limitation.

| Paired formal decode | Baseline median TPOT ms | Candidate V4 ms |
|---|---:|---:|
| Repeat 1, 64/64 requests | 37.64855795 | 36.52812658 |
| Repeat 2, 64/64 requests | 36.38672786 | 36.23626778 |
| Median of repeat medians | 37.01764291 | 36.38219718 |

Candidate aggregate improves 1.717%; both matched 2% ceilings and aggregate
37.75799576 ms ceiling pass. No failed requests. Performance uses fake prefill
and synthetic acceptance exclusively; accuracy uses real prefill/acceptance.
Both baseline and V4 pass HumanEval first32 **32/32**, no errors/truncations,
identical task/prompt identities. Candidate profiler exports eight traces and
exits0. Baseline exports eight traces but its optional PyTorch table finalizer
hangs (workflow143); disable dump_cuda_time_total in subsequent captures.

All eight candidate ranks show 400 target mega-gate, 400 image-route correction,
400 shared activation fusions and 800 masked grouped GEMMs over ten target
steps. Grouped GEMM means310.9..324.8us vsSGL312.7..324us. Shared activation
11.3..11.4us vsSGL5.4us remains slower: **full stage alignment is incomplete**.
Do not infer draft timing from target or count FP32 SB GEMMs as gate-only work.

Operator verification: 50 gate/shared/vision GPU cases plus one actual native
MLP constructor case passed. Full native Triton Linear consumers have identical
FP8 codes/scales and BF16 output; gate IDs unchanged in measured cases. Shared
coverage includes all finite BF16 gate values crossed with nine multipliers,
clamp boundaries, signed zero and changed-input graph replay. The original
LightOp shared fusion fails the frozen max_abs0.01 threshold; the accepted
implementation preserves SiLU and product BF16 rounding. Hint7->16 is bitwise
on active expert rows but measured gain<0.1%; no speedup is attributed to it.

`PROTOCOL.md` fixes the controls/gates. `profile_fp8_paired.sh` runs two formal
repeats and an independent profiler; `accuracy_fp8.sh` runs the fixed HE32.
`stage_report.py` separates GPU target/draft phases by annotations;
`summarize_he32.py` audits item outcomes and prompt identity. Operator tools
report eager synchronized wall time separately from device event times.
The original single-node graph replay timings include CPU submission gaps;
`unrolled_graph_gpu_median_us` now measures repeated calls within one graph to
resolve small kernel differences. No standalone number replaces the serving
TPOT gate.

Raw results, failed starts, exported traces and measurement snapshots:
/public/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118.
V4 performance: candidate_pair_v4/20261118_234250_channel_fp8_prZgJS.
V4 accuracy: candidate_he32/20261118_235300/20261118_235802.
Baseline performance: baseline_pair_v4/20261118_221529_prof_w4_W0HNBm.
Baseline accuracy: baseline_he32/20261118_224344/20261118_224853.

A 2304-column tail-specialized shared kernel is under validation after V4;
it is not covered by V4's serving verdict. Graph mode and DSpark transport
experiments follow operator validation. Both stacks already use DeepGEMM;
the draft transport differs (vLLM AG-RS vsSGL DeepEP-LL).
