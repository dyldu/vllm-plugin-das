# Channel-FP8 MoE validation on nmz18

Accepted production source: **5575178**, comprising native integration8e5ae71,
2304-wide tail specializationfd9a311 and bounded division correction5575178.
Adapter4e361c7 fixes AITER Torch2.13 library cleanup without changing tensor
arithmetic. Container duyl-033-1112 uses8 x BW1101 with95.98GiB per device.
Baseline isb5d88fb with adapter35c397a. SGLang's reference ran onBW1100 with
larger memory; its stage ratios have a hardware limitation.

| Formal decode | Baseline median TPOT ms | Accepted V6 ms |
|---|---:|---:|
| Repeat1,64/64 requests | 37.64855795 | 37.40793839 |
| Repeat2,64/64 requests | 36.38672786 | 36.44600387 |
| Median of repeat medians | 37.01764291 | 36.92697113 |

Both matched2% ceilings and aggregate37.75799576ms ceiling pass. No failed
requests. The observed0.245% aggregate difference is too small to claim stable
end-to-end improvement. No tests or heavy analysis overlapped these repeats.
Performance uses fake prefill and synthetic acceptance exclusively; accuracy
uses real prefill/acceptance. Baseline andV6 both pass HumanEval first32 **32/32**,
with identical prompts, no errors and no truncations. Profile and workflow
exit0; eight traces exported. The adapter cleanup exception is absent.

Every rank executes400target mega-gates,400image-route corrections,400shared
activation fusions and800masked grouped GEMMs in ten target steps. Shared
serving-kernel mean **7.593..7.679us**, versus11.29..11.43us before tail/division
specialization (V4, about33% lower). SGLang shared is5.4us: **full stage alignment
is incomplete**. Grouped means315.53..327.48us versusSGL312.7..324us; mega57.32..
57.79us plus vision5.02..5.27us; expert activation~24us matchesSGL. Do not infer
draft timing from target or classify mixed FP32 SB GEMMs as gate-only work.

51GPU regressions pass, including full native Triton Linear outputs, changed
input graph replay, signed zeros, clamp boundaries and all finite BF16 gate
values crossed with nine multipliers. All30stock activation/quant ABI cases
have identical FP8 bytes and scales; full native Linear BF16 output is bitwise.
Gate IDs are unchanged in measured cases. The original LightOp shared fusion
fails the frozen max_abs0.01 gate; the replacement preserves both native BF16
rounding boundaries. Hint7->16 is bitwise on active expert rows, with measured
gain<0.1%; no speedup is attributed to that hint.

The bounded division helper also matches E4M3 bytes over292324506 valid signed
BF16 value/amax pairs through100 (1120468992 combinations before the valid-pair
mask). It applies only to width2304/limit10 after both BF16 boundaries. Scale
retains RN division; other widths/limits retain the generic RN path. A reciprocal
product alone changes FP8 ties; residual correction and signed-zero restoration
are required. LUT and wave-count variants were not adopted.

The single-process unrolled-graph M48 stock split10.067us versus fused3.588us
is an operator result. Python eager wall64.7->78.4us does not improve. The old
single-node replay timings include CPU submission gaps; repeated calls within
one graph resolve microkernel differences. No standalone number substitutes
for serving TPOT. V4 previously passed all serving gates with aggregate36.382ms;
that run and V5's CPU-overlapped first repeat are retained for audit, and neither
replaces the isolated V6 acceptance pair.

`PROTOCOL.md` freezes controls and gates. `profile_fp8_paired.sh` runs two formal
repeats plus independent profiling; `accuracy_fp8.sh` runs fixed HE32.
`stage_report.py` separates GPU target/draft phases by annotations.
`summarize_he32.py` checks item results and prompt identity;
`quant_div_validation.py` exhausts the production bounded division domain.
Compact evidence is in candidate_v6_{perf,he32,operator}_summary.json.

Raw artifacts: /public/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118.
V6 performance: candidate_pair_v6/20261119_011443_channel_fp8_hkPCEp.
V6 accuracy: candidate_v6_he32/20261119_010741/20261119_011235.
Baseline performance: baseline_pair_v4/20261118_221529_prof_w4_W0HNBm.
Baseline accuracy: baseline_he32/20261118_224344/20261118_224853.
Baseline exported eight traces but its optional PyTorch table finalizer hung
(workflow143); subsequent captures disable dump_cuda_time_total and exit0.

Next, test FULL_AND_PIECEWISE without changing draft transport. Both stacks
already use DeepGEMM; draft transport is vLLM AG-RS versusSGL DeepEP-LL. Those
graph/transport experiments are outside this accepted operator result.
