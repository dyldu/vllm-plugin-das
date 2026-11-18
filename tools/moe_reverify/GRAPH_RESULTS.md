# FP4 decode FULL_AND_PIECEWISE validation

Plugin b4ad8ce and adapter 80c6642 build on the accepted MoE source 5575178
and AITER schema fix 4e361c7. All changes use separate worktrees. The initial
FULL_AND_PIECEWISE start was refused by FP4 preallocation admission before
weights loaded; it was not a no-kernel failure.

The adapter now accepts an owner-published FP4 decode graph contract. The
plugin checks the native V4.1 TP1/DP8/EP8 DSpark topology, FP4 selection,
breakable PIECEWISE prefill and existing query/verify/capture capacities.
The host-metadata FP8 reference keeps its original full-capture rejection.
The reader refuses prefill metadata inside FULL capture.

Real GPU replay exposed an additional correctness defect: the native FP4
decode provider uses clean_logits=False and leaves columns beyond device
context lengths unwritten. Top-k could select graph-private allocation
garbage. The reader masks those columns before selection, including zero
length/padded rows. Tests poison the tail with large values and compare real
LightOp eager/graph results while changing queries, cache contents, lengths
and block tables across page boundaries, for physical 64- and 32-record pages.

Validation on nmz18 / duyl-033-1112, 8 x BW1101, 95.98 GiB per device:

- 73 CPU admission/constructor/reader checks and 3 real GPU graph checks pass.
  CPU suites disable GPU visibility and platform plugin detection to preserve
  their existing no-CUDA-initialization assertion. The GPU suite uses the real
  HCU platform and native LightOp providers.
- HumanEval first 32: **32/32**, identical baseline prompts/tasks, zero errors
  and truncations, real prefill and real speculative acceptance. Workflow exit 0.
- Target FULL and DSpark FULL graphs capture successfully on every rank. No
  OOM/no-kernel error in either the accuracy or performance start.
- Two isolated, unprofiled formal repeats complete 64/64 requests each with
  zero failures. Median TPOT **32.40832604 / 32.09548897 ms**, aggregate
  **32.25190751 ms**, versus original baseline **37.01764291 ms** (12.874%
  observed decrease). Both matched 2% ceilings and aggregate 37.75799576 ms
  ceiling pass. The FP4 tail fix is also present; same-source PIECEWISE control
  is pending to isolate the graph-mode effect.
- Independent profile exports 8 traces; profile/workflow exit 0. Each rank has
  400 target mega-gates, image corrections and shared fusions, and 800 masked
  grouped GEMMs in 10 target steps.

With the same profiler settings, median phase idle fractions across ranks
are 1.87--1.97% for draft plus sampling and 2.25--2.30% for target. Rank 0's
prior V6 PIECEWISE values were 84.95% and 23.86%. These are profiler diagnostics,
not unprofiled latency estimates; stack collection can inflate CPU gaps.

MoE stage alignment remains incomplete: target shared mean is 8.416--8.503 us
in this capture versus the SGLang reference's approximately 5.4 us. Grouped
means are 310.08--321.14 us. The SGLang device is BW1100, whereas these paired
runs use BW1101. No general kernel speedup or complete SGLang parity is claimed.
Draft transport remains AG/RS; both frameworks already use DeepGEMM.

Frozen controls and gates are in PROTOCOL.md. CUDAGRAPH_MODE selects PIECEWISE
(default) or FULL_AND_PIECEWISE in both harnesses. Compact evidence is in
graph_v2_{perf,he32}_summary.json. Raw artifacts, failed starts and unit logs:
/public/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118.

Accuracy: graph_v2_he32/20261119_014711/20261119_015203.
Performance: graph_v2_pair/20261119_015351_channel_fp8_j45kIT.
Units: fp4_full_decode_{cpu,gpu}_v4.log.
