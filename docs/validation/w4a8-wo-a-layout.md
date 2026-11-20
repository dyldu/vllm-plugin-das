# W4A8 WO_A output layout

Native BF16 WO_A einsum produces group-major storage; flattening for the actual
INT8 WO_B consumer then copies it. Write the same native BMM directly into
contiguous token-major storage and flatten by view. Keep the cached BF16 weight,
BMM arithmetic, native INT8 quantizer and native WO_B scheme unchanged.

Admission is restricted to the verified DeepSeek V4.1 W4A8 quantization, H5120,
TP1/DP8/EP8/PP1/PCP1, gfx938/64CU, BF16 heads64/128 with64 used heads,
8 groups/rank1024 and rows8/48/64. Native source/function identities, known NK/NN
weight-cache wrapper, Torch build, consumer and weight/cache layout are checked.
Gradient, CUDA autocast, external GEMM overrides and unsupported contracts use
the prior native path. Both flat and actual nested text configurations are
covered. No new launch flag is required. Draft attention remains native.

## Operator and complete-service results

`tools/moe_reverify/probe_w4a8_wo_a_layout.py --owned` tests real checkpoint
layers0/20/39 and the actual patched CompressedTensorsW8A8Int8 weight loading,
quantization and GEMM.288 eager cases and240 changed-input graph replays retain
byte-identical BF16 activations, INT8 codes, FP32 scales and native BF16 outputs.
The cases include NK/worker-NN storage, NaN-padded heads and six magnitudes.
56 focused CPU contracts and139 additional activation/admission regressions pass.

Measured sources: Plugin55fca625cb31f2d6c2e1583a66be823b124d22b2 and
Adapter18c695303a0548b134a39a1a34c4f0ce4b472391. Complete real-prefill HumanEval
passes159/164 (96.95%),first32=32,zero errors/missing/truncations. Actual8-rank
traces each verify40 native WO_A BMMs per target step and40->0 following reorder
copies, while retaining FULL,target40/draft3 DeepEP/activation and83/6 Post calls.

With fake zero KV and5 literal unconditional0.6 acceptance rates, two unprofiled
TPOT medians are19.932703/19.343608ms,p9923.705752/24.118971ms. Against the preceding
mHC-launch composition19.923699/19.587913ms, all frozen median2%,p995%,mean2%
regression limits pass. The mean improves0.5955%; this does not pass the separate
2%-per-repeat speedup-claim gate. Profiled DP0 target-through-draft step is
79.812687ms,target73.346637ms,draft6.401201ms. SGLang65.098274ms remains a stage
reference with hardware,batch,context and provider differences; full alignment
is unfinished. Profile timing is separate from unprofiled request TPOT.

Capacity remains memory0.85,block64,max-seqs32,max-len32768,max-batched-tokens256,
capture1/4/8/16/32/40/48/64,FULL_AND_PIECEWISE. Accuracy uses real prefill/rejection;
synthetic rates apply only to performance. An earlier startup failure occurred
before loading because another user's task allocated the GPUs; it is preserved,
and the unchanged-source retry above passed. No DeepGEMM OOM is inferred.

Hashes, complete paired metrics, all8 trace checks and source pins are in
[w4a8-wo-a-layout-evidence.json](w4a8-wo-a-layout-evidence.json).
