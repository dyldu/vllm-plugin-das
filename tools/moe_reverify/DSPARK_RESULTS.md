# Bounded IPC draft LL validation

The explicit `DSPARK_DRAFT_BACKEND=deepep_low_latency` harness option now runs
DSpark through native LL dispatch/combine and masked DeepGEMM. The default
remains AG/RS with contiguous DeepGEMM. Both paths already use DeepGEMM; their
communication and expert activation layouts differ.

On nmz18 / duyl-033-1112 the installed DeepEP wheel can construct two Buffers,
but constructing a second, draft-sized Buffer invalidates the first target
dispatch in an isolated reproduction with about 88 GiB still free. This is not
proof of a general allocator OOM cause. A single native target-sized IPC Buffer
works for both layouts, provided a native clean precedes layout reuse. Keep its
original cache key and destructor; do not insert duplicate cache aliases or
mutate QP configuration. The policy rejects GDA/internode, fault tolerance,
MNVL and effectively enabled layered routes. The baseline's layered request is
inert for eight ranks in this wheel and remains unchanged.

This validation is limited to H5120, target384/draft128 experts, cap64,
TP1/DP8/EP8 and same-node IPC. It does not establish cap256 admission or solve
other memory configurations. Both HE32 and paired performance starts complete
all eight ranks' graph captures without OOM or invalid access.

## Precision

Raw vendor LL token FP8 quantization differs from the existing draft's stock
quantizer: an isolated random BF16 comparison reached max_abs0.03515625.
Providing stock scales did not repair it; FP8 sender input is unsupported by
this wheel. These variants are rejected. Existing target quantization is
unchanged. New draft LL sends BF16 and quantizes only counts-active receive
rows with the stock FP32 scale minimum, RN division and BF16/E4M3 boundaries.
Inactive capacity rows are unspecified and are never consumed by masked GEMM.

Source50b3ff2, adapter80c6642:

- 252 actual stock FP8 code/scale comparisons pass exactly, including zero,
  changing counts/payloads in a graph, skewed/full/empty experts, and all finite
  BF16 bit patterns at multiple row maxima. Typical128active rows: stock full
  capacity227.7us versus masked6.4us; empty5.4us; full8192rows219.7us vs227.7us.
  These are isolated unrolled graph measurements, not serving TPOT.
- 2,000 eight-rank graph replay output comparisons pass, max_abs0, changing
  payload/routes across rows0/1/40/48/64 and magnitudes0/.001/.25/1/8. The
  target-side wire probe uses stock receiver quantization as an independent
  layout reference; production target vendor math is retained.
- 30 focused CPU integration/metadata/ownership checks pass on this source;
  the preceding native LL source also passed89CPU checks.
- HumanEval first32 passes32/32, identical baseline prompts/tasks, zero errors
  and truncations, real prefill and real speculative acceptance, workflow0.

## Frozen serving gates

Keep the original baseline SLA and the additional, pre-measurement FULL+AG/RS
matched ceilings33.05649256564706 /32.73739875246603ms, aggregate32.89694565905654ms.
The aggregate is the mean of two repeat medians.

The first all-capacity receiver version a8b09d8 passed HE32 but failed the
first stricter matched performance ceiling:33.11171170 /32.58374500ms. It was
not accepted and the threshold was not changed.

The active-row version50b3ff2 passes both matched and aggregate gates:
32.88293272 /32.42328721ms, aggregate32.65310997ms;64/64requests each, failed0.
This is1.244% above the accepted FULL+AG/RS control32.25190751ms, within the
frozen2% allowance. Do not claim an end-to-end speedup for this transport yet.
Workflow0, profile0, eight traces. Each rank records60draft LL dispatch and
combine kernels,60masked GEMMs and30masked expert activations over10steps;
draft contiguous GEMM is absent. Draft masked GEMM means87.17..110.87us;
shared activation7.35..7.45us remains above SGLang5.3us, so full stage parity
is not established. Profiling is separate from the unprofiled formal repeats.

Artifacts: run/dsv4.1/033/moe_reverify_20261118/{dspark_masked_quant_v4_*,
masked_token_quant_v2.*,deepep_probe_shared_graph_masked_quant_v1.log}.
The per-dispatch native clean is conservative; model-entry cleanup is a
separate, unvalidated follow-up at this snapshot.
