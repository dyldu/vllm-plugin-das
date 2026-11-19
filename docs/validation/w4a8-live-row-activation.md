# DSV4.1 W4A8 live-row INT8 activation

The earlier FP8 activation experiment's 2048+256 column decomposition is
reused for the W4A8 routed expert width2304. Quantization is rewritten to
preserve the native clamped SiLU's two BF16 rounding boundaries, INT8 RNE
codes and FP32 token-scale bits. Contiguous execution skips expert-alignment
padding using native row IDs. Masked execution uses current device counts;
expected_m only controls scheduling and never truncates active rows.

Admission requires the verified LightOp source and binary, gfx938 with64CUs,
H5120, gate/up width4608, clamp10 and either16 contiguous experts or16/48
masked experts with fixed256-per-rank receive capacity on8ranks. Unknown
contracts retain the native operator. The DSpark AG/RS backend, target
communication, capacity, request limits and graph buckets are unchanged.

GPU operator evidence on nmz18 (2026-11-19):

- 75 cases passed byte equality for active INT8 codes and FP32 scales,
  input immutability, zero/tiny/saturated/random inputs, all finite BF16 gate
  bit patterns, changing graph counts through2048 and actual packed W4A8
  down-projection consumption.
- Contiguous16-expert median147us ->19.49/21.06/29.15us for8/32/64
  live rows per expert-alignment segment.
- Masked16-expert10.72/25.25/44.51us ->6.54/14.02/24.32us.
- Masked48-expert21.63/61.56/111.18us ->12.93/35.62/65.22us.

Each latency is a CUDA-event median over80 graph replays with10 launches per
replay. These are operator measurements, not an end-to-end performance claim.
Evidence: `run/dsv4.1/033/w4a8_reverify_20261119/nmz18/moe_reuse_probes_20261119_v3/clamped_int8.json`.

Serving validation (same node and recorded runtime environment):

- Real-KV HumanEval:159/164 (96.9512%), first32 32/32, zero API errors,
  missing outputs or truncations. All164 prompt hashes and pass/fail outcomes
  match the accepted baseline;24 generated output lengths differ, so this
  does not assert whole-model output byte equality.
- Original-code control Plugin526b626 and candidate dec9f79, both with
  Adapter8a1c05c and HSA_USERPTR_FOR_PAGED_MEM=0. This variable came from the
  user's SGLang launch; no claim that it caused the earlier startup failures.
- Fixed model/capacity, TP1/DP8/EP8, block64, max-seqs32, max-len32768,
  max-batched-tokens256 and all8 capture buckets retained; FULL_AND_PIECEWISE
  and the accepted MoE expected-M scheduling retained.

| Unprofiled repeat | Control median TPOT ms | Candidate median TPOT ms | Control p99 ms | Candidate p99 ms |
| --- | ---: | ---: | ---: | ---: |
| 1 |38.764281|37.145193|45.553758|41.874221|
| 2 |39.231968|37.170023|43.676180|41.916902|

Mean of the two medians:38.998125 ->37.157608ms, an incremental4.7195%
reduction. Each repeat passed the predeclared2% median/5% p99 regression
limits and the minimum2% improvement requirement. Both had64/64 successful
requests, zero failures, actual4096 input/128 output tokens, concurrency64,
8 warmups, seed0 and temperature0. Performance uses zero-KV DecodeBench and
synthetic DSpark acceptance; it is not production real-KV latency or a
statistical significance claim.

All8 independently profiled exports passed CRC and confirm40 target plus3
draft live-row activation calls per span. Target80 W4A8 GEMMs still share one
native FULL graph launch per target span. The original-code control has no
candidate activation kernels. DP0 profiled target/draft-and-sampling walls
are83.304503/7.469039ms; profile times are not used for the TPOT speedup gate.

Evidence under `run/dsv4.1/033/w4a8_reverify_20261119/`:

- `nmz18/activation_userptr0_20261119/he164.gate.json`
- `nmz18/activation_userptr0_accuracy_equivalence.json`
- `nmz18/activation_same_environment.audit.json`
- `nmz18/activation_userptr0_trace.audit.json`
- `ACTIVATION_SAME_ENV_EXPERIMENT.json`

Experimental shared draft LL, Q/RoPE, MLA and draft final Post schedules are
not part of this promotion. SGLang remains a stage reference because its
trace's quantization/provider and workload differ.
