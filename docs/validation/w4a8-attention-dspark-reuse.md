# W4A8 attention and DSpark reuse validation

The Q/RoPE, MLA scheduling and draft final Post composition passed the frozen
W4A8 service gates on nmz18. These measurements use the INT4 W4A8 checkpoint;
historical FP8 experiments are not used as model validation.

## Changes and admission

- Target Q/RoPE is split from native KV insertion only for the byte-verified
  native build, gfx938 geometry and cache layout. Unsupported cases use the
  original native operation. Current W4A8 validation includes block64 and
  padded584-byte SWA pages:322 eager cases and115 changed-input graph replays.
- Adapter45a32f3 avoids redundant MLA query copies, fuses inverse-RoPE with
  exact BF16 mask/writeback, and prepares fresh native plans from current GPU
  counts. Core FlashMLA computation is unchanged. Dynamic plans are not cached.
- The draft final Post path uses the admitted H5120 Post
  kernel for the final draft Post. Private constructor/forward namespaces keep
  native globals intact.48 actual native/private forward cases and48 changing
  graph cases passed exact BF16 bytes and input immutability. It now defaults
  on for the admitted deployment; explicit false retains the original path.
  Existing layout/autocast/precision fallback applies.
- The admitted DSpark DeepEP path shares the native target buffer and emits
  one native IPC clean at each complete model entry, including captured graph
  execution. It retains capacity256, target384/topk6 and draft128/topk3 layouts.
  Draft BF16 transport invokes the actual native AG/RS row quantizer after
  receive; direct draft INT8 transport was rejected on exactness grounds.

## Service results on the earlier geometric-rate workload

The following TPOT/step measurements use literal synthetic rates
`[0.6,0.36,0.216,0.1296,0.07776]`. The user subsequently required
`[0.6,0.6,0.6,0.6,0.6]`; that same-rate comparison has now passed separately.
These absolute TPOT values must not be compared across the two workloads
as a code improvement. Full-model accuracy uses real prefill and real sampling.

Tested AG/RS composition: Plugin9bc97af + Adapter45a32f3. Full real-KV HumanEval
passed159/164 (96.9512%), first32=32, with zero errors, missing outputs or
truncations. All164 prompt hashes and pass/fail outcomes match the preceding
accepted activation stack;28 output token counts differ. Full-model generated
outputs are not claimed byte-identical.

Two unprofiled median TPOT runs37.145193/37.170023ms became
33.456961/33.405761ms. Their mean37.157608 ->33.431361ms is a10.0282%
incremental reduction for the combined changes. Both repeats meet the frozen
median/p99 limits and minimum2% improvement. Each completed64/64 requests.
Eight independent traces passed CRC, single-FULL-target and actual-dispatch
checks: target Q40, fresh-plan40 and inverse-RoPE/mask40 calls per span;
draft fresh-plan3, inverse-RoPE/mask3 and Post6 calls per span.

DP0 target-start through following draft-end median is81.162764ms, down from
90.776181ms. The supplied SGLang stage reference is65.098274ms. Batch,
quantized kernel paths and hardware/runtime context differ; this is not a
controlled SGLang parity claim. Profiled step durations are separate from
unprofiled TPOT. The performance workload uses fake KV and synthetic DSpark
acceptance and does not establish production real-KV latency.

The further DeepEP composition, tested Pluginbb1a36e + Adapter45a32f3, passed
full160/164 (97.5610%), first32=32, zero errors/missing/truncations. Its earlier geometric-rate
paired medians were33.333352/33.402917ms, mean33.368134ms versus the
AG/RS composition33.431361ms. All eight actual-dispatch/FULL/clean-count
checks passed and the frozen regression SLA passed. The0.1891% mean
difference fails the minimum2% speedup claim gate; no end-to-end transport
speedup is established. Those measurements explicitly selected DeepEP.
The user subsequently requested DeepEP by default; that policy update is
included and now passes the separate complete default-policy GPU validation.
Explicit `hcu_dsv41_dspark_draft_backend="allgather_reducescatter"` restores
AG/RS. See `w4a8-dspark-default-selection.md` for the distinct validation state.

## User-corrected five-position acceptance workload

The repeat comparison uses fake zero-filled KV, five literal unconditional
rates `[0.6,0.6,0.6,0.6,0.6]`, 4096 input and 128 output tokens, 64 requests,
64 client concurrency and eight warmup requests. The model, native packages,
complete native Python hashes, serving capacities, graph mode and common
environment are identical between these three versions. Each version runs
two unprofiled repeats and a separate eight-rank trace pass.

| Version | Plugin / Adapter | Median TPOT repeats, ms | Mean of medians, ms |
| --- | --- | --- | --- |
| Previous published activation | b544abc / 8a1c05c | 22.509010 / 22.116227 | 22.312618 |
| Attention + Post, AG/RS | 9bc97af / 45a32f3 | 20.329348 / 19.632577 | 19.980962 |
| Attention + Post, DeepEP | bb1a36e / 45a32f3 | 20.347785 / 19.557936 | 19.952860 |

The AG/RS composition reduces the mean by10.4499% relative to the published
activation stack; both repeats pass the frozen median/p99 and minimum2%
improvement gates. DeepEP reduces it by10.5759% against that same stack.
Against the otherwise identical AG/RS composition the DeepEP difference is
only0.1406%, with one repeat slightly slower: regression limits pass, but no
incremental end-to-end transport speedup is established. All six unprofiled
runs complete64/64 requests with zero failures. All24 traces pass actual
FULL/activation/MoE dispatch checks, including fresh-plan, mask and Post
dispatch on the reuse versions. The DeepEP draft uses three LL layers per
span; the AG/RS drafts use none.

These are performance reruns of unchanged source commits. Their existing
complete real-prefill accuracy gates are reused explicitly:159/164 with
first32=32 for AG/RS,160/164 with first32=32 for DeepEP, and no evaluation
errors, missing outputs or truncations. A synthetic acceptance run is not a
new accuracy evaluation.

The DeepEP DP0 profiled target-to-following-draft-end median is81.045581ms,
against the supplied SGLang reference65.098274ms. Target logical MoE interval
sums are35.353454ms versus SGLang35.638894ms; draft sums are1.257917ms versus
1.335116ms. These are dispatch-to-final-combine windows, including computation
and intervening work, not pure communication time. They do not establish
controlled MoE or whole-step parity. Trace metadata identifies BW1101 on
nmz18 and BW1100 in the SGLang trace; batch and quantized provider paths also
differ. Whole-step improvement remains unfinished.

## Frozen configuration and evidence

TP1/DP8/EP8, deep_gemm computation, memory fraction0.85, block64, max-seqs32,
max-model-len32768, max-batched-tokens256, capture buckets1/4/8/16/32/40/48/64,
FULL_AND_PIECEWISE and DSpark5 are retained. Both sides of the paired comparison
use HSA_USERPTR_FOR_PAGED_MEM=0. No extra rocBLAS/hipBLASLt tuning package or
library override is enabled; no new W4A8 OOM was observed.

Raw evidence under `run/dsv4.1/033/w4a8_reverify_20261119/`:

- `nmz18/reuse_stack_v2.acceptance.json`
- `nmz18/reuse_stack_v2_accuracy_equivalence.json`
- `nmz18/reuse_stack_v2_single_step_reference.json`
- `nmz18/qsplit_current_native_20261119.json`
- `nmz18/attention_reuse_probes_20261119/`
- `nmz18/draft_final_post_reuse_20261119/draft_post.json`
- `nmz18/ll_entry_reset_wire.audit.json`
- `nmz18/dspark_deepep_stack_20261119/he164.gate.json`
- `nmz18/uniform06.acceptance.json`
- `nmz18/uniform06_20261119/`
- `nmz18/uniform06_deepep_single_step_reference.json`
- `nmz18/uniform06_deepep_sglang_reference.json`
- `PROF_D_SCRIPT_COMPARISON.md`

The reviewed publication tree preserves tested runtime/test bytes; publication
documentation and CI coverage are recorded separately from measured snapshots.
