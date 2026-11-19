# Fixed-capacity W4A8 DSpark LL experiment

Status: experimental, disabled by default. No serving speedup is admitted.

The default draft retains native DeepGEMM computation with AG/RS communication.
An explicit additional config
`{"hcu_dsv41_dspark_draft_backend":"deepep_low_latency"}` retains native LL
draft construction and passes an instance-local shared-buffer manager into
the native prepare factory. Target and draft request the same native384-expert
cache key, with one native allocation/destructor. Each dispatch emits native
buffer cleaning, including inside captured graphs; no Python last-layout state
is used. No native factory, communicator or source global is replaced.

The experiment admits CT W4A8 only, native TP1/DP8/EP8, single-node legacy IPC,
H5120, target384/topk6 and draft128/topk3 buffer layouts, capacity256,
max-seqs32, speculative-tokens5. Unknown topology, transport and geometry fail
before another buffer allocation. The older FP8 capacity64 experiment is not
used as W4A8 validation evidence. All existing graph buckets and service memory,
KV and scheduler parameters stay fixed.

Focused CPU tests cover unmodified default AG/RS, native constructor/config
identity, bounded native-cache ownership, unsupported transport/geometry and
clean/dispatch/hook order across changing target/draft layouts. Meta tensors
cover full frozen receive extents without substituting a smaller capacity.
The existing native hint probe also accepts16local experts for a separate
DSpark masked-GEMM check. Its passing independent oracle and fresh-count
graphs admit hints16/32/48 for this exact draft geometry. Larger initialization
shapes retain the native2048 hint; every kernel still consumes actual counts.

Required remaining GPU gates, in order:

1. Compare active INT8 codes and FP32 scale bits with the native AG/RS quantizer,
   including zero/tiny/tie/skew inputs, unequal and empty ranks and full256
   capacity. A wire mismatch requires correction, not a relaxed threshold.
2. Exercise alternating target/draft graphs with changed payload/routing/counts
   and actual native W4A8 expert GEMM consumers. Check all frozen buckets and
   above-hint skew. Retain rejected or failed records.
3. Establish operator/transport performance and memory use. Only then run real
   KV HumanEval32, followed by the full164 gate and two unprofiled TPOT runs
   plus an independent eight-rank trace under the existing frozen SLA.

Published-HEAD baseline remains Plugin526b626 + Adapter8a1c05c and is verified
separately. Experimental code is isolated in its own Git worktree.

An additional uninstalled activation experiment reuses the earlier 2304-column
split (2048+256), retaining the native clamp10, BF16 SiLU/product boundaries,
INT8 RNE and minimum absmax1e-10. The contiguous version uses native negative
row IDs to skip expert padding; the masked version follows device counts with
a grid-stride loop, so the expected row count never bounds valid rows.
`tests/vllm033/probe_w4a8_clamped_int8.py --consumer` compares native code/scale
bytes, finite BF16 gate sweeps, changing graph counts up to2048, actual packed
W4A8 down GEMMs and graph replay latency. The75-case GPU gate passed; this
worktree does not install its serving activation hook.


2026-11-19 GPU update: the direct INT8 wire failed exact AG/RS comparison
on zero/minimum scales, half-way rounding and some normal scale bits. It is
not admitted. The16-expert native masked GEMM hint probe passed active byte
comparison, signed-INT4 independent samples, native down consumers, changing
counts and full2048-row skew. A separate activation branch passed75 exact
operator cases; its end-to-end evaluation is separate from this LL experiment.

The current draft alternative sends BF16 and invokes the source-pinned actual
AG/RS row JIT at the receiver, masked by fresh counts. Target384-expert INT8
wire remains unchanged. Quantization is deferred until the recv hook completes.
All graph/wire/consumer/serving gates above still apply to this alternative.

Receiver GPU gate,2026-11-19: all8 ranks passed16 eager transactions each
and60 changing graph replays each (128 and480 total). Graph buckets include
0/1/4/8/16/32/40/48/64/160/192/256 rows, alternating target384 and draft128
layouts on the same buffer with per-dispatch clean. Eager cases cover zero,
tiny, half-way rounding, uneven/empty ranks and full2048-row expert skew.
INT8 codes and FP32 scale bits match the actual AG/RS row JIT. Combine round
trips match original input quantization; graph outputs match eager bytes.

The initial cross-transaction oracle sorted by source token index alone.
Those indices repeat across source ranks under full skew, making row pairing
ambiguous. The corrected oracle quantizes the same received BF16 rows and
separately verifies the combine round trip against source inputs. Thresholds
were not relaxed; the repeated-index case is reported in the JSON evidence.

Reproduce on an authorized idle eight-GPU node with the pinned providers:

```bash
python tests/vllm033/probe_w4a8_dspark_ll_wire.py \
  --bf16-receiver --graph --output /path/to/new-output-directory
```

Evidence: `run/dsv4.1/033/w4a8_reverify_20261119/nmz18/moe_reuse_probes_20261119_v5/ll_wire/`.
This transport/quantization gate does not establish model accuracy, native
expert-path integration, AG/RS performance parity or serving memory use.
