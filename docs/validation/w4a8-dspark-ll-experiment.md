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
DSpark masked-GEMM check; no16-expert serving hint is enabled by this patch.

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
