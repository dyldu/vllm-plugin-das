# Fixed-capacity W4A8 DSpark DeepEP validation

The optional draft DeepEP path passed exact transport/consumer gates,
complete real-prefill HumanEval, frozen performance regression limits and
eight-rank actual-dispatch checks. It remains disabled by default: paired
end-to-end TPOT does not demonstrate the minimum 2% incremental speedup.

## Communication and ownership

The default draft retains native DeepGEMM computation with AG/RS
communication. Explicit additional config
`{"hcu_dsv41_dspark_draft_backend":"deepep_low_latency"}` selects native LL
draft construction and passes an instance-local shared-buffer manager to
the native prepare factory. Target and draft request the same native
384-expert cache key, retaining one allocation and one destructor. Owned
complete-model entries emit one native IPC clean per invocation, including
inside captured graphs. Unsupported entries retain cleaning before each
dispatch. Python state never predicts the GPU buffer's previous layout.
Native factories, communicator globals and provider sources remain intact.

Admission requires CT W4A8, native TP1/DP8/EP8, single-node legacy IPC,
H5120, target384/topk6, draft128/topk3, capacity256, max-seqs32 and DSpark5.
Unknown topology, transport or geometry fails before another buffer
allocation. Service capacity and frozen graph buckets remain unchanged.

Draft BF16 transport invokes the source-pinned actual AG/RS row quantizer
after receive, masked by fresh device counts. Target INT8 transport remains
unchanged. Direct draft INT8 transport was rejected because minimum-scale,
half-way rounding and normal-scale bits differed from native AG/RS; its
thresholds were not relaxed.

## Operator and graph evidence

The receiver probe passed 16 eager transactions and 60 changed-input graph
groups per rank across eight ranks: 128 eager transactions and 480 graph
groups total. Cases cover empty and uneven ranks, zero/tiny/tie inputs,
full256 capacity, all frozen buckets and2048-row expert skew. Active INT8
codes, FP32 scale bits and combine outputs match the actual native oracle.
The oracle quantizes the same received BF16 rows and independently verifies
the combine round trip, avoiding ambiguous repeated token IDs across ranks.

The later model-entry reset probe passed another 128 eager transactions and
480 changed graph groups, comprising 1440 independent model replays. It
alternates target/draft layouts on the shared native buffer and compares
against conservative per-dispatch cleaning. Outputs remain byte-exact.
CPU contracts cover owner/cache lifetime, native identity, entry/dispatch
order and unsupported topology. Native W4A8 consumers also validate
expected-M hints16/32/48 against actual counts and above-hint skew.

Reproduce the receiver probe with the pinned native providers:

```bash
python tests/vllm033/probe_w4a8_dspark_ll_wire.py \
  --bf16-receiver --graph --output /path/to/new-output-directory
```

## Complete-model and performance results

The combined reuse stack Pluginbb1a36e + Adapter45a32f3 passed real-prefill
HumanEval160/164 (97.5610%), first32=32, with zero errors, missing outputs or
truncations. The otherwise identical AG/RS stack9bc97af passed159/164 and
first32=32. Full generated texts are not claimed byte-identical.

With the user-corrected literal acceptance rates `[0.6,0.6,0.6,0.6,0.6]`,
fake zero KV,4k input/128 output and64 requests/concurrency, the unprofiled
median TPOT repeats are:

| Draft transport | Repeat1, ms | Repeat2, ms | Mean, ms |
| --- | --- | --- | --- |
| AG/RS | 20.329348 | 19.632577 | 19.980962 |
| DeepEP | 20.347785 | 19.557936 | 19.952860 |

Both DeepEP repeats pass the frozen median/p99 regression limits. The mean
difference is0.1406% and the first repeat is slightly slower, so the minimum
2% incremental speedup gate fails. Default AG/RS is retained. All eight
DeepEP traces confirm one native clean at each target/draft model entry,
40 target and3 draft LL layers, a shared buffer and single FULL target graph.
The eight AG/RS traces confirm no draft LL layers. Existing exact-commit
real-prefill accuracy gates are reused for this performance-only rerun.

Earlier geometric rates `[0.6,0.36,0.216,0.1296,0.07776]` also passed
regression/accuracy gates, with paired mean TPOT33.431361ms AG/RS versus
33.368134ms DeepEP. Those values are a different workload and must not be
compared to the corrected-rate20ms values as a code improvement.

Evidence under `run/dsv4.1/033/w4a8_reverify_20261119/`:

- `nmz18/moe_reuse_probes_20261119_v5/ll_wire/`
- `nmz18/ll_entry_reset_wire.audit.json`
- `nmz18/dspark_deepep_stack_20261119/he164.gate.json`
- `nmz18/dspark_deepep_stack.acceptance.json`
- `nmz18/uniform06.acceptance.json`
- `nmz18/uniform06_20261119/`

The SGLang trace is a stage reference with different batch, quantized kernel
providers and hardware context. Logical MoE interval sums include computation
and overlap; they are not pure communication time or controlled parity proof.
