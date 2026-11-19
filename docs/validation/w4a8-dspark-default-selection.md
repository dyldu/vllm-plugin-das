# W4A8 draft defaults: Post fusion and DeepEP

The verified H5120/HC4/DSpark5 native CT W4A8 draft now selects final Post
fusion without an additional launch flag. Other model/topology/geometry
contracts keep their existing path. Explicit
`{"hcu_dsv41_dspark_final_post":false}` disables the schedule; an explicit
`true` request still rejects an unsupported configuration. Tensor admission,
autocast and matmul precision checks, Adapter kill switches and private native
source/identity checks are unchanged.

The admitted native CT W4A8 TP1/DP8/EP8, capacity256/max-seqs32/DSpark5
deployment also defaults draft communication to DeepEP low latency. Explicit
`{"hcu_dsv41_dspark_draft_backend":"allgather_reducescatter"}` restores AG/RS.
Unrecognized model/quantization/geometry keeps its prior path; an explicit
unsupported backend request still fails clearly. The target config is not
mutated and native shared-buffer ownership is unchanged.

This changes selection policy only. Implementations and native arithmetic
are unchanged from the full-164-validated explicit Post/DeepEP composition.
The default-selection and native loading/resource/LL tests passed65 CPU tests
with GPUs hidden. Post-only automatic selection separately passed real-prefill
HumanEval159/164, first32=32, with zero errors/missing/truncations.

Per user instruction this default-policy revision is pushed before its new
combined full-service validation completes. Its automatic Post+DeepEP
HumanEval/performance and matched VRAM comparison are pending; earlier
explicit-configuration results are not relabeled as tests of this revision.
The existing explicit DeepEP composition passes regression limits but has no
minimum2% incremental end-to-end speedup claim. The user explicitly selected
DeepEP as the default despite that small overall TPOT difference.

Performance keeps fake prefill and five literal unconditional synthetic rates
`[0.6,0.6,0.6,0.6,0.6]`. Earlier geometric-rate measurements must not be
presented as results for this workload. Accuracy uses real prefill and real
rejection sampling. GPU memory utilization remains0.85; VRAM comparison must
also report KV capacity because automatic cache sizing can absorb differences.
