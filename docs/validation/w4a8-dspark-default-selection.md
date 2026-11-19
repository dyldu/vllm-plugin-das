# W4A8 draft default selection

The verified H5120/HC4/DSpark5 native CT W4A8 draft now selects final Post
fusion without an additional launch flag. Other model/topology/geometry
contracts keep their existing path. Explicit
`{"hcu_dsv41_dspark_final_post":false}` disables the schedule; an explicit
`true` request still rejects an unsupported configuration. Tensor admission,
autocast and matmul precision checks, Adapter kill switches and private native
source/identity checks are unchanged.

This changes selection policy only. The implementation and native arithmetic
are unchanged from the full-164-validated explicit Post composition. The
default admission and existing native loading/LL tests passed55 CPU tests with
GPUs hidden. A new real-prefill service validation of this default policy is
required before publishing it as accepted.

DeepEP draft selection remains an explicit transport option while the user's
new workload is revalidated: fake prefill and literal per-position synthetic
rates `[0.6,0.6,0.6,0.6,0.6]`. Earlier geometric-rate measurements must not be
presented as results for this workload. HumanEval accuracy evaluations retain
real prefill and real rejection sampling.
