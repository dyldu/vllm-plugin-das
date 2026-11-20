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

The revision was pushed before combined service validation, as requested.
The subsequent no-additional-config run of Plugin a5bc072 / Adapter45a32f3
passes full real-prefill HumanEval159/164 (96.9512%), first32=32, with zero
errors, missing outputs or truncations. Published runtime and test bytes match
these measured snapshots; documentation/CI changes are recorded separately.

The same source was measured with only draft transport changed. Each transport
completed two independent unprofiled64-request repeats without failure,
followed by a separate eight-rank trace pass. All16 traces pass actual FULL,
activation, plan, mask, Post and transport checks.

| Draft transport | Median TPOT repeats, ms | P99 TPOT repeats, ms |
| --- | --- | --- |
| Explicit AG/RS | 20.118194 / 19.713255 | 23.459538 / 24.397416 |
| Default DeepEP | 20.122414 / 19.570560 | 23.792888 / 24.297764 |

Mean medians19.915725 ->19.846487ms differ by0.3477%. Both repeats pass the
frozen median2% / p995% regression limits; this does not meet the minimum2%
incremental speedup claim gate. The user explicitly selected DeepEP as default.

The earlier Post-only repeat measured a2.378% median regression, exceeding
its2% limit. That failed gate is retained. A separately preregistered follow-up
also compares the new AG/RS Post-default measurements with the original
explicit-Post AG/RS control. It passes the same unchanged limits; no threshold
was relaxed or failed record replaced.

Two-second SMI sampling reports steady whole-device memory87.628–88.378GiB
with AG/RS and87.813–88.563GiB with DeepEP, depending on rank: an increase
of0.185547GiB (190MiB) per card,1.484375GiB across eight cards. Observed
peak differences are0.675–0.696GiB per card. Native rounded accounting shows
weights/non-Torch +0.29–0.30GiB, activation peak +0.28GiB, graph +0.53GiB
and KV capacity -0.57–0.58GiB per card (about270,640 fewer KV tokens).
Automatic cache sizing means the total difference is not an isolated DeepEP
communication-buffer allocation; sampling may miss transient peaks.

The latest DP0 target-start through following draft-end median is80.861678ms
(target74.370376ms, draft6.459199ms), versus supplied SGLang65.098274ms.
Hardware, batch and quantized provider paths differ, so this does not establish
controlled parity. Whole-step alignment remains unfinished.

Performance keeps fake prefill and five literal unconditional synthetic rates
`[0.6,0.6,0.6,0.6,0.6]`. Earlier geometric-rate measurements must not be
presented as results for this workload. Accuracy uses real prefill and real
rejection sampling. GPU memory utilization remains0.85; VRAM comparison must
also report KV capacity because automatic cache sizing can absorb differences.

Evidence under `run/dsv4.1/033/w4a8_reverify_20261119/`:

- `nmz18/defaults_deepep_20261119_v2/acceptance.json`
- `nmz18/defaults_deepep_20261119_v2/he164.gate.json`
- `nmz18/defaults_deepep_20261119_v2/memory_comparison.json`
- `nmz18/defaults_deepep_single_step_reference.json`
- `nmz18/default_post.acceptance.json` (prior failed gate)
