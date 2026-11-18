# H5120 mHC Post follow-up

Use companion adapter branch
[fix/mhc-post-5120](https://github.com/dyldu/adapter-das/tree/fix/mhc-post-5120),
production89cc71d / validated3bb8bfe. It includes the earlier FP4 graph and AITER
schema fixes. No native H5120 Post precision guard is removed.

The new gfx938 Triton Post retains the Torch FP32 accumulation order and BF16
storage boundary. 74CPU checks,123actual helper/runtime cases and40changed-input
graph replays pass byte-exactly. Direct native AITER Post failed the fixed gate
and remains rejected. Isolated M48 graph cost194.27us ->7.34us.

With plugin902a1fa, FULL_AND_PIECEWISE and explicit native IPC draft LL:

| Configuration | Repeat median TPOTs, ms | Mean of repeat medians, ms |
| --- | --- | --- |
| Accepted model-entry clean, adapter80c6642 | 31.87492367 /31.56523106 | 31.72007737 |
| Same plugin + new Post | 28.18078645 /27.10076389 | 27.64077517 |

Each repeat completes64/64requests, failed0. Original, DSpark and the additional
pre-measurement Post matched/aggregate SLA gates all pass. HE32 passes32/32,
same tasks/prompts, zero errors/truncations, real prefill/acceptance, workflow0.
Observed aggregate reduction12.860% vs the entry-clean control,25.331% vs the
original PIECEWISE baseline; this is not a statistical confidence estimate.

Independent profile/workflow0, eight traces. Every rank replaces830target FP32
Post BMMs per10steps with830new calls (9.51..9.67us mean). Draft replaces50calls
(7.58..7.87us); its10final direct Torch Post calls remain. Profiled target phase
median106.42..106.59ms, draft+sampling7.65..7.79ms. Median idle fractions are
2.54..2.62% and1.99..2.05%. Use unprofiled formal TPOT for the SLA.

Full SGLang stage/step parity remains incomplete. Shared activation is still
about8.5us target /7.5us draft versus SGLang5.4us /5.3us. SGLang uses BW1100,
this run BW1101; the supplied trace has no BMM input dimensions. Request-batch
annotations alone do not establish equal intermediate shapes or context lengths.

Operator source/probe and detailed evidence are in the companion adapter's
[validation document](https://github.com/dyldu/adapter-das/blob/fix/mhc-post-5120/docs/validation/mhc-post-5120.md).
Raw `mhc_post_*` artifacts are in
`/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118`.
