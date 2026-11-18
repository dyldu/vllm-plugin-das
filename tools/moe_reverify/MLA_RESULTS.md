# Exact sparse MLA selector compaction

The adapter replaces Torch CSR/dense repacking, validity filtering, sorting and
padding with one gfx938 Triton kernel. Duplicates remain, invalid slots become
-1 and counts are int32. Metadata stays on device; no plan caching or native
FlashMLA/KV change is introduced. Unsupported layouts/devices retain Torch.

Companion adapter production `161e27b`, branch
[fix/mla-decode-selectors](https://github.com/dyldu/adapter-das/tree/fix/mla-decode-selectors),
includes the previously validated H5120 Post and FULL graph fixes. Use it with
this PR and the unchanged paired harness settings in PROTOCOL.md.

32 CPU checks, 480 integer cases, 480 changing-metadata graph replays and 40
complete native FlashMLA BF16 consumer comparisons pass exactly. HumanEval
first32 is 32/32 with baseline-identical tasks/prompts, zero errors/truncations,
real prefill and speculative acceptance.

| Fixed unprofiled workload | Repeat1 median TPOT, ms | Repeat2 | Mean of repeat medians |
| --- | --- | --- | --- |
| Accepted target Q/KV control | 27.185255 | 26.169627 | 26.677441 |
| Same plugin + fused selectors | 22.870670 | 22.569728 | 22.720199 |

Both repeats complete64/64 requests with zero failures; all earlier ceilings and
additional 2% matched/aggregate SLA frozen at05:31 before measurement pass.
Observed aggregate reduction is14.834% versus Q/KV and38.623% versus the original
37.017643ms baseline. These are two repeats without a confidence claim.

Separate profile/workflow exit0 with eight rank traces. Every rank has780target
selector calls per10target steps (40SWA+38extra per step), mean8.206..8.320us.
Draft has27calls on rank4 and30on other ranks, exactly matching each rank's
native draft Q/KV call count, mean6.495..6.692us. Target phase medians are
85.534..85.690ms; draft/sampling6.868..7.029ms. Median GPU idle fractions are
1.662..1.711% /1.366..1.478%.

The stage reporter singleton tuple for native Qnorm classification was fixed
and a full-symbol regression check added. Both this trace and the previous
Q/KV traces were recomputed: target Qsplit400/KVonly400 and no old combined
Q/KV, draft native Qnorm30/no-Qnorm30 per10steps remain confirmed on all ranks.
This reporting correction does not change raw formal TPOT or serving source.

Full SGLang step parity remains incomplete. The reference lacks the input
geometry/context metadata needed for a matched phase claim and uses BW1100,
while this environment uses BW1101. Raw evidence is retained under
`/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/mla_selectors_*`.
