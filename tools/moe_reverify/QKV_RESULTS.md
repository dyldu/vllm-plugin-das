# Q/KV preprocessing follow-up

The owned FP4 attention class splits Q RoPE from the unchanged native KV-only
operator. Admission requires gfx938, the verified native binary, contiguous
BF16 Q[T,64,512] and KV[T,512], T=1..64, and a 584-byte fp8_ds_mla KV record.
Blocks of 32/64 tokens may have padded page strides. PCP, interleaved Q, other
geometries and unknown binaries retain Core. The compilation factor includes
the new implementation and native-build admission.

The native 64-head kernel reports `private_segment_fixed_size: 428`,
`vgpr_count: 64`, and `vgpr_spill_count: 224`; its disassembly contains scratch
loads/stores. The separate Q schedule avoids this fused schedule. Q preserves
the verified binary's mixed FMA order, including the unfused first even element
in each 16-element lane group. This depends on the compiler, so the native
binary SHA gate is intentional.

The initial 528-byte prototype passed 240 additional cases across eight seeds
(253,231,104 Q elements), with Q/cache bytes exact. That layout is not used by
this gfx938 service: the first candidate would fall back. Its HE workflow was
deliberately stopped (exit 143) and is not a validation pass. Alternate FMA/RTZ
variants failed the byte gate and were rejected.

For the actual 584-byte helper and owned class, 644 eager cases and 230 graph
replays pass byte-exactly. Coverage includes all eight capture buckets, blocks
of 32 with 320 padding bytes/page and blocks of 64 with 64 padding bytes/page,
empty/partial/negative slots, signed zero, cancellation and magnitudes 0..100.
Inputs and page padding remain unchanged. Isolated M48 graph time changes from
104.93 us to 14.42 us (block 32), or 14.37 us (block 64).

71 CPU admission/fallback checks pass. Two packer default-policy test failures
also reproduce on unchanged parent f4152cc; both reports are retained. The
additional TPOT SLA was frozen before serving measurement against accepted
Post: matched ceilings 28.74440218/27.64277917 ms, aggregate 28.19359067 ms.
All previous precision and SLA gates remain.

Corrected candidate 2a37d2e passes HE32: 32/32, same tasks/prompts, zero errors
or truncations, real prefill/acceptance, workflow exit 0.

| Unprofiled configuration | Repeat median TPOTs, ms | Mean of repeat medians, ms |
| --- | --- | --- |
| Accepted Post, plugin f4152cc | 28.18078645 / 27.10076389 | 27.64077517 |
| Post + target Q/KV split, 2a37d2e | 27.18525538 / 26.16962667 | 26.67744103 |

Both repeats complete 64/64 requests with zero failures. Every original and
additional frozen SLA gate passes. The observed aggregate reduction is 3.485%
versus Post and 27.933% versus the original baseline, from two repeats without
a statistical confidence estimate.

Independent profiling exits 0 and exports eight traces. All eight ranks replace
400 target fused-Q/KV calls per 10 steps with 400 Q and 400 native KV-only calls.
Their combined mean is 22.33..22.60 us, versus 124.04..124.47 us previously.
The target phase median is 102.27..102.45 ms, with 2.65..2.71% idle. Draft and
sampling remain 7.63..7.81 ms: their separate native attention classes retain
the original Q/KV path. This change establishes target selection only.

Full SGLang step parity is still incomplete. Trace context/shape differences
and BW1100 versus BW1101 hardware also limit direct phase comparisons.
Raw evidence: `/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/qkv_split_*`.
