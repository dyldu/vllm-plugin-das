# Q/KV preprocessing follow-up

Production scope: owned FP4 attention, gfx938, verified installed native binary,
BF16 Q[T,64,512] and KV[T,512], T1..64, fp8_ds_mla KV block32/64, record584,
including padded page strides,
no PCP or interleaved-query path. Unknown binaries and other layouts retain
native Core. The compile factor includes this implementation and native-build
admission, so changing the binary cannot reuse a previously captured split.

The native 64-head no-Q-norm kernel has private_segment_fixed_size428 bytes
and224spilled VGPRs, plus scratch load/store traffic in its disassembly. Splitting
Q RoPE from KV avoids that schedule. KV quantization/write uses the unchanged
native KV-only operator. Q preserves the verified binary's mixed FMA order,
including the unfused first even element in each 16-element lane group. This is
compiler dependent; the binary SHA gate is intentional.

Prototype528-byte layout:240additional cases across eight seeds,253231104Q
elements, Q/cache byte-exact. This prototype layout is not the current service
layout: gfx938 uses the584-byte record; the first narrow528-only candidate
would fall back. Its HE workflow was deliberately aborted (exit143), and is
not an accuracy or performance pass. Alternate FMA/RTZ variants failed and
were rejected.

Actual584-byte helper and owned class:644eager cases and230changed-input
CUDA Graph replays pass byte-exactly, including all eight capture buckets,
block32 with320padding bytes/page and block64 with64padding bytes/page,
empty/partial/negative slots, signed zero, cancellation and magnitudes0..100.
Inputs and page padding remain unchanged. Isolated graph M48 native104.93us
-> owned14.42us (block32),14.37us (block64). This is not yet a serving claim.

71CPU admission/fallback checks pass. The broader packer test file has two stale
default-policy failures reproduced on unchanged parent f4152cc; both reports
are retained. No precision gate is relaxed. The additional TPOT SLA is frozen
before serving measurement against accepted Post: matched ceilings
28.74440218/27.64277917ms, aggregate28.19359067ms. All previous gates remain.

HumanEval32 and independent formal/profile measurements are pending.
Raw evidence: `/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/qkv_split_*`.
