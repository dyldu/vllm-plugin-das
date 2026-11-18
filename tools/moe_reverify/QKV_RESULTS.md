# Q/KV preprocessing follow-up

Production scope: owned FP4 attention, gfx938, verified installed native binary,
BF16 Q[T,64,512] and KV[T,512], T1..64, MXFP8 KV block64/record528, contiguous,
no PCP or interleaved-query path. Unknown binaries and other layouts retain
native Core. The compile factor includes this implementation and native-build
admission, so changing the binary cannot reuse a previously captured split.

The native 64-head no-Q-norm kernel has private_segment_fixed_size428 bytes
and scratch load/store traffic in the supplied binary's disassembly. Splitting
Q RoPE from KV avoids that schedule. KV quantization/write uses the unchanged
native KV-only operator. Q preserves the verified binary's mixed FMA order,
including the unfused first even element in each 16-element lane group. This is
compiler dependent; the binary SHA gate is intentional.

Prototype:240additional cases across eight seeds,253231104Q elements, Q BF16
bytes and entire KV cache bytes exact. Alternate FMA/RTZ variants failed and
were rejected. Actual helper and owned class:322eager cases and115changed-input
CUDA Graph replays pass byte-exactly, including all eight capture buckets,
empty/partial/negative slots, signed zero, cancellation and magnitudes0..100.
Inputs remain unchanged. Isolated graph M48 native104.16us -> owned14.15us;
M1 10.07us ->8.19us. This is not yet a serving-performance claim.

CPU admission/fallback checks pass. The broader packer test file has two stale
default-policy failures reproduced on unchanged parent f4152cc; both reports
are retained. No precision gate is relaxed. The additional TPOT SLA is frozen
before serving measurement against accepted Post: matched ceilings
28.74440218/27.64277917ms, aggregate28.19359067ms. All previous gates remain.

HumanEval32 and independent formal/profile measurements are pending.
Raw evidence: `/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/qkv_split_*`.
