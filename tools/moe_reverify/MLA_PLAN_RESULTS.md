# Fresh native MLA decode plans on gfx938

The source-pinned native FlashMLA consumer is retained. A fresh native metadata
object and GPU plan are produced for each call from current main/extra counts;
no shape-only or stale plan cache is used. The one-CTA Triton planner implements
the observed block64, F5, 64-partition recurrence and a runtime equivalent
uniform-block specialization. Admission is bounded to gfx938/64CUs, native padded H64,
1..64 rows, int32 contiguous counts, widths divisible by64 up to2048 and
cache blocks32/64. Unknown interface/binary/function identities, native H128 and other
unsupported layouts keep the fresh native path.

The installed native interface and binary SHA256 are pinned in source. The
schema and recurrence were checked against96actual native-plan probes;
upstream schema/algorithm references are:
https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/params.h and
https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu .
These references do not establish compatibility with other HCU wheels.

832direct plan/attention cases cover all8capture buckets, main widths
64/128/512/2048, extra0/512/2048, empty/full/ramp/alternating/random/reversed/
edge counts, actual padded320/64selector widths, and both32/64cache blocks.
Native active descriptors(first7fields), activation and all cumulative splits
match exactly. Field7 is native uninitialized padding and inactive descriptors
are not all specified; raw full-buffer equality is not claimed. Complete native
attention outputs are also BF16 byte-exact in eager and changed-count Graph.
An additional1,152complete bridge cases include caller inverse RoPE, all8
capture buckets, H4/H8/H64/H128, contiguous/strided/misaligned layouts,
both attention modes and6magnitudes. Every comparison checks input immutability
and eager/dynamic Graph output bytes. All pass.49focused CPU checks pass.

M48/H64 complete bridge Graph time, same current inputs and native attention:
SWA123.400->106.876us; combined193.756->176.369us. These microbenchmarks do not
replace serving SLA. Source9bee65c, unit outputmla_plan_gpu_v4.json/log;
earlier v1/v3 compile errors are preserved and corrected, v2 serial plan passed
before the runtime uniform specialization was added.

Serving admission completed; frozen2026-11-19 07:31:54 CST against the accepted
21.906416365/21.540173914ms repeats, with2% matched ceilings
22.344544692/21.970977393ms and aggregate22.157761042ms. All previous SLA and
HE32 same-task/prompt32/32, zero errors/truncations remain mandatory.

HE32 passes32/32 on identical baseline tasks/prompts, zero errors/truncations.
Formal medians 21.606007875053/21.098323208537ms, aggregate21.352165541795ms;
both repeats complete64requests with0failures. Every frozen SLA gate passes.
formal_1/formal_2/profile/workflow exits all0. Eight rank traces each show
400target fresh-plan calls and3per actual draft forward, zero native-plan
generator calls in those phases; Q/KV/Post/selector/mask-RoPE counts also pass.
Target phase 80.604158..80.765512ms, draft/sampling 6.233600..6.335278ms;
median GPU idle 0.913591..0.951911% / 0.816667..0.841571%.
Aggregate is1.708441% below the previous accepted composition and42.318949%
below original. Two repeats are not a statistical confidence claim. Full
SGLang phase parity is still incomplete, with reference70.508/4.712ms on
BW1100/144GiB versus this BW1101/95.98GiB and incomplete reference geometry.

Formal run: mla_plan_v1_pair/20261119_074008_channel_fp8_uVC0HC .
Real precision run: mla_plan_v1_he32/20261119_073201/20261119_073809 .
Raw evidence and frozen SLA: mla_plan_v1_*, mla_plan_preflight_sla.json .
Reproduce GPU operator checks: tests/tools/probe_mla_native_plan.py --baseline
<accepted4678199>/hcu_adapter/kernels/mla_sparse_decode_dtk.py --output <json>.
Serving commands use pluginf83ff54 with adapter9bee65c, FULL_AND_PIECEWISE
and DSPARK_DRAFT_BACKEND=deepep_low_latency; all baseline request settings
remain unchanged. See PROTOCOL.md and reproduce_validated.py in raw artifacts.
