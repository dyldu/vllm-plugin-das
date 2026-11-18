# Exact draft Q RoPE and final Post schedules

The explicitly admitted native IPC-LL draft path uses private, source-pinned
native constructor/forward namespaces to select an owned attention subclass
and the already verified H5120 Post for the final direct call. Core globals,
parameter names, adapter loading/enumeration and native context precompute
remain unchanged. Default AG/RS construction is unchanged.

112 focused CPU checks pass. All eight capture buckets and both actual padded
584-byte cache layouts pass 644 target and 644 draft eager records plus 460 dynamic
graph replays, with exact Q/cache/page bytes and immutable inputs. The actual
private/native final forward body, including native hc_collapse, passes48
changed-input cases in eager and captured replay. Its isolated M48 fixture
moves 212.95us to 24.25us; draft Q/KV moves 104.90us to 14.39..14.44us. These are
isolated timings. HumanEval passes 32/32 with identical baseline prompts/tasks,
zero errors and truncations. Both formal clients complete 64 requests with zero
failures, and the profiling workflow exits zero with eight traces.

**Not admitted for serving performance.** Formal repeat medians are
24.850544674 / 22.494046866 ms (mean 23.672295770 ms). The frozen matched
ceilings are 23.328083424 / 23.021122288 ms and aggregate 23.174602856 ms,
so repeat 1 and the aggregate fail. No ceiling was changed and no result was
dropped. The accepted PR6 remains a4687f8 with adapter 759bb3f.

All eight traces prove the intended selection: 30 draft Q split and 30 native
KV-only calls, zero old no-Qnorm combined calls, 60 owned Post calls per ten
actual draft forwards; the 30 native context Qnorm calls remain. Target
400 Q split / 400 KV-only / 830 Post calls are retained. Profile phase
medians are target 84.537..84.691 ms and draft/sampling 6.471..6.596 ms.
These phase improvements do not override the formal TPOT rejection.

A context-KV-only shortcut was **rejected and removed**: a three-layer
precompute graph test produced one different FP8 cache byte atB64/T64,
layer0/replay4. The saved input independently reproduces native Qnorm=true
byte155 versus KV-only byte156 at[1,31,200] in eager mode. The original native
context method remains inherited. No tolerance was relaxed. Separately, the
accepted no-Qnorm target/draft path matches on this counterexample and10,000
additional stress cases acrossB32/B64, magnitudes.001/.25/1/8/100, changing
positions/cos-sin/slots:327,680,000KV elements, zero byte differences.

Raw probes, failed context prototype/input and logs are retained under
`/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/`:
`dspark_schedule_{target,draft}_b{32,64}_v1.json`,
`dspark_schedule_forward_v3.json`, `dspark_schedule_cpu_v3.log`,
`dspark_schedule_production_v2.json.failure.pt`,
`dspark_context_diagnosis_v1.log`, `qkv_kv_rounding_stress.{json,log}` and
`experiments/context_kvonly_rejected/`.
