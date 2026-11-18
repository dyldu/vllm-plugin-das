# Exact draft Q RoPE and final Post schedules

The explicitly admitted native IPC-LL draft path uses private, source-pinned
native constructor/forward namespaces to select an owned attention subclass
and the already verified H5120 Post for the final direct call. Core globals,
parameter names, adapter loading/enumeration and native context precompute
remain unchanged. Default AG/RS construction is unchanged.

112 focused CPU checks pass. All eight capture buckets and both actual padded
584-byte cache layouts pass644target and644draft eager records plus460dynamic
graph replays, with exact Q/cache/page bytes and immutable inputs. The actual
private/native final forward body, including native hc_collapse, passes48
changed-input cases in eager and captured replay. Its isolated M48 fixture
moves212.95us to24.25us; draft Q/KV moves104.90us to14.39..14.44us. These are
isolated timings; serving HE32 and paired/profile results are pending.

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
