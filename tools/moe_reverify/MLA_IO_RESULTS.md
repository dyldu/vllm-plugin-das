# Native MLA data movement: accepted on gfx938

The companion adapter passes full contiguous H64/H128 queries and sinks to
native FlashMLA without redundant zero/padding copies, guarded by gfx938,
1..64 rows and 256-byte alignment. Other layouts retain copies. A guarded
kernel combines BF16 empty-row masking with output writeback. Native attention,
KV bytes, selectors, fresh per-call plans and inverse-RoPE behavior are retained.

Production adapter03ef92e includes accepted selector759bb3f/Post/graph fixes.
45CPU checks,160all-BF16-bit mask cases and1,152full native consumer cases
pass exactly in eager and changed-input Graph replay, with immutable inputs.
Final-source M48 wholebridge SWA151.005->117.906us, combined219.794->185.072us.

HE32 passes32/32, identical baseline tasks/prompts, zero errors/truncations,
real prefill/acceptance. Formal medians22.223018363/21.874368957ms,
aggregate22.048693660ms. Both complete64requests,zero failures. Frozen
matched ceilings23.328083424/23.021122288ms, aggregate23.174602856ms and all
prior gates pass. Observed aggregate is2.956% below selector control and
40.437% below original; two repeats do not establish statistical confidence.

Separate profile/workflow exit0 with8traces. Each rank selects400target and
30draft mask/copy calls/10steps. Mean target8.217..8.361us, draft6.532..6.731us.
Existing target400Qsplit/400KVonly/830Post calls and native draft30no-Qnorm,
30context-Qnorm/50Post calls remain. Target phase82.611..82.812ms, draft with
sampling6.755..6.902ms; median idle0.881..0.922%/0.746..0.780%.

Measured plugin ec93021 (model code a4687f8). A later CRLF/EOF-only cleanup
preserves the full Python AST. Held draft-schedule207817d is excluded.
SGLang uses different hardware/context metadata; full step parity is unproven.

Reproduce with companion adapter tests/tools/probe_mla_decode_io.py, baseline
759bb3f bridge. Frozen settings and admission:
mla_io_preflight_sla.json, mla_io_v1_he32_summary.json,
mla_io_v1_perf_summary.json. Raw artifacts remain under
/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/mla_io_*.
