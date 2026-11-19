# W4A8 DSpark final Post reuse

Experimental, disabled by default. Explicit additional config
`{"hcu_dsv41_dspark_final_post":true}` applies the already verified Adapter
H5120 Post kernel to the native draft's final Post call. It requires the owned
W4A8 target scope, TP1/DP8/EP8, HC4 and DSpark5. The private AG/RS construction
lease remains validated. Unsupported tensor layouts, autocast or precision
settings retain the native Post through the Adapter's existing selector.

This reuses the private-namespace principle from207817d/2820632. Only the native
outer constructor's model-class global and that model's final Post global are
substituted. Both native function sources are pinned and their identities are
rechecked before construction. Decoder initialization, forward arithmetic,
weight enumeration/loading, scale aliases and the final native hc_collapse
remain inherited. Core globals are unchanged.

The current native function source hashes match the historical validated
schedule. Earlier FP8 service figures do not validate this W4A8 candidate.
This worktree retains the activation and target Q/RoPE candidates from its
base; the optional final Post is the additional change.

GPU fixture command, on an authorized idle GPU with the pinned sources:

```bash
python tools/moe_reverify/verify_w4a8_dspark_post.py --output /path/to/result.json
```

The fixture invokes the actual native/private three-layer draft forward and
native hc_collapse, across every frozen capture bucket, six input magnitudes,
eager execution and changing-input graphs. It checks BF16 bytes, finite outputs
and input immutability, then measures both schedules with CUDA graphs.
GPU fixture and W4A8 HE32/full164/paired TPOT gates are still required before
promotion or a serving speedup claim.
