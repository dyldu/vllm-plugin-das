# W4A8 DSpark final Post reuse

Enabled by default for the admitted deployment. It applies the already verified
Adapter H5120 Post kernel to the native draft's final Post call. Explicit
`{"hcu_dsv41_dspark_final_post":false}` restores the original Post. It requires the owned
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
Current W4A8 validation passed48 eager native/private forward cases and48
changed-input graph cases. Combined Plugin9bc97af/Adapter45a32f3 passed full
159/164, first32=32 and paired TPOT/actual eight-rank dispatch gates. These
results validate the complete Q/RoPE + MLA + Post composition and do not
attribute its end-to-end improvement to final Post alone. See
`w4a8-attention-dspark-reuse.md` for measured sources and limitations.
