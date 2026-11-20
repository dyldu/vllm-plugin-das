# FP4 prefill logits budget and 128K validation

Core metadata chunks query rows using actual compressed KV extents and `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (default 512 MiB). The plugin previously passed maximum model capacity to the native FP4 scorer, which allocates FP32 `[query_rows, width]`; that mismatch defeats Core's byte budget.

Use the metadata's CPU `max_seqlen_k`, rounded to a growing power-of-two pitch (also satisfying native four-column alignment), and subdivide query rows by `budget // (pitch * 4)`. These runtime allocation buckets keep selector JIT variants reusable as successive prefill chunks grow; the pitch is less than twice the actual extent, except for the native minimum of four columns. Query offsets, causal bounds and candidate-bank slices follow every tile. Release each logits tensor before allocating the next and release gathered KV before the next chunk. No device-to-host length read is added. The decode branch is unchanged, including device valid-prefix selection and the tail mask for unsupported selection fallbacks.

There is no fixed 32K/128K context, sequence-count, scheduler-budget or 512-MiB equality check. The budget is configurable. Model and native ABI requirements still apply.

## Native memory and correctness

On nmz18/gfx938, 2048 query rows, KV extents 131073/65539, model capacity 1048576, physical page records32: the old native call allocates 8 GiB of logits; the complete old operation's incremental allocator peak is 8.074 GiB.

| Per-logits budget | Query tiles | Complete operation incremental allocator peak |
|---|---:|---:|
| 64 MiB | 32 | 77.48 MiB |
| 512 MiB | 4 | 531.55 MiB |

All score bytes match the independently retained original native output, and selected score ranks, valid counts, unique in-range IDs, padding and short-row ordering match. The 64/512-MiB tests include non-power-of-two extents and query tiles crossing request boundaries. CPU validation: 93 affected cases pass, including release-before-next-allocation and candidate/global-row slicing checks. The peak includes gather and selector allocations; a 512-MiB logits budget does not cap the whole operation or device at 512 MiB.

An initial exact-extent implementation passed the native oracle and full HumanEval but compiled 215 new TopK score/output variants during its first long request. That owned trial was deliberately stopped, rather than counted as a completed long-input run. The final implementation reuses one selector pitch across each growing power-of-two interval; a CPU check covers successive 128-token growth steps across the 32K-to-64K interval. No paired cold-start latency claim is made.

## Real KV

Full HumanEval: 159/164 (96.95%), previous published real-KV run 158/164; frozen reference 159/164. All prompt hashes agree. Changed outcomes vs previous run: `[{"task_id": "HumanEval/93", "before": false, "after": true}]`; vs frozen reference: `[]`. Zero API errors, missing outputs, truncations or execution timeouts. This meets the existing minimum 95% gate; it is not proof of identical accuracy or 128K quality.

With `--max-model-len` omitted (effective capacity 1048576), a real 131072-token prompt completes 8 output tokens in 199.30s with no OOM. During that phase the largest per-device 100-ms VRAM sample is 91706.92 MiB. Sampled peaks can miss shorter transients; the native operation's allocator peak is separately measured above.

## Separate matched decode experiment

Before the prefill-only patch, published plugin `bc7cf5f` / adapter `a682367` were compared with the runnable memory-only baseline `afc4588` / `e733437`. Both omit max-model-len, use actual scheduling length 131072, output 128, global concurrency 64, seq64/budget256, TP1/DP8/EP8 and identical graph settings/native Core/packages. Only source overlays and profiler output paths differ. This isolates the latest live-length FP4 decode changes; earlier RMS/WO/MoE optimizations are present on both sides.

Three formal repeats average median TPOT 27.5192 -> 17.8237 ms (35.23% reduction). Output throughput 838.63 -> 970.40 tokens/s. The prior 4K pair reduced TPOT by about 35.55%; the 128K gain remains similar. Each formal pass completes 64/64 requests, with zero failures or OOMs.

Both sides actually execute FULL target graphs on all 8 ranks. Median complete target+draft GPU step across ranks: 128.030 -> 76.909 ms. Rank0 kernel duration sums per complete step: staging 6.1993 -> 2.2185 ms, native FP4 logits 6.3228 -> 1.1330 ms, torch multiblock TopK 23.4647 -> 0.6553 ms. Kernel sums can overlap and are not additive wall-time components.

This decode benchmark uses dummy KV and synthetic acceptance `[0.6]*5`; it does not measure model-computed prefill or real acceptance. The real-KV tests above are separate. The decode comparison predates this prefill-only patch, whose decode branch is unchanged. User launch scripts `laibao/test_d.sh` and `laibao/prof_d.sh` were not edited.

## Reproduction and evidence

Functional plugin source: `42e29f33dd84fd00c4db16943d6b6e8bb78272f7`; adapter `7b7b1dd34997419b2580d04dd49bebe145895d42`. Full commands, environment, artifact paths, score outcomes and measurements are in [the evidence JSON](dsv41-prefill-budget-evidence.json).

Native probe: `tests/vllm033/probe_fp4_prefill_budget.py --output result.json`, with the recorded source PYTHONPATH overlay and a single idle HCU. Full service/eval arguments are preserved in `/public2/home/duyunlong/run/dsv4.1/033/prefill_budget_fix_20261120/accuracy/he164/20261120_183843/serve_cmd.txt` and `eval_cmd.txt`; decode commands/traces are under `/public2/home/duyunlong/run/dsv4.1/033/context_128k_compare_20261120`. Raw whole-device samples: `/public2/home/duyunlong/run/dsv4.1/033/prefill_budget_fix_20261120/vram_100ms.jsonl`.

The screenshot's original 8.5GB failed allocation was not available, and its upstream PR 426 URL was not supplied. This validation measures a separate native 8GiB logits allocation; it does not claim to reproduce that original failure.
