# Real 4K/128K input: matched performance and dispatch validation

Real model-computed KV and real DSpark acceptance on nmz18, TP1/DP8/EP8. Client concurrency 1, 256 output tokens, identical deterministic technical-document inputs, prefix caching off. Both services omit max-model-len (effective model capacity 1048576), use seq64/budget256 and the same graph settings. At each input length, a separate profiled warm-up precedes two unprofiled serial formal requests. Only formal requests contribute to the following arithmetic means.

| Input | Metric | Baseline | Candidate | Reduction |
|---|---|---:|---:|---:|
| 4096 | TTFT (s) | 6.125 | 6.221 | -1.58% |
| 4096 | TPOT (ms) | 10.483 | 9.157 | 12.65% |
| 4096 | E2E (s) | 8.798 | 8.557 | 2.74% |
| 131072 | TTFT (s) | 197.829 | 197.380 | 0.23% |
| 131072 | TPOT (ms) | 15.465 | 10.277 | 33.55% |
| 131072 | E2E (s) | 201.773 | 200.001 | 0.88% |

TPOT includes real acceptance, routing, scheduling and streaming effects. The two repeats have varying output hashes even within the same source snapshot; these are workload observations, not a guarantee for other inputs or concurrency. The earlier 35.23% C64 dummy-KV/synthetic-acceptance result is a separate experiment.

## Execution paths

The final profiled prefill chunks at 4K and 128K both use query M=256 before the native SWA replay cut and M=128 in later layers. The native model writes all KV through layer 20, then computes the rest of that layer and layers 21–39 on each request’s last SWA128 rows. Input length changes the number of chunks (16 versus 512 at a 256-token query budget), not the projection GEMMs into M=131072.

The indexer’s actual end-of-prompt key extents grow from 2048/4096 (compression ratios 2/1) to 65536/131072. FP4 scoring, key gather, TopK and candidate pruning therefore change in work and storage. Sparse MLA keeps model Top512 plus SWA128, with native padding; it does not attend densely to all 128K keys per query. Prefill gathers BF16 keys; decode uses native paged FP8 sparse attention.

Candidate pruning can retain 2048 blocks of 8 positions: the 4K candidates fit, while 128K may require a subset. Very early chunks also have the native short-context bypass when the entire compressed candidate set fits Top512. These are model conditions, not a 32K optimization whitelist.

RMSNorm+INT8 fusion, token-major WO_A and W4A8 MoE are observed at both lengths on both sources. The actual six-query target decode spans execute all 80 masked MoE GEMMs through one hipGraphLaunch, confirming FULL target graphs remain active. MoE routing and real DSpark acceptance may change with the content even when operator dimensions stay the same.

The candidate FP4 prefill passes explicit valid-prefix lengths to native TopK. This retains native selection at long widths instead of reaching the generic unbounded selector’s 32768-column guard. Main index TopK is native at both measured lengths; remaining Torch TopK calls and dimensions are retained in the evidence (candidate-block selection remains a separate path).

Profiled warm-up target spans (not complete DSpark/client steps):

| Input | Phase | Baseline target median (ms) | Candidate target median (ms) |
|---|---|---:|---:|
| 4096 | prefill | 433.622 | 428.350 |
| 4096 | decode | 40.122 | 34.743 |
| 131072 | prefill | 436.944 | 422.502 |
| 131072 | decode | 40.499 | 34.237 |

Kernel duration sums can overlap and are not additive wall-time components. For the candidate, 4K →128K prefill KV-gather kernel sums increase from 0.947 to 12.705 ms per target, while the sparse MLA attention sums change from 7.454 to 7.666 ms. This directly distinguishes longer KV gathering from a growth to dense full-context attention.

Formal decode wall time divided by reported draft steps provides a separate service-cycle check (stream boundaries/final partial output prevent treating it as an exact GPU timer):

| Input | Baseline ms/reported step | Candidate ms/reported step | Reduction |
|---|---:|---:|---:|
| 4096 | 43.835 | 34.861 | 20.47% |
| 131072 | 42.415 | 35.655 | 15.94% |

TPOT also depends on real acceptance and generated content, so its percentage is not a pure kernel acceleration. Prefill TTFT shows no clear gain in this pair: the 4K mean is about 1.6% slower and the 128K mean about 0.2% faster. With only two serial repeats and automatic DP routing, small differences do not establish a steady-state improvement. The 128K total-time reduction is about 0.9%, despite its larger TPOT reduction.

## Correctness, memory and scope

Full HumanEval: 159/164 (96.95%), with no API errors, missing outputs, truncations or execution timeouts. All prompts match the previous and frozen references. Changed pass/fail outcomes are retained in the evidence; HumanEval is not a 128K quality evaluation. Affected CPU validation: 93 cases pass.

The native 2048-query oracle covers KV extents 131073/65539 at 1M capacity. All score bytes, selected score ranks, valid counts, unique IDs, padding and short-row order match; the probe asserts native wide-prefix selection rather than silently accepting fallback. The original logits require 8 GiB, with 8.074 GiB complete-operation incremental allocator peak. The new 64/512 MiB per-logits budgets produce complete-operation incremental peaks 141.00/1038.76 MiB. The native selector’s mapped scores coexist with logits, so this budget does not cap the entire operation or device. Persistent inputs/oracle tensors are excluded from the incremental probe peak.

This comparison isolates the latest FP4 live-length staging/scoring/TopK changes plus the prefill budget/native-selection fix against the runnable memory-only baseline. Earlier RMS/WO/MoE optimization code exists on both sides. It does not measure the cumulative gain against the original unoptimized repository.

## Sources and reproduction

Baseline plugin `afc458876a4928cd4e994b6d1aa4e0f472ecb031`, adapter `e73343750e0c124e015cb11de1f5f4792d1081a3`; candidate functional plugin `3a310b4c40f33cd0178c93f30533c6469eb52d7c`, adapter `7b7b1dd34997419b2580d04dd49bebe145895d42`. Native Core Python manifests and package versions agree; effective environments differ only in source overlays. User scripts `laibao/test_d.sh` and `laibao/prof_d.sh` remain byte-identical.

Raw commands, inputs/hashes, formal SSE timings/counters, profile hashes/shapes, GPU oracle, accuracy outcomes and allocator/sample measurement limits are retained under `/public2/home/duyunlong/run/dsv4.1/033/real_128k_compare_20261120`. Each role’s timestamped `service` directory contains `serve_cmd.txt`, `effective_env.txt` and request records; candidate `eval_cmd.txt` is the full accuracy command. `bench.py` is the shared real-input client. The aggregate [evidence JSON](dsv41-real-128k-evidence.json) is published with this report.
