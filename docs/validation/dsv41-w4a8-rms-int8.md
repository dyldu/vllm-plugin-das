# W4A8 owned attention RMS and dynamic INT8

The target attention norm previously wrote BF16 values, then its native CT
QKV projection reread them to produce dynamic INT8 codes/scales. The paired
Plugin/Adapter change computes both in one kernel. It retains the native BF16
storage boundary, weight loading and selected GEMM. The BF16 normalized value
is still available to the unchanged BF16 indexer consumer. A call-scoped,
tensor-identity handoff expires at decoder exit, including exceptions.

Admission is the owned target attention norm, M=1..64, H=5120, epsilon=1e-20,
contiguous aligned BF16 inputs, gfx938/64 CU, dynamic symmetric per-channel CT
INT8, native RMS priority and CompilationMode.NONE. CUDA graphs still capture
the normal invocation. Autocast, gradient mode, residual/fused seam, sequence
parallel and unsupported consumers retain their existing paths. Draft norms
are unchanged. No new serving flag or environment variable is required.

The tested clean sources are Plugin `f1d530c` and Adapter `56206d3`. Validation
covers 2,752 real-weight normalized cases, 544 edge cases, 256 native QKV
consumers, 256 changing-input graph replays, 64 BF16 consumers and 20 fallbacks.
BF16/INT8/FP32 outputs are byte exact, including overflow and NaN behavior.
The paired CPU suite passes 53 tests; Adapter L0 passes with 30 existing
warnings. At M48 the complete norm/quant/native-QKV consumer median changes
24.2127 to 20.4720 us (15.45%).

Real-prefill/rejection HumanEval at the original client batch16 passes
159/164 (96.95%), first32 all correct. A profiler-only repeat passes160/164.
All eight ranks show 40 target fused RMS/INT8 kernels per complete target
span, with native RMS41 and INT8 quant170; draft counters remain unchanged.

A supplementary client batch64 run initially scored83/164. This was a review
failure: 78 of its81 failures were execution timeouts. EvalScope1.9.1 uses
`eval_batch_size` for both inference and review; deprecated `judge_worker_num`
did not independently limit its64 review workers. Reusing the **same164 saved
predictions**, the official cached-review pipeline with four review workers
passes **159/164, first32=32, zero timeouts**. The per-program timeout stays
four seconds. No model requests or GPU work run during review. Prediction
SHA256 is `dfa2b4965c1e6435acf4c886edae137c7aad718309a1123a7c063e0410c43b66`.
Extracted programs and problem metadata/tests remain identical to the original
batch64 review; all164 pass/fail outcomes match the same candidate's original
batch16 result. Original83/164 artifacts remain.

`tools/moe_reverify/rejudge_evalscope_cached.py` reproduces this review-only
repair for the saved164-answer batch64 run, in a fresh output directory:

```bash
python tools/moe_reverify/rejudge_evalscope_cached.py \
  --source /path/to/original/evalscope/run \
  --output /path/to/new/review-only-run --workers 4
```

The tool forbids prediction calls and verifies unchanged prediction bytes.
For future high-concurrency checks, generate at the intended request
concurrency, then review the complete saved predictions separately; changing
the overall eval batch to4 before generation would change the workload.

Fixed performance remains fake zero KV, five literal unconditional0.6 rates,
4096/128 input/output,64 requests/concurrency, TP1/DP8/EP8 and the frozen server
capacity settings. Two unprofiled repeats complete64/64 with zero failures:
TPOT medians17.8993/17.3512 ms and p99s20.9428/21.5518 ms. Frozen regression
limits pass. Mean median17.6252 ms versus17.7456 ms is a0.68% difference;
the separate >=2%-gain-in-each-repeat criterion does not pass. Profiled complete
step is71.0260 ms versus71.3833 ms. SGLang65.0983 ms is a stage reference with
different context, batch, hardware and providers; parity remains unfinished.

Evidence under `run/dsv4.1/033/w4a8_reverify_20261119`:
`RMS_INT8_REJUDGED_ACCEPTANCE.json`, `RMS_INT8_UNIT_RESULTS.json`,
`nmz18/rms_int8_owned_service_20261120/acceptance.json`,
`nmz18/rms_int8_owned_accuracy_dispatch_20261120_v2/acceptance.json`,
`nmz18/batch64_evalscope_rejudge_20261120/rejudge_summary.json` and
`nmz18/rms_int8_owned_step_reference_20261120.json`.
This validates the fixed workload and guarded contracts, not a complete
concurrency/context sweep or universal dispatch of every optimization.
