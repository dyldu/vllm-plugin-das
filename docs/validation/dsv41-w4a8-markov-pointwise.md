# W4A8 greedy DSpark pointwise sampling

The native greedy draft loop materializes a BF16 sum, launches a vocabulary
argmax, then gathers the next Markov embedding at each of five positions.
The owned W4A8 model now fuses those pointwise operations. Every Markov bias
projection still calls the model's original `markov_bias`; its GEMM, weights,
dtype and logits processor remain native.

Admission requires the existing verified TP1/DP8/EP8 W4A8 draft configuration,
gfx938/64 CU, V=129280/R=256, five positions, BF16 contiguous Markov weights,
and the fingerprinted native greedy loop. Stochastic sampling, watermarks,
reduced vocabulary, confidence/acceptance heads, output transforms, autocast
and unknown implementations retain the native loop before custom writes.
Scratch belongs to the invocation/graph pool; no global GPU buffers or new
communicators are allocated. No additional launch flag is required.

On nmz18 in `duyl-033-w4a8-1119`, operator validation passed 147 special-value
pointwise cases, 84 real-checkpoint five-step cases and 27 graph replays with
changing inputs. IDs and gathered embeddings were exact, including NaN,
infinity, overflow and first-index ties. The actual native DSpark entry passed
48 cases and 18 changing-input replays, covering native int32 and int64 anchors
without casting and preserving unused token-buffer rows; 84 unsupported-mode
checks returned without writes. Existing DSpark CPU
regressions passed 65 tests.

For M=8, three graph timing samples gave median 683.813 us native versus
339.472 us fused at tile 2048. This isolated, cache-hot measurement does not
establish an end-to-end speedup. The experiment replacing the native BF16 GEMM
with MFMA tiles failed constructed rounding ties and is excluded from this PR.

The tested composition is Plugin `1fd0242` and Adapter `ec4dc86`, both clean
worktrees. Full real-prefill/rejection HumanEval passes 160/164 (97.56%), first
32 all correct, zero errors/missing/truncated outputs. Performance separately
uses fake zero KV, five literal unconditional acceptance rates of 0.6, and the
unchanged serving parameters. Both unprofiled repeats complete 64/64 requests.
TPOT medians are 18.1226/17.3686 ms and p99 values 21.1876/21.9254 ms. The frozen
regression limits pass against the previous accepted composition. Mean of
medians decreases from 17.8827 to 17.7456 ms (0.77%); the separate claim requiring
at least 2% improvement in each repeat does not pass.

All eight traces verify five partial-argmax and five argmax/embedding kernels
per draft span, with zero native ArgMaxOps. FULL target execution, target/draft
DeepEP, native BF16 bias GEMMs, mHC boundary, FP4 scoring/selection and WO_A
output ownership remain. Profiled DP0 complete-step median decreases from
71.7187 to 71.3833 ms; current target is 65.4921 ms and draft 5.9645 ms. This is
separate from unprofiled TPOT. SGLang's supplied trace is 65.0983 ms, leaving
6.2850 ms, with different hardware, batch/context and provider paths.

Evidence under `run/dsv4.1/033/w4a8_reverify_20261119`:
`nmz18/markov_pointwise_service_20261120_v2/acceptance.json` (driver exit 0),
`nmz18/markov_pointwise_step_reference_20261120.json`, and
`MARKOV_POINTWISE_RESULTS.json`. Documentation followups preserve the tested
runtime and test files exactly.
