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
24 cases and nine changing-input replays, preserving unused token-buffer rows;
42 unsupported-mode checks returned without writes. Existing DSpark CPU
regressions passed 65 tests.

For M=8, three graph timing samples gave median 683.813 us native versus
339.472 us fused at tile 2048. This isolated, cache-hot measurement does not
establish an end-to-end speedup. The experiment replacing the native BF16 GEMM
with MFMA tiles failed constructed rounding ties and is excluded from this PR.

Full-model validation for this increment is pending: real-prefill HumanEval
164 >=156, first 32 all correct, zero errors/missing/truncated outputs; separate
fake-zero-KV performance with five literal unconditional acceptance rates of
0.6, two repeats and all-eight-rank traces. Fixed serving parameters and frozen
TPOT limits remain unchanged. Revert this increment if those gates fail.
