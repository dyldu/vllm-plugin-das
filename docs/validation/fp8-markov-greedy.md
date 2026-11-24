# Channel-FP8 draft Markov sampling

The Channel-FP8 draft uses the same BF16 Markov embedding, bias projection,
and logits ABI as the previously admitted native-LL draft. Enable the existing
guarded pointwise producer for this owned quantization route as well. The
native bias GEMM remains unchanged. Each position fuses BF16 logits addition
with partial argmax, then finishes first-index argmax and the next embedding
gather. No serving option is added.

Admission still checks the actual model/speculator methods, BF16 head, logits
processor, device, topology, token buffer layout, and greedy sampling contract.
Probabilistic sampling, watermarking, vocabulary remapping, confidence heads,
unsupported shapes, and modified methods retain their native path. Batch and
number of speculative positions vary at runtime; the existing bounded admission
and native fallback remain in force.

## Operator validation on nmz10, 2026-11-24

`tests/vllm033/probe_fp8_markov_owned.py` loads the actual
`mtp.2.markov_head.embed.weight` and `head.weight` from
`DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8`. The independent oracle calls the
installed native sequential loop without the optional model sampling hook.
It exercises the owned model hook and requires every output token and untouched
tail row to match exactly.

- 256 exact cases: batch 1/2/7/9/17/32/63/64, draft positions 1/3/5/7,
  int32/int64 anchors, logits magnitudes 0/0.01/1/8.
- 36 changing CUDA Graph replays, including batch 1/7/64.
- 448 unsupported sampling/layout checks leave the destination untouched;
  autocast also retains the native path.
- 15 maintained pointwise tests cover non-aligned vocabularies, first-index
  ties, NaNs/infinities, BF16 addition, destination guards, and three changing
  graph replays per test.

Five-position graph timings, seven alternating native/candidate measurements,
no profiler (microseconds; the native bias projection is included):

| Batch | Native | Fused | Time reduction |
|---:|---:|---:|---:|
| 1 | 621.529 | 327.685 | 47.28% |
| 7 | 701.169 | 356.365 | 49.18% |
| 17 | 1054.449 | 654.759 | 37.91% |
| 32 | 930.519 | 490.145 | 47.33% |
| 64 | 1331.644 | 800.584 | 39.88% |

These are operator measurements. Whole-model accuracy, actual trace invocation,
and unprofiled serving performance are separately required for promotion.

Run in the HCU image with the adapter/plugin source overlay:

```bash
python3 tests/vllm033/probe_fp8_markov_owned.py \
  --model /models/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 \
  --output /tmp/fp8-markov-owned.json
python3 -m pytest -q tests/vllm033/test_dspark_greedy_pointwise.py
```
