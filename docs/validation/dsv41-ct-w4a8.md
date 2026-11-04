# CT W4A8 HCU migration on vLLM 0.33

## Scope and source anchors

Plugin starts at `a6b8900`; Core is read-only `368701db06`, Adapter read-only
`3257a934`. Source contracts are `95d04bc7`, `3f79b3c6`, `a340111f` from the
0.32 integration. The work is split into packed loading and execution routing.
No Core/private oracle/global factory rebinding or legacy package import is
allowed. Existing ordinary CT methods remain official owners.

Native `compressed_tensors_moe_w4a8_int8.py` is CPU-only and allocates unpacked
INT8 q values, whereas the HCU checkpoint contract keeps INT32-packed CT weights
until in-place HIPC conversion. Native RoutedExperts already loads packed expert
weights, channel scales and shape metadata through its generic public loader.

Exact byte conversion: CT stores offset-8 low-first INT4; convert each uint8 view
byte by nibble swap then XOR 0x88, returning an INT8 storage view. HIPC represents
q as q*16, so BOTH derived FP32 channel scales are s_ct/16. Original checkpoint
scales must remain unchanged. SlimQuant scales use a different domain.

Core DeepEP HT post-dispatch quantization hardcodes per_act_token_quant=False;
Core LL lacks the old use_int8_dispatch constructor flag. The installed vendor
LL ABI instead requires real topk_weight and quant_type=1, quant_group_size=0.
Those weights feed dispatch handle metadata, so synthesizing ones is not an
acceptable ABI workaround. Plugin-owned prepare/finalize must preserve the
public modular interface and actual INT8 dispatch, or explicitly reject the ABI.

## Failure modes before implementation

| ID | Failure | Planned verification |
| --- | --- | --- |
| W01 | int32.to(int8) truncates packed values; sign/order is wrong. | 200 seeds and all nibble values vs independent logical pack reference. |
| W02 | Conversion/packer clones model weights or changes shape/storage. | Native packed loading, exact parameter shape/dtype/storage checks, rejecting fake cloning packers. |
| W03 | Scale compensation omitted, applied twice, or checkpoint scales mutated. | Exact derived scale = original/16, storage guard and original bytes unchanged for both projections. |
| W04 | Gate/up shards, down K, expert IDs or weight_shape load incorrectly. | Real generic RoutedExperts weight_loader with CPU EP-map fixture; distinct shards and metadata. |
| W05 | Repacking/hot reload corrupts already packed storage. | Same-owner idempotence; mutation/reload and failed packing remain rejected. |
| W06 | HIPC dimensions/scales are incompatible. | Reject wrong H/I alignment, rank/dtype/shape/scale/device before destructive conversion. |
| W07 | LL N32 view copies or misinterprets storage. | Exact N32 dimensions and owner pointer, reject fake clone/malformed view. |
| W08 | New quant name never selected, or captures other models/schemes. | Actual Core selector and independent registration; HCU/V41/CT schema only, native ignores and other paths preserved. |
| W09 | CPU CT constructor/oracle or legacy runtime is reused on HCU. | Owned method/kernel construction and source/symbol identity checks. |
| W10 | HT becomes per-tensor or LL becomes BF16/FP8 wire dispatch. | Owned public prepare calls, exact token scale shape/INT8 dtype and vendor LL quant_type=1/real routing weights. |
| W11 | Dispatch/combine router weights applied twice, handles/hooks lost. | CPU communication doubles with actual modular kernel, hook order, handle and weight assertions. |
| W12 | Int4 scale/activation/permute/metadata changes silently alter math. | Full tiny MoE pipeline vs independent CPU reference; provider GEMM/pack/communication doubled and disclosed. |
| W13 | Capacity, EP membership or graph bucket exceeds allocation. | Invariant checks without DP whitelist; reject unsupported TP/PP/DCP/LoRA/EPLB/Elastic EP, PCP>1 non-eager. |
| W14 | Provider import/ABI failure leaves partially packed reusable weights. | Resolve providers before packing, sticky packing failure, no silent backend fallback. |
| W15 | Zero local tokens skip required collectives or launch empty kernels. | Keep prepare/finalize calls, skip only local expert compute, fixed output and metadata. |
| W16 | CPU source checks are reported as device/model acceptance. | Report sources/SHAs/metadata/no CUDA init; no GPU, Graph, model accuracy or performance claims. |

Replays are written before the corresponding production subsystem. They must
reload production modules and leave repeatable scripts plus JSON/log artifacts.
No unit cases or post-implementation assertions will be added. Existing package
metadata expectations may only be synchronized before adding a new public entry.

## Verification results

Packed-weight stage: the pre-implementation replay failed 13/13 because the
production module was absent. After implementation, the same cases passed
**13/13** (`weights-green-final/`), including 200 seeds each for independent
byte conversion and scale-domain checks. The first post-implementation run
passed 8/13; its CPU native-loader fixture lacked `tp_shard_with_padding=False`.
That required TP1 fixture field was added without new cases or weakened
assertions; the failed run remains in `weights-green/`.

Run with the prepared Python 3.12 environment:

```bash
VLLM_TEST_PYTHON=/path/to/python3.12 \
  bash tests/vllm033/run_ct_w4a8_cpu.sh /path/to/output weights
```

Evidence root is `tests/vllm033/.cache/ct-w4a8/`. JSON records revisions, source
hashes, each case and `cuda_initialized=false`; stdout is retained alongside it.
The HIPC packer/view are CPU doubles. This proves raw CT conversion, ownership
and loader contracts, not the vendor's native permutation or GPU numerics.
The module is not yet selected by a model: public routing and execution follow
in the next functional slice. No Core or Adapter source changed. No GPU,
service, container creation, installation, kernel compilation or remote Git
operation is part of this task.
