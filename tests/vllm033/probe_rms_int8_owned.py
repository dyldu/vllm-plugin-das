# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual owned norm/CT handoff with checkpoint weights and graphs."""
import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from hcu_adapter import rms_int8_scope as scope
from hcu_adapter.dsv41_int8_linear import HcuDSV41Int8LinearMethod
from hcu_adapter.kernels.rms_int8_cpp import module
from safetensors import safe_open
from vllm import _custom_ops as ops, ir
from vllm.config import CompilationConfig, CompilationMode, VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from hcu_vllm_das.rms_int8 import install_owned_decoder
from hcu_vllm_das.rms_norm import HcuRMSNorm


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def bench_pair(native, candidate):
    graphs = {}
    for key, fn in [('native', native), ('candidate', candidate)]:
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(8):
                fn()
        graphs[key] = graph
    samples = {key: [] for key in graphs}
    for repeat in range(7):
        for key in (('native', 'candidate') if repeat % 2 else ('candidate', 'native')):
            for _ in range(5):
                graphs[key].replay()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            for _ in range(30):
                graphs[key].replay()
            end.record()
            end.synchronize()
            samples[key].append(begin.elapsed_time(end) * 1000 / 240)
    return dict(samples_us=samples,
                medians_us={key: statistics.median(v) for key, v in samples.items()})


def edge_patterns(rows):
    x = torch.zeros((rows, 5120), dtype=torch.bfloat16, device='cuda')
    yield 'zero', x
    yield 'negative_zero', -x
    for value in (1.e-38, 1.e-12, 1., 1.e10, torch.finfo(torch.bfloat16).max):
        yield 'constant_' + str(value), torch.full_like(x, value)
        sparse = x.clone()
        sparse[:, 0], sparse[:, -1] = value, -value
        yield 'sparse_' + str(value), sparse
    for seed in (20261130, 20261131):
        torch.manual_seed(seed + rows)
        outlier = torch.randn_like(x)
        outlier[:, 1023], outlier[:, 4095] = 64., -64.
        yield 'outlier_' + str(seed), outlier
    for value in (float('inf'), -float('inf'), float('nan')):
        sparse = x.clone()
        sparse[:, 639] = value
        yield 'nonfinite_' + str(value), sparse


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, output = Path(args.model), Path(args.output)
    assert not output.exists()
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    result = dict(passed=False, completed=False, normalized_cases=[], consumer_cases=[],
                  graphs=[], fallback_cases=[], edge_cases=[], bf16_consumers=[],
                  timings=[], production_admitted=False)

    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')

    def load(key):
        with safe_open(str(root / index[key]), framework='pt', device='cpu') as f:
            return f.get_tensor(key).cuda().contiguous()

    save()
    try:
        from hcu_adapter.bridges.vllm_scaled_mm import install
        assert install(), 'Actual native CT -> LightOp bridge required'
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend='gloo',
                                     distributed_init_method='file://' + str(output.with_suffix('.dist')))
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, backend='gloo')
        fn = module().apply
        norm_keys = sorted(k for k in index if k.endswith(('attn_norm.weight', 'ffn_norm.weight')))
        assert len(norm_keys) == 86
        # Verify the final production kernel across every actual normalization
        # weight, including all graph capture row buckets and finite extremes.
        for key in norm_keys:
            w = load(key)
            for rows in (1, 4, 8, 16, 32, 40, 48, 64):
                for seed, magnitude in ((20261132, 0.), (20261133, 1.e-12),
                                        (20261134, 1.), (20261135, 1.e4)):
                    torch.manual_seed(seed + rows)
                    x = (torch.randn((rows, 5120), device='cuda') * magnitude).bfloat16()
                    before, wb = x.clone(), w.clone()
                    y = ir.ops.rms_norm(x, w, 1.e-20)
                    q, s, _ = ops.scaled_int8_quant(y)
                    actual = fn(x, w, 1.e-20)
                    check = all(exact(a, b) for a, b in zip(actual, (y, q, s)))
                    row = dict(weight=key, rows=rows, seed=seed, magnitude=magnitude,
                               exact=check, immutable=exact(x, before) and exact(w, wb))
                    result['normalized_cases'].append(row)
                    assert row['exact'] and row['immutable'], row
            save()
        for layer_id in (0, 2, 14, 39):
            w = load('layers.%d.attn_norm.weight' % layer_id)
            for rows in (1, 4, 8, 16, 32, 40, 48, 64):
                for label, x in edge_patterns(rows):
                    before, wb = x.clone(), w.clone()
                    y = ir.ops.rms_norm(x, w, 1.e-20)
                    q, s, _ = ops.scaled_int8_quant(y)
                    actual = fn(x, w, 1.e-20)
                    row = dict(layer=layer_id, rows=rows, pattern=label,
                               exact=all(exact(a, b) for a, b in zip(actual, (y, q, s))),
                               immutable=exact(x, before) and exact(w, wb))
                    result['edge_cases'].append(row)
                    assert row['exact'] and row['immutable'], row
            save()
        for layer_id in (0, 2, 14, 39):
            names = ['layers.%d.attn.%s' % (layer_id, k) for k in ('wq_a', 'wkv')]
            widths = [load(k + '.weight').shape[0] for k in names]
            projection = torch.nn.Module()
            projection.scheme = CompressedTensorsW8A8Int8('channel', False, True)
            projection.quant_method = HcuDSV41Int8LinearMethod(None)
            projection.input_size_per_partition = 5120
            projection.output_size_per_partition = sum(widths)
            projection.output_partition_sizes = widths
            with torch.device('cuda'):
                projection.scheme.create_weights(projection, widths, 5120, torch.bfloat16, default_weight_loader)
            projection.weight.copy_(torch.cat([load(k + '.weight') for k in names]))
            projection.weight_scale.copy_(torch.cat([load(k + '.weight_scale') for k in names]))
            projection.quant_method.process_weights_after_loading(projection)
            before_weight, before_scale = projection.weight.clone(), projection.weight_scale.clone()
            source = HcuRMSNorm(5120, 1.e-20, dtype=torch.bfloat16).cuda()
            source.weight.copy_(load('layers.%d.attn_norm.weight' % layer_id))
            decoder = SimpleNamespace(attn_norm=source, use_sequence_parallel=False,
                                      fuse_seam_norm=False, mono=None,
                                      attn=SimpleNamespace(fused_wqa_wkv=projection, prefix=str(layer_id),
                                                           _wqa_wkv_scale=None))
            compiled = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE))
            with set_current_vllm_config(compiled):
                assert not install_owned_decoder(decoder)
                assert decoder.attn_norm is source
            result['fallback_cases'].append(dict(layer=layer_id, label='Inductor_compilation', exact=True))
            assert install_owned_decoder(decoder), 'Actual producer admission must take the new path'
            assert decoder.attn_norm.weight is source.weight
            assert list(decoder.attn_norm.state_dict()) == ['weight']
            for rows in (1, 4, 8, 16, 32, 40, 48, 64):
                for bias_enabled in (False, True):
                    bias = torch.randn((sum(widths),), device='cuda', dtype=torch.bfloat16) if bias_enabled else None
                    x = torch.randn((rows, 5120), device='cuda', dtype=torch.bfloat16)
                    for magnitude in (0., 1.e-12, 1., 1.e4):
                        x.normal_().mul_(magnitude)
                        y = source(x)
                        expected = projection.scheme.apply_weights(projection, y, bias)
                        with scope.attention_norm_scope():
                            yy = decoder.attn_norm(x)
                            assert scope.lookup(yy) is not None
                            assert projection.quant_method._paired_input_supported(projection, yy, bias)
                            actual = projection.quant_method.apply(projection, yy, bias)
                            row = dict(layer=layer_id, rows=rows, bias=bias_enabled,
                                       magnitude=magnitude, output_exact=exact(actual, expected),
                                       normalized_exact=exact(yy, y))
                            result['consumer_cases'].append(row)
                            assert row['output_exact'] and row['normalized_exact'], row
                        assert scope.lookup(yy) is None
                    def candidate():
                        with scope.attention_norm_scope():
                            yy = decoder.attn_norm(x)
                            return projection.quant_method.apply(projection, yy, bias)
                    def native():
                        return projection.scheme.apply_weights(projection, source(x), bias)
                    if layer_id == 2 and not bias_enabled:
                        x.normal_()
                        result['timings'].append(dict(layer=layer_id, rows=rows,
                                                       **bench_pair(native, candidate)))
                    candidate()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_output = candidate()
                    for replay, magnitude in enumerate((0., 1.e-12, 1., 1.e4)):
                        x.normal_().mul_(magnitude)
                        graph.replay()
                        expected = projection.scheme.apply_weights(projection, source(x), bias)
                        row = dict(layer=layer_id, rows=rows, bias=bias_enabled, replay=replay,
                                   output_exact=exact(graph_output, expected))
                        result['graphs'].append(row)
                        assert row['output_exact'], row
                    del graph
                    assert exact(projection.weight, before_weight) and exact(projection.weight_scale, before_scale)
                save()
            # Checkpoint indexer.weights_proj is BF16 and has no INT8 scale.
            # It continues to consume normalized BF16 through the same native
            # BF16 linear, while QKV consumes the scoped INT8 codes/scales.
            bf16_key = 'layers.%d.attn.indexer.weights_proj.weight' % layer_id
            if bf16_key in index:
                assert bf16_key.removesuffix('.weight') + '.weight_scale' not in index
                weight = load(bf16_key)
                assert weight.dtype == torch.bfloat16
                for rows in (1, 4, 8, 16, 32, 40, 48, 64):
                    for magnitude in (0., 1.e-12, 1., 1.e4):
                        x = torch.randn((rows, 5120), device='cuda', dtype=torch.bfloat16) * magnitude
                        expected = torch.nn.functional.linear(source(x), weight)
                        with scope.attention_norm_scope():
                            y = decoder.attn_norm(x)
                            actual = torch.nn.functional.linear(y, weight)
                        row = dict(layer=layer_id, rows=rows, magnitude=magnitude,
                                   exact=exact(actual, expected))
                        result['bf16_consumers'].append(row)
                        assert row['exact'], row
            # Runtime policy/layout fallbacks must keep native normalization
            # without publishing quantized data into the owned call.
            for label, x in (
                ('rows65', torch.randn((65, 5120), device='cuda', dtype=torch.bfloat16)),
                ('strided_rows', torch.randn((16, 5120), device='cuda', dtype=torch.bfloat16)[::2]),
            ):
                with scope.attention_norm_scope():
                    y = decoder.attn_norm(x)
                    check = scope.lookup(y) is None and exact(y, source(x))
                    result['fallback_cases'].append(dict(layer=layer_id, label=label, exact=check))
                    assert check
            with ir.ops.rms_norm.set_priority(['native']):
                x = torch.randn((8, 5120), device='cuda', dtype=torch.bfloat16)
                with scope.attention_norm_scope():
                    y = decoder.attn_norm(x)
                    check = scope.lookup(y) is None and exact(y, source(x))
                    result['fallback_cases'].append(dict(layer=layer_id, label='changed_IR_provider', exact=check))
                    assert check
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with scope.attention_norm_scope():
                    x = torch.randn((8, 5120), device='cuda', dtype=torch.bfloat16)
                    y = decoder.attn_norm(x)
                    check = scope.lookup(y) is None and exact(y, source(x))
                    result['fallback_cases'].append(dict(layer=layer_id, label='autocast', exact=check))
                    assert check
            save()
            print('actual owned pair exact', layer_id, flush=True)
        assert len(result['normalized_cases']) == 2752
        assert len(result['edge_cases']) == 544
        assert len(result['consumer_cases']) == 256
        assert len(result['graphs']) == 256
        assert len(result['bf16_consumers']) == 64
        assert len(result['timings']) == 8
        assert len(result['fallback_cases']) == 20
        result['passed'] = True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result['error'] = repr(exc)
    result['completed'] = True
    save()
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    from vllm.kernels import vllm_c
    cfg = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with set_current_vllm_config(cfg), ir.ops.rms_norm.set_priority(['vllm_c', 'native']):
        main()
