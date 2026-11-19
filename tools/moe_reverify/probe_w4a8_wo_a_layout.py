#!/usr/bin/env python3
"""Verify token-major WO_A against real W4A8 WO_B weights and its native scheme.

Run only in an idle, authorized GPU window. This does not admit serving code.
"""
import argparse
import hashlib
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 20, 39])
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(194831)
    from compressed_tensors.quantization import QuantizationStrategy
    from safetensors import safe_open
    from triton.testing import do_bench_cudagraph
    from vllm.models.deepseek_v41.amd import rocm
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as native
    from vllm_hcu.patch.worker.op_opt import patch_compressed_tensors_w8a8_int8

    patch_compressed_tensors_w8a8_int8.apply()
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8
    from lightop.quant import per_token_quant_int8
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment, destroy_model_parallel,
        init_distributed_environment, initialize_model_parallel,
    )

    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx938'
    assert torch.cuda.get_device_properties(0).multi_processor_count == 64
    index = json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    checkpoint = []

    def load(name):
        with safe_open(args.model/index[name], framework='pt', device='cpu') as handle:
            value = handle.get_tensor(name)
            checkpoint.append(dict(index_key=name, shard_key=name, shape=list(value.shape), dtype=str(value.dtype)))
            return value.cuda()

    def exact(a, b, label):
        assert a.shape == b.shape and a.dtype == b.dtype, label
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), label

    class Consumer(torch.nn.Module):
        def __init__(self, weight, scale):
            super().__init__()
            self.scheme = CompressedTensorsW8A8Int8(
                strategy=QuantizationStrategy.CHANNEL,
                is_static_input_scheme=False, input_symmetric=True)
            with torch.device('cuda'):
                self.scheme.create_weights(
                    self, output_partition_sizes=[5120], input_size_per_partition=8192,
                    params_dtype=torch.bfloat16, weight_loader=lambda *a, **k: None)
            self.weight.data.copy_(weight)
            self.weight_scale.data.copy_(scale.reshape(self.weight_scale.shape))
            self.params_dtype = torch.bfloat16
            self.scheme.process_weights_after_loading(self)
            assert self.weight.dtype == torch.int8 and self.weight.shape == (5120, 8192)
            assert self.weight.is_contiguous() and self.weight_scale.dtype == torch.float32
            assert self.scheme.supports_quanted_inputs()
            self.last_input = None

        def forward(self, x):
            self.last_input = x
            return self.scheme.apply_weights(self, x, bias=None)

    def capture(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fn()
        return graph, output

    report = dict(unit_gate_passed=False, serving_admitted=False, eager=[], graph_replays=[],
                  timing=[], checkpoint_tensors=checkpoint,
                  native_consumers={name:dict(module=fn.__module__, sha256=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest())
                                    for name, fn in [('wo_a', native.rocm_inv_rope_einsum),
                                                     ('wo_b', CompressedTensorsW8A8Int8.apply_weights),
                                                     ('activation_quant', per_token_quant_int8)]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendezvous = tempfile.TemporaryDirectory(prefix='w4a8-wo-a-tp1-')
    config_scope = set_current_vllm_config(VllmConfig())
    config_scope.__enter__()
    try:
        # ModelWeightParameter records the real TP rank during native loading.
        # Initialize the actual TP1 environment instead of mocking that owner.
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
            distributed_init_method='file://' + str(Path(rendezvous.name) / 'store'),
            backend='gloo')
        initialize_model_parallel(1, 1, backend='gloo')
        with torch.inference_mode():
            for layer in args.layers:
                prefix = f'layers.{layer}.attn.'
                wa, wb, scale = (load(prefix+suffix) for suffix in ('wo_a.weight', 'wo_b.weight', 'wo_b.weight_scale'))
                assert wa.dtype == torch.bfloat16 and wa.shape == (8192, 4096)
                assert wb.dtype == torch.int8 and wb.shape == (5120, 8192)
                consumer = Consumer(wb, scale)
                owner = object.__new__(rocm.DeepseekV41ROCMAiterMLAAttention)
                torch.nn.Module.__init__(owner)
                owner.n_local_heads, owner.n_local_groups, owner.o_lora_rank = 64, 8, 1024
                owner.rope_head_dim = 64
                owner.rotary_emb = SimpleNamespace(cos_sin_cache=None)
                owner.wo_a = SimpleNamespace(weight=wa)
                owner._wo_b_scale = None
                owner.wo_b = consumer
                cached = native._get_cached_wo_a_bf16(owner.wo_a, 8, 1024, 4096)
                original = cached.clone()
                for rows in (1, 8, 48, 64):
                    for heads in (64, 128):
                        x = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device='cuda')
                        positions = torch.arange(rows, device='cuda', dtype=torch.int64)

                        def execute(variant, audit=False):
                            if variant == 'native':
                                out = rocm.DeepseekV41ROCMAiterMLAAttention._o_proj(owner, x, positions)
                            else:
                                inputs = x[:, :64].reshape(rows, 8, 4096)
                                z = torch.empty((rows, 8, 1024), dtype=x.dtype, device=x.device)
                                torch.bmm(inputs.transpose(0, 1), cached.transpose(1, 2), out=z.transpose(0, 1))
                                out = owner._wo_b_after_wo_a(z.flatten(1))
                            if audit:
                                activation = consumer.last_input
                                codes, scales = per_token_quant_int8(activation)
                                return out, activation, codes, scales
                            return out

                        for magnitude in (0., .001, .25, 1., 8., 100.):
                            x.normal_().mul_(magnitude)
                            if heads == 128:
                                x[:, 64:] = torch.nan
                            saved = x.clone()
                            expected, actual = execute('native', True), execute('token_major', True)
                            for i, (a, b) in enumerate(zip(actual, expected)):
                                exact(a, b, f'layer{layer} rows{rows} heads{heads} magnitude{magnitude} consumer field{i}')
                            exact(x, saved, 'input modified')
                            assert actual[1].is_contiguous()
                            report['eager'].append(dict(layer=layer, rows=rows, heads=heads, magnitude=magnitude, exact=True))
                        base_graph, expected = capture(lambda: execute('native', True))
                        new_graph, actual = capture(lambda: execute('token_major', True))
                        for replay, magnitude in enumerate((.001, .25, 1., 8., 100.)):
                            x.normal_().mul_(magnitude)
                            if heads == 128:
                                x[:, 64:] = torch.nan
                            cached.copy_(original * (-.5 if replay % 2 else 1.))
                            for graph in (base_graph, new_graph):
                                graph.replay()
                            for i, (a, b) in enumerate(zip(actual, expected)):
                                exact(a, b, f'graph replay{replay} consumer field{i}')
                            report['graph_replays'].append(dict(layer=layer, rows=rows, heads=heads, replay=replay, exact=True))
                        cached.copy_(original)
                        for variant in ('native', 'token_major'):
                            us = do_bench_cudagraph(lambda: execute(variant), rep=100) * 1000
                            report['timing'].append(dict(layer=layer, rows=rows, heads=heads, variant=variant, us=us))
                        del base_graph, new_graph
                exact(cached, original, 'static cached weight modified')
                exact(consumer.weight, wb, 'INT8 WO_B weight modified')
                exact(consumer.weight_scale.reshape(-1), scale.reshape(-1), 'WO_B scale modified')
        report['unit_gate_passed'] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        destroy_model_parallel()
        destroy_distributed_environment()
        config_scope.__exit__(None, None, None)
        rendezvous.cleanup()
    print(json.dumps(dict(unit_gate_passed=True, serving_admitted=False,
                          eager=len(report['eager']), graph_replays=len(report['graph_replays']))))


if __name__ == '__main__':
    main()
