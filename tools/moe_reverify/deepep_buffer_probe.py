"""Actual-wheel 8-rank LL buffer/precision reproducer. Run on idle GPUs.

Validated stock-precision replay:
  python3 deepep_buffer_probe.py --shared --clean --stress --stock-random \
    --receiver-quant --magnitudes 0 .001 .25 1 8

--dual reproduces the installed wheel's second-Buffer invalid access.
--shared without --clean reproduces dirty-layout invalid access.
These are explicit diagnostic modes, not serving configurations.
"""
import argparse
import datetime
import json
import itertools
import math
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, init, args):
    torch.cuda.set_device(rank)
    import deep_ep
    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=8,
                            timeout=datetime.timedelta(seconds=90))
    def emit(stage, **kw):
        print(json.dumps(dict(rank=rank, stage=stage, time=time.time(),
                              free=torch.cuda.mem_get_info()[0], **kw)), flush=True)
    buffers = []
    experts = [args.experts] if args.single or args.shared else [384, 128]
    for e in experts:
        n = deep_ep.Buffer.get_low_latency_rdma_size_hint(64, 5120, 8, e)
        emit('construct_start', experts=e, rdma=n)
        b = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=args.nvl_mb * 1024**2,
                           num_rdma_bytes=n, low_latency_mode=True,
                           num_qps_per_rank=e // 8,
                           allow_nvlink_for_low_latency_mode=True,
                           explicitly_destroy=True)
        buffers.append((b, e))
        emit('construct_done', experts=e)
    if args.shared:
        buffers.append((buffers[0][0], 128))
    entry_models = {}
    if args.model_entry_reset:
        # This isolates the actual production hook and vendor ABI. Config
        # admission is covered separately by the CPU/native constructor tests.
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager
        from hcu_vllm_das.dspark_ll import install_model_entry_reset
        manager = object.__new__(DeepEPLLAll2AllManager)
        manager.internode = manager.support_fault_tolerance = False
        manager.world_size = manager.dp_world_size = 8
        def get_handle(kwargs):
            assert kwargs == dict(max_num_tokens_per_dp_rank=64, token_hidden_size=5120,
                num_ep_ranks=8, num_global_experts=384, num_local_experts=48)
            return buffers[0][0]
        manager.get_handle = get_handle
        hf = NS(hidden_size=5120, n_routed_experts=384, num_experts_per_tok=6,
            dspark_n_routed_experts=128, dspark_num_experts_per_tok=3,
            num_hidden_layers=40)
        cfg = NS(model_config=NS(hf_config=hf))
        class Entry(torch.nn.Module):
            def forward(self):
                pass
        with patch('hcu_vllm_das.dspark_backend.native_draft_ll_requested', return_value=True), \
             patch('vllm.model_executor.layers.fused_moe.all2all_utils.get_ep_all2all_manager', return_value=manager):
            for _, e in buffers:
                model = Entry()
                assert install_model_entry_reset(model, cfg, e)
                entry_models[e] = model
    if args.stress:
        assert args.clean
        failures = []
        for magnitude, rows in itertools.product(args.magnitudes, [0, 1, 40, 48, 64]):
            inputs = []
            for b, e in buffers:
                k = 6 if e == 384 else 3
                m = max(0, rows - rank)
                pattern = torch.tensor([1., -.5, .25, -.125], device='cuda').repeat(1280)
                x = ((rank + 1) / 8 * pattern).bfloat16().expand(m, -1).clone()
                if args.stock_random:
                    torch.manual_seed(1900 + rank + e + rows)
                    x = (torch.randn((m, 5120), device='cuda') * magnitude).bfloat16()
                ids = ((torch.arange(m, device='cuda')[:, None] * 17 +
                        torch.arange(k, device='cuda')[None, :] * 43 + rank) % e).long()
                w = torch.full((m, k), 1. / k, dtype=torch.float32, device='cuda')
                inputs.extend([(b, e, x, ids, w)] * args.same_layout_repeats)
            def execute(clean_each_dispatch=False):
                outputs = []
                for index, (b, e, x, ids, w) in enumerate(inputs):
                    if args.model_entry_reset and not clean_each_dispatch:
                        if index % args.same_layout_repeats == 0:
                            entry_models[e]()
                    else:
                        b.clean_low_latency_buffer(64, 5120, e, num_topk=ids.shape[1])
                    wire = x
                    extra = {}
                    if args.prequant and x.numel():
                        from vllm import _custom_ops as ops
                        quantized, scales = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                        if args.dequant_wire:
                            wire = (quantized.float() * scales).bfloat16()
                        extra['x_scales'] = scales
                    receiver_quant = args.receiver_quant and not (args.native_target_quant and e == 384)
                    q, counts, handle, _, hook = b.low_latency_dispatch(
                        wire, ids, w, 64, e, quant_type=0 if receiver_quant else 2, quant_group_size=0,
                        fp8_round_scale=False, async_finish=False, return_recv_hook=True, **extra)
                    hook()
                    if receiver_quant:
                        from vllm import _custom_ops as ops
                        shape = q.shape
                        if args.masked_receiver_quant and e == 128:
                            from hcu_vllm_das.moe.masked_token_quant import masked_token_quant
                            values, scales = masked_token_quant(q, counts)
                        else:
                            values, scales = ops.scaled_fp8_quant(
                                q.reshape(-1, 5120), use_per_token_if_dynamic=True)
                            values = values.reshape(shape)
                    else:
                        values, scales = q
                    y = (values.float() * scales.reshape(*values.shape[:2], 1)).bfloat16()
                    out, _, hook = b.low_latency_combine(
                        y, ids, w, handle, async_finish=False, return_recv_hook=True)
                    hook()
                    outputs.append(out)
                return outputs
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                eager = execute()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                outputs = execute()
            for replay in range(5):
                # Each repeated layout aliases the same changing payload.
                for _, e, x, ids, w in inputs[::args.same_layout_repeats]:
                    x.neg_()
                    ids.add_(19).remainder_(e)
                g.replay()
                torch.cuda.synchronize()
                # Compare a pure reset scheduling change against the prior
                # per-dispatch-clean native combine, including signed zeros.
                # Stock dequantized identity is a value oracle for draft FP8;
                # it does not preserve native combine's zero sign bits.
                references = execute(clean_each_dispatch=True) if args.native_target_quant else [None] * len(outputs)
                torch.cuda.synchronize()
                for out, reference, (_, e, x, _, _) in zip(outputs, references, inputs):
                    expected = x
                    if args.native_target_quant and e == 384:
                        # Keep the pre-existing vendor target math. This
                        # assertion checks eager/captured layout replay exactly.
                        expected = reference
                    elif args.stock_random and x.numel():
                        from vllm import _custom_ops as ops
                        qref, sref = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                        expected = (qref.float() * sref).bfloat16()
                    err = (out.float() - expected.float()).abs().max().item() if x.numel() else 0.
                    byte_reference = reference if args.model_entry_reset else expected
                    equal_bytes = torch.equal(out.view(torch.uint8), byte_reference.view(torch.uint8))
                    emit('graph_replay', experts=e, rows=rows, replay=replay, max_abs=err,
                         magnitude=magnitude, equal_bytes=equal_bytes)
                    if (not math.isfinite(err) or err > .01
                            or (args.model_entry_reset and not equal_bytes)):
                        failures.append((e, rows, replay, err))
        dist.barrier()
        buffers[0][0].destroy()
        dist.destroy_process_group()
        emit('STRESS_FAIL' if failures else 'STRESS_PASS', failures=failures)
        assert not failures, failures
        return
    for iteration in range(3):
        for b, e in buffers:
            k = 6 if e == 384 else 3
            x = torch.full((8, 5120), (rank + 1) / 8,
                           dtype=torch.bfloat16, device='cuda')
            ids = ((torch.arange(8, device='cuda')[:, None] * 17 +
                    torch.arange(k, device='cuda')[None, :] * 43 + rank) % e).long()
            weights = torch.full((8, k), 1. / k, dtype=torch.float32, device='cuda')
            if args.clean:
                b.clean_low_latency_buffer(64, 5120, e, num_topk=k)
            emit('dispatch_start', experts=e, iteration=iteration)
            q, counts, handle, event, hook = b.low_latency_dispatch(
                x, ids, weights, 64, e, quant_type=2, quant_group_size=0,
                fp8_round_scale=False, async_finish=False, return_recv_hook=True)
            hook()
            torch.cuda.synchronize()
            emit('dispatch_done', experts=e, iteration=iteration)
            values, scales = q
            y = (values.float() * scales.reshape(*values.shape[:2], 1)).bfloat16()
            out, event, hook = b.low_latency_combine(
                y, ids, weights, handle, async_finish=False, return_recv_hook=True)
            hook()
            torch.cuda.synchronize()
            err = (out.float() - x.float()).abs().max().item()
            emit('combine_done', experts=e, iteration=iteration, max_abs=err,
                 tokens=counts.sum().item())
            assert err <= 0.01, err
            torch.cuda.empty_cache()
    dist.barrier()
    for b in dict.fromkeys(b for b, _ in reversed(buffers)):
        b.destroy()
    dist.destroy_process_group()
    emit('PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--single', action='store_true')
    p.add_argument('--dual', action='store_true')
    p.add_argument('--nvl-mb', type=int, default=1024)
    p.add_argument('--shared', action='store_true')
    p.add_argument('--clean', action='store_true')
    p.add_argument('--stress', action='store_true')
    p.add_argument('--stock-random', action='store_true')
    p.add_argument('--prequant', action='store_true')
    p.add_argument('--dequant-wire', action='store_true')
    p.add_argument('--experts', type=int, default=384)
    p.add_argument('--receiver-quant', action='store_true')
    p.add_argument('--magnitudes', type=float, nargs='+', default=[.25])
    p.add_argument('--masked-receiver-quant', action='store_true',
                   help='owned E128 receiver: quantize only native counts-active rows')
    p.add_argument('--model-entry-reset', action='store_true')
    p.add_argument('--native-target-quant', action='store_true')
    p.add_argument('--same-layout-repeats', type=int, default=1)
    args = p.parse_args()
    if sum((args.single, args.shared, args.dual)) != 1:
        p.error('choose exactly one of --single, --shared, or --dual')
    if args.masked_receiver_quant and not (args.shared and args.receiver_quant and args.stress):
        p.error('--masked-receiver-quant requires --shared --receiver-quant --stress')
    if args.same_layout_repeats < 1 or (args.model_entry_reset and not args.shared):
        p.error('model-entry reset requires shared Buffer and positive layout repeats')
    with tempfile.TemporaryDirectory(prefix='deepep-dual-') as d:
        mp.spawn(worker, args=('file://' + d + '/group', args), nprocs=8, join=True)
