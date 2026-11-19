"""Isolated fixed-capacity W4A8 draft/target LL transport diagnostic.

Compare actual LL INT8 bytes/scales to the serving AG/RS quantizer, aligning
received rows by native source IDs. No production backend is changed here.
This first gate is eager transport/quantization only; graph and expert GEMM
consumer checks are separate gates before serving admission.
"""
import argparse
import datetime
import json
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


CAPACITY, HIDDEN, WORLD = 256, 5120, 8


def worker(rank, init, output, graph_enabled, bf16_receiver):
    torch.cuda.set_device(rank)
    import deep_ep
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=WORLD,
                            timeout=datetime.timedelta(seconds=120))
    records = []
    output_path = Path(output, 'rank%d.json' % rank)

    def emit(stage, **kw):
        row = dict(rank=rank, stage=stage, time=time.time(),
                   free_bytes=torch.cuda.mem_get_info()[0], **kw)
        records.append(row)
        output_path.write_text(json.dumps(dict(status='running', records=records), indent=2)+'\n')
        concise = {k:v for k,v in row.items() if k not in ('mismatches','failures')}
        concise['mismatch_count'] = len(row.get('mismatches', []))
        print(json.dumps(concise), flush=True)

    def quant(x):
        return moe_kernel_quantize_input(x, None, torch.int8, True, None)

    def quant_receiver(x, counts):
        from hcu_vllm_das.dspark_ll_quant import quantize_receiver
        return quantize_receiver(x, counts)

    nbytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(CAPACITY, HIDDEN, WORLD, 384)
    emit('construct_start', rdma_bytes=nbytes)
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=1024**3,
        num_rdma_bytes=nbytes, low_latency_mode=True, num_qps_per_rank=48,
        allow_nvlink_for_low_latency_mode=True, explicitly_destroy=True)
    emit('construct_done')
    failures = []

    def routing(rows, experts, topk, skew):
        ids = ((torch.arange(rows, device='cuda')[:, None] * (0 if skew else 17)
                + torch.arange(topk, device='cuda')[None, :] * (1 if skew else 43)
                + (0 if skew else rank)) % experts)
        weights = torch.zeros((rows, topk), device='cuda', dtype=torch.float32)
        weights[:, 0] = 1.
        return ids.long(), weights

    def dispatch(x, ids, weights, experts, quant_type):
        buffer.clean_low_latency_buffer(CAPACITY, HIDDEN, experts,
                                        quant_group_size=0, num_topk=ids.shape[1])
        value, counts, handle, _, hook = buffer.low_latency_dispatch(
            x, ids, weights, CAPACITY, experts, quant_type=quant_type,
            quant_group_size=0, fp8_round_scale=False,
            async_finish=False, return_recv_hook=True)
        hook()
        return value, counts, handle

    cases = [(0, 'random', False), (1, 'zero', False), (8, 'random', False),
             (40, 'tiny', False), (64, 'ties', False), (160, 'random', False),
             (192, 'random', False), (256, 'random', True)]
    for rows, pattern, skew in cases:
        for experts, topk in ((384, 6), (128, 3)):
            local_rows = rows if skew else max(0, rows-rank)
            torch.manual_seed(7351 + rows + experts + rank)
            x = torch.randn((local_rows, HIDDEN), device='cuda').bfloat16()
            if pattern == 'zero':
                x.zero_()
            elif pattern == 'tiny':
                x.mul_(2.**-22)
            elif pattern == 'ties':
                x.copy_(torch.tensor([127., 1.5, -1.5, .5, -.5],
                         device='cuda', dtype=torch.bfloat16).repeat(1024))
            ids, weights = routing(local_rows, experts, topk, skew)
            raw, counts, handle = dispatch(x, ids, weights, experts, 0)
            torch.cuda.synchronize()
            references = []
            for e, count in enumerate(counts.cpu().tolist()):
                if not count:
                    references.append(None)
                    continue
                source = handle[0][e, :count].clone()
                order = source.argsort()
                q, scale = quant(raw[e, :count].contiguous())
                references.append((source[order], q[order].clone(), scale[order].clone()))
            saved_counts = counts.clone()
            # Finish the BF16 transaction before switching wire/layout.
            _, _, hook = buffer.low_latency_combine(raw, ids, weights, handle,
                async_finish=False, return_recv_hook=True)
            hook()
            torch.cuda.synchronize()
            dist.barrier()
            received, counts, handle = dispatch(x, ids, weights, experts,
                                               0 if bf16_receiver else 1)
            if bf16_receiver:
                # A source token index may repeat across ranks. Comparing two
                # independently packed transactions by that index alone is
                # ambiguous under full expert skew. Use the *current* received
                # BF16 rows for the native row-quantization oracle; the separate
                # combine round trip still verifies routing against original x.
                references = []
                for e, count in enumerate(counts.cpu().tolist()):
                    if not count:
                        references.append(None)
                        continue
                    source = handle[0][e, :count].clone()
                    order = source.argsort()
                    qref, sref = quant(received[e, :count].contiguous())
                    references.append((source[order], qref[order].clone(), sref[order].clone()))
            q, scale = quant_receiver(received, counts) if bf16_receiver else received
            scale = scale.reshape(experts // WORLD, -1, 1)
            torch.cuda.synchronize()
            same_counts = torch.equal(saved_counts, counts)
            mismatches = []
            for e, reference in enumerate(references):
                if reference is None:
                    continue
                source, expected_q, expected_scale = reference
                count = len(source)
                actual_source = handle[0][e, :count]
                order = actual_source.argsort()
                if not torch.equal(source, actual_source[order]):
                    mismatches.append(dict(expert=e, reason='source_ids'))
                    continue
                actual_q, actual_scale = q[e, :count][order], scale[e, :count][order]
                if not torch.equal(expected_q, actual_q):
                    where = (expected_q != actual_q).nonzero()[:8]
                    mismatches.append(dict(expert=e, reason='int8_codes',
                        elements=(expected_q != actual_q).sum().item(),
                        examples=[dict(row=int(i), column=int(j),
                            expected=int(expected_q[i, j]), actual=int(actual_q[i, j]),
                            expected_scale=float(expected_scale[i, 0]),
                            actual_scale=float(actual_scale[i, 0]))
                            for i, j in where.cpu().tolist()]))
                if not torch.equal(expected_scale.view(torch.int32), actual_scale.view(torch.int32)):
                    mismatches.append(dict(expert=e, reason='fp32_scale_bytes',
                        max_abs=(expected_scale-actual_scale).abs().max().item()))
            y = (q.float() * scale).bfloat16()
            out, _, hook = buffer.low_latency_combine(y, ids, weights, handle,
                async_finish=False, return_recv_hook=True)
            hook()
            torch.cuda.synchronize()
            if x.numel():
                qref, sref = quant(x)
                expected = (qref.float() * sref).bfloat16()
                roundtrip_exact = torch.equal(out, expected)
                roundtrip_max_abs = (out.float()-expected.float()).abs().max().item()
            else:
                roundtrip_exact, roundtrip_max_abs = out.shape == x.shape, 0.
            ok = same_counts and not mismatches and roundtrip_exact
            emit('case', experts=experts, rows=rows, local_rows=local_rows,
                 pattern=pattern, skew=skew, same_counts=same_counts,
                 reference_mode='same_received_bf16' if bf16_receiver else 'cross_wire_source_index',
                 source_ids_unique=all(ref is None or len(ref[0].unique()) == len(ref[0]) for ref in references),
                 mismatches=mismatches, roundtrip_exact=roundtrip_exact,
                 roundtrip_max_abs=roundtrip_max_abs, passed=ok)
            if not ok:
                failures.append(dict(experts=experts, rows=rows, pattern=pattern,
                                     mismatches=mismatches))
            dist.barrier()
    # All ranks must agree before entering collective graph transactions.
    # A local quantization mismatch must not leave other ranks inside LL recv.
    wire_failures = [None] * WORLD
    dist.all_gather_object(wire_failures, failures)
    global_wire_ok = not any(wire_failures)
    emit('wire_gate', global_passed=global_wire_ok,
         failed_ranks=[i for i, rows in enumerate(wire_failures) if rows])
    if graph_enabled and global_wire_ok:
        # Native outputs from more than two transactions cannot retain views
        # into the LL buffer. Keep only the independently allocated combine
        # outputs, completing each transaction before the next layout.
        for rows in (0, 1, 4, 8, 16, 32, 40, 48, 64, 160, 192, 256):
            transactions = []
            for experts, topk in ((384, 6), (384, 6), (128, 3), (128, 3), (384, 6)):
                m = max(0, rows-rank)
                x = torch.randn((m, HIDDEN), device='cuda').bfloat16()
                ids, weights = routing(m, experts, topk, False)
                transactions.append((experts, x, ids, weights))

            def execute():
                outputs = []
                for experts, x, ids, weights in transactions:
                    use_receiver = bf16_receiver and experts == 128
                    received, counts, handle = dispatch(x, ids, weights, experts,
                                                       0 if use_receiver else 1)
                    q, scale = quant_receiver(received, counts) if use_receiver else received
                    y = (q.float() * scale.reshape(*q.shape[:2], 1)).bfloat16()
                    out, _, hook = buffer.low_latency_combine(y, ids, weights, handle,
                        async_finish=False, return_recv_hook=True)
                    hook()
                    outputs.append(out)
                return outputs

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                execute()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = execute()
            for replay in range(5):
                for experts, x, ids, weights in transactions:
                    x.normal_().mul_((0., 2.**-22, .25, 8., 100.)[replay])
                    ids.add_(19).remainder_(experts)
                graph.replay()
                torch.cuda.synchronize()
                expected = execute()
                torch.cuda.synchronize()
                exact = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                            for a, b in zip(captured, expected))
                finite = all(torch.isfinite(a).all().item() for a in captured)
                emit('graph', rows=rows, replay=replay, native_bytes_exact=exact,
                     finite=finite, passed=exact and finite)
                if not exact or not finite:
                    failures.append(dict(rows=rows, replay=replay, reason='graph_combine_bytes'))
            del graph, captured, expected, transactions
    all_failures = [None] * WORLD
    dist.all_gather_object(all_failures, failures)
    global_passed = not any(all_failures)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()
    emit('PASS' if global_passed else 'FAIL', failures=failures,
         failed_ranks=[i for i, rows in enumerate(all_failures) if rows])
    output_path.write_text(json.dumps(
        dict(passed=global_passed, records=records), indent=2)+'\n')
    assert global_passed, {'failed_ranks':[i for i, rows in enumerate(all_failures) if rows],
                           'details':'See per-rank JSON records'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--bf16-receiver', action='store_true')
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='w4a8-ll-wire-') as directory:
        mp.spawn(worker, args=('file://'+directory+'/group', args.output, args.graph,
                               args.bf16_receiver),
                 nprocs=WORLD, join=True)
