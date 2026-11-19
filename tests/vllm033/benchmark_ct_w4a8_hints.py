#!/usr/bin/env python3
"""Native HIPC hint probe; never imports or changes the serving implementation.

Native baseline/candidate
GEMMs must have identical active BF16 bytes. An independent signed-INT4 CPU
dot product checks sampled coordinates exactly using power-of-two scales.
The INT8 activation probe preserves native arithmetic and checks exact active
codes/scales and the following native down GEMM, including fresh graph counts.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

import torch


def exact(a, b, message):
    if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
        raise AssertionError(message)


def timing(call, warmup=15, repeats=80):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    # Ten native launches in each replay amortize Python/event submission.
    # This is also the execution mode used for the captured serving fragments.
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(10):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / 10)
    ordered = sorted(values)
    return {'median_us': statistics.median(values),
            'p10_us': ordered[int(len(ordered) * .1)],
            'p90_us': ordered[int(len(ordered) * .9)], 'samples_us': values,
            'method': 'CUDA events around graph replay; 10 native launches per replay'}


def guarded_scale(shape, value, device):
    # Match serving's HIPC channel-scale allocation/read-ahead contract.
    size = 1
    for extent in shape:
        size *= extent
    storage = torch.zeros(size + (2 * 1024 * 1024) // 4, device=device, dtype=torch.float32)
    result = storage[:size].view(shape)
    result.fill_(value)
    return result


def run(args, report):
    import deepgemm
    from deepgemm.configs.w4a8_masked_hipc import select_w4a8_hipc_masked_config
    from lightop import fuse_silu_mul_clamp_quant_ep

    torch.set_num_threads(8)
    torch.manual_seed(9231)
    report['provider_files'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(deepgemm.__file__), Path(__file__))}
    report['package_versions'] = {}
    for name in ('torch', 'deepgemm', 'lightop'):
        try:
            report['package_versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['package_versions'][name] = None
    e, capacity, k, n = 48, 2048, 5120, 4608
    device = 'cuda'
    props = torch.cuda.get_device_properties(0)
    report['hardware'] = {'name': props.name, 'arch': props.gcnArchName,
                          'cus': props.multi_processor_count, 'vram_bytes': props.total_memory}
    report['shape'] = {'experts': e, 'capacity': capacity, 'hidden': k, 'gateup': n}
    logical = torch.randint(-8, 8, (e, n, k), dtype=torch.int8, device=device)
    oracle_columns = [0, 1, n // 2, n - 1]
    oracle_weights = logical[:, oracle_columns].cpu().to(torch.int64)
    # Canonical checkpoint order: high nibble stores the first K value.
    packed = (((logical[..., 0::2].to(torch.int16) & 15) << 4)
              | (logical[..., 1::2].to(torch.int16) & 15)).to(torch.int8)
    del logical
    weight = deepgemm.view_w4a8_moe_hipc_weight_n32_layout(
        deepgemm.pack_w4a8_moe_hipc_weight(packed))
    weight_scale = guarded_scale((e, n, 1), 1 / 32, device)
    x = torch.randint(-64, 64, (e, capacity, k), device=device, dtype=torch.int8)
    x[0, 0].zero_()
    xs = torch.full((e, capacity, 1), 1 / 256, device=device, dtype=torch.float32)
    counts = torch.zeros(e, device=device, dtype=torch.int32)
    out_old = torch.empty((e, capacity, n), device=device, dtype=torch.bfloat16)
    out_new = torch.empty_like(out_old)
    row = torch.arange(capacity, device=device)[None, :]

    def gemm(output, hint):
        return deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc(
            (x, xs), (weight, weight_scale), output, counts, hint)

    # Above-hint skew explicitly verifies that expected_m is not a row bound.
    layouts = {'decode': [4 + i % 5 for i in range(e)],
               'verify': [24 + i % 9 for i in range(e)],
               'skew': [capacity, 0, 1, 129] + [0] * (e - 4),
               'empty': [0] * e}
    report['gemm_cases'] = []
    for name, values in layouts.items():
        counts.copy_(torch.tensor(values, device=device, dtype=torch.int32))
        active = row < counts[:, None]
        gemm(out_old, capacity)
        original_x, original_xs = x.clone(), xs.clone()
        original_weight, original_weight_scale = weight.clone(), weight_scale.clone()
        for hint in (16, 32, 48):
            gemm(out_new, hint)
            torch.cuda.synchronize()
            if not torch.isfinite(out_new[active]).all():
                raise AssertionError('Nonfinite active GEMM output')
            exact(out_old[active], out_new[active], 'GEMM active bytes differ: %s hint%d' % (name, hint))
            report['gemm_cases'].append({'layout': name, 'hint': hint,
                'active_rows': sum(values), 'active_bytes_equal': True,
                'config': select_w4a8_hipc_masked_config(hint, n, k, e)})
        for value, before in ((x, original_x), (xs, original_xs),
                              (weight, original_weight), (weight_scale, original_weight_scale)):
            exact(value, before, 'Native GEMM modified an input')
        del original_x, original_xs, original_weight, original_weight_scale
        # Independent exact integer dot-product oracle on first/middle/last rows.
        checked = 0
        for expert, count in enumerate(values):
            for token in sorted({0, count // 2, count - 1}) if count else []:
                q = x[expert, token].cpu().to(torch.int64)
                integer_dot = (oracle_weights[expert] * q[None, :]).sum(-1)
                expected = (integer_dot.float() * (16 / (32 * 256))).to(torch.bfloat16)
                actual = out_old[expert, token, oracle_columns].cpu()
                exact(actual, expected, 'Independent GEMM oracle mismatch')
                checked += len(oracle_columns)
        report['gemm_cases'].append({'layout': name, 'independent_oracle_coordinates': checked,
                                     'independent_oracle_exact': True})

    # Power-of-two scales establish the independent exact oracle above. Real
    # non-power-of-two scales must independently preserve the native BF16 bits
    # when the hint selects another tile/epilogue implementation.
    weight_scale.uniform_(1 / 512, 1 / 16)
    xs.uniform_(1e-5, .03)
    for name in ('decode', 'verify', 'skew'):
        counts.copy_(torch.tensor(layouts[name], device=device, dtype=torch.int32))
        active = row < counts[:, None]
        gemm(out_old, capacity)
        for hint in (16, 32, 48):
            gemm(out_new, hint)
            torch.cuda.synchronize()
            exact(out_old[active], out_new[active], 'Real-scale GEMM bytes differ')
        report['gemm_cases'].append({'layout': name, 'non_power_of_two_scales_exact': True})

    # A fresh graph must see changed expert counts without any host .item().
    counts.copy_(torch.tensor(layouts['verify'], device=device, dtype=torch.int32))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gemm(out_new, 32)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        gemm(out_new, 32)
    for name in ('empty', 'skew', 'decode', 'verify'):
        counts.copy_(torch.tensor(layouts[name], device=device, dtype=torch.int32))
        x.random_(-64, 64)
        xs.uniform_(1e-5, .03)
        graph.replay()
        gemm(out_old, capacity)
        torch.cuda.synchronize()
        active = row < counts[:, None]
        exact(out_old[active], out_new[active], 'GEMM fresh graph mismatch: ' + name)
    report['fresh_gemm_graph'] = True
    report['gemm_timings'] = {}
    for name in ('decode', 'verify'):
        counts.copy_(torch.tensor(layouts[name], device=device, dtype=torch.int32))
        report['gemm_timings'][name] = {str(h): timing(lambda h=h: gemm(out_new, h))
                                        for h in (capacity, 16, 32, 48)}

    del weight, packed, oracle_weights, x, xs, graph, out_old
    torch.cuda.empty_cache()
    # Use independent BF16 data with inactive poison; only actual counts are read.
    gateup = out_new
    gateup.normal_(0, 8)
    w2_logical = torch.randint(-8, 8, (e, k, n // 2), dtype=torch.int8, device=device)
    down_columns = [0, 1, k // 2, k - 1]
    down_oracle_weights = w2_logical[:, down_columns].cpu().to(torch.int64)
    w2_packed = (((w2_logical[..., 0::2].to(torch.int16) & 15) << 4)
                 | (w2_logical[..., 1::2].to(torch.int16) & 15)).to(torch.int8)
    del w2_logical
    w2 = deepgemm.view_w4a8_moe_hipc_weight_n32_layout(
        deepgemm.pack_w4a8_moe_hipc_weight(w2_packed))
    w2s = guarded_scale((e, k, 1), 1 / 32, device)
    down_old = torch.empty((e, capacity, k), device=device, dtype=torch.bfloat16)
    down_new = torch.empty_like(down_old)

    def consumer(activation, output):
        deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc(
            activation, (w2, w2s), output, counts, capacity)

    # Down-projection has a different N/K and native configuration table entry.
    # Validate its hint independently, including a signed-INT4 integer oracle.
    probe = torch.randint(-64, 64, (e, capacity, n // 2), device=device, dtype=torch.int8)
    probe_s = torch.full((e, capacity, 1), 1 / 256, device=device, dtype=torch.float32)
    counts.copy_(torch.tensor(layouts['verify'], device=device, dtype=torch.int32))
    consumer((probe, probe_s), down_old)
    torch.cuda.synchronize()
    for expert, count in enumerate(layouts['verify']):
        for token in (0, count - 1):
            q = probe[expert, token].cpu().to(torch.int64)
            dot = (down_oracle_weights[expert] * q[None, :]).sum(-1)
            expected = (dot.float() * (16 / (32 * 256))).to(torch.bfloat16)
            exact(down_old[expert, token, down_columns].cpu(), expected,
                  'Independent down-GEMM integer oracle differs')
    report['down_independent_oracle_coordinates'] = e * 2 * len(down_columns)
    del probe, probe_s, down_oracle_weights
    w2s.uniform_(1 / 512, 1 / 16)
    report['down_gemm_cases'] = []
    report['down_gemm_timings'] = {}
    report['activation_cases'] = []
    report['activation_timings'] = {}
    for name, values in layouts.items():
        counts.copy_(torch.tensor(values, device=device, dtype=torch.int32))
        active = row < counts[:, None]
        gateup[0, 0].zero_()
        gateup.masked_fill_(~active[..., None], float('nan'))
        original_gateup = gateup.clone()
        baseline = fuse_silu_mul_clamp_quant_ep(gateup, 10., counts, capacity)
        consumer(baseline, down_old)
        for hint in (16, 32, 48):
            deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc(
                baseline, (w2, w2s), down_new, counts, hint)
            torch.cuda.synchronize()
            exact(down_old[active], down_new[active], 'Down-GEMM hint changed active bytes')
            report['down_gemm_cases'].append({'layout': name, 'hint': hint,
                'active_bytes_equal': True,
                'config': select_w4a8_hipc_masked_config(hint, k, n // 2, e)})
        if name in ('decode', 'verify'):
            def down_call(h):
                return deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc(
                    baseline, (w2, w2s), down_new, counts, h)
            report['down_gemm_timings'][name] = {
                str(h): timing(lambda h=h: down_call(h)) for h in (capacity, 16, 32, 48)}
        for hint in (16, 32, 48):
            candidate = fuse_silu_mul_clamp_quant_ep(gateup, 10., counts, hint)
            torch.cuda.synchronize()
            exact(baseline[0][active], candidate[0][active], 'INT8 activation codes differ')
            exact(baseline[1][active], candidate[1][active], 'INT8 activation scales differ')
            if not torch.isfinite(candidate[1][active]).all():
                raise AssertionError('Nonfinite active activation scale')
            consumer(candidate, down_new)
            torch.cuda.synchronize()
            exact(down_old[active], down_new[active], 'Native down-GEMM consumer differs')
            report['activation_cases'].append({'layout': name, 'hint': hint,
                'active_codes_and_scale_bytes_equal': True, 'native_consumer_bytes_equal': True})
        exact(gateup, original_gateup, 'Activation modified its input or poison padding')
        del original_gateup
        if name in ('decode', 'verify'):
            report['activation_timings'][name] = {
                str(h): timing(lambda h=h: fuse_silu_mul_clamp_quant_ep(gateup, 10., counts, h))
                for h in (capacity, 16, 32, 48)}
        # Restore rows after a previous empty/skew poison before changing counts.
        gateup.normal_(0, 8)

    # Capture activation once, then change device metadata and all input bytes.
    counts.copy_(torch.tensor(layouts['verify'], device=device, dtype=torch.int32))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fuse_silu_mul_clamp_quant_ep(gateup, 10., counts, 32)
    torch.cuda.current_stream().wait_stream(stream)
    activation_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(activation_graph, stream=stream):
        graph_result = fuse_silu_mul_clamp_quant_ep(gateup, 10., counts, 32)
    for name in ('empty', 'skew', 'decode', 'verify'):
        counts.copy_(torch.tensor(layouts[name], device=device, dtype=torch.int32))
        gateup.normal_(0, 8)
        active = row < counts[:, None]
        gateup.masked_fill_(~active[..., None], float('nan'))
        activation_graph.replay()
        baseline = fuse_silu_mul_clamp_quant_ep(gateup, 10., counts, capacity)
        torch.cuda.synchronize()
        exact(baseline[0][active], graph_result[0][active], 'Fresh activation graph codes differ')
        exact(baseline[1][active], graph_result[1][active], 'Fresh activation graph scales differ')
        consumer(baseline, down_old)
        deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc(
            graph_result, (w2, w2s), down_new, counts, 32)
        torch.cuda.synchronize()
        exact(down_old[active], down_new[active], 'Fresh graph native consumer differs')
    report['fresh_activation_graph'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = {'started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'operator_contract': 'exact active bytes vs unchanged native provider',
              'status': 'running'}
    try:
        run(args, report)
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
