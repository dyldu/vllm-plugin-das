# SPDX-License-Identifier: Apache-2.0
"""Real 1M-context reservation, >2 GiB addressing and native-consumer oracle."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from hcu_vllm_das import fp4_indexer_lightop as reader


def main(args):
    torch.cuda.set_device(0)
    torch.manual_seed(520116)
    spec = importlib.util.spec_from_file_location('old_fp4_reader', args.original)
    old = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    providers = reader.load_providers()
    cfg = NS(cache_config=NS(block_size=64),
             scheduler_config=NS(max_num_seqs=64, max_num_batched_tokens=256),
             speculative_config=NS(num_speculative_tokens=5))
    width, rows, records = 1048576 // 2, 256, 32
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    pool = reader.reserve_decode_staging(cfg, width, 2, torch.device('cuda:0'))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    assert peak < 9 * 1024**3, peak
    assert pool.numel() == 256 * 8192 * 4352
    report = dict(passed=False, profile_peak_bytes=peak,
                  persistent_bytes=pool.numel(), eager=0, graphs=0, consumers=0)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    save()
    cache = torch.empty(512, records, 68, dtype=torch.uint8, device='cuda')
    raw = cache.view(512, -1)
    raw[:, :records * 64].random_(0, 256)
    raw[:, records * 64:] = 127
    table = torch.arange(rows * (width // records), dtype=torch.int32,
                         device='cuda').reshape(rows, -1).remainder_(512)
    lengths = torch.full((rows,), 129, dtype=torch.int32, device='cuda')
    selected = (0, 1, 63, 64, 65, 127, 128, 255)

    def run():
        return reader.stage_decode_pages(cache, table, lengths, width, providers.gather)

    def check(output, graph=False):
        pages, out_table = output
        for row in selected:
            live = (int(lengths[row]) + 63) // 64
            if not live:
                continue
            expected, _ = old.stage_decode_pages(
                cache, table[row:row+1], lengths[row:row+1], live * 64, providers.gather)
            torch.testing.assert_close(pages[row*8192:row*8192+live], expected,
                                       rtol=0, atol=0)
            expected_table = torch.arange(row*8192, row*8192+live,
                                          dtype=torch.int32, device='cuda')
            torch.testing.assert_close(out_table[row, :live], expected_table, rtol=0, atol=0)
            report['graphs' if graph else 'eager'] += 1
        torch.cuda.synchronize()
        save()

    with torch.inference_mode():
        output = run()
        check(output)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run()
        for iteration in range(2):
            lengths.fill_(37 + iteration)
            lengths[-1] = width - iteration
            table.copy_(table.roll(1, 1))
            raw[:, :records * 64].random_(0, 256)
            graph.replay()
            check(output, graph=True)
        # Native scorer consumes pages whose physical IDs point beyond 2 GiB.
        for row in (65, 255):
            expected, et = old.stage_decode_pages(
                cache, table[row:row+1], lengths[row:row+1], width, providers.gather)
            q = torch.randint(256, (1, 1, 32, 64), dtype=torch.uint8, device='cuda')
            sf = torch.full((1, 1, 32), 0x7f7f7f7f, dtype=torch.int32, device='cuda')
            weights = torch.rand(1, 32, device='cuda') / 32
            a = providers.decode(q, sf, expected.unsqueeze(2), weights,
                                 lengths[row:row+1], et, None, width, 64, None, 1)
            schedule = providers.plan(lengths[row:row+1], 64,
                torch.cuda.get_device_properties(0).multi_processor_count)
            b = providers.decode(q, sf, output[0].unsqueeze(2), weights,
                                 lengths[row:row+1], output[1][row:row+1],
                                 schedule, width, 64, None, 1)
            n = int(lengths[row].item())
            assert torch.equal(a[:, :n].view(torch.uint8), b[:, :n].view(torch.uint8))
            assert torch.isfinite(b[:, :n]).all().item()
            report['consumers'] += 1
        report['passed'] = True
        save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
