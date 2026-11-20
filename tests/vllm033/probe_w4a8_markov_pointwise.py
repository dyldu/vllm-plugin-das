# SPDX-License-Identifier: Apache-2.0
"""Strict BF16 add/argmax/embedding and real-weight five-step Markov probe."""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
from safetensors import safe_open


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.model), Path(args.output)
    assert not output.exists()
    from hcu_vllm_das.dspark_greedy_pointwise import greedy, step
    mapping = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    weights = []
    for suffix in ('embed.weight', 'head.weight'):
        key = 'mtp.2.markov_head.' + suffix
        with safe_open(str(root/mapping[key]), framework='pt', device='cpu') as f:
            weights.append(f.get_tensor(key).cuda().contiguous())
    w1, w2 = weights
    v, rank = w1.shape
    assert w1.shape == w2.shape == (129280, 256)
    report = dict(passed=False, precision_passed=False, production_admitted=False,
                  pointwise_cases=[], sequential_cases=[], graphs=[], timings={})
    def save(): output.write_text(json.dumps(report, indent=2)+'\n')
    def native(base, anchors):
        prev, tokens = anchors, []
        for i in range(5):
            bias = torch.nn.functional.linear(torch.nn.functional.embedding(prev, w1), w2)
            prev = (base[:,i] + bias).argmax(-1)
            tokens.append(prev)
        return torch.stack(tokens, 1)
    def bench(fn):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): fn()
        stream.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): result=fn()
        values=[]
        for _ in range(3):
            for _ in range(5):graph.replay()
            a,b=(torch.cuda.Event(enable_timing=True)for _ in range(2))
            a.record()
            for _ in range(30):graph.replay()
            b.record();b.synchronize();values.append(a.elapsed_time(b)*1000/30)
        return dict(samples_us=values,median_us=statistics.median(values))
    torch.manual_seed(20261124)
    save()
    try:
        for tile in (1024, 2048, 4096):
            for m in (1,2,4,8,16,32,64):
                base=torch.randn((m,5,v),device='cuda',dtype=torch.bfloat16)
                bias=torch.empty((m,v),device='cuda',dtype=torch.bfloat16)
                tokens=torch.empty((m,5),device='cuda',dtype=torch.int64)
                embed=torch.empty((m,rank),device='cuda',dtype=torch.bfloat16)
                nb=triton.cdiv(v,tile)
                maxima=torch.empty((m,nb),device='cuda')
                indices=torch.empty((m,nb),device='cuda',dtype=torch.int32)
                for case in ('zero','random','overflow','ties','nan','inf','negative_inf'):
                    a=base[:,2]
                    a.normal_();bias.normal_()
                    if case=='zero': a.zero_();bias.zero_()
                    elif case=='overflow': a.fill_(torch.finfo(torch.bfloat16).max);bias.copy_(a)
                    elif case=='ties': a.fill_(-10);bias.zero_();a[:,17]=a[:,v-1]=10
                    elif case=='nan': a[:,v-1]=float('nan');a[:,123]=float('nan')
                    elif case=='inf': a[:,17]=a[:,v-1]=float('inf')
                    elif case=='negative_inf': a.fill_(-float('inf'));bias.zero_()
                    expected=(a+bias).argmax(-1)
                    step(a,bias,w1,tokens[:,2],embed,maxima,indices,tile=tile)
                    ids=bool(torch.equal(tokens[:,2],expected))
                    emb=bool(torch.equal(embed,torch.nn.functional.embedding(expected,w1)))
                    report['pointwise_cases'].append(dict(tile=tile,m=m,case=case,ids_exact=ids,embedding_exact=emb))
                    assert ids and emb, report['pointwise_cases'][-1]
            for m in (1,2,4,8,16,32,64):
                anchors=torch.randint(v,(m,),device='cuda')
                base=torch.randn((m,5,v),device='cuda',dtype=torch.bfloat16)
                for mag in (0,.01,1,8):
                    base.normal_().mul_(mag)
                    equal=bool(torch.equal(greedy(w1,w2,base,anchors,tile=tile),native(base,anchors)))
                    report['sequential_cases'].append(dict(tile=tile,m=m,magnitude=mag,ids_exact=equal))
                    save();assert equal,report['sequential_cases'][-1]
                if m in (1,8,32):
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):greedy(w1,w2,base,anchors,tile=tile)
                    stream.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream): actual=greedy(w1,w2,base,anchors,tile=tile)
                    for replay in range(3):
                        base.normal_();anchors.random_(v);graph.replay()
                        equal=bool(torch.equal(actual,native(base,anchors)))
                        report['graphs'].append(dict(tile=tile,m=m,replay=replay,ids_exact=equal))
                        save();assert equal,report['graphs'][-1]
            m=8
            anchors=torch.randint(v,(m,),device='cuda')
            base=torch.randn((m,5,v),device='cuda',dtype=torch.bfloat16)
            report['timings'][str(tile)]={'native':bench(lambda:native(base,anchors)),
                'candidate':bench(lambda:greedy(w1,w2,base,anchors,tile=tile))}
            save()
        report['precision_passed']=True
        report['faster_tiles']=[int(k) for k,r in report['timings'].items()
                               if r['candidate']['median_us']<r['native']['median_us']]
        report['passed']=bool(report['faster_tiles'])
    except Exception as exc:
        report['error']=repr(exc)
    save()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__': main()
