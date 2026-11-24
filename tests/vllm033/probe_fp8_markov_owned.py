# SPDX-License-Identifier: Apache-2.0
"""Validate BF16 Markov reuse with a Channel-FP8 checkpoint and native oracle.

Run in the HCU image with the plugin source overlay and --model/--output.
Every token, untouched tail row, and graph replay must match exactly.
"""
import argparse
import json
import hashlib
import statistics
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from safetensors import safe_open


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.model);out=Path(args.output)
    assert not out.exists()
    from hcu_vllm_das.dspark_greedy_admission import supported, sample_sequential, qualify_markov
    from hcu_vllm_das.models.dsv41_dspark import HcuDSparkDeepseekV41ForCausalLM as Owned
    from vllm.models.deepseek_v41.amd.dspark import DSparkDeepseekV4ForCausalLM as Native
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator as Spec
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
    mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    weights=[]
    for suffix in ('embed.weight','head.weight'):
        key='mtp.2.markov_head.'+suffix
        with safe_open(str(root/mapping[key]),framework='pt',device='cpu') as f:
            weights.append(f.get_tensor(key).cuda().contiguous())
    w1,w2=weights;v,rank=w1.shape
    model=Owned.__new__(Owned);torch.nn.Module.__init__(model)
    model.model=torch.nn.Module()
    head=DSparkMarkovHead.__new__(DSparkMarkovHead);torch.nn.Module.__init__(head)
    model.model.markov_head=head
    head.markov_w1=torch.nn.Embedding(v,rank,device='cuda',dtype=torch.bfloat16)
    head.markov_w1.weight=torch.nn.Parameter(w1,requires_grad=False)
    head.markov_w2=torch.nn.Module()
    head.markov_w2.weight=torch.nn.Parameter(w2,requires_grad=False)
    head.markov_w2.tp_size=1
    head.markov_w2.quant_method=UnquantizedEmbeddingMethod()
    processor=LogitsProcessor.__new__(LogitsProcessor);torch.nn.Module.__init__(processor)
    processor.head_dtype=None;processor.scale=1.;processor.soft_cap=None
    processor.logits_as_input=False;processor.org_vocab_size=v;processor.vocab_size=v
    model.logits_processor=processor;model._hcu_markov_pointwise=True
    report=dict(passed=False,model=str(root),exact_token_gate=True,rows=[],graphs=[],fallbacks=[],timings=[],weight_sha256=[hashlib.sha256(w.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for w in weights])
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    torch.manual_seed(20261125);save()
    try:
        for m, steps, anchor_dtype in ((m, steps, dtype) for steps in (1,3,5,7)
                                    for dtype in (torch.int32,torch.int64)
                                    for m in (1,2,7,9,17,32,63,64)):
            base=torch.empty((m*steps,v),device='cuda',dtype=torch.bfloat16)
            model.compute_draft_logits=MethodType(lambda self,hidden:base,model)
            spec=Spec.__new__(Spec)
            if isinstance(spec,torch.nn.Module):torch.nn.Module.__init__(spec)
            spec.model=model;spec.num_speculative_steps=steps;spec._draft_topk=None
            spec.draft_logits=None;spec.draft_watermarker=None;spec._d2t_scatter_index=None
            spec.use_confidence_head=False;spec.acceptance_estimator=None
            spec.sample_indices=torch.arange(m*steps,device='cuda')
            spec.sample_idx_mapping=torch.arange(m,device='cuda').repeat_interleave(steps)
            spec.sample_pos=torch.ones(m*steps,device='cuda',dtype=torch.int64)
            spec._step_cols=list(range(steps))
            spec.input_buffers=SimpleNamespace(input_ids=torch.randint(v,(m,),device='cuda',dtype=anchor_dtype))
            spec._anchor_idx=torch.arange(m,device='cuda')
            spec.draft_tokens=torch.empty((m+1,steps),device='cuda',dtype=torch.int64)
            hidden=torch.zeros((m*steps,5120),device='cuda',dtype=torch.bfloat16)
            if not supported(model,spec,m,hidden):
                assert qualify_markov(model,spec), 'Native source preflight failed'
            assert supported(model,spec,m,hidden),'Actual native entry not admitted'
            # The oracle has the actual loaded head and native methods, without
            # the optional model sampling hook queried by the speculator.
            native_model=SimpleNamespace(compute_draft_logits=model.compute_draft_logits,
                markov_embed=model.markov_embed,markov_bias=model.markov_bias,
                map_draft_to_target=model.map_draft_to_target)
            def native_loop():
                spec.model=native_model
                try: Spec._sample_sequential(spec,m,hidden)
                finally: spec.model=model
            def owned_loop():
                assert model.sample_sequential(spec,m,hidden)
            for mag in (0,.01,1,8):
                base.normal_().mul_(mag);spec.draft_tokens.fill_(-9)
                native_loop()
                expected=spec.draft_tokens.clone()
                spec.draft_tokens.fill_(-9)
                owned_loop()
                equal=bool(torch.equal(spec.draft_tokens,expected))
                report['rows'].append(dict(m=m,steps=steps,anchor_dtype=str(anchor_dtype),magnitude=mag,all_ids_and_tail_exact=equal))
                save();assert equal,report['rows'][-1]
            if m in (1,7,64) and anchor_dtype == torch.int64:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream): sample_sequential(model,spec,m,hidden)
                stream.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): sample_sequential(model,spec,m,hidden)
                for replay in range(3):
                    base.normal_();spec.input_buffers.input_ids.random_(v)
                    native_loop();expected=spec.draft_tokens.clone()
                    spec.draft_tokens.fill_(-9);graph.replay()
                    equal=bool(torch.equal(spec.draft_tokens,expected))
                    report['graphs'].append(dict(m=m,steps=steps,anchor_dtype=str(anchor_dtype),replay=replay,all_ids_and_tail_exact=equal))
                    save();assert equal,report['graphs'][-1]
            if steps == 5 and anchor_dtype == torch.int64 and m in (1,7,17,32,64):
                graphs=[]
                for function in (native_loop,owned_loop):
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(5):function()
                    stream.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(4):function()
                    graphs.append(graph)
                samples=[[],[]]
                for turn in range(7):
                    for which in ((0,1) if turn%2==0 else (1,0)):
                        begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        begin.record()
                        for _ in range(8):graphs[which].replay()
                        end.record();end.synchronize()
                        samples[which].append(begin.elapsed_time(end)*1000/32)
                baseline,candidate=map(statistics.median,samples)
                report['timings'].append(dict(m=m,steps=steps,native_us=baseline,
                    candidate_us=candidate,speedup=baseline/candidate,samples_us=samples))
                del graphs
                save()
            print(json.dumps(dict(m=m,steps=steps,dtype=str(anchor_dtype),cases=len(report['rows']))),flush=True)
            changes={'draft_logits':torch.empty(1,device='cuda'),'draft_watermarker':object(),
                     '_d2t_scatter_index':object(),'_draft_topk':object(),
                     'use_confidence_head':True,'acceptance_estimator':object(),
                     'num_speculative_steps':8}
            for name,value in changes.items():
                old=getattr(spec,name);setattr(spec,name,value);spec.draft_tokens.fill_(-7)
                assert not sample_sequential(model,spec,m,hidden)
                assert bool((spec.draft_tokens==-7).all())
                setattr(spec,name,old);report['fallbacks'].append(dict(m=m,steps=steps,anchor_dtype=str(anchor_dtype),field=name,untouched=True))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                assert not sample_sequential(model,spec,m,hidden)
            save()
        report['passed']=True
    except Exception as exc:report['error']=repr(exc)
    save();raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
