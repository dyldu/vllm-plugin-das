# SPDX-License-Identifier: Apache-2.0
"""Check the owned sampling entry against the actual native DSpark loop."""
import argparse
import json
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
    from hcu_vllm_das.dspark_greedy_admission import supported, sample_sequential
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
    model=Native.__new__(Native);torch.nn.Module.__init__(model)
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
    report=dict(passed=False,rows=[],graphs=[],fallbacks=[])
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    torch.manual_seed(20261125);save()
    try:
        for m in (1,2,4,8,16,32):
            base=torch.empty((m*5,v),device='cuda',dtype=torch.bfloat16)
            model.compute_draft_logits=MethodType(lambda self,hidden:base,model)
            spec=Spec.__new__(Spec)
            if isinstance(spec,torch.nn.Module):torch.nn.Module.__init__(spec)
            spec.model=model;spec.num_speculative_steps=5;spec._draft_topk=None
            spec.draft_logits=None;spec.draft_watermarker=None;spec._d2t_scatter_index=None
            spec.use_confidence_head=False;spec.acceptance_estimator=None
            spec.sample_indices=torch.arange(m*5,device='cuda')
            spec.sample_idx_mapping=torch.arange(m,device='cuda').repeat_interleave(5)
            spec.sample_pos=torch.ones(m*5,device='cuda',dtype=torch.int64)
            spec._step_cols=list(range(5))
            spec.input_buffers=SimpleNamespace(input_ids=torch.randint(v,(m,),device='cuda'))
            spec._anchor_idx=torch.arange(m,device='cuda')
            spec.draft_tokens=torch.empty((m+1,5),device='cuda',dtype=torch.int64)
            hidden=torch.zeros((m*5,5120),device='cuda',dtype=torch.bfloat16)
            assert supported(model,spec,m,hidden),'Actual native entry not admitted'
            for mag in (0,.01,1,8):
                base.normal_().mul_(mag);spec.draft_tokens.fill_(-9)
                Spec._sample_sequential(spec,m,hidden)
                expected=spec.draft_tokens.clone()
                spec.draft_tokens.fill_(-9)
                assert sample_sequential(model,spec,m,hidden)
                equal=bool(torch.equal(spec.draft_tokens,expected))
                report['rows'].append(dict(m=m,magnitude=mag,all_ids_and_tail_exact=equal))
                save();assert equal,report['rows'][-1]
            if m in (1,8,32):
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream): sample_sequential(model,spec,m,hidden)
                stream.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): sample_sequential(model,spec,m,hidden)
                for replay in range(3):
                    base.normal_();spec.input_buffers.input_ids.random_(v)
                    Spec._sample_sequential(spec,m,hidden);expected=spec.draft_tokens.clone()
                    spec.draft_tokens.fill_(-9);graph.replay()
                    equal=bool(torch.equal(spec.draft_tokens,expected))
                    report['graphs'].append(dict(m=m,replay=replay,all_ids_and_tail_exact=equal))
                    save();assert equal,report['graphs'][-1]
            changes={'draft_logits':torch.empty(1,device='cuda'),'draft_watermarker':object(),
                     '_d2t_scatter_index':object(),'_draft_topk':object(),
                     'use_confidence_head':True,'acceptance_estimator':object(),
                     'num_speculative_steps':4}
            for name,value in changes.items():
                old=getattr(spec,name);setattr(spec,name,value);spec.draft_tokens.fill_(-7)
                assert not sample_sequential(model,spec,m,hidden)
                assert bool((spec.draft_tokens==-7).all())
                setattr(spec,name,old);report['fallbacks'].append(dict(m=m,field=name,untouched=True))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                assert not sample_sequential(model,spec,m,hidden)
            save()
        report['passed']=True
    except Exception as exc:report['error']=repr(exc)
    save();raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
