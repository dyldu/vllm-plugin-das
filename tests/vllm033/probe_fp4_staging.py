"""Native gather/repack parity and actual LightOp decode-consumer probe."""
import argparse,hashlib,importlib.util,json,random,statistics,sys
from pathlib import Path

import torch
from hcu_vllm_das import fp4_indexer_lightop as current


def load_original(path):
    name='_w4_fp4_staging_original'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def capture(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):fn()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=fn()
    stream.synchronize()
    return graph,out


def exact(a,b):
    return a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def run(args):
    torch.cuda.set_device(0);torch.manual_seed(83651)
    original=load_original(args.original)
    provider=current.load_providers()
    report=dict(passed=False,original=str(args.original),original_sha256=hashlib.sha256(args.original.read_bytes()).hexdigest(),eager=[],graphs=[],consumer=[],timing=[])
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    with torch.inference_mode():
        for records in (32,128):
            for rows in (1,8,48,64):
                for context in (65,4096,32768):
                    count=(context+records-1)//records;pages=rows*count
                    # Outer padding and offset detect assumptions about contiguous
                    # pages; native cache is planar within each physical page.
                    stride=records*68+256
                    backing=torch.randint(0,256,(pages,stride+4),dtype=torch.uint8,device='cuda')
                    cache=backing[:,4:4+records*68].as_strided((pages,records,68),(backing.stride(0),68,1))
                    table=torch.arange(pages,device='cuda',dtype=torch.int32).reshape(rows,count)
                    lengths=torch.empty(rows,device='cuda',dtype=torch.int32)
                    def changed():return current.stage_decode_pages(cache,table,lengths,context,provider.gather)
                    def baseline():return original.stage_decode_pages(cache,table,lengths,context,provider.gather)
                    for case in range(3):
                        table.copy_(table.flip(0));backing.random_(0,256)
                        lens=[(0,1,records-1,records,records+1,context-1,context)[(case+i)%7] for i in range(rows)]
                        lengths.copy_(torch.tensor(lens,device='cuda',dtype=torch.int32))
                        frozen=backing.clone();expected,et=baseline();got,gt=changed();torch.cuda.synchronize()
                        assert exact(expected,got) and exact(et,gt)
                        assert exact(frozen,backing)
                        report['eager'].append(dict(records=records,rows=rows,context=context,case=case,bytes_exact=True,input_unmodified=True));save()
                    graph,(got,gt)=capture(changed)
                    for case in range(3):
                        table.copy_(table.roll(1,0));backing.random_(0,256)
                        lengths.copy_(torch.tensor([(context,0,records+1,1)[(case+i)%4] for i in range(rows)],device='cuda',dtype=torch.int32))
                        frozen=backing.clone();graph.replay();torch.cuda.synchronize()
                        expected,et=baseline();torch.cuda.synchronize()
                        assert exact(expected,got) and exact(et,gt) and exact(frozen,backing)
                        report['graphs'].append(dict(records=records,rows=rows,context=context,case=case,bytes_exact=True,changing_metadata=True));save()
                    del graph,got,gt,expected,et,frozen,backing,cache,table,lengths
        # Native downstream scorer sees identical canonical bytes and tables.
        for records in (32,128):
            for rows in (8,48):
                context=4096;count=context//records;pages=rows*count
                cache=torch.empty(pages,records,68,device='cuda',dtype=torch.uint8)
                raw=cache.view(pages,-1);raw[:,:records*64].random_(0,256);raw[:,records*64:]=127
                table=torch.arange(pages,device='cuda',dtype=torch.int32).reshape(rows,count).flip(0).contiguous()
                lengths=torch.tensor([(0,37,1025,4096)[i%4] for i in range(rows)],device='cuda',dtype=torch.int32)
                q=torch.randint(0,256,(rows,1,32,64),device='cuda',dtype=torch.uint8)
                sf=torch.full((rows,1,32),0x7F7F7F7F,device='cuda',dtype=torch.int32)
                weights=torch.rand(rows,32,device='cuda')/32
                old,ot=original.stage_decode_pages(cache,table,lengths,context,provider.gather)
                new,nt=current.stage_decode_pages(cache,table,lengths,context,provider.gather)
                x=provider.decode(q,sf,old.unsqueeze(2),weights,lengths,ot,None,context,64,None,1)
                y=provider.decode(q,sf,new.unsqueeze(2),weights,lengths,nt,None,context,64,None,1)
                valid=torch.arange(context,device='cuda')[None,:]<lengths[:,None]
                assert exact(x[valid],y[valid]) and torch.isfinite(y[valid]).all().item()
                report['consumer'].append(dict(records=records,rows=rows,native_valid_logits_bytes_exact=True));save()
                # Original and changed graphs execute the full staging wrapper.
                pairs=[('original',original),('fused',current)];graphs=[]
                for label,module in pairs:
                    def batched(module=module):
                        result=None
                        for _ in range(8):
                            result=module.stage_decode_pages(cache,table,lengths,context,provider.gather)
                        return result
                    graph,out=capture(batched)
                    graphs.append((label,graph,out))
                for repeat in range(3):
                    random.Random(83651+repeat).shuffle(graphs)
                    for label,graph,out in graphs:
                        for _ in range(10):graph.replay()
                        torch.cuda.synchronize()
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(100):graph.replay()
                        end.record();end.synchronize()
                        report['timing'].append(dict(records=records,rows=rows,context=context,label=label,repeat=repeat,transactions_per_graph=8,us=begin.elapsed_time(end)*10/8));save()
                del graphs,graph,out,cache,raw,table,lengths,old,ot,new,nt,x,y
    report['passed']=True;save();print(json.dumps({k:(len(v) if isinstance(v,list) else v) for k,v in report.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();run(a)
