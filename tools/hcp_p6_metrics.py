#!/usr/bin/env python3
"""Full-vocabulary matched-context metrics and crossed-block bootstrap."""
from __future__ import annotations
import hashlib,math,pathlib,struct
import numpy as np
MAGIC=0x50364c47

def read_dump(path, *, positions:int, vocab:int=102400):
    path=pathlib.Path(path)
    if positions<1 or vocab<2: raise ValueError('invalid expected shape')
    if path.stat().st_size != positions*(16+4*vocab):
        raise ValueError('incomplete, extra or duplicated logit bytes')
    rows=[]
    with path.open('rb') as f:
        for pos in range(positions):
            magic,slot,actual,n=struct.unpack('<4I',f.read(16))
            if (magic,slot,actual,n)!=(MAGIC,0,pos,vocab):
                raise ValueError('logit identity, slot, position, or vocabulary mismatch')
            row=np.frombuffer(f.read(4*vocab),dtype='<f4')
            if row.size!=vocab or not np.isfinite(row).all():
                raise ValueError('nonfinite or incomplete logits')
            rows.append(row)
    return np.stack(rows)

def logprob(x):
    x=np.asarray(x,dtype=np.float64)
    z=x-x.max(axis=-1,keepdims=True)
    return z-np.log(np.exp(z).sum(axis=-1,keepdims=True))

def logit_metrics(reference, candidate):
    a=np.asarray(reference);b=np.asarray(candidate)
    if a.ndim!=2 or a.shape!=b.shape or a.size==0:
        raise ValueError('matched nonempty logit matrices required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite logits')
    sq=base=kl=dot=bb=0.0;maximum=0.0;agreements=0
    for i in range(0,a.shape[0],8):
        x=a[i:i+8].astype(np.float64);y=b[i:i+8].astype(np.float64)
        d=y-x;sq+=float(np.square(d).sum());base+=float(np.square(x).sum())
        dot+=float((x*y).sum());bb+=float(np.square(y).sum())
        maximum=max(maximum,float(np.max(np.abs(d))))
        pa=logprob(x);pb=logprob(y)
        kl+=float((np.exp(pa)*(pa-pb)).sum())
        agreements+=int((x.argmax(1)==y.argmax(1)).sum())
    return {'positions':int(a.shape[0]),'vocabulary':int(a.shape[1]),
            'rmse':math.sqrt(sq/a.size),'relative_l2':math.sqrt(sq/max(base,1e-300)),
            'max_abs':maximum,'cosine':dot/max(math.sqrt(base*bb),1e-300),
            'kl_af_to_candidate_mean':max(0.0,kl/a.shape[0]),
            'top1_agree_count':agreements,'top1_agreement':agreements/a.shape[0]}

def teacher_forced(logits, token_ids):
    x=np.asarray(logits)
    ids=np.asarray(token_ids,dtype=np.int64)
    if x.ndim!=2 or len(ids)<2 or x.shape[0]!=len(ids):
        raise ValueError('one logit row per original input position required')
    if ids.min()<0 or ids.max()>=x.shape[1]:raise ValueError('token outside vocabulary')
    loss=0.0;correct=0;n=len(ids)-1
    for i in range(0,n,8):
        j=min(i+8,n); lp=logprob(x[i:j]); truth=ids[i+1:j+1]
        loss-=float(lp[np.arange(j-i),truth].sum())
        correct+=int((x[i:j].argmax(1)==truth).sum())
    return {'predicted_input_tokens':n,'nll_sum':loss,'mean_nll':loss/n,
            'perplexity':math.exp(loss/n),'correct_next_input_tokens':correct,
            'next_input_token_accuracy':correct/n,
            'is_task_answer_accuracy':False}

def distribution(values):
    a=np.asarray(values,dtype=float)
    if a.size==0 or not np.isfinite(a).all():raise ValueError('finite measurements required')
    return {'count':int(a.size),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),
            'p95':float(np.quantile(a,.95)),'max':float(a.max())}

def crossed_ci(matrix,*,seed=20261014,draws=20000):
    x=np.asarray(matrix,dtype=float)
    if x.ndim!=2 or min(x.shape)<2 or not np.isfinite(x).all():
        raise ValueError('multiple complete blocks and prompts required')
    rng=np.random.default_rng(seed);nb,nprompt=x.shape
    wb=rng.multinomial(nb,np.full(nb,1/nb),size=draws)/nb
    wp=rng.multinomial(nprompt,np.full(nprompt,1/nprompt),size=draws)/nprompt
    means=np.einsum('bi,ij,bj->b',wb,x,wp,optimize=True)
    return {'mean_B_minus_A_ms':float(x.mean()),
            'crossed_prompt_block_bootstrap_95ci_ms':[float(v) for v in np.quantile(means,[.025,.975])],
            'blocks':nb,'prompts':nprompt,'draws':draws,'seed':seed,
            'upper_ci_below_zero':bool(np.quantile(means,.975)<0)}

def summarize_speed(records,*,blocks=6,prompts=12):
    if len(records)!=blocks*prompts*4:raise ValueError('not full preregistered speed sample')
    result={}
    for phase in ('switch','steady','equal_mix'):
        rows=[r for r in records if phase=='equal_mix' or r['phase']==phase]
        policies={name:{k:distribution([r[k] for r in rows if r['policy']==name])
                        for k in ('admission_wall_ms','engine_ms','roundtrip_ms','transition_ms')}
                  for name in ('A','B')}
        comparisons={}
        for metric in ('admission_wall_ms','engine_ms','roundtrip_ms'):
            delta=np.zeros((blocks,prompts))
            for block in range(blocks):
                for prompt in range(prompts):
                    pair=[r for r in rows if r['block']==block and r['prompt_index']==prompt]
                    expected=2 if phase!='equal_mix' else 4
                    if len(pair)!=expected:raise ValueError('missing or duplicate paired measurement')
                    d={name:np.mean([r[metric] for r in pair if r['policy']==name]) for name in ('A','B')}
                    delta[block,prompt]=d['B']-d['A']
            comparisons[metric]=crossed_ci(delta)
        result[phase]={'per_policy':policies,'paired_comparisons':comparisons}
    return result
