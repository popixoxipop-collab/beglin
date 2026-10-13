#!/usr/bin/env python3
"""Fresh-process real DeepSeek corpus: true post-softmax Top-6/7 boundary per layer.

Only XOX run-local scratch contains raw h2, full logits, routing and all p6/p7
records. Upload only aggregate predictions and their SHA-bound receipts.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,subprocess,time
from transformers import AutoTokenizer

SCHEMA="beglin-p3-120prompt-full-softmax-boundary-v1"
BANK_FILES=("astronomy_physics_chemistry.json","biology_mathematics_computing.json",
            "ml_robotics_energy.json","economics_history_language.json")
EXPECTED_INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
HIDDEN=2048
ROUTED_LAYERS=tuple(range(1,27))
PROMPT_COUNT=120

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):h.update(part)
    return h.hexdigest()

def catalog(directory):
    directory=pathlib.Path(directory)
    items=[];categories=set()
    for filename in BANK_FILES:
        p=directory/filename
        if not p.is_file() or p.is_symlink():raise ValueError("untrusted/missing prompt bank")
        data=json.loads(p.read_text())
        if not isinstance(data,dict) or len(data)!=3:raise ValueError("exactly three domains per file")
        for category,texts in data.items():
            if category in categories or len(texts)!=10:raise ValueError("prompt domain count or duplicate")
            categories.add(category)
            for j,text in enumerate(texts):
                if not isinstance(text,str) or len(text)<35:raise ValueError("invalid source prompt")
                split="train" if j<6 else ("validation" if j<8 else "holdout")
                items.append((split,category,text))
    if len(items)!=PROMPT_COUNT or len(categories)!=12 or len({t[2].casefold() for t in items})!=PROMPT_COUNT:
        raise ValueError("invalid exact preregistered 120-prompt catalog")
    return items

def execution(binary,model_root,tok,ids,root,*,hook:bool):
    root.mkdir(parents=True,exist_ok=False)
    values=[int(z) for z in ids]
    text=",".join(map(str,values))
    if len(values)<8 or len(values)>24 or len(text)>=900 or any(z<0 or z>=102400 for z in values):
        raise ValueError("invalid real prompt token sequence")
    env={k:os.environ[k] for k in ("PATH","HOME","TMPDIR","LANG") if k in os.environ}
    env.update({"QWEN_MOE_BASE":str(model_root),
                "QWEN_MOE_PROMPT_IDS":text,
                "QWEN_MOE_NEARTIE_CORRECT":"0",
                "BEGLIN_CAPTURE_PRE_FFN_INPUT":str(root/"true_h2.bin")})
    if hook:
        env["BEGLIN_P3_ROUTER_BOUNDARY"]=str(root/"true_boundary.txt")
    start=time.monotonic()
    result=subprocess.run([str(binary)],cwd=root,env=env,text=True,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100)
    if result.returncode:
        raise RuntimeError(f"isolated DeepSeek scratch forward exit={result.returncode}; tail={result.stderr[-240:]}")
    out=(root/"moe3a_c_logits.bin",root/"moe3a_c_routing.txt",root/"true_h2.bin")
    if any(not p.is_file() or p.is_symlink() or p.stat().st_size==0 for p in out):
        raise ValueError("missing exact scratch forward output")
    if (root/"true_h2.bin").stat().st_size!=len(values)*(12+HIDDEN*4):
        raise ValueError("layer-1 hidden token count mismatch")
    trace=root/"true_boundary.txt"
    if hook:
        if not trace.is_file() or trace.is_symlink():
            raise ValueError("true full-softmax boundary trace missing")
        # One observation per routed layer per token. Do not export raw trace.
        lines=trace.read_text().splitlines()
        if len(lines)!=len(values)*len(ROUTED_LAYERS):
            raise ValueError(f"true rank7 trace size mismatch {len(lines)} != {len(values)*len(ROUTED_LAYERS)}")
    return {"positions":len(values),
            "elapsed_s":float(time.monotonic()-start),
            "h2_sha256":sha(out[2]),
            "routing_sha256":sha(out[1]),
            "logits_sha256":sha(out[0]),
            "boundary_sha256":sha(trace) if hook else None}

def capture(binary,index,tokenizer_dir,model_root,prompt_bank_dir,out):
    binary=pathlib.Path(binary).resolve(strict=True)
    index=pathlib.Path(index).resolve(strict=True)
    model_root=pathlib.Path(model_root).resolve(strict=True)
    tokenizer_dir=pathlib.Path(tokenizer_dir).resolve(strict=True)
    if sha(index)!=EXPECTED_INDEX_SHA:
        raise ValueError("DeepSeek checkpoint changed from preregistered model")
    if out.exists():raise ValueError("scratch suite already exists (never overwrite evidence)")
    out.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir),trust_remote_code=False,
                                             local_files_only=True,use_fast=True)
    entries=catalog(prompt_bank_dir);seen=set();records=[];start=time.monotonic()
    for i,(split,category,text) in enumerate(entries):
        values=tokenizer.encode(text,add_special_tokens=True)[:24]
        key=",".join(map(str,values))
        if key in seen:raise ValueError("duplicate tokenized prompt")
        seen.add(key)
        pid=f"p{i:03d}";directory=out/pid
        data=execution(binary,model_root,tokenizer,values,directory,hook=True)
        # Raw logits/route hashes must be IDENTICAL with/without the boundary
        # instrumentation on the FIRST prompt, proving the hook does not alter
        # selected TopK or the original full-forward logits.
        if i==0:
            old=execution(binary,model_root,tokenizer,values,out/"hook_off_control",hook=False)
            if old["routing_sha256"]!=data["routing_sha256"] or old["logits_sha256"]!=data["logits_sha256"]:
                raise ValueError("router measurement hook changed model output")
            print("P3_TRUE_BOUNDARY_HOOK_NONINVASIVE_PASS",
                  json.dumps({"identical_logits_and_routing":True,"tokens":data["positions"]}),flush=True)
        records.append({"id":pid,"split":split,"domain":category,
                 "positions":data["positions"],"h2_sha256":data["h2_sha256"],
                 "routing_sha256":data["routing_sha256"],
                 "boundary_sha256":data["boundary_sha256"],
                 "logits_sha256":data["logits_sha256"],
                 "token_sequence_sha256":hashlib.sha256(key.encode()).hexdigest()})
        if (i+1)%12==0:
            print("P3_REAL_PROMPT_PROGRESS",json.dumps({"prompts":i+1,
                 "tokens":sum(r["positions"] for r in records),
                 "elapsed_s":round(time.monotonic()-start,1)}),flush=True)
    counts={s:sum(x["split"]==s for x in records) for s in ("train","validation","holdout")}
    if counts!={"train":72,"validation":24,"holdout":24}:
        raise ValueError("split cardinality failed")
    doc={"schema":SCHEMA,"source":"actual quantized AF DeepSeek layer-1 h2 and all routed layer rank6/7 probabilities BEFORE topk renormalization",
        "source_index_sha256":sha(index),"source_binary_sha256":sha(binary),
        "prompt_split_counts":counts,"prompt_count":len(records),
        "all_routed_layers":list(ROUTED_LAYERS),
        "causal_fresh_process_per_prompt":True,
        "capture_positions_total":sum(r["positions"] for r in records),
        "raw_logits_uploaded":False,"raw_hidden_uploaded":False,
        "production_touched":False,"auto_promotion_allowed":False,
        "records":records}
    (out/"runs_p3.json").write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n")
    print("P3_REAL_CORPUS_COMPLETE",json.dumps({"prompts":len(records),
        "tokens":doc["capture_positions_total"],"split":counts,
        "elapsed_s":round(time.monotonic()-start,2)},sort_keys=True),flush=True)
    return doc

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("binary","checkpoint-index","tokenizer-dir","model-root","prompt-bank-dir","output"):
        p.add_argument("--"+name,required=True,type=pathlib.Path)
    a=p.parse_args()
    capture(a.binary,a.checkpoint_index,a.tokenizer_dir,a.model_root,a.prompt_bank_dir,a.output)
