#!/usr/bin/env python3
"""Frozen real DeepSeek long-ish prompt registry: 36 disjoint questions × 12 domains.

Uses source-committed original prompts, no generated PII or external data.
The single-forward Beglin verifier has MOE_MAXPOS=160 but its comma-IDs
environment parser copies into a char buf[1024] without a length rejection.
Therefore cap at 112 token IDs and enforce comma-string <900 bytes.
This is a 112-token context stress pilot, NOT a 4K long-context certification.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib
from transformers import AutoTokenizer

SCHEMA="beglin-hcp-p4-36prompt-112token-v1"
MIN_TOKENS=80
MAX_TOKENS=112
MAX_ENV_BYTES=900
BANK_FILES=(
    "astronomy_physics_chemistry.json",
    "biology_mathematics_computing.json",
    "ml_robotics_energy.json",
    "economics_history_language.json")
PAIRS={"train":(0,1),"validation":(4,5),"holdout":(8,9)}

def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def prepare(bank_dir,tokenizer_dir):
    bank_dir=pathlib.Path(bank_dir)
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir),
        trust_remote_code=False,local_files_only=True,use_fast=True)
    domains={};bank_sha={}
    for fn in BANK_FILES:
        p=bank_dir/fn
        if not p.is_file() or p.is_symlink():
            raise ValueError("prompt bank missing or symlinked")
        bank_sha[fn]=digest(p.read_bytes())
        obj=json.loads(p.read_text())
        if len(obj)!=3:raise ValueError("expected 3 domains per source file")
        for domain,questions in obj.items():
            if domain in domains or len(questions)!=10 or len(set(questions))!=10:
                raise ValueError("domain questions duplicated or malformed")
            domains[domain]=questions
    if len(domains)!=12:raise ValueError("expected twelve domains")
    records=[];seen=set()
    for domain in sorted(domains):
        for split,(left,right) in PAIRS.items():
            a,b=domains[domain][left],domains[domain][right]
            # Both unique original questions occur EARLY, before any 112-token cutoff.
            prompt=(
                f"Research note in {domain}. Question one: {a} "
                f"Question two: {b} "
                "Compare these two questions carefully. Begin by defining the "
                "technical terms and the evidence needed to answer each one. "
                "Give a concrete example with a measurable result and a limitation "
                "or counterexample. Separate causal explanations from correlations. "
                "Explain which parts would generalize to a different context and "
                "which parts depend on specific assumptions. Finish by identifying "
                "what a careful experiment would observe before claiming success.")
            whole=tokenizer.encode(prompt,add_special_tokens=True)
            ids=whole[:MAX_TOKENS]
            serialized=",".join(map(str,ids))
            if (not MIN_TOKENS<=len(ids)<=MAX_TOKENS or
                len(serialized.encode())>=MAX_ENV_BYTES or
                any(not isinstance(x,int) or x<0 or x>=102400 for x in ids)):
                raise ValueError(f"source prompt is outside safe verifier constraints: {domain}/{split} length={len(ids)} env={len(serialized)}")
            token_sha=digest(serialized.encode())
            if token_sha in seen:raise ValueError("duplicated tokenized input across train/val/holdout")
            seen.add(token_sha)
            records.append({"id":"p"+str(len(records)).zfill(3),"domain":domain,"split":split,
                "tokens":len(ids),"source_text_sha256":digest(prompt.encode()),
                "token_sha256":token_sha,"truncated_from_tokens":len(whole),
                "token_ids":ids})
    if len(records)!=36 or len(seen)!=36:raise ValueError("invalid 36 independent prompt registry")
    if {s:sum(r["split"]==s for r in records) for s in PAIRS}!={"train":12,"validation":12,"holdout":12}:
        raise ValueError("disjoint split broken")
    return {"schema":SCHEMA,"model":"DeepSeek-V2-Lite",
        "scope":"112-token offline XOX pilot; not general long-context support",
        "prompts":records,
        "bank_file_sha256":bank_sha,
        "distinct_tokenized_input_count":36,
        "prompt_splits":{"train":12,"validation":12,"holdout":12},
        "actual_max_tokens":max(r["tokens"] for r in records),
        "minimum_tokens":min(r["tokens"] for r in records),
        "source_questions_reused_across_splits":False,
        "production_touched":False}

def public_manifest(obj):
    # No raw prompt token list in GitHub Actions artifacts.
    return {**{k:v for k,v in obj.items() if k!="prompts"},
        "prompts":[{k:v for k,v in row.items() if k!="token_ids"} for row in obj["prompts"]]}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--bank-dir",required=True,type=pathlib.Path)
    ap.add_argument("--tokenizer-dir",required=True,type=pathlib.Path)
    ap.add_argument("--public-output",required=True,type=pathlib.Path)
    opts=ap.parse_args()
    corpus=prepare(opts.bank_dir,opts.tokenizer_dir)
    dest=opts.public_output
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(public_manifest(corpus),sort_keys=True,indent=2)+"\n")
    print("P4_LONG_PROMPT_REGISTRY",json.dumps({
        "prompts":len(corpus["prompts"]),
        "splits":corpus["prompt_splits"],
        "tokens":[corpus["minimum_tokens"],corpus["actual_max_tokens"]],
        "bank_sha256":corpus["bank_file_sha256"],
        "raw_token_ids_exported":False},sort_keys=True))
