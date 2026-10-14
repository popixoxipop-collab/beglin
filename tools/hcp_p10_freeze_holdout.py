#!/usr/bin/env python3
"""P10 freeze of first-use short-answer questions (no model inference).

Run exactly once before native trials; verify no SHA overlap with sealed P4/P6
original prompts. No raw prompt token IDs are exported in the manifest.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib
from transformers import AutoTokenizer

MODEL="DeepSeek-V2-Lite"
SCHEMA="beglin-hcp-p10-new-heldout-token-manifest-v1"
OLD_P4="c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def digest_ids(ids):
    return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()

def freeze(bank,tokenizer,p4,output):
    bank,tokenizer,p4,output=map(pathlib.Path,(bank,tokenizer,p4,output))
    if output.exists() or any(not x.is_file() or x.is_symlink() for x in (bank,p4)):
        raise ValueError("new holdout must be frozen once from regular inputs")
    data=json.loads(bank.read_text())
    if data.get("schema")!="beglin-hcp-p10-preregistered-new-short-answer-v1" or data.get("seed")!=20261014:
        raise ValueError("unrecognized precommitted holdout bank")
    if len(data["items"])!=12 or len({r["domain"] for r in data["items"]})!=12:
        raise ValueError("missing distinct domain questions")
    if sha(p4)!=OLD_P4:raise ValueError("P4 original manifest changed")
    old={x["token_sha256"] for x in json.loads(p4.read_text())["runs"]}
    assert len(old)==36
    tok=AutoTokenizer.from_pretrained(str(tokenizer),local_files_only=True,trust_remote_code=False,use_fast=True)
    new=[];seen=set()
    for row in data["items"]:
        prompt=(data["instruction"]+" Task domain: "+row["domain"]+
          ". Question: "+row["question"]+" Answer:")
        ids=tok.encode(prompt,add_special_tokens=True)
        if not (20<=len(ids)<=112 and len(ids)+int(data["token_budget"])<128):
            raise ValueError("P10 input context exceeds research KV128")
        if any(type(x) is not int or x<0 or x>=102400 for x in ids):
            raise ValueError("invalid DeepSeek token identifier")
        encoded=",".join(map(str,ids))
        digest=digest_ids(ids)
        if digest in old or digest in seen:
            raise ValueError("P10 new test overlaps sealed P4/P6")
        seen.add(digest)
        new.append({"id":row["id"],"domain":row["domain"],
                    "token_sha256":digest,"tokens":len(ids),
                    "prompt_text_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                    "reference_answer_sha256":hashlib.sha256(row["expected_short_answer"].encode()).hexdigest(),
                    "generation_budget":int(data["token_budget"])})
    report={"schema":SCHEMA,"seed":20261014,"bank_sha256":sha(bank),
       "reference_P4_sha256":OLD_P4,"model":MODEL,"tokenizer_index_sha256":"d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf",
       "first_use_only_relative_to_original_P4":True,
       "not_external_independent_benchmark":True,"full_BF16_reference_not_measured":True,
       "records":new,"count":12,"raw_token_ids_exported":False,
       "automatic_live_promotion":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as f:json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
    print("P10_PREREGISTERED_HELDOUT_FROZEN",json.dumps({
      "count":12,"new_domain_sha_disjoint_from_P4":True,
      "lengths":[min(x["tokens"] for x in new),max(x["tokens"] for x in new)],
      "bank_sha256":sha(bank),"manifest_sha256":sha(output)},sort_keys=True))

if __name__=="__main__":
    a=argparse.ArgumentParser()
    for k in ("bank","tokenizer","p4","output"):a.add_argument("--"+k,required=True)
    n=a.parse_args();freeze(n.bank,n.tokenizer,n.p4,n.output)
