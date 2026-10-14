#!/usr/bin/env python3
"""P17 one-shot tokenizer-precise frozen multistep gold eval.

Requires preauthored SHA-locked question bank before ANY inference. Detects
gold-token BPE boundary change and overlap with P4/P10/P15 prior tested prompts.
Only frozen input/prefix hashes and answer token ids are used by native observer.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib
from transformers import AutoTokenizer
BANK_SHA="dee6f0741f7e4b9e47fcd68c87cf69296add7a30df457ffc36c06908efce5744"
P4_SHA="c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53"
P10_SHA="93f6f9f21983cb0c93e78b813b2972724aff45501b773943be2020574ada3b9f"
P15_SHA="c85d16c113780d73692fd0bdb9b1afaacb40bda9a998876420f361899d3cbb8b"
MODEL_INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def tokenhash(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def freeze(bank,model,p4,p10,p15,output):
    paths=[pathlib.Path(p) for p in (bank,model,p4,p10,p15,output)]
    bank,model,p4,p10,p15,output=paths
    if output.exists() or any(p.is_symlink() or not p.is_file() for p in (bank,p4,p10,p15)):
        raise ValueError("P17 input freeze must create new file from regular preknown sources")
    if any(a!=b for a,b in ((sha(bank),BANK_SHA),(sha(p4),P4_SHA),(sha(p10),P10_SHA),(sha(p15),P15_SHA),(sha(model/"model.safetensors.index.json"),MODEL_INDEX_SHA))):
        raise ValueError("P17 source input corpus/model/prior SHA drifted")
    j=json.loads(bank.read_text())
    if (j.get("schema")!="beglin-hcp-p17-amended-before-inference-internal-holdout-v2"
        or j.get("total")!=12 or len(j.get("items",[]))!=12):
        raise ValueError("P17 bank differs from original 12 question prereg")
    tok=AutoTokenizer.from_pretrained(str(model),trust_remote_code=False,local_files_only=True,use_fast=True)
    old4={x["token_sha256"] for x in json.loads(p4.read_text())["runs"]}
    old10={x["token_sha256"] for x in json.loads(p10.read_text())["records"]}
    old15={tokenhash(x["token_ids"]) for x in json.loads(p15.read_text())["entries"]}
    if len(old4)!=36 or len(old10)!=12 or len(old15)!=2:
        raise ValueError("P17 prior prompt count changed")
    exclude=old4|old10|old15
    seen=set();prefixes=set();questions=[]
    for item in j["items"]:
        prompt=j["prompt_template"].replace("{question}",item["question"])
        whole=prompt+item["expected_answer"]
        ids=tok.encode(prompt,add_special_tokens=True)
        completed=tok.encode(whole,add_special_tokens=True)
        if completed[:len(ids)]!=ids:
            raise ValueError(f"BPE prefix/answer boundary changes input token ids: {item['id']}")
        answer_ids=completed[len(ids):]
        if (len(answer_ids)<1 or len(answer_ids)>8 or len(ids)<14 or len(ids)+len(answer_ids)>112
            or len(ids)+j["generation_budget"]>=128):
            raise ValueError(f"P17 unsafe context/answer length: {item['id']}, prompt={len(ids)}, answer={len(answer_ids)}")
        if any(type(x)!=int or not 0<=x<102400 for x in completed):
            raise ValueError("P17 unexpected token id")
        psha=tokenhash(ids)
        if psha in seen or psha in exclude:raise ValueError("P17 prompt overlaps prior observed corpus")
        seen.add(psha)
        forced=[]
        for step,target in enumerate(answer_ids):
            prefix=ids+answer_ids[:step]
            digest=tokenhash(prefix)
            if digest in prefixes or digest in exclude:
                raise ValueError("P17 repeated teacher-forced input SHA")
            prefixes.add(digest)
            forced.append({"step":step,"prefix_len":len(prefix),"prefix_token_sha256":digest,
                            "gold_next_token_id":target})
        questions.append({"id":item["id"],"domain":item["domain"],
             "prompt":prompt,"reference_answer":item["expected_answer"],
             "prompt_ids":ids,"prompt_token_sha256":psha,
             "gold_answer_ids":answer_ids,"gold_answer_sha256":tokenhash(answer_ids),
             "teacher_full_context_ids":completed,
             "teacher_full_context_token_sha256":tokenhash(completed),
             "expected_gold_token_count":len(answer_ids),
             "forced_prefixes":forced,"generation_budget":j["generation_budget"]})
    if (len(seen)!=12 or len(prefixes)!=sum(x["expected_gold_token_count"] for x in questions)
        or sum(x["expected_gold_token_count"]>=2 for x in questions)<11):
        raise ValueError("P17 source not unique or missing forced prefixes")
    record={"schema":"beglin-hcp-p17-token-exact-frozen-12-internal-multitoken-gold-eval-v1",
        "source_bank_SHA256":BANK_SHA,"original_model_index_sha256":MODEL_INDEX_SHA,
        "source_P4_frozen_sha256":P4_SHA,"source_P10_new_frozen_sha256":P10_SHA,
        "source_P15_frozen_sha256":P15_SHA,"gold_token_prefix_count":len(prefixes),
        "first_use_disjoint_from_P4_P10_P15_exact_token_SHA":True,
        "question_count":len(questions),"items":questions,
        "source_answers_were_precommitted_before_inference":True,
        "external_independent_benchmark":False,"unknown_original_pretraining_contamination":True,
        "reference_arithmetic":"BF16 CPU original P15 whole 27-layer model",
        "fixed_policy_A_comparator":True,"policy_B_research_challenger":True,
        "margin_nll_nats_per_token":float(j["evaluation"]["noninferiority_margin_nats_per_token"]),
        "paired_bootstrap_seed":20261015,"paired_bootstrap_resamples":10000,
        "strict_answer_gold_token_prefix_accuracy":True,
        "automatic_promotion":False,"production_touched":False}
    with output.open("x") as f:json.dump(record,f,sort_keys=True,indent=2);f.write("\n")
    print("P17_TOKEN_EXACT_PREREGISTRATION_SUCCESS",json.dumps({
        "questions":12,"forced_gold_tokens":len(prefixes),
        "prompt_lengths":[min(len(x["prompt_ids"]) for x in questions),max(len(x["prompt_ids"]) for x in questions)],
        "gold_lengths":[min(x["expected_gold_token_count"] for x in questions),max(x["expected_gold_token_count"] for x in questions)],
        "output_sha256":sha(output)},sort_keys=True))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for f in ("bank","model","p4","p10","p15","output"):ap.add_argument("--"+f,required=True)
    o=ap.parse_args();freeze(o.bank,o.model,o.p4,o.p10,o.p15,o.output)
