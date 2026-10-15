#!/usr/bin/env python3
"""Independent genuine P17 original BF16 12-question/31-gold-logit verification.

Recalculates teacher token NLL from every 102400-dimensional checkpoint
full-model logit row and source input tokens; does not invent native A/B data.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,statistics
import numpy as np
FROZEN="a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521"
MODEL="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def tok(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def audit(frozen,folder):
    frozen,folder=map(pathlib.Path,(frozen,folder))
    if sha(frozen)!=FROZEN or not folder.is_dir() or folder.is_symlink():
        raise ValueError("P17 preregistered token set or private genuine evidence root missing")
    bank=json.loads(frozen.read_text())
    if bank.get("question_count")!=12 or bank.get("gold_token_prefix_count")!=31:
        raise ValueError("P17 original cohort wrong")
    first=[];hashes=[];all_nll=[]
    observed_files=set()
    for row in bank["items"]:
        name=row["id"]
        report=folder/(name+".json")
        logits=folder/(name+".bf16_logits.npy")
        source=folder/(name+".source_hashes.json")
        if any(p.is_symlink() or not p.is_file() for p in (report,logits,source)):
            raise ValueError("P17 missing original BF16 complete model result "+name)
        observed_files|={report.name,logits.name,source.name}
        z=json.loads(report.read_text())
        if (z.get("schema")!="beglin-hcp-p17-original-complete-BF16-teacher-gold-span-v1" or
            z.get("status")!="FULL_BF16_DEEPSEEK_27L_5291T_GOLD_LOGIT_WINDOW_COMPLETE" or
            z.get("whole_model_teacher_executed") is not True or
            z.get("decoder_layers_forwarded")!=27 or
            z.get("original_source_tensors_exactly_once")!=5291 or
            z.get("original_full_vocab")!=102400 or
            z.get("reference_arithmetic")!="BF16" or
            z.get("source_checkpoint_index_sha256")!=MODEL or
            z.get("input_token_sha256")!=row["prompt_token_sha256"] or
            z.get("full_context_token_sha256")!=row["teacher_full_context_token_sha256"] or
            z.get("gold_answer_token_sha256")!=row["gold_answer_sha256"] or
            z.get("gold_answer_token_count")!=len(row["gold_answer_ids"]) or
            z.get("prompt_tokens")!=len(row["prompt_ids"]) or
            z.get("logits_NPY_sha256")!=sha(logits) or
            z.get("source_hashes_manifest_sha256")!=sha(source) or
            z.get("production_touched") is not False or
            z.get("GPU_used") is not False or
            z.get("automatic_live_promotion") is not False or
            z.get("end_to_end_noninferiority_certified") is not False):
            raise ValueError("P17 source/teacher whole-model semantics incomplete "+name)
        live={"stage_sha256":STAGE,"route_sha256":ROUTE,"pids":[35481,35493]}
        if z["production_before"]!=live or z["production_after"]!=live:
            raise ValueError("P17 native production changed")
        value_hashes=json.loads(source.read_text())
        if len(value_hashes)!=5291 or any(len(s)!=64 for s in value_hashes.values()):
            raise ValueError("Original BF16 value SHAs for all source tensors missing")
        hashes.append(sha(source))
        a=np.load(logits,allow_pickle=False)
        gold=row["gold_answer_ids"]
        if a.dtype!=np.float32 or a.shape!=(len(gold),102400) or not np.isfinite(a).all():
            raise ValueError("P17 full teacher logits malformed")
        t=a.astype(np.float64)
        m=t.max(axis=1)
        lse=m+np.log(np.exp(t-m[:,None]).sum(axis=1))
        nll=lse-t[np.arange(len(gold)),gold]
        if (not np.isfinite(nll).all() or not np.all(nll>=0) or
            not np.allclose(nll,z["gold_nll_nats"],rtol=1e-10,atol=1e-7) or
            not math.isclose(float(nll.mean()),z["mean_gold_nll_nats"],abs_tol=1e-7)):
            raise ValueError("P17 original torch teacher gold losses not independently reproducible")
        if (z["logits_shape"]!=list(a.shape) or len(z["per_layer"])!=27 or
            [d["layer"] for d in z["per_layer"]]!=list(range(27)) or
            z["peak_process_RSS_bytes"]>6*1024**3 or
            z["max_additional_swap_bytes"]>1024**3):
            raise ValueError("P17 full teacher numerics/memory/causal layer witness invalid")
        all_nll.extend(nll.tolist())
        first.append({"item_id":name,"domain":row["domain"],"gold_token_count":len(gold),
             "teacher_BF16_mean_gold_NLL_nats":float(nll.mean()),
             "token_gold_NLL_nats":[float(x) for x in nll],
             "source_teacher_report_sha256":sha(report),
             "actual_original_full_logits_SHA256":sha(logits),
             "source_5291_BF16_weight_hashes_manifest_SHA256":sha(source),
             "full_decoder_layers":27,
             "peak_process_RSS_bytes":z["peak_process_RSS_bytes"]})
    if len(set(hashes))!=1 or len(first)!=12 or len(all_nll)!=31:
        raise ValueError("P17 original BF16 tensor source drift or incomplete 31 gold token cohort")
    domain={}
    for row in first:
        domain.setdefault(row["domain"],[]).append(row["teacher_BF16_mean_gold_NLL_nats"])
    result={"schema":"beglin-hcp-p17-independent-original-BF16-goldteacher-NLL-audit-v1",
        "status":"ALL_12_REAL_DEEPSEEK_27_LAYER_TEACHERS_31_GOLD_LOGITS_INDEPENDENTLY_VERIFIED",
        "frozen_input_sha256":FROZEN,
        "question_count":12,"source_tokens_gold":31,"full_vocab_per_position":102400,
        "original_BF16_decoder_layers_per_question":27,
        "original_BF16_source_tensors_per_question":5291,
        "all_12_source_checkpoint_tensor_value_hashes_equal":True,
        "full_model_teacher_question_mean_NLL_nats":statistics.mean([r["teacher_BF16_mean_gold_NLL_nats"] for r in first]),
        "full_model_teacher_pooled_token_mean_NLL_nats":statistics.mean(all_nll),
        "domain_question_means":{d:statistics.mean(v) for d,v in domain.items()},
        "records":first,"teacher_full_model_verified":True,
        "actual_native_A_B_logits_audited":False,
        "noninferiority_certified":False,"external_independent_benchmark":False,
        "production_touched":False,"automatic_live_promotion":False}
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--frozen",required=True);p.add_argument("--private",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();res=audit(a.frozen,a.private)
    with pathlib.Path(a.output).open("x") as f:json.dump(res,f,sort_keys=True,indent=2);f.write("\n")
    print("P17_INDEPENDENT_ORIGINAL_BF16_12_GOLD_NLL_AUDIT_PASS",json.dumps({
       "questions":res["question_count"],"gold":res["source_tokens_gold"],
       "mean_NLL":res["full_model_teacher_question_mean_NLL_nats"],
       "full_teacher":True,"native_AB_compared":False},sort_keys=True))
