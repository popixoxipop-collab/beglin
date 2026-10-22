#!/usr/bin/env python3
"""Separate Qv4 receipt auditor: new parser, pinned inputs and native C outputs.

Checks actual log numbers against independent arithmetic and test provenance.
Does not independently recompute logits from checkpoint; no final quality claim.
"""
from __future__ import annotations
import hashlib, json, math, pathlib, re, stat

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v4_masked_answer_20261022"
PRE=ROOT/"configs/research/BEGLIN_Q_V4_MASKED_ANSWER_DEV_PRELOCK_20261022.json"
EXPECTED_PRE="984adbb205bc14e3ff76c1654a3143529fdd07a8ba017e445462cb3707c3677c"
NAMES=("f32","q4","role_random","v2_diag","v3_cov")
EXPECTED={
    "f32":[196,0,0], "q4":[0,0,196],
    "role_random":[0,28,168],"v2_diag":[0,28,168],"v3_cov":[0,28,168],
}
ROW=re.compile(r"^\[qv4-masked\] window (\d+) prefix (\d+) nll ([0-9eE.+-]+) tokens (\d+)$",re.M)
FINAL=re.compile(r"^qv4_masked_nll ([0-9eE.+-]+) tokens (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
COUNTS=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def digest(b):return hashlib.sha256(b).hexdigest()
def secure(path):
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode & 0o777)!=0o600:
        raise RuntimeError("unsafe or missing private evidence: "+str(path))
    return path.read_bytes()
def run():
    if digest(PRE.read_bytes())!=EXPECTED_PRE:raise RuntimeError("preregistration drift")
    prereg=json.loads(PRE.read_bytes())
    if prereg["production_promotion"] is not False or len(prereg["examples"])!=8:
        raise RuntimeError("candidate is not research-only")
    frozen=json.loads(secure(LAB/"inputs_frozen.json"))
    if frozen["preregistration_sha256"]!=EXPECTED_PRE or len(frozen["windows"])!=16:
        raise RuntimeError("wrong held input cohort")
    rawdata=[]; outcomes={}; bins={}
    for name in NAMES:
        meta=json.loads(secure(LAB/name/"native_masked_answer_receipt.json"))
        transcript=secure(LAB/name/"native_masked_nll_log.txt")
        if meta["mode"]!=name or meta["log_sha256"]!=digest(transcript):
            raise RuntimeError("provenance mismatch: "+name)
        if meta["original_bf16_revision"]!="4ae77cb209f06199b8df1c94e21ff341332a3a89":
            raise RuntimeError("BF16 source drift")
        if meta["fixed_input_receipt_sha256"]!=digest(secure(LAB/"inputs_frozen.json")):
            raise RuntimeError("mode input cohort differs")
        if meta["projected_roles"]!=EXPECTED[name] or meta["model_layers"]!=28:
            raise RuntimeError("not original 28-layer precision policy")
        kind="base" if name in ("f32","q4") else "mixed"
        binary=LAB/"bin"/("beglin_qv4_"+kind)
        if digest(binary.read_bytes())!=meta["native_binary_sha256"]:
            raise RuntimeError("native binary changed")
        bins[name]=meta["native_binary_sha256"]
        log=transcript.decode("utf-8")
        counts=COUNTS.findall(log)
        if len(counts)!=1 or list(map(int,counts[0]))!=EXPECTED[name]:
            raise RuntimeError("native C had wrong quant roles")
        matches=ROW.findall(log)
        if len(matches)!=16:
            raise RuntimeError("no evidence of 16 genuine answer-token logits")
        costs=[]
        for index,hit in enumerate(matches):
            serial,prefix,value,num=hit
            frozenw=frozen["windows"][index]
            if int(serial)!=index or int(prefix)!=frozenw["prefix"] or int(num)!=1:
                raise RuntimeError("native loss not solely one suffix token")
            ids=secure(LAB/name/f"eval/ppl_work/w{index}.i32")
            if digest(ids)!=frozenw["ids_sha256"]:
                raise RuntimeError("evaluation prompt tokens drift")
            k=secure(LAB/name/f"eval/ppl_work/w{index}.prefix")
            if k!=(str(frozenw["prefix"])+"\n").encode():
                raise RuntimeError("mask boundary edited")
            nll=float(value)
            if not math.isfinite(nll) or nll<0:raise RuntimeError("loss nonfinite")
            costs.append(nll)
        sums=FINAL.findall(log)
        if len(sums)!=1:raise RuntimeError("native C total line not unique")
        total,ns,avg,ppl=map(float,sums[0])
        if int(ns)!=16 or abs(sum(costs)-total)>1e-8 or \
           abs(avg-total/16)>1e-10 or abs(math.log(ppl)-avg)>1e-9:
            raise RuntimeError("C native masked loss arithmetic wrong")
        good=[];bad=[];win=0
        for j in range(0,16,2):
            a,b=frozen["windows"][j:j+2]
            if a["case"]!=b["case"] or a["is_correct"]==b["is_correct"]:
                raise RuntimeError("wrong paired labels")
            good_j,bad_j=(costs[j],costs[j+1]) if a["is_correct"] else (costs[j+1],costs[j])
            good.append(good_j);bad.append(bad_j);win+=int(good_j<bad_j)
        exact={"accuracy":win/8,"correct_answer_mean_nll":sum(good)/8,
               "mean_margin":sum((b-a) for a,b in zip(good,bad))/8}
        if abs(meta["scores"]["choice_accuracy"]-exact["accuracy"])>1e-12 or \
           abs(meta["scores"]["correct_answer_mean_nll"]-exact["correct_answer_mean_nll"])>1e-10 or \
           abs(meta["scores"]["mean_correct_logprob_margin"]-exact["mean_margin"])>1e-10:
            raise RuntimeError("original evaluator claim not supported")
        outcomes[name]=exact
    if bins["f32"]!=bins["q4"] or len({bins["role_random"],bins["v2_diag"],bins["v3_cov"]})!=1:
        raise RuntimeError("native binary mismatch confounds precision A/B")
    summary={
        "schema":"BEGLIN-Qv4-native-separate-loss-auditor/1",
        "status":"PASS_SEPARATE_LOG_ARITHMETIC_AND_SOURCE_BINDING",
        "evaluation_type":"post-hoc exploratory development, not held-out generalization",
        "prelocked_public_prompt_count":8,"candidate_answer_windows":16,
        "native_28_layers_all_modes":True,
        "exact_policy_source_locked_before_eval":True,
        "native_same_binary_per_precision_family":True,
        "source_sha256":EXPECTED_PRE,
        "results":outcomes,
        "strongest_development_policy_by_correct_suffix_NLL":
            min(NAMES,key=lambda m:outcomes[m]["correct_answer_mean_nll"]),
        "v3_better_than_random_by_correct_suffix_NLL":
            outcomes["v3_cov"]["correct_answer_mean_nll"]<outcomes["role_random"]["correct_answer_mean_nll"],
        "v3_better_than_Q4_by_correct_suffix_NLL":
            outcomes["v3_cov"]["correct_answer_mean_nll"]<outcomes["q4"]["correct_answer_mean_nll"],
        "limitations":[
            "Only eight authored easy binary questions; A/B one-token constrained protocol",
            "Independent parser checks native reported NLL; independent BF16 logits are not recomputed",
            "No new precision masks learned, no statistical generalization or IID/OOD acceptance",
            "Full-prompt historical PPL and masked answer-only NLL are incomparable",
            "No reliable runtime speed/energy measurement, no live promotion",
        ],
        "operational_release_modified":False,
    }
    dest=LAB/"qv4_separate_independent_audit.json"
    with dest.open("xb") as f:
        f.write((json.dumps(summary,indent=2,sort_keys=True)+"\n").encode())
        f.flush()
    dest.chmod(0o600)
    print("QV4_SEPARATE_AUDIT_PASS",json.dumps({
        "receipt_sha256":digest(secure(dest)),"ranked":[(k,outcomes[k]) for k in sorted(NAMES,key=lambda n:outcomes[n]["correct_answer_mean_nll"])],
        "v3_beats_random":summary["v3_better_than_random_by_correct_suffix_NLL"]},sort_keys=True),flush=True)

if __name__=="__main__":run()
