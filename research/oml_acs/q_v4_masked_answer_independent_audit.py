#!/usr/bin/env python3
"""Independent, separately implemented Qv4 native answer-mask verifier.

No import from the evaluator. Recomputes every paired choice probability,
source/policy identity, Qwen A/B suffix boundaries, native transcript hashes,
and quality ranking from readonly native outputs. Never refits allocation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
ORIGIN=Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
LAB=ROOT/"lab_local/q_v4_masked_answer_20261022"
PRELOCK=ROOT/"configs/research/BEGLIN_Q_V4_MASKED_ANSWER_DEV_PRELOCK_20261022.json"
PROTO_HASH="984adbb205bc14e3ff76c1654a3143529fdd07a8ba017e445462cb3707c3677c"
WINDOW_HASH="a6e6ab1e0b2054faabaf460fde280f19d9411ed26862c1cd1aa2b3118fa88f39"
MASK_HASH={
 "role_random":"72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5",
 "v2_diag":"e044675ea5b47a5aca0ca75980a55070b932d1a4dc330815bbb097acbf75df7b",
 "v3_cov":"82ae93e06afa7380175afa57535c1f0d1f7fd4320b8d28f981cdc8ee0cf24ef3",
}
POLICIES=("f32","q4","role_random","v2_diag","v3_cov")
LOG_EXPR=re.compile(r"^\[qv4-masked\] window (?P<index>\d+) prefix (?P<prefix>\d+) nll (?P<loss>\S+) tokens (?P<count>\d+)$",re.M)
AGG_EXPR=re.compile(r"^qv4_masked_nll (?P<total>\S+) tokens (?P<count>\d+) mean (?P<mean>\S+) ppl (?P<ppl>\S+)$",re.M)
ROLE_EXPR=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
EXPECTED_COUNTS={"f32":(196,0,0),"q4":(0,0,196),
                 "role_random":(0,28,168),"v2_diag":(0,28,168),"v3_cov":(0,28,168)}

def sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def readonly(p:Path)->bytes:
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("missing/unprotected immutable native evidence "+str(p))
    return p.read_bytes()

def interpret(log:str,windows:list[dict],policy:str)->dict:
    match=ROLE_EXPR.search(log)
    if not match or tuple(map(int,match.groups()))!=EXPECTED_COUNTS[policy]:
        raise RuntimeError("incorrect actual native projection tiers")
    raw=LOG_EXPR.findall(log)
    if len(raw)!=16 or len(windows)!=16:
        raise RuntimeError("incomplete fixed 16 native A/B score transcript")
    vals=[]
    for ix,(idtxt,k,loss,n) in enumerate(raw):
        original=windows[ix]
        scalar=float(loss)
        if int(idtxt)!=ix or int(k)!=original["prefix"] or int(n)!=1 or \
           not math.isfinite(scalar) or scalar<0:
            raise RuntimeError("native answer-only token count, prefix or nll invalid")
        vals.append(scalar)
    match=AGG_EXPR.findall(log)
    if len(match)!=1:
        raise RuntimeError("nonunique/missing native answer aggregate")
    total,n,mean,ppl=map(float,match[0])
    if abs(total-sum(vals))>1e-8 or n!=16 or \
       abs(total/n-mean)>1e-8 or abs(math.exp(mean)-ppl)>1e-8:
        raise RuntimeError("native per-answer loss does not reconcile")
    hits=0; positives=[]; margins=[]
    for i in range(0,16,2):
        left,right=windows[i:i+2]
        if left["case"]!=right["case"] or left["answer"]!="A" or \
           right["answer"]!="B" or left["is_correct"]==right["is_correct"]:
            raise RuntimeError("A/B answer pairing tampered")
        yes,no=(vals[i],vals[i+1]) if left["is_correct"] else (vals[i+1],vals[i])
        positives.append(yes)
        margins.append(no-yes)
        hits+=int(no>yes)
    return {"case_count":8,"accuracy":hits/8,
            "correct_only_nll":sum(positives)/8,
            "mean_answer_margin":sum(margins)/8,
            "per_case_margins":margins}

def calculate()->dict:
    if sha(PRELOCK)!=PROTO_HASH:
        raise RuntimeError("original preregistration drifted")
    inputfile=LAB/"inputs_frozen.json"
    if sha(inputfile)!=WINDOW_HASH:
        raise RuntimeError("evaluation inputs not identical to pre-inference freeze")
    manifest=json.loads(readonly(inputfile))
    if (manifest["preregistration_sha256"]!=PROTO_HASH
        or manifest["mask_sha256"]!=MASK_HASH
        or manifest["state"]!="INPUTS_FROZEN_BEFORE_ANY_NATIVE_QV4_INFERENCE"):
        raise RuntimeError("not source-locked 16 native A/B windows")
    windows=manifest["windows"]
    if len(windows)!=16:raise RuntimeError("expected eight two-arm tasks")
    out={}
    all_bin={}
    for p in POLICIES:
        work=LAB/p
        rawlog=readonly(work/"native_masked_nll_log.txt")
        receipt=json.loads(readonly(work/"native_masked_answer_receipt.json"))
        if sha(work/"native_masked_nll_log.txt")!=receipt["log_sha256"] or \
           receipt["mode"]!=p or receipt["fixed_input_receipt_sha256"]!=WINDOW_HASH or \
           receipt["classification"]!="EXPLORATORY_DEVELOPMENT_ONLY" or \
           receipt["no_original_model_or_live_modification"] is not True:
            raise RuntimeError("original native evidence/receipt identity mismatch "+p)
        kind="base" if p in ("f32","q4") else "mixed"
        original_build=json.loads(readonly(LAB/"bin"/(kind+"_build_receipt.json")))
        binfile=LAB/"bin"/("beglin_qv4_"+kind)
        if sha(binfile)!=original_build["exe_sha256"] or \
           sha(binfile)!=receipt["native_binary_sha256"] or \
           original_build["legacy_ppl_unchanged"] is not True:
            raise RuntimeError("actual source-compiled Qwen worker drift "+p)
        all_bin[p]=sha(binfile)
        for item in windows:
            idx=item["idx"]; ref=work/f"eval/ppl_work/w{idx}.i32"
            pref=work/f"eval/ppl_work/w{idx}.prefix"
            if hashlib.sha256(readonly(ref)).hexdigest()!=item["ids_sha256"] or \
               readonly(pref)!=(str(item["prefix"])+"\n").encode():
                raise RuntimeError("source-locked A/B token bytes changed")
        actual=interpret(rawlog.decode(),windows,p)
        original=receipt["scores"]
        for key,ref in (("accuracy","choice_accuracy"),
                        ("correct_only_nll","correct_answer_mean_nll"),
                        ("mean_answer_margin","mean_correct_logprob_margin")):
            if abs(actual[key]-original[ref])>1e-9:
                raise RuntimeError("independently recalculated native quality mismatch")
        if any(abs(a-b)>1e-9 for a,b in zip(actual["per_case_margins"],original["per_case_margin"])):
            raise RuntimeError("per-question native error provenance mismatch")
        out[p]=actual
    return {"schema":"BEGLIN-Qv4-independent-A-B-suffix-verifier/1",
        "status":"PASS_INDEPENDENT_SEPARATE_IMPLEMENTATION",
        "prelock_sha256":PROTO_HASH,"input_sha256":WINDOW_HASH,
        "original_Qwen_bf16_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "native_binary_sha256_by_policy":all_bin,
        "source_mixed_policy_SHA256":MASK_HASH,
        "score_kind":"ASSISTANT_ONE_TOKEN_A_B_DEVELOPMENT_ONLY",
        "conditions":out,
        "v2_outperforms_seed73_on_dev_correct_NLL":
           out["v2_diag"]["correct_only_nll"]<out["role_random"]["correct_only_nll"],
        "v3_outperforms_seed73_on_dev_correct_NLL":
           out["v3_cov"]["correct_only_nll"]<out["role_random"]["correct_only_nll"],
        "old_V6_24_not_opened":True,"no_runtime_mutation":True,
        "no_sample_generalization_claim":True,"independent_code_not_evaluator_import":True}

def main()->None:
    out=calculate()
    target=LAB/"qv4_independent_second_implementation_audit.json"
    if target.exists():raise FileExistsError("independent verifier cannot overwrite previous run")
    raw=(json.dumps(out,indent=2,sort_keys=True)+"\n").encode()
    os.umask(0o077)
    with target.open("xb") as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    target.chmod(0o600)
    print("QV4_SECOND_IMPLEMENTATION_NATIVE_AUDIT_PASS",json.dumps({
        "sha256":hashlib.sha256(raw).hexdigest(),
        "quality":{k:{"acc":v["accuracy"],"correct_nll":v["correct_only_nll"]} for k,v in out["conditions"].items()},
        "no_production_promotion":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
