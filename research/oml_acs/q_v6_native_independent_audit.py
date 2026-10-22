#!/usr/bin/env python3
"""Separate Qv6 native OpenBookQA48 audit. Does not import native evaluator.

Verifies 4 real BEGLIN C 28-layer process transcripts, original Qv5 binary
SHA, four-choice logits, prompt/answer token SHA, source-only 28-bit masks,
original 128 training gradient source authority and paired losses.
"""
import hashlib,json,math,os,pathlib,re,struct
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv6_fisher_scaling_20261022"
RUN=LAB/"native_openbookqa_48_four_arms"
HELD=LAB/"openbookqa_val48_frozen.json"
SRC_AUDIT=LAB/"independent_128_original_Fisher_source_audit.json"
PLAN=ROOT/"docs/research/BEGLIN_QV6_NESTED_64_128_SOURCE_POLICY_20261022.json"
BF16=ROOT/"docs/research/BEGLIN_QV6_ORIGINAL_BF16_OPENBOOKQA48_REFERENCE_20261022.json"
PRIVATE=LAB/"independent_all48_native_policy_comparison_receipt.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV6_CROSSCORPUS_NATIVE_INDEPENDENT_PUBLIC_20261022.json"
MODES=("q4","v5_fisher_24","v6_fisher_64","v6_fisher_128")
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
AGG=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
REV="c09831bf86c6eedebed433dc2a0398a0992afb19fddc9f86750c2d137d71d1b0"

def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
    if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
        raise RuntimeError("unsafe/missing original native evidence "+str(p))
    return p.read_bytes()

def original_native_score(text:str,mode:str,manifest,source):
    levels=ROLE.findall(text)
    target=(0,0,196) if mode=="q4" else (0,28,168)
    if len(levels)!=1 or tuple(map(int,levels[0]))!=target:
        raise RuntimeError("native C program used unexpected source precision tier")
    rows=ROW.findall(text)
    if len(rows)!=48:raise RuntimeError("missing 48 source-locked OpenBookQA answers")
    nll=[];hit=[];expected=[];predicted=[]
    for ix,(n,f,y,z,p) in enumerate(rows):
        item=source["unseen_questions"][ix]
        spec=manifest["all_48_prompt_choice_hashes"][ix]
        val=float(f)
        if int(n)!=ix or int(y)!=("ABCD".index(item["correct"])) or \
           int(p)!=spec["prompt_length"] or not 0<=int(z)<4 or \
           not math.isfinite(val) or val<0:
            raise RuntimeError("native source answer-only NLL or correct label mismatch")
        nll.append(val);hit.append(int(y==z));expected.append(int(y));predicted.append(int(z))
    sums=AGG.findall(text)
    if len(sums)!=1:raise RuntimeError("duplicate/missing native C 48 aggregate")
    total,n,correct,mean,ppl=map(float,sums[0])
    if n!=48 or int(correct)!=sum(hit) or abs(total-sum(nll))>1e-7 or \
       abs(mean-total/48)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
        raise RuntimeError("native quantized Qwen arithmetic/prompt count inconsistent")
    return {"correct":int(correct),"accuracy":int(correct)/48,
      "correct_answer_NLL":mean,"per_question_NLL":nll,"per_question_right":hit}

def verify():
    plan=json.loads(PLAN.read_text())
    if plan["status"]!="FROZEN_64_128_POLICIES_BEFORE_NEW_OPENBOOKQA_VALIDATION" or \
       not plan["evaluation_48_OpenBookQA_logits_not_opened"]:
        raise RuntimeError("Fisher tensor choices were not frozen before evaluation")
    audit=json.loads(sealed(SRC_AUDIT))
    if audit["status"]!="PASS_ALL_196_TENSOR_SCORES_128_SOURCE_INPUTS":
        raise RuntimeError("independent 24/64/128 original-BF16 source gradients not verified")
    if sha(sealed(HELD))!=REV:raise RuntimeError("OpenBookQA 48 data not pre-frozen")
    held=json.loads(sealed(HELD)); manifest=json.loads(sealed(RUN/"frozen_natural_science_48_manifest.json"))
    if manifest["status"]!="FROZEN_FOUR_ARMS_BEFORE_FIRST_CROSSCORPUS_NATIVE_FORWARD" or \
       manifest["source_only_two_policies_document_SHA256"]!=sha(PLAN.read_bytes()) or \
       manifest["cross_validation_48_private_SHA256"]!=REV:
        raise RuntimeError("pre-native token identity or source-only policy mismatch")
    results={};bins={}
    for mode in MODES:
        folder=RUN/mode
        rec=json.loads(sealed(folder/"openbookqa48_native_receipt.json"))
        raw=sealed(folder/"openbookqa48_native_log.txt")
        if rec["mode"]!=mode or rec["actual_native_log_SHA256"]!=sha(raw) or \
           rec["frozen_input_manifest_SHA256"]!=sha(sealed(RUN/"frozen_natural_science_48_manifest.json")):
            raise RuntimeError("source C actual run receipt does not match original transcript")
        nativefile=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin")/("beglin_qv5_base" if mode=="q4" else "beglin_qv5_mixed")
        bsha=sha(nativefile.read_bytes())
        expected=manifest["native_Qv5_verbatim_source_binary_SHA256"]["base" if mode=="q4" else "mixed"]
        if bsha!=expected or bsha!=rec["origin_qv5_28layer_native_binary_SHA"]:
            raise RuntimeError("different source native C executable")
        bins[mode]=bsha
        for i,item in enumerate(held["unseen_questions"]):
            spec=manifest["all_48_prompt_choice_hashes"][i]
            data=sealed(folder/f"eval/ppl_work/w{i}.i32")
            correct=(" ".join(str(item["suffix_ids"][ch]) for ch in "ABCD")+
                    " "+str("ABCD".index(item["correct"]))+"\n").encode()
            choice=sealed(folder/f"eval/ppl_work/w{i}.choices")
            if sha(data)!=spec["prompt_sha256"] or \
               data!=struct.pack("<"+"i"*len(item["prefix_ids"]),*item["prefix_ids"]) or \
               choice!=correct or sha(choice)!=spec["choice_sha256"]:
                raise RuntimeError("original model inputs/labels changed across C experiment arms")
        x=original_native_score(raw.decode(),mode,manifest,held)
        if rec["metrics"]["correct"]!=x["correct"] or abs(rec["metrics"]["answer_only_correct_letter_NLL"]-x["correct_answer_NLL"])>1e-12:
            raise RuntimeError("Qv6 native evaluator does not match independent transcript")
        results[mode]=x
    if len({bins[k] for k in MODES if k!="q4"})!=1:
        raise RuntimeError("non-identical mixed precision binary across 24/64/128")
    return results,bins,plan,audit

def run():
    if PRIVATE.exists() or PUBLIC.exists():raise FileExistsError("original independent native audit already persisted")
    scores,bins,plan,src=verify()
    ref=scores["q4"]
    rng=np.random.default_rng(20261022)
    ids=rng.integers(0,48,size=(5000,48))
    diffs={}
    for mode in MODES[1:]:
        x=scores[mode]
        d=np.array(x["per_question_NLL"])-np.array(ref["per_question_NLL"])
        boot=d[ids].mean(axis=1)
        diffs[mode]={
          "candidate_minus_Q4_NLL_mean":float(d.mean()),
          "paired_bootstrap_95_percentile":[float(v) for v in np.percentile(boot,[2.5,97.5])],
          "candidate_lower_NLL_in_items":int((d<0).sum()),
          "candidate_minus_Q4_correct_count":x["correct"]-ref["correct"]}
    bf16=json.loads(BF16.read_text())
    if bf16["frozen_crosscorpus_data_SHA256"]!=REV or bf16["dataset_revision"]!="388097ea7776314e93a529163e0fea805b8a6454":
        raise RuntimeError("BF16 comparison source is not same cohort")
    record={
      "schema":"BEGLIN-Qv6-independent-source-and-native-crosscorpus-audit/1",
      "status":"PASS_4_POLICY_48_REAL_APPLE_C_NATIVE_EXECUTIONS_AND_BF16_SOURCE_AUDIT",
      "source_input_frozen_sha256":REV,
      "Fisher_source_independent_audit_sha256":sha(sealed(SRC_AUDIT)),
      "source64_and_128_milp_sha256":sha(PLAN.read_bytes()),
      "orig_native_binary_sha256":bins,"native_six":False,
      "actual_native_48_question_metrics":scores,
      "paired_q4_comparisons":diffs,
      "source_odd_even_reliability":src["scores"],
      "constrained_policy_overlap":plan["constrained_selected_overlap"],
      "HF_original_BF16_correct":bf16["correct"],
      "HF_original_BF16_NLL":bf16["original_BF16_correct_oneletter_NLL"],
      "generalization_quality_superiority_proven":False,
      "automated_live_promotion":False,
      "notes":"One model and one other science-domain corpus, 48 development questions; no speed/Metal claim"}
    os.umask(0o077)
    private=(json.dumps(record,indent=2,sort_keys=True)+"\n").encode()
    with PRIVATE.open("xb") as f:f.write(private);f.flush();os.fsync(f.fileno())
    PRIVATE.chmod(0o600)
    public={k:v for k,v in record.items() if k not in ("actual_native_48_question_metrics","orig_native_binary_sha256")}
    public["native_summary"]={mode:{key:val for key,val in obj.items() if not key.startswith("per_question")}
       for mode,obj in scores.items()}
    public["original_private_audit_sha256"]=sha(private)
    doc=(json.dumps(public,indent=2,sort_keys=True)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(doc);f.flush();os.fsync(f.fileno())
    print("QV6_INDEPENDENT_NATIVE_CROSSDATASET_48_PASS",json.dumps({
      "private_sha":sha(private),"public_sha":sha(doc),
      "correct":{n:v["correct"] for n,v in scores.items()},
      "nll":{n:round(v["correct_answer_NLL"],6) for n,v in scores.items()},
      "q4_difference":diffs,"no_live_activation":True},sort_keys=True),flush=True)
if __name__=="__main__":run()
