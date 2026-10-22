#!/usr/bin/env python3
"""Separate Qv8 native four-policy HellaSwag48 transcript and source SHA audit.

No imports from source estimator, MILP/1-swap optimizer or original C scorer.
Re-reads all 48×4 original real BEGLIN C logs, compares original per-prompt
token SHA and candidate answer tokens, binaries, selected source masks and
source-only signed-gradient evidence. Deterministic paired bootstrap and exact
McNemar are exploratory, do NOT assert generalization or serving readiness.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,struct
import numpy as np
from scipy.stats import binomtest

ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
LAB=ROOT/"lab_local/qv8_joint_signed_20261022"
PRE=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_SOURCE_PRELOCK_20261022.json"
SRC=LAB/"pinned_original_ARC_train96.json"
HELD=LAB/"pinned_HellaSwag_val48.json"
SCORE=ROOT/"docs/research/BEGLIN_QV8_SOURCE_JOINT_SIGNED_INDEPENDENT_PUBLIC_20261022.json"
POL=ROOT/"docs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_OPTIMIZER_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
ERR=ROOT/"docs/research/BEGLIN_QV8_SOURCE_START_LABEL_ERRATUM_20261022.json"
REF=ROOT/"docs/research/BEGLIN_QV8_ORIGINAL_BF16_HELLASWAG48_REFERENCE_20261022.json"
REFPRIVATE=LAB/"original_BF16_Qwen_HellaSwag48_oracle_private.json"
RUN=LAB/"native_hellaswag48_four_modes"
OUT=LAB/"independent_full_native_48x4_task_quality_audit.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV8_NARRATIVE48_NATIVE_INDEPENDENT_PUBLIC_20261022.json"
MODES=("q4","qv6_diag64","qv6_diag128","qv8_signed96")
BINS={
"base":(OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base",
      "408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"),
"mixed":(OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed",
      "f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221")}
MASKS={
"qv6_diag64":(ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",
"74fb949e45d13f63a26d412effc39377492ea71bda20ae3690f56fddf6303e08"),
"qv6_diag128":(ROOT/"configs/research/BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt",
"8b9e2404f9a38d7b57784b7d663efa19661553ccaa7dc79bfcbf6e6180fc7997"),
"qv8_signed96":(MASK,None)}
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
COUNT=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
PRE_SHA="9b833f5fc2729d13320c87e368baaff952c890ed068586622badfc7da63b4db1"
HELD_SHA="45c95170a2eb01b4c79ad44719ab9a1877e568f2583a90d84340d7448e0d1a33"
SRC_SHA="8da2f96b6ec02a5a45a2308e84074a1de14e7a93dc30e9517960596e3b2cee0c"
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("Qv8 original private native/model receipt unsealed "+str(p))
    return p.read_bytes()
def reference():
    public=json.loads(REF.read_text())
    b=sealed(REFPRIVATE)
    if public["private_original_BF16_receipt_SHA256"]!=sha(b):
        raise RuntimeError("external original BF16 MPS model arithmetic receipt missing")
    raw=json.loads(b)
    if raw["held_input_SHA256"]!=HELD_SHA or raw["case_count"]!=48 or \
       raw["status"]!="REFERENCE_ORIGINAL_BF16_48_NARRATIVE_ANSWERS":
        raise RuntimeError("separate HF original BF16 evaluation did not read identical held inputs")
    val=json.loads(sealed(HELD))["held_validation48"]
    if len(raw["per_case"])!=48:raise RuntimeError("not 48 actual source model BF16 answers")
    for a,q in zip(raw["per_case"],val):
        if a["id_sha"]!=sha(q["id"].encode()) or a["correct"]!=q["correct"]:
            raise RuntimeError("original HF BF16 and BEGLIN C used different HellaSwag samples")
    return raw
def native(mode,manifest,held,policy):
    dest=RUN/mode
    log=sealed(dest/"actual_native_hellaswag48_log.txt")
    r=json.loads(sealed(dest/"native_hellaswag48_receipt.json"))
    if r["mode"]!=mode or r["status"]!="NATIVE_REAL_28_LAYER_FULL_VOCAB_Q4_N8_48_HELLASWAG_PASS" or \
       r["original_native_log_SHA256"]!=sha(log) or \
       r["frozen_eval_manifest_SHA"]!=sha(sealed(RUN/"locked_model_48x4_manifest.json")):
        raise RuntimeError("native original Qwen full model log/receipt not immutable "+mode)
    original=BINS["base" if mode=="q4" else "mixed"]
    if sha(original[0].read_bytes())!=original[1] or r["original_native_executable_SHA256"]!=original[1]:
        raise RuntimeError("not original Qv5 audited native C binary "+mode)
    if mode!="q4":
        f,expected=MASKS[mode]
        h=expected or policy["mask_sha256"]
        if sha(f.read_bytes())!=h or r["source_only_28n8_mask_SHA"]!=h:
            raise RuntimeError("quantized model evaluated with different candidate bitmask "+mode)
    roles=COUNT.findall(log.decode())
    counts=(0,0,196) if mode=="q4" else (0,28,168)
    if len(roles)!=1 or tuple(map(int,roles[0]))!=counts:
        raise RuntimeError("executed original C wrong Q4/n8 projectors "+mode)
    rows=ROW.findall(log.decode())
    if len(rows)!=48:raise RuntimeError("native Qwen 48 new-domain tasks incomplete")
    losses=[];right=[];answers=[]
    for ix,(i,loss,y,guess,n) in enumerate(rows):
        q=held[ix]; meta=manifest["all_48_input_sha"][ix]
        ids=sealed(dest/f"eval/ppl_work/w{ix}.i32")
        alternatives=sealed(dest/f"eval/ppl_work/w{ix}.choices")
        original_ids=struct.pack("<"+"i"*len(q["prefix_ids"]),*q["prefix_ids"])
        original_choices=(" ".join(str(q["suffix_ids"][l]) for l in "ABCD")+
            " "+str("ABCD".index(q["correct"]))+"\n").encode()
        val=float(loss)
        if (int(i)!=ix or int(y)!="ABCD".index(q["correct"]) or not 0<=int(guess)<4 or
            int(n)!=len(q["prefix_ids"]) or not math.isfinite(val) or val<0 or
            ids!=original_ids or alternatives!=original_choices or
            sha(ids)!=meta["prompt_i32_SHA"] or sha(alternatives)!=meta["choices_SHA"]):
            raise RuntimeError("native C full model used different held prompts/candidate correct letter "+mode)
        losses.append(val);right.append(int(int(y)==int(guess)));answers.append(int(guess))
    total=SUM.findall(log.decode())
    if len(total)!=1:raise RuntimeError("not a unique actual 28layer native NLL aggregate")
    amount,n,correct,mean,ppl=map(float,total[0])
    if n!=48 or int(correct)!=sum(right) or abs(amount-sum(losses))>1e-7 or \
       abs(mean-amount/48)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
        raise RuntimeError("source actual NLL arithmetic/accuracy wrong "+mode)
    if r["scores"]["correct"]!=sum(right) or \
       abs(r["scores"]["mean_correct_answer_NLL"]-mean)>1e-12:
        raise RuntimeError("source native original C quality scorer contradicted by independent parser "+mode)
    return {"N":48,"correct":int(correct),"accuracy":int(correct)/48,
        "mean_correct_token_NLL":mean,"per_question_losses":losses,
        "per_question_right":right,"per_question_guesses":answers}
def audit():
    if OUT.exists() or PUBLIC.exists():raise FileExistsError("Qv8 final native audit already exists")
    if sha(PRE.read_bytes())!=PRE_SHA or sha(sealed(SRC))!=SRC_SHA or sha(sealed(HELD))!=HELD_SHA:
        raise RuntimeError("cross dataset original source and freeze mismatch")
    pol=json.loads(POL.read_text())
    src=json.loads(SCORE.read_text())
    if src["status"]!="PASS_RAW_96x196_SIGNED_CROSS_TENSOR_SOURCE_AND_MASK_COST_CONTRACT" or \
       pol["status"]!="SOURCE_ONLY_28N8_STRICT_FEASIBLE_LOCAL_SWAP_DONE_NO_EVAL_OPENED" or \
       pol["mask_sha256"]!=sha(MASK.read_bytes()) or \
       not pol["no_held_data_labels_or_logits_loaded"]:
        raise RuntimeError("source-only gradient localization allocation not independently validated")
    correction=json.loads(ERR.read_text())
    if correction["status"]!="CORRECTED_START_LABEL_ONLY_NO_MASK_OR_RESULT_CHANGED" or \
       correction["new_source_report_SHA256"]!=sha(POL.read_bytes()) or \
       correction["selected_mask_sha256_unchanged"]!=pol["mask_sha256"]:
        raise RuntimeError("before-evaluation source-only typo correction not transparently proven")
    manifest=json.loads(sealed(RUN/"locked_model_48x4_manifest.json"))
    if manifest["status"]!="FROZEN_SOURCE_ONLY_MASKS_AND_NEW_HELD_INPUT_BYTES_PRE_NATIVE" or \
       manifest["joint_source_policy_document_sha256"]!=sha(POL.read_bytes()) or \
       manifest["held_file_SHA"]!=HELD_SHA or len(manifest["all_48_input_sha"])!=48:
        raise RuntimeError("original joint source policy or HellaSwag case changed after native run")
    held=json.loads(sealed(HELD))["held_validation48"]
    out={mode:native(mode,manifest,held,pol) for mode in MODES}
    ref=reference()
    rng=np.random.default_rng(20261022)
    sample=rng.integers(0,48,size=(5000,48))
    comparisons={}
    candidate=out["qv8_signed96"]
    for name in MODES[:-1]:
        other=out[name]
        paired=np.asarray(candidate["per_question_losses"])-np.asarray(other["per_question_losses"])
        ci=np.percentile(paired[sample].mean(axis=1),[2.5,97.5])
        a=np.asarray(other["per_question_right"]);b=np.asarray(candidate["per_question_right"])
        gain=int(((a==0)&(b==1)).sum());lost=int(((a==1)&(b==0)).sum())
        comparisons[name]={
            "joint_minus_baseline_mean_NLL":float(paired.mean()),
            "paired_bootstrap95_percentile":[float(ci[0]),float(ci[1])],
            "new_mask_lower_NLL_case_count":int((paired<0).sum()),
            "accuracy_delta_new_minus_base":candidate["correct"]-other["correct"],
            "gain_only_cases":gain,"loss_only_cases":lost,
            "McNemar_exact_two_sided_p":float(binomtest(min(gain,lost),n=gain+lost,p=.5).pvalue) if gain+lost else 1.}
    report={
       "schema":"BEGLIN-Qv8-independent-signed-first-order-source-and-4-arm-native-HellaSwag48/1",
       "status":"PASS_96x196_SIGNED_SOURCE_AND_4x48_ACTUAL_QWEN_CPU_NATIVE_LOGS",
       "prelock_SHA256":PRE_SHA,"source_96_SHA256":SRC_SHA,"held_48_SHA256":HELD_SHA,
       "source_independent_public_SHA256":sha(SCORE.read_bytes()),
       "joint_selected_mask_SHA256":pol["mask_sha256"],
       "source_pre_eval_erratum_SHA256":sha(ERR.read_bytes()),
       "source_joint_local_only_not_global_optimum":True,
       "all_four_native_actual_results":out,"paired_joints":comparisons,
       "original_BF16_external_reference_correct":ref["correct_count"],
       "original_BF16_external_reference_NLL":ref["mean_correct_answer_NLL"],
       "original_model_HF_revision":ref["source_revision"],
       "separate_C_widenedF32_vs_HF_BF16_numerical_paths":True,
       "new_dataset_different_domain_but_development_only":True,
       "quality_generalization_established":False,
       "no_actual_Apple_Metal_AutoBitSwitch":True,
       "no_valid_cpu_speed_claim_from_quality_benchmark":True,
       "main_and_operator_release_unchanged":True,
       "no_auto_promotion":True}
    os.umask(0o077)
    b=(json.dumps(report,indent=2,sort_keys=True)+"\n").encode()
    with OUT.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    pub={k:v for k,v in report.items() if k!="all_four_native_actual_results"}
    pub["native_summary"]={key:{j:v for j,v in data.items() if not j.startswith("per_question")}
       for key,data in out.items()}
    pub["private_original_full_independent_audit_SHA256"]=sha(b)
    doc=(json.dumps(pub,indent=2,sort_keys=True)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(doc);f.flush();os.fsync(f.fileno())
    print("QV8_ORIGINAL_QWEN_SIGNED_FIRSTORDER_HELLASWAG48_INDEPENDENT_AUDIT_PASS",
        json.dumps({"metrics":{k:(v["correct"],v["mean_correct_token_NLL"]) for k,v in out.items()},
         "comparisons":comparisons,
         "private_SHA":sha(b),"public_SHA":sha(doc),"not_live_promotion":True},sort_keys=True),flush=True)
if __name__=="__main__":audit()
