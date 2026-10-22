#!/usr/bin/env python3
"""Independent source-Fisher + native 40-ARC six-arm audit, fail-closed.

No imports from source scorer, policy optimizer or native evaluation driver.
Checks original BF16 gradient/Fisher arithmetic and original C process
transcripts including every one-token answer & four-option choice; validates
SHA-frozen split, 196 source modules, source-only bit policy and native binary.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re
import numpy as np
from scipy.stats import spearmanr

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv5_fisher_20261022"
NATIVE=LAB/"native_40val_6mode"
PROTO=ROOT/"configs/research/BEGLIN_QV5_FISHER_GRAD_PRELOCK_20261022.json"
SRC=LAB/"source_train_frozen.json"
DEV=LAB/"development_val_frozen.json"
FSRC=LAB/"fisher_196_train_private_receipt.json"
FPUB=ROOT/"docs/research/BEGLIN_QV5_FISHER_SOURCE_AGGREGATE_20261022.json"
MILP=ROOT/"docs/research/BEGLIN_QV5_FISHER_HIGHS_SOURCE_MILP_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt"
FREEZE=NATIVE/"pre_native_sixmode_manifest.json"
OUT=LAB/"qv5_independent_fisher_and_native_audit.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV5_FISHER_40ARC_NATIVE_PUBLIC_RESULT_20261022.json"

MODES=("q4","role_random","v5_fisher","v2_diag","v3_cov","f32")
EXPECTED={"f32":(196,0,0),"q4":(0,0,196),
 "role_random":(0,28,168),"v2_diag":(0,28,168),"v3_cov":(0,28,168),"v5_fisher":(0,28,168)}
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
PROTOCOL="a886418affc9f665d64ce4ca9eb0b5ea2aef4ae97bc59bf58ea1affb53bf19ef"
SOURCE_SHA="631bb02d22b696ddcabfd3d7ab759218fd75e1e8f3d8384cbf4a5df4de3f6994"
DEV_SHA="a00985cc73f80518f448c2b8a1295cf894283723f87b5af53321d486ce0187ae"
DATA_PUB_SHA="43f693c8697b4e8889f807780034093c62c5813a3ad31e7a3dea7428083a3f84"
LOG_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
LOG_TOT=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
LOG_ROLES=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")

def sha(b):return hashlib.sha256(b).hexdigest()
def secure(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("unsealed original Qv5 source / output: "+str(p))
    return p.read_bytes()

def check_source()->tuple[dict,dict]:
    if sha(PROTO.read_bytes())!=PROTOCOL or sha(secure(SRC))!=SOURCE_SHA or \
       sha(secure(DEV))!=DEV_SHA:
        raise RuntimeError("original source/dev split or preregistration drift")
    frozen=json.loads(secure(SRC)); held=json.loads(secure(DEV))
    if (len(frozen["questions"])!=24 or len(held["questions"])!=40 or
        frozen["dataset_sha"]!=held["dataset_sha"] or
        len({x["id"] for x in frozen["questions"]})!=24 or
        len({x["id"] for x in held["questions"]})!=40):
        raise RuntimeError("source train or developmental ARC split invalid")
    if {x["question_sha256"] for x in frozen["questions"]}&{x["question_sha256"] for x in held["questions"]}:
        raise RuntimeError("source and validation text overlap")
    fisher=json.loads(secure(FSRC))
    source_pub=json.loads(FPUB.read_text())
    if fisher["status"]!="SOURCE_TRAIN_ONLY_ALL_196_NATIVE_GRADIENTS_AND_QUANT_ERRORS" or \
       fisher["source_train_SHA256"]!=SOURCE_SHA or fisher["unopened_dev_cohort_sha256"]!=DEV_SHA or \
       source_pub["private_full_source_receipt_SHA256"]!=sha(secure(FSRC)):
        raise RuntimeError("original BF16 directional gradients identity mismatch")
    rows=fisher["source_score_rows"]
    if len(rows)!=196 or len({(r["role"],r["layer"]) for r in rows})!=196:
        raise RuntimeError("expected 196 actual original BF16 projection gradients")
    odds=[];evens=[]; gains=[]
    for i,role in enumerate(ROLES):
        for layer in range(28):
            row=rows[i*28+layer]
            if row["role"]!=role or row["layer"]!=layer:raise RuntimeError("source role ordering drift")
            v4=np.asarray(row["per_source_item_q4_sq"],dtype=np.float64)
            v8=np.asarray(row["per_source_item_n8_sq"],dtype=np.float64)
            if v4.shape!=(24,) or v8.shape!=(24,) or \
               not (np.isfinite(v4).all() and np.isfinite(v8).all() and np.min(v4)>=0 and np.min(v8)>=0):
                raise RuntimeError("missing original BF16 per-case Fisher direction values")
            g=float((v4-v8).mean())
            if abs(g-row["gain_8bit_vs_4bit"])>1e-8 or \
               abs(g-fisher["quality_upgrade_by_role_layer"][role][layer])>1e-8:
                raise RuntimeError("Fisher source aggregate arithmetic tampered")
            odds.append(float((v4[::2]-v8[::2]).mean()))
            evens.append(float((v4[1::2]-v8[1::2]).mean()))
            gains.append(g)
    rho=float(spearmanr(odds,evens).statistic)
    if not math.isfinite(rho) or abs(rho-fisher["split_half_odd_even_spearman"])>1e-12:
        raise RuntimeError("split-half original Fisher rank arithmetic mismatch")
    policy=json.loads(MILP.read_text());src=np.asarray(gains,dtype=np.float64)
    if (policy["status"]!="FROZEN_SOURCE_ONLY_EXACT_MILP_NOT_YET_ARC_VALIDATION_SCORED" or
        policy["original_fisher_source_pub_sha256"]!=sha(FPUB.read_bytes()) or
        policy["preregistration_sha256"]!=PROTOCOL or
        policy["original_input_lock_sha256"]!=DATA_PUB_SHA or
        policy["source_only_objective"].find("grad")<0 or
        policy["development_ARC_40_labels_unopened"] is not True):
        raise RuntimeError("native policy was not frozen from original Fisher before testing")
    if abs(policy["score_scale_to_max_abs"]-float(np.max(np.abs(src))))>1e-10:
        raise RuntimeError("Fisher MILP source score scaling not original")
    rawmask=MASK.read_bytes()
    if sha(rawmask)!=policy["mask_sha256"]:
        raise RuntimeError("source-only 28bit mask identity drift")
    entries=[line.split() for line in rawmask.decode().splitlines()]
    if len(entries)!=28 or len({tuple(x) for x in entries})!=28 or \
       any(len(x)!=2 or x[0] not in ROLES or not x[1].isdigit() or int(x[1])>=28 for x in entries):
        raise RuntimeError("not 28 real unique tensor upgrades")
    chosen=set((x[0],int(x[1])) for x in entries)
    if set(tuple(x) for x in policy["optimizer"]["selected_ids"])!=chosen or \
       policy["optimizer"]["estimated_projection_weight_bytes"]>830570496 or \
       policy["optimizer"]["original_fullmodel_serial_estimated_ms"]>100 or \
       policy["optimizer"]["isolated_async_CPU_stage_total_ms"]>50.000001 or \
       policy["optimizer"]["mip_gap"]>1e-7:
        raise RuntimeError("actual native budgets / MILP acceptance invalid")
    return policy,{"gain":src.tolist(),"split_half_spearman":rho,
       "score_mean":float(src.mean()),"score_nonpositive_count":int((src<=0).sum()),
       "selected_source_normalized_gain":float(sum(fisher["quality_upgrade_by_role_layer"][role][layer] for role,layer in chosen)/np.max(np.abs(src))),
       "train_original_correct_answer_nll":fisher["source_full_correct_answer_avg_nll"]}

def decode_native(log:str,mode:str,frozen:dict)->dict:
    rolecounts=LOG_ROLES.findall(log)
    if len(rolecounts)!=1 or tuple(map(int,rolecounts[0]))!=EXPECTED[mode]:
        raise RuntimeError("wrong actual 28-layer C quant projection count "+mode)
    found=LOG_ROW.findall(log)
    if len(found)!=40:raise RuntimeError("missing 40 actual per-prompt C result rows")
    vals=[];yes=[]
    for i,(idx,loss,correct,pred,prefix) in enumerate(found):
        meta=frozen["validation"][i]
        n=float(loss)
        if (int(idx)!=i or not 0<=int(correct)<4 or not 0<=int(pred)<4 or
            not math.isfinite(n) or n<0 or int(prefix)!=meta["prompt_length"]):
            raise RuntimeError("C response log / exact eval input length mismatch")
        vals.append(n);yes.append(int(correct==pred))
    agg=LOG_TOT.findall(log)
    if len(agg)!=1:raise RuntimeError("missing native full aggregate")
    total,n,acc,avg,ppl=map(float,agg[0])
    if abs(total-sum(vals))>1e-7 or n!=40 or acc!=sum(yes) or \
       abs(avg-total/40)>1e-7 or abs(math.log(ppl)-avg)>1e-7:
        raise RuntimeError("independent Qv5 native C arithmetic failed")
    return {"NLL":avg,"accuracy":sum(yes)/40.,
        "correct_count":sum(yes),"per_case_nll":vals,"per_case_right":yes}

def check_native(policy:dict)->dict:
    manifest=json.loads(secure(FREEZE))
    if (manifest["status"]!="INPUTS_AND_SOURCE_FISHER_POLICY_FROZEN_BEFORE_NATIVE_RESULTS" or
        manifest["source_fisher_policy_SHA256"]!=policy["mask_sha256"] or
        manifest["source_DEV_file_SHA256"]!=DEV_SHA or len(manifest["validation"])!=40):
        raise RuntimeError("native evaluation manifest was not source SHA locked")
    outputs={}
    binary_shas={}
    for mode in MODES:
        folder=NATIVE/mode
        original=json.loads(secure(folder/"native_40choice_receipt.json"))
        raw=secure(folder/"native_40choice_log.txt")
        if original["native_precision_policy"]!=mode or \
           original["native_program_log_SHA256"]!=sha(raw) or \
           original["dev_frozen_manifest_SHA256"]!=sha(secure(FREEZE)) or \
           original["model_layers_executed"]!=28:
            raise RuntimeError("native C original execution transcript mismatch "+mode)
        kind="base" if mode in ("f32","q4") else "mixed"
        binary=NATIVE/"bin"/f"beglin_qv5_{kind}"
        rec=json.loads(secure(NATIVE/"bin"/f"{kind}_build_receipt.json"))
        if sha(binary.read_bytes())!=rec["actual_engine_binary_SHA256"] or \
           original["native_binary_SHA256"]!=rec["actual_engine_binary_SHA256"] or \
           not rec["original_ppl_unchanged"] or not rec["Qv4_masked_nll_unchanged"]:
            raise RuntimeError("source/binary ancestry mismatch "+mode)
        binary_shas[mode]=sha(binary.read_bytes())
        for i,meta in enumerate(manifest["validation"]):
            if (sha(secure(folder/f"eval/ppl_work/w{i}.i32"))!=meta["prefix_sha256"] or
                sha(secure(folder/f"eval/ppl_work/w{i}.choices"))!=meta["choices_SHA256"]):
                raise RuntimeError("native source/answer bytes tampered "+mode)
        x=decode_native(raw.decode(),mode,manifest)
        o=original["scores"]
        if x["correct_count"]!=o["correct"] or \
           abs(x["NLL"]-o["correct_answer_mean_nll"])>1e-12 or \
           abs(x["accuracy"]-o["accuracy"])>1e-12:
            raise RuntimeError("native original tool score differs from independent parser")
        outputs[mode]=x
    if len({binary_shas[k] for k in MODES if k not in ("f32","q4")})!=1 or \
       binary_shas["f32"]!=binary_shas["q4"]:
        raise RuntimeError("different C program among policy comparison arms")
    return {"modes":outputs,"binary_shas":binary_shas,
            "input_manifest_sha256":sha(secure(FREEZE))}


def independently_compare_BF16_oracle(outcomes)->dict:
    """External Transformers BF16 vs native BEGLIN widened-F32 is diagnostic only."""
    oracle_path=NATIVE/"independent_original_BF16_HF_mps_reference.json"
    pub_path=ROOT/"docs/research/BEGLIN_QV5_ORIGINAL_BF16_HF_40_ARC_AGGREGATE_20261022.json"
    if not pub_path.is_file():
        raise RuntimeError("independent BF16 MPS reference aggregate absent")
    meta=json.loads(pub_path.read_text())
    blob=secure(oracle_path)
    oracle=json.loads(blob)
    if meta["original_private_oracle_sha256"]!=sha(blob) or \
       oracle["dev_frozen_sha256"]!=DEV_SHA or \
       oracle["source_only_fisher_mask_sha256"]!=outcomes["source_policy_mask_sha256"] or \
       len(oracle["individual_examples"])!=40 or not oracle["no_any_model_weight_changes"]:
        raise RuntimeError("original BF16 MPS baseline not independently bound to Qv5 policy")
    dev=json.loads(secure(DEV))["questions"]
    for q,x in zip(dev,oracle["individual_examples"]):
        if sha(q["id"].encode())!=x["question_id_sha"] or \
           x["correct"]!=q["correct"] or not math.isfinite(x["mean_answer_nll"]):
            raise RuntimeError("independent BF16 original oracle used different validation question")
    f32=outcomes["modes"]["f32"]
    b=np.array([row["mean_answer_nll"] for row in oracle["individual_examples"]])
    a=np.array(f32["per_case_nll"])
    bfcorrect=np.array([int(x["correct_prediction"]) for x in oracle["individual_examples"]])
    cc=np.array(f32["per_case_right"])
    return {"independent_BF16_correct":int(bfcorrect.sum()),
       "independent_BF16_accuracy":float(bfcorrect.mean()),
       "independent_BF16_correct_answer_mean_NLL":float(b.mean()),
       "native_BF16_source_widenedF32_correct":int(cc.sum()),
       "native_widenedF32_correct_answer_mean_NLL":float(a.mean()),
       "native_minus_torch_BF16_mean_NLL":float(a.mean()-b.mean()),
       "native_vs_torch_BF16_per_case_absolute_NLL_mean":float(np.mean(np.abs(a-b))),
       "native_vs_torch_BF16_per_case_correctness_agreement":int(np.sum(cc==bfcorrect)),
       "not_approximate_software_execution_parity_certification":True,
       "torch_BF16_original_private_SHA256":sha(blob)}

def compare_paired(outcomes):
    val=outcomes["modes"]["v5_fisher"]
    results={}
    rng=np.random.default_rng(20261022)
    sampleidx=rng.integers(0,40,size=(5000,40))
    for mode in MODES:
        if mode=="v5_fisher":continue
        other=outcomes["modes"][mode]
        paired=np.array(other["per_case_nll"])-np.array(val["per_case_nll"])
        boot=np.mean(paired[sampleidx],axis=1)
        results[mode]={
          "baseline_minus_Fisher_NLL_mean":float(np.mean(paired)),
          "bootstrap95_interval":list(map(float,np.percentile(boot,[2.5,97.5]))),
          "nll_fisher_better_on_cases":int(sum(paired>0)),
          "native_accuracy_delta_Fisher_minus_baseline":val["accuracy"]-other["accuracy"]}
    return results

def main():
    if OUT.exists() or PUBLIC.exists():raise FileExistsError("cannot overwrite original independent native audit")
    policy,src=check_source()
    runs=check_native(policy)
    comparisons=compare_paired(runs)
    runs["source_policy_mask_sha256"]=policy["mask_sha256"]
    oracle_crosscheck=independently_compare_BF16_oracle(runs)
    winning=sum(q["baseline_minus_Fisher_NLL_mean"]>0 for q in comparisons.values())
    report={"schema":"BEGLIN-Qv5-two-level-original-BF16-Fisher-native-ARC40-independent-audit/1",
      "status":"PASS_ORIGINAL_BF16_GRADIENT_AND_NATIVE_C_REPLAY_AUDIT_NOT_FINAL_RESEARCH_SUCCESS",
      "source_preregistration_SHA":PROTOCOL,"source_fisher_manifest_SHA":sha(secure(FSRC)),
      "source_24_train_private_SHA":SOURCE_SHA,"frozen_40_val_SHA":DEV_SHA,
      "policy_SHA":policy["mask_sha256"],"source_fisher":src,
      "native_results":runs,"paired_bootstrap":comparisons,"BF16_external_oracle_crosscheck":oracle_crosscheck,
      "Fisher_policy_better_mean_NLL_than_n_baselines":winning,
      "Qv5_fisher_research_promotion_allowed":False,
      "Qv5_fisher_result_generalization_established":False,
      "old_V6_unknown_holdout_excluded_from_data_access":True,
      "native_CPU_latency_not_remeasured_in_this_40_held_eval":True,
      "legacy_main_and_immutable_release_modified":False,
      "limitations":[
         "Only one Qwen2.5-1.5B model and 24 training/40 development ARC-Challenge multiple-choice questions; not generalization",
         "Empirical directional Fisher E[(grad·delta)^2] is a first-order original BF16 approximation, not exact Hessian or causal quantized-model output loss",
         "BF16 MPS autograd weights/gradients rounded to BF16; native C dequantized residuals stored as FP16 with F64 dot accumulation",
         "All source gradients from teacher-forced correct one-letter label; not multi-token generative quality",
         "Actual native C Q4/n8 six-policy results measured, but per-question BF16 source gradients not recomputed in an independent framework",
         "No established accuracy/speed/energy advantage and no original V6 24 final holdout provenance"
      ]}
    os.umask(0o077)
    priv=(json.dumps(report,sort_keys=True,indent=2)+"\n").encode()
    with OUT.open("xb") as f:f.write(priv);f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    public={k:v for k,v in report.items() if k not in ("native_results",)}
    public["native_summary"]={mode:{"correct":data["correct_count"],"total":40,
      "accuracy":data["accuracy"],"answer_only_mean_NLL":data["NLL"]}
      for mode,data in runs["modes"].items()}
    public["private_independent_full_audit_SHA256"]=sha(priv)
    p=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(p);f.flush();os.fsync(f.fileno())
    print("QV5_INDEPENDENT_BF16_FISHER_AND_40_ARC_NATIVE_AUDIT_PASS",json.dumps({
      "public_sha256":sha(p),"private_audit_SHA256":sha(priv),
      "Qv5_correct":runs["modes"]["v5_fisher"]["correct_count"],
      "baseline_to_fisher_mean_NLL_diffs":{k:round(v["baseline_minus_Fisher_NLL_mean"],5) for k,v in comparisons.items()},
      "no_live_promotion":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
