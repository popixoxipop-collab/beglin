#!/usr/bin/env python3
"""Independent SHA/arithmetical audit of real TinyLlama 154x64 Fisher and C test48.

No imports from model calibrator, policy optimizer or native evaluator.
Reads original source-only 64 BF16 loss gradients, rechecks 154 gain means,
mask bit counts/cost constraints and all 96 genuine C native next-token NLL
and four-way predictions against frozen inputs. Dev only; not generalization.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,struct
import numpy as np
from scipy.stats import spearmanr,binomtest

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv7_tinyllama_20261022"
PRE=ROOT/"configs/research/BEGLIN_QV7_CROSSFAMILY_CALIBRATION_PRELOCK_20261022.json"
TRAIN=LAB/"source_arc64_tinyllama_frozen.json"
TEST=LAB/"eval_openbookqa_test48_tinyllama_frozen.json"
FISH=LAB/"TinyLlama_original_BF16_154_directional_Fisher_source64_private.json"
FISHPUB=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_SOURCE_154_FISHER_PUBLIC_20261022.json"
COST=ROOT/"docs/research/BEGLIN_QV7_LLAMA_NATIVE_SEVEN_ROLE_COSTS_20261022.json"
POL=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_154_COST_FISHER_POLICY_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV7_TINYLLAMA_22N8_FISHER_FROZEN_POLICY_20261022.txt"
BF16=LAB/"TinyLlama_original_BF16_test48_private.json"
RUN=LAB/"native_openbookqa_test48_two_arms"
OUT=LAB/"crossfamily_tinyllama_154x64_2native_independent_audit.json"
PUB=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_FISHER_NATIVE48_AUDIT_PUBLIC_20261022.json"
MODES=("q4","fisher_22n8")
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
PRE_SHA="1b5fa904cf11f25292720efc75f658fc10cedc8923236b4d5b0ae7e70d3e88c6"
SOURCE_SHA="59c1d22197ba83ac3769049c01de4ae98414c678066ef6faa48fa2a0859bbe41"
TEST_SHA="8807002e79afadcb2d150b32dffdefa204afb0d70678375a608b7900f18d0ab0"
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
COUNTS=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def sealed(p):
    if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
        raise RuntimeError("unsealed original BF16 source or C model output "+str(p))
    return p.read_bytes()
def sources():
    if sha(PRE.read_bytes())!=PRE_SHA or sha(sealed(TRAIN))!=SOURCE_SHA or sha(sealed(TEST))!=TEST_SHA:
        raise RuntimeError("model family original preregistration or frozen source/test altered")
    model=json.loads(sealed(FISH));public=json.loads(FISHPUB.read_text())
    if (model["status"]!="SOURCE_ONLY_REAL_LLAMA_ORIGINAL_BF16_64_GRAD_AND_NATIVE_C_QUANT" or
        not model["forward_always_original_unquantized_BF16"] or
        public["private_original_source_SHA256"]!=sha(sealed(FISH))):
        raise RuntimeError("unverified source only actual TinyLlama gradients")
    if len(model["per_source_case_Fisher_Q4_by_tensor"])!=154 or len(model["per_source_case_Fisher_n8_by_tensor"])!=154:
        raise RuntimeError("missing 154 original Llama projection grads")
    m4=np.asarray(model["per_source_case_Fisher_Q4_by_tensor"],np.float64)
    m8=np.asarray(model["per_source_case_Fisher_n8_by_tensor"],np.float64)
    if m4.shape!=(154,64) or m8.shape!=(154,64) or not (np.isfinite(m4).all() and np.isfinite(m8).all()):
        raise RuntimeError("actual Llama train sample too short/nonfinite")
    gain=(m4-m8).mean(axis=1)
    for rid,r in enumerate(ROLES):
        if not np.allclose(gain[rid*22:(rid+1)*22],public["quality_upgrade_by_role_layer"][r],rtol=0,atol=1e-10):
            raise RuntimeError("independent original BF16 gain arithmetic differs "+r)
    rho=float(spearmanr((m4-m8)[:,::2].mean(axis=1),(m4-m8)[:,1::2].mean(axis=1)).statistic)
    if abs(rho-public["source_split_half_odd_even_spearman"])>1e-12:
        raise RuntimeError("source Llama model rank split not reproducible")
    pol=json.loads(POL.read_text());cost=json.loads(COST.read_text())
    if pol["status"]!="SOURCE_ONLY_EXACT_LLAMA_22N8_MASK_FROZEN_BEFORE_CROSSCORPUS_NATIVE_QUALITY" or \
       pol["source_154_public_sha256"]!=sha(FISHPUB.read_bytes()) or \
       pol["source_original_Llama_seven_native_costs_sha256"]!=sha(COST.read_bytes()) or \
       pol["solver_status"]!=0 or pol["mip_gap"]>1e-7:
        raise RuntimeError("source-only Fisher exact MILP not source-locked")
    mask=MASK.read_bytes()
    if sha(mask)!=pol["mask_sha256"]:raise RuntimeError("source selected 22 n8 SHA altered")
    ids=[tuple(line.split()) for line in mask.decode().splitlines()]
    if len(ids)!=22 or len(set(ids))!=22 or any(len(x)!=2 or x[0] not in ROLES or not 0<=int(x[1])<22 for x in ids):
        raise RuntimeError("not exact original Llama 154 split 22n8/132q4")
    source_before=pol["source_q4_projection_cost_ms"]
    if pol["selected_projection_cost_ms"]>pol["source_measured_projection_cost_budget_ms"]+1e-6 or \
       pol["selected_projection_bytes"]>pol["rolebalanced_22_upgrade_memory_cap_bytes"]:
        raise RuntimeError("source-only native cost constraints violated")
    return model,pol,rho

def actual_native(pol):
    orig=json.loads(sealed(TEST))["test48"]
    manifest=json.loads(sealed(RUN/"frozen_test48_native_manifest.json"))
    if manifest["n8_mask_SHA256"]!=pol["mask_sha256"] or \
       manifest["held48_SHA256"]!=TEST_SHA or len(manifest["tokens"])!=48:
        raise RuntimeError("Llama evaluation input locks not original")
    outputs={};sha_binary={}
    for mode in MODES:
        folder=RUN/mode
        original=json.loads(sealed(folder/"native_fullmodel_quality_receipt.json"))
        raw=sealed(folder/"native_tinyllama_48choice_log.txt")
        if original["mode"]!=mode or original["native_log_sha256"]!=sha(raw) or \
           original["frozen_test_manifest_SHA256"]!=sha(sealed(RUN/"frozen_test48_native_manifest.json")) or \
           original["model_layers"]!=22:
            raise RuntimeError("source original actual BEGLIN native process transcript changed "+mode)
        role=COUNTS.findall(raw.decode())
        shape=(0,0,154) if mode=="q4" else (0,22,132)
        if len(role)!=1 or tuple(map(int,role[0]))!=shape:
            raise RuntimeError("wrong actual C q4/n8 mix projection tier")
        binary=(pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base")
             if mode=="q4" else LAB/"native_mixed_22/beglin_tinyllama_22layer_mixed")
        if sha(binary.read_bytes())!=original["binary_sha256"]:
            raise RuntimeError("real native model executable drift")
        sha_binary[mode]=sha(binary.read_bytes())
        logs=ROW.findall(raw.decode())
        if len(logs)!=48:raise RuntimeError("Llama actual forward not 48 cases")
        vals=[];right=[]
        for i,(ix,nll,y,pred,np_) in enumerate(logs):
            q=orig[i]
            md=manifest["tokens"][i]
            pbytes=sealed(folder/f"eval/ppl_work/w{i}.i32")
            expected=struct.pack("<"+"i"*len(q["prefix_ids"]),*q["prefix_ids"])
            choice=(" ".join(str(q["suffix_ids"][l]) for l in "ABCD")+
                    " "+str("ABCD".index(q["correct"]))+"\n").encode()
            candidate=sealed(folder/f"eval/ppl_work/w{i}.choices")
            v=float(nll)
            if int(ix)!=i or pbytes!=expected or sha(pbytes)!=md["ids_sha"] or \
               candidate!=choice or sha(candidate)!=md["choices_sha"] or \
               int(np_)!=md["length"] or int(y)!="ABCD".index(q["correct"]) or \
               not 0<=int(pred)<4 or not math.isfinite(v) or v<0:
                raise RuntimeError("real source query or choice logits modified in "+mode)
            vals.append(v);right.append(int(int(y)==int(pred)))
        totals=SUM.findall(raw.decode())
        if len(totals)!=1:raise RuntimeError("native C actual metric not attested")
        total,count,cnt,mean,ppl=map(float,totals[0])
        if count!=48 or int(cnt)!=sum(right) or abs(total-sum(vals))>1e-7 or \
           abs(mean-total/48)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
            raise RuntimeError("independent native C fullvocab logits-derived NLL disagrees")
        if original["metrics"]["correct"]!=sum(right) or \
            abs(original["metrics"]["correct_answer_NLL"]-mean)>1e-12:
            raise RuntimeError("native scorer arithmetic mismatch "+mode)
        outputs[mode]={"N":48,"correct":sum(right),"NLL":mean,
          "per_question_NLL":vals,"per_question_right":right}
    return outputs,sha_binary

def summarize():
    if OUT.exists() or PUB.exists():raise FileExistsError("independent native audit already exists")
    model,pol,rho=sources()
    out,binaries=actual_native(pol)
    a=out["q4"];b=out["fisher_22n8"]
    diff=np.asarray(b["per_question_NLL"])-np.asarray(a["per_question_NLL"])
    rng=np.random.default_rng(20261022)
    idx=rng.integers(0,48,size=(5000,48))
    ci=np.percentile(diff[idx].mean(axis=1),[2.5,97.5]).tolist()
    a_right=np.asarray(a["per_question_right"]);b_right=np.asarray(b["per_question_right"])
    gained=int(np.sum((a_right==0)&(b_right==1)))
    lost=int(np.sum((a_right==1)&(b_right==0)))
    mcnemar=float(binomtest(min(gained,lost),gained+lost,p=.5,alternative="two-sided").pvalue) if gained+lost else 1.
    bf16=json.loads(sealed(BF16))
    test=json.loads(sealed(TEST))["test48"]
    if bf16["frozen_test48_sha256"]!=TEST_SHA or len(bf16["per_case_results"])!=48 or \
       bf16["status"]!="ACTUAL_ORIGINAL_LLAMA_FAMILY_BF16_UNMODIFIED_MODEL_COMPLETE":
        raise RuntimeError("original TinyLlama BF16 fullmodel reference not same corpus")
    for i,q in enumerate(test):
        if bf16["per_case_results"][i]["id_sha256"]!=sha(q["id"].encode()):
            raise RuntimeError("independent true BF16 reference used different prompt")
    summary={"schema":"BEGLIN-Qv7-independent-cross-family-22N8-Llama-native-test48/1",
      "status":"PASS_SOURCE_ONLY_154x64_FISHER_MILP_AND_TWO_NATIVE_LLAMACPU_48",
      "original_model_revision":model["official_rev"],"source_64_sha256":model["frozen_train64_sha256"],
      "held_48_sha256":TEST_SHA,"source_154_fisher_full_sha256":sha(sealed(FISH)),
      "source_64_half_rank_correlation":rho,
      "source_milp_SHA256":sha(POL.read_bytes()),"frozen_22mask_SHA256":pol["mask_sha256"],
      "actual_Llama_native_binary_SHA256":binaries,
      "actual_native_results":out,
      "cross_family_original_BF16_reference":{"correct":bf16["correct_count"],"NLL":bf16["correct_label_NLL"]},
      "Fisher_minus_Q4_mean_NLL":float(diff.mean()),
      "Fisher_minus_Q4_95pct_bootstrap_interval":[float(v) for v in ci],
      "Fisher_correct_delta_vs_Q4":b["correct"]-a["correct"],
      "Fisher_mcnemar_gain_only":gained,"Fisher_mcnemar_lost_only":lost,
      "Fisher_mcnemar_exact_p":mcnemar,
      "fisher_quality_gain_proven":False,
      "single_test_case_prior_Q4_smoke_consumption_declared":True,
      "cross_family_kernel_inference_implemented":True,
      "cross_family_mixed_accuracy_generalization_established":False,
      "live_operator_and_main_unchanged":True}
    os.umask(0o077)
    raw=(json.dumps(summary,sort_keys=True,indent=2)+"\n").encode()
    with OUT.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    public={k:v for k,v in summary.items() if k not in ("actual_native_results","actual_Llama_native_binary_SHA256")}
    public["actual_native_summary"]={mode:{k:v for k,v in row.items() if not k.startswith("per_question")} for mode,row in out.items()}
    public["private_full_independent_audit_sha256"]=sha(raw)
    x=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
    with PUB.open("xb") as f:f.write(x);f.flush();os.fsync(f.fileno())
    print("QV7_INDEPENDENT_ACTUAL_LLAMA_C_MODEL_FAMILY_AUDIT_DONE",
      json.dumps({"score":public["actual_native_summary"],"Fisher_minus_Q4_NLL":float(diff.mean()),
       "95pct_CI":ci,"McNemar_p":mcnemar,"src_rho":rho,
       "public_sha256":sha(x)},sort_keys=True),flush=True)
if __name__=="__main__":summarize()
