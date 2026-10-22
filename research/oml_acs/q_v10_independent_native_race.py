#!/usr/bin/env python3
"""Independent SHA/authentication and arithmetic replay, 48 RACE × 3 C arms.

No imports from native evaluation or grouped precision optimizer. Recomputes
exact correct-letter full-vocab NLL logs, four-choice argmax, source Q4/n8 mask
identity, original C binary SHA and identical unique full RACE passages;
paired bootstrap over 48 independent article IDs. Never modifies source.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,struct
import numpy as np
from scipy.stats import binomtest
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv10_topk_trust_20261023"
DIR=LAB/"native_RACE48_3arm_original_BEG_C"
RACE=LAB/"RACE_middle_validation_48_locked.json"
POL=ROOT/"docs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_SOURCE_POLICY_20261023.json"
SOURCE_AUDIT=ROOT/"docs/research/BEGLIN_QV10_48SOURCE_DIAGNOSTIC_INDEPENDENT_PUBLIC_20261023.json"
BF16=LAB/"original_Qwen_BF16_MPS_RACE48_reference_private.json"
BFP=ROOT/"docs/research/BEGLIN_QV10_ORIGINAL_BF16_RACE48_REFERENCE_PUBLIC_20261023.json"
OUT=LAB/"qv10_independent_actual_RACE48_3arm_native_audit_private.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV10_RACE48_NATIVE_3ARM_INDEPENDENT_PUBLIC_20261023.json"
BIN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin")
BINARY={
 "base":(BIN/"beglin_qv5_base","408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"),
 "mixed":(BIN/"beglin_qv5_mixed","f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221")}
MASKS={
 "fisher64":ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",
 "qv10_top2other_trust":ROOT/"configs/research/BEGLIN_QV10_TOP2OTHER_TRUST_28N8_SOURCE_POLICY_20261023.txt"}
ARMS=("q4","fisher64","qv10_top2other_trust")
ROWN=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUMM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
RACE_SHA="acdfd36477df4f440eeb9d03ad68342a34ce2435028eff1b8cef0de4c9f859de"
def sha(b):return hashlib.sha256(b).hexdigest()
def private(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
  raise RuntimeError("private model or original RACE input proof not protected "+str(p))
 return p.read_bytes()
def native(mode,manifest,inputs):
 folder=DIR/mode
 log=private(folder/"original_Qwen_C_RACE48_native_log.txt")
 rec=json.loads(private(folder/"original_RACE48_native_receipt.json"))
 binary,expected=BINARY["base" if mode=="q4" else "mixed"]
 if sha(binary.read_bytes())!=expected or rec["original_audited_C_binary_SHA"]!=expected:
  raise RuntimeError("native source C actual executable does not match original SHA")
 if rec["real_original_48_window_C_log_SHA256"]!=sha(log) or \
    rec["frozen_input_manifest_SHA256"]!=sha(private(DIR/"RACE48_native_original_preexecution_lock.json")) or \
    rec["mode"]!=mode:
  raise RuntimeError("original RACE fullmodel native C log not same model/inputs")
 if mode!="q4" and sha(MASKS[mode].read_bytes())!=rec["source_only_mask_SHA256"]:
  raise RuntimeError("source-only bit mask changed after evaluation")
 role=ROLE.findall(log.decode())
 if len(role)!=1 or tuple(map(int,role[0]))!=((0,0,196) if mode=="q4" else (0,28,168)):
  raise RuntimeError("real Qwen 28layer Q4/n8 projection count not true")
 allrows=ROWN.findall(log.decode())
 if len(allrows)!=48:raise RuntimeError("not exactly 48 original full native model RACE forwards")
 nll=[];hits=[];choicepred=[]
 for i,(seq,loss,y,pred,prefix) in enumerate(allrows):
  q=inputs[i];info=manifest["all_48_inputs"][i]
  raw=struct.pack("<"+"i"*len(q["prompt_token_ids"]),*q["prompt_token_ids"])
  opts=(" ".join(str(q["option_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(q["correct"]))+"\n").encode()
  if int(seq)!=i or int(y)!="ABCD".index(q["correct"]) or not 0<=int(pred)<4 or int(prefix)!=len(q["prompt_token_ids"]):
   raise RuntimeError("true C native RACE answer letter or prompt boundary mismatch")
  if raw!=private(folder/f"eval/ppl_work/w{i}.i32") or \
    opts!=private(folder/f"eval/ppl_work/w{i}.choices") or \
    sha(raw)!=info["prefix_sha256"] or sha(opts)!=info["fouranswer_sha256"]:
   raise RuntimeError("real native C RACE Qwen frozen prompt or answer bytes changed")
  val=float(loss)
  if not math.isfinite(val) or val<0:raise RuntimeError("native C valid next token loss required")
  nll.append(val);hits.append(int(y==pred));choicepred.append(int(pred))
 total=SUMM.findall(log.decode())
 if len(total)!=1:raise RuntimeError("RACE native full model aggregate invalid")
 summed,n,correct,mean,ppl=map(float,total[0])
 if n!=48 or correct!=sum(hits) or abs(summed-sum(nll))>1e-7 or \
    abs(mean-summed/48)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
  raise RuntimeError("independent fullvocab RACE mean NLL differs from native C")
 native_rec=rec["measurements"]
 if native_rec["correct"]!=correct or abs(native_rec["correct_letter_full_vocab_mean_NLL"]-mean)>1e-10:
  raise RuntimeError("source native scorer itself made incorrect claim")
 return {"n":48,"correct":int(correct),"accuracy":float(correct/48),
  "correct_letter_mean_fullvocab_NLL":float(mean),"native_bin_sha256":expected,
  "per_case_correct":hits,"per_case_NLL":nll}
def run():
 if OUT.exists() or PUBLIC.exists():raise FileExistsError("cannot replace original independent native RACE quality result")
 if sha(private(RACE))!=RACE_SHA:
  raise RuntimeError("new RACE original locked 48 articles changed")
 inputs=json.loads(private(RACE))["questions"]
 if len(inputs)!=48 or len(set(x["passage_original_id"] for x in inputs))!=48:
  raise RuntimeError("dataset pseudo-replication: one original RACE article only once")
 pol=json.loads(POL.read_text())
 source=json.loads(SOURCE_AUDIT.read_text())
 if pol["status"]!="SOURCE_ONLY_REAL_TOPK_OTHER_GROUPED_KL_28N8_MASK_FROZEN_BEFORE_DIAG_OR_RACE" or \
    source["status"]!="PASS_SOURCE_24_AND_DIAG_24_BF16_7GROUP_SIGNED_NATIVE_RESIDUAL_RECALC" or \
    source["mask_sha256"]!=pol["mask_sha256"]:
  raise RuntimeError("original BF16 grouped source and independent diagnostic not bound to native RACE")
 manifest=json.loads(private(DIR/"RACE48_native_original_preexecution_lock.json"))
 if manifest["status"]!="ALL_3_NATIVE_ARM_INPUTS_AND_SOURCE_MASKS_FROZEN_BEFORE_FORWARD" or \
    manifest["top2other_source_policy_sha"]!=sha(POL.read_bytes()) or manifest["source_RACE48_SHA256"]!=RACE_SHA:
  raise RuntimeError("original model native RACE input/mask SHA not frozen")
 scores={mode:native(mode,manifest,inputs) for mode in ARMS}
 rng=np.random.default_rng(20261023);draw=rng.integers(0,48,size=(5000,48))
 candidate=scores["qv10_top2other_trust"]
 paired={}
 for name in ARMS[:-1]:
  old=scores[name]
  change=np.asarray(candidate["per_case_NLL"])-np.asarray(old["per_case_NLL"])
  a=np.asarray(candidate["per_case_correct"]);b=np.asarray(old["per_case_correct"])
  gain=int(np.sum((a==1)&(b==0)));lost=int(np.sum((a==0)&(b==1)))
  paired[name]={
    "qv10_minus_baseline_mean_fullvocab_NLL":float(change.mean()),
    "paired_bootstrap95_interval":[float(z) for z in np.percentile(change[draw].mean(axis=1),[2.5,97.5])],
    "qv10_lower_NLL_in_articles":int(np.sum(change<0)),
    "qv10_minus_baseline_correct":candidate["correct"]-old["correct"],
    "McNemar_exact_two_sided_p":float(binomtest(min(gain,lost),gain+lost,p=.5).pvalue) if gain+lost else 1.}
 b=private(BF16);ref=json.loads(b)
 refpublic=json.loads(BFP.read_text())
 if refpublic["original_private_ref_SHA256"]!=sha(b) or \
    ref["source_RACE48_SHA256"]!=RACE_SHA or len(ref["per_question_model_outputs"])!=48:
  raise RuntimeError("external original BF16 RACE not same source cohort")
 for item,q in zip(ref["per_question_model_outputs"],inputs):
  if item["case_id_SHA256"]!=sha(q["id"].encode()) or item["correct_letter"]!=q["correct"]:
   raise RuntimeError("BF16 model reference different original RACE question labels")
 audit={"schema":"BEGLIN-Qv10-independent-original-BF16-7group-source-and-3arm-C-RACE48-audit/1",
  "status":"PASS_144_REAL_ORIGINAL_BEGLIN_C_RACE_NEXTTOKEN_LOGS_AND_BF16_SOURCE_AUDIT",
  "source_grouped_BF16_train_diag_public_SHA256":sha(SOURCE_AUDIT.read_bytes()),
  "source_only_Top2OTHER_policy_public_SHA256":sha(POL.read_bytes()),
  "frozen_RACE48_original_sha256":RACE_SHA,
  "model_original_Qwen_BF16_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  "native_original_144_real_RACE_results":scores,
  "paired_comparisons":paired,
  "separate_original_BF16_model_correct":ref["correct"],
  "separate_original_BF16_correct_token_mean_NLL":ref["full_vocab_correct_letter_mean_NLL"],
  "each_eval_article_distinct":True,
  "RACE_same_model_family_Qwen_not_multi_model_generalization":True,
  "system_load_not_controlled_for_latency":True,
  "full_vocab_prob_mass_preserved_only_at_original_BF16_grouping":True,
  "native_generalization_and_speed_superiority_unproved":True,
  "no_auto_promotion":True}
 os.umask(0o077)
 blob=(json.dumps(audit,sort_keys=True,indent=2)+"\n").encode()
 with OUT.open("xb") as f:f.write(blob);f.flush();os.fsync(f.fileno())
 OUT.chmod(0o600)
 pub={k:v for k,v in audit.items() if k!="native_original_144_real_RACE_results"}
 pub["native_summary"]={mode:{k:v for k,v in x.items() if not k.startswith("per_case")} for mode,x in scores.items()}
 pub["private_independent_audit_SHA256"]=sha(blob)
 data=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
 with PUBLIC.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 print("QV10_INDEPENDENT_ORIGINAL_C_RACE48_3ARM_NATIVE_AUDIT_PASS",
    json.dumps({"correct":{n:x["correct"] for n,x in scores.items()},
      "NLL":{n:round(x["correct_letter_mean_fullvocab_NLL"],6) for n,x in scores.items()},
      "paired":paired,"public_sha":sha(data),"private_sha":sha(blob)},sort_keys=True),flush=True)
if __name__=="__main__":run()
