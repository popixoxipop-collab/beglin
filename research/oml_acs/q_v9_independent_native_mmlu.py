#!/usr/bin/env python3
"""Independent SHA/exact-token/numerical replay of 5 real C MMLU48 arms."""
import hashlib,json,math,os,pathlib,re,struct
import numpy as np
from scipy.stats import binomtest
R=pathlib.Path(__file__).resolve().parents[2];L=R/"lab_local/qv9_ggn_20261023"
N=L/"MMLU48_5arm_original_CPU_native";D=R/"docs/research";C=R/"configs/research"
V=L/"MMLU_psych_CS_48_eval_frozen.json"
PO=D/"BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
SA=D/"BEGLIN_QV9_INDEPENDENT_OUTPUT_GGN_TRAIN_DIAGNOSTIC_20261023.json"
O=L/"independent_original_5arm_MMLU48_native_audit.json"
P=D/"BEGLIN_QV9_MMLU48_NATIVE_FIVE_ARM_INDEPENDENT_PUBLIC_20261023.json"
MODES=("q4","qv6_64","qv6_128","qv8_joint96","qv9_ggn24")
MASK={"qv6_64":C/"BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",
      "qv6_128":C/"BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt",
      "qv8_joint96":C/"BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt",
      "qv9_ggn24":C/"BEGLIN_QV9_GGN_28N8_FROZEN_MASK_20261023.txt"}
BIN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin")
BINS={"q4":(BIN/"beglin_qv5_base","408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"),
 "mixed":(BIN/"beglin_qv5_mixed","f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221")}
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
TIER=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def sha(x):return hashlib.sha256(x).hexdigest()
def read(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:raise RuntimeError("not original 0600 BEGLIN native evidence "+str(p))
 return p.read_bytes()
def reparse(mode,items,manifest):
 folder=N/mode;blob=read(folder/"real_mmlu48_native_C_log.txt")
 rec=json.loads(read(folder/"original_mmlu48_native_receipt.json"))
 if rec["mode"]!=mode or rec["native_log_SHA256"]!=sha(blob) or rec["frozen_input_manifest_SHA256"]!=sha(read(N/"frozen_5x48_manifest.json")):
  raise RuntimeError("original C process log SHA/report mismatch "+mode)
 name="q4" if mode=="q4" else "mixed"
 exe,h=BINS[name]
 if sha(exe.read_bytes())!=h or rec["original_binary_SHA256"]!=h:raise RuntimeError("different Qwen original C executable "+mode)
 tiers=TIER.findall(blob.decode())
 if len(tiers)!=1 or tuple(map(int,tiers[0]))!=((0,0,196) if mode=="q4" else (0,28,168)):
  raise RuntimeError("native C quant precision mask wrong "+mode)
 if mode!="q4" and (sha(MASK[mode].read_bytes())!=rec["source_mask_SHA256"] or rec["source_mask_SHA256"]!=manifest["all_policy_masks_SHA256"][mode]):
  raise RuntimeError("original SHA frozen policy changed "+mode)
 rows=ROW.findall(blob.decode())
 if len(rows)!=48:raise RuntimeError("not 48 actual native MMLU source prompts "+mode)
 scores=[];rights=[]
 for i,(seq,nll,y,pred,length) in enumerate(rows):
  q=items[i];f=manifest["inputs"][i]
  data=struct.pack("<"+"i"*len(q["prefix_ids"]),*q["prefix_ids"])
  choices=(" ".join(str(q["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(q["correct"]))+"\n").encode()
  if int(seq)!=i or int(y)!="ABCD".index(q["correct"]) or not 0<=int(pred)<4 or int(length)!=len(q["prefix_ids"]):
   raise RuntimeError("real MMLU original candidate tokens not coherent")
  if read(folder/f"eval/ppl_work/w{i}.i32")!=data or read(folder/f"eval/ppl_work/w{i}.choices")!=choices or sha(data)!=f["prompt_SHA256"] or sha(choices)!=f["choices_SHA256"]:
   raise RuntimeError("prelocked BF16-native model prompt SHA tampered")
  v=float(nll)
  if not math.isfinite(v) or v<0:raise RuntimeError("wrong native NLL")
  scores.append(v);rights.append(int(int(y)==int(pred)))
 totals=SUM.findall(blob.decode())
 if len(totals)!=1:raise RuntimeError("native scorer aggregate missing")
 total,n,correct,mean,ppl=map(float,totals[0])
 if int(n)!=48 or correct!=sum(rights) or abs(total-sum(scores))>1e-7 or abs(mean-total/48)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
  raise RuntimeError("native MMLU next-token NLL/accuracy arithmetic mismatch")
 if abs(rec["metrics"]["mean_correct_fullvocab_next_token_NLL"]-mean)>1e-12 or rec["metrics"]["correct"]!=sum(rights):
  raise RuntimeError("original native scorer and independent evaluator disagree")
 return {"correct":int(correct),"n":48,"mean_fullvocab_correct_nexttoken_NLL":mean,
         "per_question_NLL":scores,"per_question_correct":rights}
def run():
 if O.exists() or P.exists():raise FileExistsError("no repeat of finalized independent audit")
 original=json.loads(read(V))["items"];policy=json.loads(PO.read_text())
 source=json.loads(SA.read_text())
 if len(original)!=48 or policy["status"]!="GGN_SOURCE_ONLY_FROZEN_BEFORE_DIAG_OR_MMLU" or source["status"]!="PASS_48_ACTUAL_BF16_SOURCE_CASES_AND_144_JACOBIAN_BACKWARDS_REPLAYED":
  raise RuntimeError("model/source-only provenance not confirmed")
 m=json.loads(read(N/"frozen_5x48_manifest.json"))
 if m["status"]!="FROZEN_SOURCE_ONLY_GGN_MASKS_AND_48_MMLU_INPUTS_BEFORE_C_FORWARD" or m["original_GGN_source_policy_doc_SHA256"]!=sha(PO.read_bytes()):
  raise RuntimeError("Qv9 source-only mask learned from validation")
 out={k:reparse(k,original,m) for k in MODES}
 rng=np.random.default_rng(20261023);ind=rng.integers(0,48,size=(5000,48))
 j=out["qv9_ggn24"];comparison={}
 for k in MODES[:-1]:
  b=out[k]
  delta=np.asarray(j["per_question_NLL"])-np.asarray(b["per_question_NLL"])
  a=np.asarray(b["per_question_correct"]);z=np.asarray(j["per_question_correct"])
  gain=int(((a==0)&(z==1)).sum());lost=int(((a==1)&(z==0)).sum())
  comparison[k]={"GGN_minus_baseline_mean_fullvocab_NLL":float(delta.mean()),
    "paired_bootstrap95_percent":[float(x) for x in np.percentile(delta[ind].mean(axis=1),[2.5,97.5])],
    "accuracy_difference":j["correct"]-b["correct"],"GGN_lower_NLL_cases":int(sum(delta<0)),
    "McNemar_exact_p":float(binomtest(min(gain,lost),gain+lost,p=.5).pvalue) if gain+lost else 1.}
 ref=json.loads(read(L/"original_BF16_Qwen_MMLU48_reference_private.json"))
 if len(ref["per_case_original_BF16"])!=48 or ref["pinned_MMLU_frozen_inputs_SHA256"]!="2203bf54dfbc20e07488a234a012011efd78b098876e981de8ebed12cf8045c9":
  raise RuntimeError("BF16 original MMLU source reference differs")
 for a,q in zip(ref["per_case_original_BF16"],original):
  if a["original_question_id_sha256"]!=sha(q["id"].encode()):raise RuntimeError("BF16 comparison different MMLU case")
 record={"schema":"BEGLIN-Qv9-independent-source-GGN-and-five-native-MMLU48-audit/1",
  "status":"PASS_SOURCE_AND_ORIGINAL_MMLU48_FIVE_NATIVE_POLICY_AUDIT",
  "source_48_BF16_GGN_audit_SHA256":sha(SA.read_bytes()),
  "Qv9_source_policy_sha256":sha(PO.read_bytes()),"mask_sha256":policy["mask_sha256"],
  "native_original_full_results":out,"paired_comparisons":comparison,
  "original_BF16_reference_correct":ref["correct"],"original_BF16_reference_NLL":ref["mean_correct_answer_full_vocab_NLL"],
  "native_fullvocab_NLL_not_conditional_choice_GGN":True,
  "generalization_established":False,"native_CPU_speed_win_established":False,
  "auto_promotion":False,"operational_release_unchanged":True}
 os.umask(0o077)
 raw=(json.dumps(record,sort_keys=True,indent=2)+"\n").encode()
 with O.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 O.chmod(0o600)
 pub={k:v for k,v in record.items() if k!="native_original_full_results"}
 pub["native_summary"]={k:{field:value for field,value in v.items() if not field.startswith("per_question")} for k,v in out.items()}
 pub["private_SHA256"]=sha(raw)
 blob=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
 with P.open("xb") as f:f.write(blob);f.flush();os.fsync(f.fileno())
 print("QV9_INDEPENDENT_5ARM_REAL_MMLU48_NATIVE_AUDIT_PASS",json.dumps({"scores":{k:(v["correct"],v["mean_fullvocab_correct_nexttoken_NLL"]) for k,v in out.items()},"paired":comparison,"private_SHA":sha(raw),"public_SHA":sha(blob)},sort_keys=True),flush=True)
if __name__=="__main__":run()
