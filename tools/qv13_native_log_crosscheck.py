#!/usr/bin/env python3
"""Independent original-C Qv13 log/input/mask cross-check. Aggregate-only stdout."""
import argparse, hashlib, json, math, pathlib, re, sys
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023")
ROOT=pathlib.Path(__file__).resolve().parents[1]
PRE=ROOT/"configs/research/BEGLIN_QV13_ACTUAL_NATIVE_EVAL_PRELOCK_20261023.json"
EXPECTED_PRE="5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a"
EXPECTED_PRIVATE="79826eb944ac2a4f87b637a430471f069a3789da37cdb12a9b7d1060d49db9b8"
EXPECTED_PUBLIC="9d045e1056e5ec8bc3a093a8a3d198277cbbc4083fb054a929b566cb2bb03c8c"
RE_ROW=re.compile(r"^\[qv5-choice\] window ([0-9]+) NLL ([^\s]+) expected ([0-9]+) predicted ([0-9]+) prefix ([0-9]+)$",re.M)
RE_TOTAL=re.compile(r"^qv5_choice NLL ([^\s]+) items ([0-9]+) correct ([0-9]+) mean ([^\s]+) ppl ([^\s]+)$",re.M)
RE_ROLES=re.compile(r"projection_role_tensors: f32=([0-9]+) n8=([0-9]+) q4=([0-9]+)")
ARMS=("q4","source_qv8","random73","random137","random211","random389","random577","original_v2")
BATCHES=("pilot16","arc128","ob128")
DELTA={"q_proj":1179648,"k_proj":196608,"v_proj":196608,"o_proj":1179648,"gate_proj":6881280,"up_proj":6881280,"down_proj":6881280}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for chunk in iter(lambda:f.read(1<<18),b""):h.update(chunk)
 return h.hexdigest()
def eq(value,expected,meaning):
 if value!=expected:raise RuntimeError(f"INDEPENDENT_AUDIT_FAIL:{meaning}: {value} != {expected}")
def scan(batch):
 if batch not in BATCHES:raise RuntimeError("unsupported batch")
 eq(sha(PRE),EXPECTED_PRE,"prereg")
 lockp=LAB/"qv13_lock_private.json"
 eq(sha(lockp),EXPECTED_PRIVATE,"original private frozen authority")
 eq(sha(ROOT/"docs/research/BEGLIN_QV13_ORIGINAL_NATIVE_256_PREEXEC_LOCK_PUBLIC_20261023.json"),EXPECTED_PUBLIC,"public freeze")
 lock=json.loads(lockp.read_text())
 eq(lock["prelock_sha256"],EXPECTED_PRE,"source prelock binding")
 man=LAB/"native_runs"/batch/"input_manifest.json"
 eq(sha(man),lock["batches"][batch]["manifest_sha256"],"frozen input manifest")
 manifest=json.loads(man.read_text())
 meta=manifest["original_items"];N=manifest["count"]
 if N!=len(meta) or N not in (16,128):raise RuntimeError("manifest item count invalid")
 results={}
 base=LAB/"native_runs"/batch
 for arm in ARMS:
  f=base/arm
  recp=f/"native_qv13_original_C_receipt.json"
  if not recp.is_file():continue
  r=json.loads(recp.read_text())
  logp=f/"native_qv13_original_C_run.log"
  eq(sha(logp),r["native_log_sha256"],"raw native C log content")
  eq(r["prelock_sha256"],EXPECTED_PRE,"receipt prelock")
  eq(r["preexecution_lock_sha256"],EXPECTED_PRIVATE,"receipt private freeze")
  eq(r["input_manifest_sha256"],sha(man),"receipt input batch")
  eq(r["arm"],arm,"receipt arm")
  eq(r["batch"],batch,"receipt batch")
  eq(r["n"],N,"receipt count")
  eq(r["status"],"ORIGINAL_NATIVE_28L_REAL_Q4_OR_28N8_SCORING_PASS","receipt status")
  kind="q4" if arm=="q4" else "mixed"
  eq(r["native_original_binary_sha256"],lock["original_C_binary_sha256"][kind],"frozen native C executable")
  if arm!="q4":
   mask=LAB/"masks"/f"{arm}.txt"
   eq(sha(mask),lock["masks"][arm]["mask_sha256"],"actual role/layer mask")
   eq(r["mask_sha256"],sha(mask),"receipt mask")
   parts=[line.split() for line in mask.read_text().splitlines()]
   if len(parts)!=28 or len(set(tuple(x) for x in parts))!=28:
    raise RuntimeError("not unique 28-bit roles")
   role_counts={}
   cost=0
   for role,num in parts:
    if role not in DELTA or not num.isdigit() or not 0<=int(num)<28:
     raise RuntimeError("invalid role/layer in original source mask")
    cost+=DELTA[role]
    role_counts[role]=role_counts.get(role,0)+1
   eq(cost,lock["masks"][arm]["incremental_packed_bytes"],"original packed-byte delta")
   eq(736985088+cost,lock["masks"][arm]["projection_packed_bytes"],"original packed total")
   eq(role_counts,lock["masks"][arm]["role_counts"],"role counts")
  else:
   eq(r["mask_sha256"],None,"q4 mask")
  text=logp.read_text()
  x=RE_ROLES.findall(text)
  eq(len(x),1,"native C roles line count")
  eq(tuple(map(int,x[0])),(0,0,196) if arm=="q4" else (0,28,168),"actual original C role precision")
  if arm!="q4" and "validated mixed 28/196 n8 tensor role-layer mask" not in text:
   raise RuntimeError("missing original C native n8 mask validation")
  rows=RE_ROW.findall(text)
  eq(len(rows),N,"native per-window records")
  losses=[];right=[]
  for i,values in enumerate(rows):
   idx,nll,answer,pred,length=values
   eq(int(idx),i,"source C window index")
   eq(int(length),meta[i]["prefix_len"],"frozen original prompt length")
   choice=(f/"eval/ppl_work"/f"w{i}.choices")
   ids=(f/"eval/ppl_work"/f"w{i}.i32")
   eq(sha(choice),meta[i]["label_sha256"],"native choices bound to original sealed input")
   eq(sha(ids),meta[i]["ids_sha256"],"native input tokens bound to original sealed input")
   if ids.stat().st_size!=4*int(length):raise RuntimeError("truncated native i32 tokens")
   choices=[int(v) for v in choice.read_text().split()]
   if len(choices)!=5 or len(set(choices[:4]))!=4:
    raise RuntimeError("invalid exact four-choice score boundary")
   eq(int(answer),choices[4],"original C gold token index")
   if not 0<=int(pred)<4:raise RuntimeError("invalid native predicted choice")
   v=float(nll)
   if not math.isfinite(v) or v<0:raise RuntimeError("nonfinite native NLL")
   eq(v,r["article_fullvocab_gold_NLL"][i],"receipt per-item gold NLL")
   iscorrect=int(int(answer)==int(pred))
   eq(iscorrect,r["article_correct"][i],"receipt per-item accuracy")
   losses.append(v);right.append(iscorrect)
  z=RE_TOTAL.findall(text)
  eq(len(z),1,"native original model final summary")
  total,n,correct,mean,ppl=map(float,z[0])
  if int(n)!=N or int(correct)!=sum(right) or abs(sum(losses)-total)>1e-7:
   raise RuntimeError("original C NLL total not independently reproduced")
  if abs(mean-total/N)>1e-8 or abs(mean-r["mean_fullvocab_gold_NLL"])>1e-8:
   raise RuntimeError("native/receipt loss mean mismatch")
  eq(int(correct),r["correct_ABCD"],"C accuracy")
  results[arm]={"n":N,"correct":int(correct),"mean_gold_fullvocab_NLL":mean,
   "native_receipt_sha256":sha(recp),
   "source_original_C_log_sha256":sha(logp),
   "projection_packed_bytes":736985088 if arm=="q4" else lock["masks"][arm]["projection_packed_bytes"]}
 return {"schema":"BEGLIN-Qv13-independent-real-original-C-log-reparse-public/1",
  "status":"PARTIAL_INDEPENDENT_C_LOG_REPARSE_PASS" if len(results)<8 else "ALL_EIGHT_NATIVE_C_LOG_REPARSE_PASS",
  "batch":batch,"n_original_questions_per_arm":N,
  "tested_arms":sorted(results),"missing_arms":[x for x in ARMS if x not in results],
  "source_prelock_sha256":EXPECTED_PRE,"original_preexecution_private_sha256":EXPECTED_PRIVATE,
  "frozen_dataset_manifest_sha256":sha(man),"public_freeze_sha256":EXPECTED_PUBLIC,
  "original_C_rows_reparsed":N*len(results),"no_raw_prompt_or_per_case_score_published":True,
  "qv13_cannot_claim_dynamic_precision_switch":True,
  "qv13_research_only_no_release_change":True,"results":results}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--batch",choices=BATCHES,default="pilot16")
 a=p.parse_args()
 print(json.dumps(scan(a.batch),ensure_ascii=False,sort_keys=True,indent=2))
