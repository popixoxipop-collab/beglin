#!/usr/bin/env python3
"""Independently recompute one P18 gold NLL from true BF16+A/B full-vocab logits.
Raw data stays on D50; no model calls or production changes.
"""
from __future__ import annotations
import hashlib,json,pathlib,re,math
import numpy as np
R=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
E=R/"native1024_real_evidence"
H={
 "teacher_report":("p18-mmlu-000.teacher_receipt.json","c7628194d2e601f64525b1c05e5c59be5848eb40c68420dbd5c935e6ec148c3b"),
 "teacher_raw":("p18-mmlu-000.teacher_full_logits.npy","7025aed3a8c23fdc024f9a2f4e4a9c4944129f09d1bc3c32633c57c21a1249b6"),
 "teacher_source":("p18-mmlu-000.original_5291_tensor_hashes.json","88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda"),
 "native_report":("p18-one-gold-native-AB-logits-receipt.json","d41531836ad8fd1fc38fb3ff798894d618365998d431ab3987140a753609eb04"),
 "native_log":("p18-one-gold-native-AB-worker.log","09a19c990347f59ffd9b75790558cd05e0ac9ac4f869193a3f469927c331c631")
}
FROZEN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
PROTO="77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c"
BIN="ed2af527c1291567b0e9e67a9940cb5f13b298d4a645d11742ff1cc34248bc6e"
STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
PAT=re.compile(r"^P18_NATIVE_GOLD_PREFIX_LOGITS rid=([a-f0-9]{32}) inputsha=([a-f0-9]{64}) plen=(\d+) pos=(\d+) next=(\d+) vocab=(102400) bytes=(409600) floatsha=([a-f0-9]{64})$")
def sha(p):
 p=pathlib.Path(p)
 if not p.is_file() or p.is_symlink():raise ValueError("Missing/linked original raw evidence")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def tok(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def nll(a,idx):
 a=np.asarray(a,dtype=np.float64)
 if a.shape!=(102400,) or not np.isfinite(a).all():raise ValueError("Actual full-vocab numeric data invalid")
 m=float(a.max());return float(m+np.log(np.exp(a-m).sum())-a[idx])
def same_stage(o,keys):
 stage=o.get(keys[0]);after=o.get(keys[1])
 expect={"stage_sha256":STAGE,"route_sha256":ROUTE,"stage_pids":[35481,35493]}
 if keys[0].startswith("original_live"):expect["scratch_pids"]=[]
 if stage!=expect or after!=expect:raise ValueError("Production route/pids/source drifted")
def check():
 if sha(R/"p18_multishot_frozen_PRIVATE.json")!=FROZEN or sha(R/"p18_protocol_FROZEN.json")!=PROTO:
  raise ValueError("Preregistered source/gate SHA drifted")
 for tag,(name,expected) in H.items():
  if sha(E/name)!=expected:raise ValueError("Original "+tag+" SHA differs")
 bank=json.loads((R/"p18_multishot_frozen_PRIVATE.json").read_text())
 q=next(x for x in bank["items"] if x["item_id"]=="p18-mmlu-000")
 ids,gold=q["prompt_ids"],q["gold_ids"]
 if len(ids)!=107 or len(gold)!=1 or tok(ids)!=q["prompt_token_sha256"] or tok(gold)!=q["gold_token_sha256"]:
  raise ValueError("P18 source item token identity or length differs")
 target=gold[0]
 teacher=json.loads((E/H["teacher_report"][0]).read_text())
 native=json.loads((E/H["native_report"][0]).read_text())
 same_stage(teacher,("original_serving_before","original_serving_after"))
 same_stage(native,("original_live_before","original_live_after"))
 if(teacher.get("original_decoder_layers")!=27 or teacher.get("original_model_source_tensors_exactly_once")!=5291
    or len(teacher.get("per_layer",[]))!=27 or teacher.get("item_id")!=q["item_id"]
    or teacher.get("input_token_sha256")!=q["prompt_token_sha256"]
    or teacher.get("gold_token_sha256")!=q["gold_token_sha256"]
    or teacher.get("raw_full_logits_npy_sha256")!=H["teacher_raw"][1]
    or teacher.get("raw_source_tensor_hashes_sha256")!=H["teacher_source"][1]
    or teacher.get("production_touched") is not False
    or native.get("native_calls")!=2 or native.get("binary_sha256")!=BIN
    or native.get("frozen_protocol_sha256")!=PROTO
    or native.get("source_480_token_bank_sha256")!=FROZEN
    or native.get("production_touched") is not False
    or native.get("automatic_live_promotion") is not False):
  raise ValueError("Original BF16/native provenance contract violated")
 t=np.load(E/H["teacher_raw"][0],allow_pickle=False)
 if t.dtype!=np.float32 or t.shape!=(1,102400):raise ValueError("Original BF16 full-vocab array malformed")
 loss_teacher=nll(t[0],target)
 if abs(loss_teacher-teacher["gold_NLL_nats"][0])>1e-6:raise ValueError("Teacher NLL not independently recomputed")
 lines=(E/H["native_log"][0]).read_text(errors="replace").splitlines()
 witnesses={}
 for line in lines:
  if "P18_NATIVE_GOLD_PREFIX_LOGITS" not in line:continue
  m=PAT.fullmatch(line.strip())
  if not m:raise ValueError("C raw source line malformed")
  rid,h,plen,pos,first,vocab,size,dig=m.groups()
  if h!=q["prompt_token_sha256"] or int(plen)!=107 or int(pos)!=106 or rid in witnesses:
   raise ValueError("Original C generated gold prefix input provenance mismatch")
  witnesses[rid+".f32"]={"sha256":dig,"first":int(first)}
 if len(witnesses)!=2:raise ValueError("Missing original full-logit A/B capture lines")
 results={}
 rows=native.get("native_rows")
 if len(rows)!=2 or [x["policy"] for x in rows]!=["A","B"]:
  raise ValueError("Unpaired native policies")
 for row in rows:
  policy=row["policy"];fname=row["raw_original_full_logits"]
  trace=witnesses.get(fname)
  if trace is None or trace["sha256"]!=row["raw_full_logits_sha256"]:
   raise ValueError("Native worker witness/raw SHA mismatch")
  file=E/fname
  if file.stat().st_size!=409600 or sha(file)!=trace["sha256"]:
   raise ValueError("Actual native F32 full-vocab raw source invalid")
  z=np.fromfile(file,dtype="<f4")
  value=nll(z,target)
  if (abs(value-row["gold_nll_nats"])>1e-7 or
      int(np.argmax(z))!=row["generated_next_token_id"] or
      int(np.argmax(z))!=trace["first"]):
   raise ValueError("Original raw native NLL vs C log source mismatch")
  if policy=="B" and (row.get("B_transition_A_to_B") is not True or row.get("precision_epoch")!=3):
   raise ValueError("Native B precision epoch/ACK source invalid")
  results[policy]={"gold_NLL_nats":value,"strict_gold_first_token":int(np.argmax(z))==target,
                   "raw_full_logits_sha256":sha(file)}
 delta=results["B"]["gold_NLL_nats"]-results["A"]["gold_NLL_nats"]
 out={"schema":"beglin-hcp-p18-one-question-independent-three-real-full-vocab-NLL-v1",
  "status":"P18_ONE_ITEM_ORIGINAL_BF16_AND_NATIVE_A_B_NLL_INDEPENDENTLY_VERIFIED_NOT_FULL_GATE",
  "item_id":"p18-mmlu-000","vocab":102400,
  "frozen_480_token_sha256":FROZEN,"frozen_protocol_sha256":PROTO,
  "original_BF16_gold_NLL_nats":loss_teacher,
  "original_BF16_logits_sha256":H["teacher_raw"][1],
  "source_5291_tensor_hash_manifest_sha256":H["teacher_source"][1],
  "policy_A":results["A"],"policy_B":results["B"],"B_minus_A_gold_NLL_nats":delta,
  "original_BF16_complete_model_forward_count":1,"native_full_vocab_F32_captures":2,
  "P18_full_480_research_noninferiority_certified":False,
  "production_touched":False,"automatic_live_promotion":False}
 targetfile=R/"P18_ONE_ITEM_REAL_BF16_A_B_INDEPENDENT_NLL_AUDIT.json"
 if targetfile.exists():raise FileExistsError("Research audit receipt already sealed")
 with targetfile.open("x") as f:json.dump(out,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
 print(json.dumps({"status":out["status"],"teacher":round(loss_teacher,6),
  "A":round(results["A"]["gold_NLL_nats"],6),"B":round(results["B"]["gold_NLL_nats"],6),
  "B_minus_A":round(delta,6),"receipt_sha256":sha(targetfile)},sort_keys=True))
if __name__=="__main__":check()
