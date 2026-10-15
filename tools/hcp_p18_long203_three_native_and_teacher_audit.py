#!/usr/bin/env python3
"""Independent immutable-data audit of P18 BF16 204-position + native A/B full-vocab logits.
No inference. Never promotes policy B or certifies 480-question noninferiority.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,re
import numpy as np

ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
R=ROOT/"native1024_real_evidence/P18_LONG203"
ITEM="p18-mmlu-018"
INPUT_SHA="374c74797344d68164ee84d3be8932ec86f7b2ffcea51b7cdbe4d6c4a72cde71"
FROZEN_SHA="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
PROTO_SHA="77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c"
BF16_RECEIPT="73091f999dafc59d53ea6a8646e2543951bca68750129d9bb7b120cb24c3d924"
BF16_F32_SHA="2cc6e216e6c734bc041c8897de1a126a5562988dd93d1022b960eb2484a6a783"
TENSOR_SHA="88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda"
AB_RECEIPT="5e5f27925d3fc5eaa978dd91df41dcf957a56dfa0fd85c967b47e9a2e092d66d"
AB_LOG="276ea1e482726cb44b29576ec6d15a8d53accd2c93bed5d0dea7d8b83c854650"
P18_LONG_SOURCE="fa8a328bedc1e56e79938460e1d4a171f0fd363535fc610183b4022e41ce0bfa"
P18_ORIGINAL_CAPTURE_SOURCE="82475145a4351b934cfccda6763112cdf25751e2ca470ac8417a16e6dc8fdb1c"
P18_LONG_BIN="bf8a70ec5f0e449c3e2c35aeeccc3932dbafe0a376322570d0ebd0a0cae8f81d"
P18_CORE_SOURCE="c2a631ee7d82feef83efce59087b0a144f7561b9da26db920e1201b45b689820"
ACTUAL={"A":("f6a353d99df241609a5cc35838618433.f32",
              "bdebf665b3111db7c6482cd3682697a62e3b9785d1dd4f66571300343dcf9bfb"),
        "B":("056ed167aebe4a878931dd1152d9dc51.f32",
             "5bf00419150101002257c713ce91eb4d44f901bec8634da81e76c84c9d13fb02")}
STAGE_RAW="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_RAW="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
LOG_RE=re.compile(r"^P18_NATIVE_GOLD_PREFIX_LOGITS rid=([a-f0-9]{32}) inputsha=([a-f0-9]{64}) plen=(\d+) pos=(\d+) next=(\d+) vocab=(102400) bytes=(409600) floatsha=([a-f0-9]{64})$")
def sha(p):
 p=pathlib.Path(p)
 if not p.is_file() or p.is_symlink():raise ValueError("Raw source missing or symlink")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def ensure(p,h):
 if sha(p)!=h:raise ValueError("Immutable SHA drift: "+str(p))
def tok(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def lprob(z):
 a=np.asarray(z,dtype=np.float64)
 if a.shape!=(102400,) or not np.isfinite(a).all():raise ValueError("Not a real finite 102400-F32 logit vector")
 m=float(a.max());den=m+math.log(float(np.exp(a-m).sum()))
 return a-den
def source_identity():
 short=ROOT/"native_p18_capture_buildroot/qwen_infer.c"
 long=ROOT/"native_p18_capture_long203_buildroot/qwen_infer.c"
 ensure(short,P18_ORIGINAL_CAPTURE_SOURCE)
 ensure(long,P18_LONG_SOURCE)
 pattern=b"(rq_plen[r] < 20 || rq_plen[r] > 112)"
 repl=b"(rq_plen[r] < 20 || rq_plen[r] >= MOE_CBATCH_MAXPOS)"
 data=short.read_bytes()
 if data.count(pattern)!=1 or data.replace(pattern,repl,1)!=long.read_bytes():
  raise ValueError("P18 >112 native C fix modified calculations or scopes beyond one bound")
 return {"source_bound_only_difference":True,"old_C_source_sha256":P18_ORIGINAL_CAPTURE_SOURCE,
         "new_C_source_sha256":P18_LONG_SOURCE}
def verify():
 scope=source_identity()
 ensure(ROOT/"p18_long203_teacher_pilot_PRIVATE.json",INPUT_SHA)
 ensure(R/"original_p18-mmlu-018_BF16_teacher_receipt.json",BF16_RECEIPT)
 ensure(R/"original_p18-mmlu-018_203BF16_logits.npy",BF16_F32_SHA)
 ensure(R/"original_p18-mmlu-018_5291_tensors.json",TENSOR_SHA)
 ensure(R/"p18-long203-native-AB-receipt.json",AB_RECEIPT)
 ensure(R/"p18-long203-native-AB-worker.log",AB_LOG)
 ensure(ROOT/"hcp_p18_extended_original_bf16_teacher_core.py",P18_CORE_SOURCE)
 src=json.loads((ROOT/"p18_long203_teacher_pilot_PRIVATE.json").read_text())
 if src.get("source_480_sha256")!=FROZEN_SHA or src.get("P18_frozen_protocol_sha256")!=PROTO_SHA:
  raise ValueError("Original 480 frozen protocol token bank drift")
 item=src["item"];ids=item["prompt_ids"];golds=item["gold_ids"]
 if(item["item_id"]!=ITEM or len(ids)!=203 or len(golds)!=1
    or item["prompt_token_sha256"]!=tok(ids)
    or item["gold_token_sha256"]!=tok(golds)):
  raise ValueError("P18 true 203prompt+1gold source identity mismatch")
 gold=golds[0]
 teacher=json.loads((R/"original_p18-mmlu-018_BF16_teacher_receipt.json").read_text())
 teacher_stage={"stage_sha256":STAGE_RAW,"route_sha256":ROUTE_RAW,"pids":[35481,35493]}
 if(teacher["original_serving_before"]!=teacher_stage or
    teacher["original_serving_after"]!=teacher_stage or
    teacher.get("source_real_decoder_layers")!=27 or
    teacher.get("source_real_tensors_exactly_once")!=5291 or
    teacher.get("prompt_tokens")!=203 or teacher.get("gold_tokens")!=1 or
    teacher.get("original_tensors_exactly_once") not in (None,5291) or
    teacher.get("real_BF16_logits_npy_sha256")!=BF16_F32_SHA or
    teacher.get("source_tensor_hash_sha256")!=TENSOR_SHA or
    teacher.get("P18_480_quality_certified") is not False):
  raise ValueError("Original full long teacher evidence 27L/5291T invalid")
 raw_teacher=np.load(R/"original_p18-mmlu-018_203BF16_logits.npy",allow_pickle=False)
 if raw_teacher.dtype!=np.float32 or raw_teacher.shape!=(1,102400):raise ValueError("Original BF16 logit span missing")
 bf_loss=-float(lprob(raw_teacher[0])[gold])
 if abs(bf_loss-float(teacher["gold_NLL_nats"]))>1e-8:raise ValueError("Original BF16 NLL re-compute mismatch")
 ab=json.loads((R/"p18-long203-native-AB-receipt.json").read_text())
 stage={"stage_sha256":STAGE_RAW,"route_sha256":ROUTE_RAW,"stage_pids":[35481,35493],"scratch_pids":[]}
 if(ab.get("status")!="P18_LONG203_ONE_ITEM_REAL_NATIVE_A_B_FULL_LOGITS_CAPTURE_PASS_NOT_480_GATE"
    or ab.get("pilot_source_sha256")!=INPUT_SHA or
    ab.get("source_480_token_bank_sha256")!=FROZEN_SHA or
    ab.get("frozen_protocol_sha256")!=PROTO_SHA or
    ab.get("binary_sha256")!=P18_LONG_BIN
    or ab.get("original_live_before")!=stage or ab.get("original_live_after")!=stage or
    ab.get("native_calls")!=2 or ab.get("automatic_live_promotion") is not False
    or ab.get("production_touched") is not False):
  raise ValueError("P18 genuine paired native full logit receipt invalid")
 lines=[s.strip() for s in (R/"p18-long203-native-AB-worker.log").read_text(errors="replace").splitlines()
        if "P18_NATIVE_GOLD_PREFIX_LOGITS" in s]
 if len(lines)!=2:raise ValueError("Not exactly two real native C full-vocab logit captures")
 logs={}
 for line in lines:
  m=LOG_RE.fullmatch(line)
  if m is None:raise ValueError("Native GPU C full vocab real output log malformed")
  rid,digest,plen,pos,gen,_,_,fhash=m.groups()
  if rid in logs:raise ValueError("Duplicate native request ID")
  logs[rid]={"token":digest,"plen":int(plen),"pos":int(pos),"next":int(gen),"hash":fhash}
 loss={}
 for row in ab.get("native_rows",[]):
  role=row["policy"]
  if role not in ACTUAL or role in loss:raise ValueError("Duplicated or unrecognized policy")
  file,hash_expected=ACTUAL[role]
  ensure(R/file,hash_expected)
  if (row.get("raw_original_full_logits")!=file or
      row.get("raw_full_logits_sha256")!=hash_expected or
      row.get("token_sha256")!=tok(ids) or
      row.get("gold_next_token_id")!=gold or
      row.get("float32_bytes")!=409600 or
      row.get("vocab")!=102400):
   raise ValueError("Actual GPU native source metadata invalid")
  v=np.fromfile(R/file,dtype="<f4")
  nll=-float(lprob(v)[gold])
  l=logs.get(file[:-4])
  if not l or l["token"]!=tok(ids) or l["plen"]!=203 or l["pos"]!=202 or l["hash"]!=hash_expected or l["next"]!=int(np.argmax(v)) or l["next"]!=row["generated_next_token_id"]:
   raise ValueError("C log source and F32 full vocab argmax differs")
  if abs(nll-float(row["gold_nll_nats"]))>1e-8:raise ValueError("P18 native NLL not independently reproducible")
  loss[role]=nll
 if set(loss)!={"A","B"}:raise ValueError("Not two actual policy losses")
 delta=loss["B"]-loss["A"]
 out={"schema":"beglin-hcp-p18-long203-real-BF16-plus-native-AB-independent-audit-v1",
      "status":"P18_LONG203_27L_BF16_AND_NATIVE_AB_FULL_102400_LOGITS_AUDIT_PASS_NOT_480_QUALITY",
      "item_id":ITEM,"input_tokens":203,"gold_token_positions":1,
      "source_frozen_bank_sha256":FROZEN_SHA,
      "source_original_teacher_tensor_values_sha256":TENSOR_SHA,
      "P18_original_27_layers_full_forwarded":True,"P18_original_5291_BF16_tensors_read":True,
      "teacher_BF16_gold_NLL_nats":bf_loss,
      "native_A_gold_NLL_nats":loss["A"],"native_B_gold_NLL_nats":loss["B"],
      "B_minus_A_nats_per_gold_token":delta,
      "teacher_F32_raw_logits_SHA256":BF16_F32_SHA,
      "native_A_F32_raw_logits_SHA256":ACTUAL["A"][1],
      "native_B_F32_raw_logits_SHA256":ACTUAL["B"][1],
      "original_C_source_change_audit":scope,
      "new_native_C_binary_SHA256":P18_LONG_BIN,
      "native_204_total_positions_verified":True,
      "frozen_480_source_count_complete":False,
      "quality_noninferiority_certified":False,
      "automatic_live_promotion":False,"production_touched":False}
 path=ROOT/"P18_LONG203_REAL_BF16_AND_AB_INDEPENDENT_AUDIT.json"
 if path.exists():raise FileExistsError("Refuse to overwrite real long-context source evidence")
 with path.open("x") as f:json.dump(out,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":out["status"],"BF16":round(bf_loss,7),"A":round(loss["A"],7),
   "B":round(loss["B"],7),"B-A":round(delta,7),"receipt_sha256":sha(path)},sort_keys=True))
if __name__=="__main__":verify()
