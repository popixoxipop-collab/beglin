#!/usr/bin/env python3
"""P18 independent REAL raw full-vocab evidence coverage audit, pre-gate.

Recomputes exactly 4 BF16 original 27L teacher windows and 10 native A/B
gold-position raw F32 vectors, binding actual native worker witness SHA,
the already-pre-registered 480 source tokens, and original production identity.
This module NEVER promotes policy B or converts incomplete one-token probes to
task accuracy. Missing 476 BF16 questions / 477 paired A/B questions blocks
the preregistered P18 480-question gate.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,re,sys
import numpy as np

ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
E=ROOT/"native1024_real_evidence"
M=E/"P18_ARC_GSM"
L=E/"P18_LONG203"
sys.path.insert(0,str(ROOT))
import hcp_p18_independent_task_scorer as scorer
FROZEN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
P18_THREE="2d9bd085bcfb28871ad9d8b42f82bb9347b02e0fcb9897a6ab6baf25ce57bd7e"
P18_ONE="3182a319d3bc370aded568895be16fe91cc5bc16be47e887c8edcbfc81d8e055"
P18_LONG_RECEIPT="73091f999dafc59d53ea6a8646e2543951bca68750129d9bb7b120cb24c3d924"
P18_LONG_TEACHER_RAW="2cc6e216e6c734bc041c8897de1a126a5562988dd93d1022b960eb2484a6a783"
TENSOR_SHA="88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda"
GPU1_RECEIPT="d41531836ad8fd1fc38fb3ff798894d618365998d431ab3987140a753609eb04"
GPU1_LOG_SHA="09a19c990347f59ffd9b75790558cd05e0ac9ac4f869193a3f469927c331c631"
GPU_MULTI_RECEIPT="d84ccb4a63bb52fa6819f2a50fc3df0dcc1227154d8473c81f9616adb38d37c5"
GPU_MULTI_LOG_SHA="3a2d5b61dbb4144a4adddcfbd3120e7625f2c6a0eea593bccec4266112edc6ce"
WITNESS_RE=re.compile(r"^P18_NATIVE_GOLD_PREFIX_LOGITS rid=([0-9a-f]{32}) inputsha=([0-9a-f]{64}) plen=(\d+) pos=(\d+) next=(\d+) vocab=(102400) bytes=(409600) floatsha=([0-9a-f]{64})$")
TEACHER={
 "p18-mmlu-000":(E/"p18-mmlu-000.teacher_full_logits.npy","7025aed3a8c23fdc024f9a2f4e4a9c4944129f09d1bc3c32633c57c21a1249b6",E/"p18-mmlu-000.teacher_receipt.json","c7628194d2e601f64525b1c05e5c59be5848eb40c68420dbd5c935e6ec148c3b",E/"p18-mmlu-000.original_5291_tensor_hashes.json","gold_NLL_nats"),
 "p18-arc_challenge-000":(M/"original_p18-arc_challenge-000_teacher_logits.npy","ffb9d59f8b9739274498c2085d679ec5d2fb4137a4a7a482705f6761da83d490",M/"original_p18-arc_challenge-000_teacher_receipt.json","17a4fc2664e4e1d7e68dafa79769c2dd28cd34c82a42c541a7c18503d96827e7",M/"original_p18-arc_challenge-000_tensor_hashes.json","gold_NLL_nats"),
 "p18-gsm8k-000":(M/"original_p18-gsm8k-000_teacher_logits.npy","ba5423d1728802ea189af1ba0ef5ebceb64aa5c9f17c6dea96c924f1b5c7ead9",M/"original_p18-gsm8k-000_teacher_receipt.json","8b4c32c2225083e9d17ac8dc81c4461d04b82568ef60b3b083b59aac8bdd4a32",M/"original_p18-gsm8k-000_tensor_hashes.json","gold_NLL_nats"),
 "p18-mmlu-018":(L/"original_p18-mmlu-018_203BF16_logits.npy",P18_LONG_TEACHER_RAW,L/"original_p18-mmlu-018_BF16_teacher_receipt.json",P18_LONG_RECEIPT,L/"original_p18-mmlu-018_5291_tensors.json","gold_NLL_nats")
}
PROD_STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
PROD_ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"

def fail(s):raise ValueError("Independent raw evidence: "+s)
def checksha(p,h):
 actual=scorer.digest(p)
 if actual!=h:fail("source SHA mismatch "+p.name)
def logprobs(raw):
 v=np.asarray(raw,dtype=np.float64)
 if v.shape!=(102400,) or not np.isfinite(v).all():fail("wrong/nonfinite real original full-vocab logits")
 peak=float(v.max());norm=peak+math.log(float(np.exp(v-peak).sum()))
 return v-norm
def parse_actual_native_log(path,sha):
 checksha(path,sha)
 found={}
 for line in path.read_text(errors="replace").splitlines():
  if "P18_NATIVE_GOLD_PREFIX_LOGITS" not in line:continue
  m=WITNESS_RE.fullmatch(line.strip())
  if not m:fail("malformed original native log witness")
  rid,tokens,plen,pos,pred,_,_,h=m.groups()
  if rid in found:fail("duplicate original native request ID")
  found[rid]={"input_sha":tokens,"prompt_len":int(plen),"pos":int(pos),"generated":int(pred),"floatsha":h}
 return found
def check_before_after(doc):
 for fields in (("original_live_before","original_live_after"),("original_serving_before","original_serving_after")):
  if fields[0] in doc:
   a,b=(doc.get(z) for z in fields)
   if a!=b or a.get("stage_sha256")!=PROD_STAGE_SHA or a.get("route_sha256")!=PROD_ROUTE_SHA:
    fail("real native original production stage/route changed")
   if sorted(a.get("stage_pids",[]))!=[35481,35493] or a.get("scratch_pids")!=[]:
    fail("unexpected original runtime worker PIDs or leaked scratch process")
   return
 fail("no authentic before/after runtime identity")
def run():
 scorer.pin_authority()
 frozen=json.loads((ROOT/"p18_multishot_frozen_PRIVATE.json").read_text())
 source={x["item_id"]:x for x in frozen["items"]}
 if len(source)!=480 or sum(x["gold_token_count"] for x in source.values())!=673:fail("frozen 480 / 673 gold units changed")
 checksha(ROOT/"P18_THREE_SOURCE_REAL_BF16_A_B_INDEPENDENT_AUDIT.json",P18_THREE)
 checksha(ROOT/"P18_ONE_REAL_BF16_A_B_INDEPENDENT_LOGIT_AUDIT.json",P18_ONE)
 golden=json.loads((ROOT/"P18_THREE_SOURCE_REAL_BF16_A_B_INDEPENDENT_AUDIT.json").read_text())
 if golden.get("question_count")!=3 or golden.get("gold_token_positions_total")!=5 or golden.get("P18_noninferiority_certified") is not False:
  fail("source 3-question independent baseline audit not pinned")
 raw_teachers={}
 for item,(npy,npy_sha,receipt,receipt_sha,tensor,loss_key) in TEACHER.items():
  q=source[item]
  checksha(npy,npy_sha);checksha(receipt,receipt_sha);checksha(tensor,TENSOR_SHA)
  d=json.loads(receipt.read_text())
  if len(json.loads(tensor.read_text()))!=5291 or d.get("production_touched") is not False:
   fail("teacher real 5291 source tensor file unverified")
  if item=="p18-mmlu-000":
   layers=d.get("original_decoder_layers");tensors=d.get("original_model_source_tensors_exactly_once")
  elif item=="p18-mmlu-018":
   layers=d.get("source_real_decoder_layers");tensors=d.get("source_real_tensors_exactly_once")
  else:
   layers=d.get("original_decoder_layers");tensors=d.get("original_tensors_exactly_once")
  if layers!=27 or tensors!=5291:fail("incomplete original BF16 model source")
  if d.get("item_id")!=item:fail("BF16 teacher applied to wrong frozen question")
  if item=="p18-mmlu-018":
   if d.get("prompt_token_sha256")!=q["prompt_token_sha256"] or d.get("gold_token_sha256")!=q["gold_token_sha256"]:
    fail("P18 long teacher source prompt/answer not frozen")
  elif item!="p18-mmlu-000":
   if d.get("prompt_token_sha256")!=q["prompt_token_sha256"] or d.get("gold_token_sha256")!=q["gold_token_sha256"]:
    fail("P18 multisource teacher not frozen")
  arr=np.load(npy,allow_pickle=False)
  if arr.dtype!=np.float32 or arr.shape!=(q["gold_token_count"],102400):fail("BF16 actual raw NPY full window shape mismatch")
  gold=[]
  for i,tok in enumerate(q["gold_ids"]):
   z=logprobs(arr[i])
   nll=-float(z[tok])
   reported=d[loss_key]
   target=float(reported[i] if isinstance(reported,list) else reported)
   if abs(nll-target)>2e-8:fail("BF16 receipt gold NLL differs from raw NPY")
   gold.append(nll)
  raw_teachers[item]={"mean_nll":sum(gold)/len(gold),"position_count":len(gold),"raw_sha256":npy_sha}
 # GPU A/B captured question+token source lives in read-only XOX native logs already SHA-copied to D50.
 native_one=E/"P18_ONE_REAL_AB_RECEIPT.json"
 native_multi=M/"native_ARC_GSM_8AB_RECEIPT.json"
 checksha(native_one,GPU1_RECEIPT);checksha(native_multi,GPU_MULTI_RECEIPT)
 n1=json.loads(native_one.read_text());n2=json.loads(native_multi.read_text())
 if (n1.get("native_calls")!=2 or n2.get("native_policy_calls")!=8 or
     n1.get("production_touched") is not False or n2.get("production_touched") is not False):
  fail("missing original native policy A/B capture receipts")
 check_before_after(n1);check_before_after(n2)
 traces1=parse_actual_native_log(E/"p18_AB_capture_worker.log",GPU1_LOG_SHA)
 traces2=parse_actual_native_log(M/"native_ARC_GSM_8AB_worker.log",GPU_MULTI_LOG_SHA)
 if len(traces1)!=2 or len(traces2)!=8:fail("native GPU original C witness vector count wrong")
 audited={}
 for src,rows,traces,folder in [
     ("mmlu",n1["native_rows"],traces1,E),
     ("arc_gsm",n2["rows"],traces2,M/"raw_original")]:
  for row in rows:
   ident=row["item_id"]
   if ident not in source:fail("native row not in frozen external corpus")
   q=source[ident];pol=row.get("policy")
   if pol not in ("A","B"):fail("original A/B policy identity incorrect")
   step=row.get("step",0)
   if type(step)!=int or not 0<=step<len(q["gold_ids"]):fail("native gold position out of bounds")
   prefix=q["prompt_ids"]+q["gold_ids"][:step]
   target=q["gold_ids"][step]
   real_file=row.get("actual_409600_F32_logits") or row.get("raw_original_full_logits")
   real_sha=row.get("raw_float32_logits_sha256") or row.get("raw_full_logits_sha256")
   if not isinstance(real_file,str) or not re.fullmatch(r"[0-9a-f]{32}\.f32",real_file) or not isinstance(real_sha,str):
    fail("native source raw F32 filename/SHA not pinned")
   # Previously copied one-MMLU GPU logits to fixed file names rather than raw C filename.
   if src=="mmlu":fp=E/("p18_AB_"+pol+"_full.f32")
   else:fp=folder/real_file
   checksha(fp,real_sha)
   if fp.stat().st_size!=409600:fail("raw native float32 file not 102400 entries")
   logits=np.fromfile(fp,dtype="<f4");lp=logprobs(logits)
   nll=-float(lp[target]);pred=int(np.argmax(logits))
   witness=traces.get(real_file[:-4])
   if (witness is None or witness["input_sha"]!=scorer.shaids(prefix) or
       witness["prompt_len"]!=len(prefix) or witness["pos"]!=len(prefix)-1 or
       witness["generated"]!=pred or witness["floatsha"]!=real_sha):
    fail("real original native log and SHA-copied full float32 output disagree")
   original_nll=row["gold_nll_nats"]
   if abs(nll-original_nll)>2e-8:fail("raw native gold NLL differs from captured receipt")
   identkey=(ident,step,pol)
   if identkey in audited:fail("repeated native source gold position")
   audited[identkey]=nll
 actual_questions=sorted({x[0] for x in audited})
 for ident in actual_questions:
  q=source[ident]
  if any((ident,i,pol) not in audited for i in range(len(q["gold_ids"])) for pol in ("A","B")):
   fail("incomplete paired A/B gold prefix source counted as full question")
 if actual_questions!=["p18-arc_challenge-000","p18-gsm8k-000","p18-mmlu-000"]:
  fail("real original full vocab A/B source set not 3 pinned questions")
 for row in golden["samples"]:
  ident=row["item_id"];goldn=len(source[ident]["gold_ids"])
  for pol in ("A","B"):
   calc=sum(audited[(ident,i,pol)] for i in range(goldn))/goldn
   if abs(calc-row[f"{pol}_gold_mean_NLL"])>2e-8:fail("prior independent 3-source NLL differs from raw evidence")
  if abs(raw_teachers[ident]["mean_nll"]-row["BF16_original_mean_NLL"])>2e-8:
   fail("prior independent 3-source BF16 source NLL differs")
 gold_native=sum(source[x]["gold_token_count"] for x in actual_questions)
 gold_teacher=sum(z["position_count"] for z in raw_teachers.values())
 if gold_native!=5 or gold_teacher!=6 or len(audited)!=10:fail("real source coverage counts incorrect")
 nll_coverage={ident:{"dataset":source[ident]["dataset"],"raw_gold_positions":source[ident]["gold_token_count"]} for ident in actual_questions}
 result={
 "schema":"beglin-hcp-p18-raw-evidence-coverage-v1",
 "status":"P18_REAL_RAW_LOGITS_COVERAGE_VERIFIED_480_GATE_BLOCKED",
 "frozen_480_bank_sha256":FROZEN,
 "expected_question_count":480,"original_BF16_question_windows_verified":4,
 "original_BF16_gold_positions_verified":gold_teacher,
 "original_BF16_gold_positions_required":673,
 "native_paired_A_B_question_windows_verified":3,
 "native_paired_gold_positions_verified":gold_native,
 "native_paired_gold_positions_required":673,
 "native_full_vocab_float32_A_B_vectors_verified":len(audited),
 "native_full_vocab_float32_A_B_vectors_required":1346,
 "original_BF16_full_vocab_float32_gold_rows_verified":gold_teacher,
 "actual_original_native_GPU_witness_SHA_match":True,
 "source_question_counts_per_dataset":{"mmlu":1,"arc_challenge":1,"gsm8k":1},
 "native_full_pair_per_question":nll_coverage,
 "complete_greedy_generations_with_native_finish_reason_verified":0,
 "BF16_original_full_generations_with_native_finish_reason_verified":0,
 "fixed_work_3_trial_warm_latency_pairs_verified":0,
 "independent_full_480_task_accuracy_verified":False,
 "statistical_480_question_noninferiority_certified":False,
 "production_touched":False,"automatic_live_promotion":False}
 path=ROOT/"P18_REAL_RAW_LOGIT_EVIDENCE_COVERAGE.json"
 if path.exists():raise FileExistsError("Refuse audit receipt overwrite")
 with path.open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":result["status"],
 "BF16_teacher_question_coverage":"4/480",
 "paired_AB_question_coverage":"3/480",
 "actual_AB_raw_vectors_verified":len(audited),
 "all_gold_prefix_positions_P18":673,
 "receipt_sha256":scorer.digest(path)},sort_keys=True))
if __name__=="__main__":run()
