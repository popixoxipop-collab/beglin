#!/usr/bin/env python3
"""P18 independent three external benchmark source real-original-logit audit.

Cross-check three ORIGINAL 27-layer BF16 teacher full-vocab F32 arrays and
ten native A/B full-vocab F32 arrays from actual Metal GPU. No inference,
no benchmark holdout reuse, no quality gate bypass. Per-question mean NLL.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,re
import numpy as np

ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
E=ROOT/"native1024_real_evidence/P18_ARC_GSM"
FROZEN_SHA="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
SOURCE_SAMPLE_SHA="7ffc9bd15eba15ff853c0d6dfd4e16ef204e6992b340a18a7e15ea5b0eb12979"
FROZEN_PROTOCOL_SHA="77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c"
ORIG_MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
ORIG_TEACHER_SOURCE_SHA="40dd5550b7759ac2760a6c374e1ddf3af35e68af637ab8d2a4f75bca8bee01d3"
TENSOR_SOURCE_SHA="88ec605a2824d0b1cb7dd905986cd06e8a2c11b5255ccd0a9b7dc47d8dfd8cda"
AB_RECEIPT_SHA="d84ccb4a63bb52fa6819f2a50fc3df0dcc1227154d8473c81f9616adb38d37c5"
AB_LOG_SHA="3a2d5b61dbb4144a4adddcfbd3120e7625f2c6a0eea593bccec4266112edc6ce"
AB_DRIVER_SHA="6b0ea7e99d6ecd93560b35f9698d2da20a9b96568c74465bc0f9b43f7e699c7f"
TEACHER_EXPECTED={
 "p18-arc_challenge-000":("ffb9d59f8b9739274498c2085d679ec5d2fb4137a4a7a482705f6761da83d490",
                          "17a4fc2664e4e1d7e68dafa79769c2dd28cd34c82a42c541a7c18503d96827e7"),
 "p18-gsm8k-000":("ba5423d1728802ea189af1ba0ef5ebceb64aa5c9f17c6dea96c924f1b5c7ead9",
                    "8b4c32c2225083e9d17ac8dc81c4461d04b82568ef60b3b083b59aac8bdd4a32")
}
GOLD_RE=re.compile(r"^P18_NATIVE_GOLD_PREFIX_LOGITS rid=([a-f0-9]{32}) inputsha=([a-f0-9]{64}) plen=(\d+) pos=(\d+) next=(\d+) vocab=(102400) bytes=(409600) floatsha=([a-f0-9]{64})$")
STAGE={"stage_sha256":"8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9",
       "route_sha256":"c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47",
       "stage_pids":[35481,35493],"scratch_pids":[]}

def sha(p):
 p=pathlib.Path(p)
 if not p.is_file() or p.is_symlink():raise ValueError("Source missing or a symlink")
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def thash(ids):return hashlib.sha256(",".join(map(str,ids)).encode("ascii")).hexdigest()
def logprob(z):
 a=np.asarray(z,dtype=np.float64)
 if a.shape!=(102400,) or not np.isfinite(a).all():raise ValueError("Raw full original 102400 logits absent/nonfinite")
 m=float(a.max());k=m+math.log(float(np.exp(a-m).sum()))
 return a-k
def require_sha(p,expected):
 if sha(p)!=expected:raise ValueError("Evidence raw SHA mismatch: "+str(p))
 return p
def audit():
 require_sha(ROOT/"p18_multishot_frozen_PRIVATE.json",FROZEN_SHA)
 require_sha(ROOT/"p18_ARC_GSM_two_pilot_PRIVATE.json",SOURCE_SAMPLE_SHA)
 require_sha(E/"native_ARC_GSM_8AB_RECEIPT.json",AB_RECEIPT_SHA)
 require_sha(E/"native_ARC_GSM_8AB_worker.log",AB_LOG_SHA)
 require_sha(E/"native_ARC_GSM_8AB_driver.py",AB_DRIVER_SHA)
 frozen=json.loads((ROOT/"p18_multishot_frozen_PRIVATE.json").read_text())
 sampled=json.loads((ROOT/"p18_ARC_GSM_two_pilot_PRIVATE.json").read_text())
 if (sampled["original_full_480_sha256"]!=FROZEN_SHA
    or sampled["original_protocol_sha256"]!=FROZEN_PROTOCOL_SHA
    or len(sampled["items"])!=2):
  raise ValueError("Original ARC/GSM sample was not frozen")
 original={x["item_id"]:x for x in frozen["items"]}
 items={x["item_id"]:x for x in sampled["items"]}
 if set(items)!=set(TEACHER_EXPECTED):raise ValueError("Wrong set of two P18 external data items")
 for ident,q in items.items():
  orig=original[ident]
  for key in ("prompt_ids","gold_ids","record_sha256","prompt_token_sha256","gold_token_sha256"):
   if orig[key]!=q[key]:raise ValueError("Original benchmark prompt/gold preimage mismatch: "+ident+" "+key)
  if thash(q["prompt_ids"])!=q["prompt_token_sha256"] or thash(q["gold_ids"])!=q["gold_token_sha256"]:
   raise ValueError("Gold or prompt token hash identity mismatch")
 n=json.loads((E/"native_ARC_GSM_8AB_RECEIPT.json").read_text())
 if (n.get("status")!="P18_TWO_MORE_REAL_ORIGINAL_A_B_FULL_LOGIT_MULTISTEP_PILOTS_PASS_NOT_480_GATE"
     or n.get("P18_frozen_480_sha256")!=FROZEN_SHA
     or n.get("input_pilot_sha256")!=SOURCE_SAMPLE_SHA
     or n.get("input_frozen_protocol_sha256")!=FROZEN_PROTOCOL_SHA
     or n.get("native_source_items")!=2 or n.get("native_gold_prefixes")!=4
     or n.get("native_policy_calls")!=8 or n.get("production_touched") is not False
     or n.get("automatic_live_promotion") is not False
     or n.get("original_serving_before")!=STAGE or n.get("original_serving_after")!=STAGE):
  raise ValueError("Raw native 8 requests not a proper frozen P18 source")
 rows=n["rows"]
 if len(rows)!=8 or len({r["actual_409600_F32_logits"] for r in rows})!=8:
  raise ValueError("Original native full logits source duplicate/incomplete")
 traces={}
 logtext=(E/"native_ARC_GSM_8AB_worker.log").read_text(errors="replace")
 for line in logtext.splitlines():
  if "P18_NATIVE_GOLD_PREFIX_LOGITS" not in line:continue
  m=GOLD_RE.fullmatch(line.strip())
  if m is None:raise ValueError("Malformed actual C native GPU log witness")
  rid,tok,plen,pos,nxt,_,_,digest=m.groups()
  if rid in traces:raise ValueError("Repeated native request id")
  traces[rid]={"tok":tok,"plen":int(plen),"pos":int(pos),"pred":int(nxt),"sha":digest}
 if len(traces)!=8 or logtext.count("GPU_VALIDATION_V1 backend=mlx_metal arch=mla correction=off finite_logits=1")!=8:
  raise ValueError("Missing native original GPU 8 logits validation messages")
 teachers={};tasks=[]
 for ident in ("p18-arc_challenge-000","p18-gsm8k-000"):
  q=items[ident]
  expected_raw,expected_receipt=TEACHER_EXPECTED[ident]
  t=E/("original_"+ident+"_teacher_logits.npy")
  meta=E/("original_"+ident+"_teacher_receipt.json")
  source=E/("original_"+ident+"_tensor_hashes.json")
  require_sha(t,expected_raw);require_sha(meta,expected_receipt);require_sha(source,TENSOR_SOURCE_SHA)
  originals=json.loads(source.read_text())
  if len(originals)!=5291:raise ValueError("Original BF16 source 5291 tensor hashes incomplete")
  data=json.loads(meta.read_text())
  if (data.get("status")!="P18_ONE_REAL_FULL_BF16_TEACHER_MULTISOURCE_PILOT_NOT_480_GATE"
      or data.get("original_decoder_layers")!=27 or data.get("original_tensors_exactly_once")!=5291
      or data.get("source_original_checkpoint_index_sha256")!=ORIG_MODEL_SHA
      or data.get("original_teacher_source_sha256")!=ORIG_TEACHER_SOURCE_SHA
      or data.get("prompt_token_sha256")!=q["prompt_token_sha256"]
      or data.get("gold_token_sha256")!=q["gold_token_sha256"]
      or data.get("raw_logits_f32_sha256")!=expected_raw or data.get("production_touched") is not False):
   raise ValueError("Original complete 27L BF16 source proof mismatch")
  arr=np.load(t,allow_pickle=False)
  if arr.shape!=(len(q["gold_ids"]),102400) or arr.dtype!=np.float32:
   raise ValueError("Original BF16 full-vocab teacher gold window shape wrong")
  gold_losses=[]
  for step,g in enumerate(q["gold_ids"]):
   lp=logprob(arr[step]);loss=-float(lp[g])
   if abs(loss-float(data["gold_NLL_nats"][step]))>1e-8:
    raise ValueError("Independent original teacher NLL disagrees with native receipt")
   gold_losses.append(loss)
  teachers[ident]={"NLL":[float(x) for x in gold_losses],
                   "mean_NLL":sum(gold_losses)/len(gold_losses),"original_raw_sha256":expected_raw,
                   "original_5291_source_SHA256":TENSOR_SOURCE_SHA}
  per_policy={}
  for kind in ("A","B"):
   losses=[];argmaxes=[]
   for step,g in enumerate(q["gold_ids"]):
    pair=[r for r in rows if r["item_id"]==ident and r["step"]==step and r["policy"]==kind]
    if len(pair)!=1:raise ValueError("Actual A/B paired gold-logit request missing/duplicated")
    row=pair[0]
    full_prefix=q["prompt_ids"]+q["gold_ids"][:step]
    phash=thash(full_prefix)
    if (row.get("gold_token_id")!=g or row.get("prefix_token_sha256")!=phash
        or row.get("prefix_len")!=len(full_prefix)
        or row.get("vocab_size")!=102400 or
        not isinstance(row.get("raw_float32_logits_sha256"),str) or
        not re.fullmatch(r"[a-f0-9]{32}\.f32",row.get("actual_409600_F32_logits",""))):
     raise ValueError("Native source gold-prefix identity invalid")
    fp=E/"raw_original"/row["actual_409600_F32_logits"]
    require_sha(fp,row["raw_float32_logits_sha256"])
    if fp.stat().st_size!=409600:raise ValueError("Native full-vocab byte count wrong")
    z=np.fromfile(fp,dtype="<f4")
    lp=logprob(z)
    loss=-float(lp[g]);pred=int(np.argmax(z))
    if abs(loss-row["gold_nll_nats"])>1e-8 or pred!=row["generated_token"]:
     raise ValueError("P18 original native gold NLL or actual argmax disagrees")
    witness=traces.get(row["actual_409600_F32_logits"][:-4])
    if (witness is None or witness["sha"]!=row["raw_float32_logits_sha256"]
       or witness["tok"]!=phash or witness["plen"]!=len(full_prefix) or
       witness["pos"]!=len(full_prefix)-1 or witness["pred"]!=pred):
     raise ValueError("Original native GPU C witness differs from raw evidence")
    losses.append(loss);argmaxes.append(pred)
   per_policy[kind]={"losses":losses,"question_mean_NLL":sum(losses)/len(losses),
                     "gold_argmax_count":sum(int(a==b) for a,b in zip(argmaxes,q["gold_ids"]))}
  diff=per_policy["B"]["question_mean_NLL"]-per_policy["A"]["question_mean_NLL"]
  tasks.append({"item_id":ident,"dataset":q["dataset"],"gold_count":len(q["gold_ids"]),
      "BF16_original_mean_NLL":teachers[ident]["mean_NLL"],
      "A_gold_mean_NLL":per_policy["A"]["question_mean_NLL"],
      "B_gold_mean_NLL":per_policy["B"]["question_mean_NLL"],
      "B_minus_A_gold_mean_NLL":diff,
      "A_gold_argmax_count":per_policy["A"]["gold_argmax_count"],
      "B_gold_argmax_count":per_policy["B"]["gold_argmax_count"],
      "source_BF16_logits_SHA256":teachers[ident]["original_raw_sha256"],
      "A_gold_token_losses":per_policy["A"]["losses"],
      "B_gold_token_losses":per_policy["B"]["losses"]})
 # Reuse separately authenticated first real MMLU triple source, no new inference.
 first=ROOT/"P18_ONE_REAL_BF16_A_B_INDEPENDENT_LOGIT_AUDIT.json"
 require_sha(first,"3182a319d3bc370aded568895be16fe91cc5bc16be47e887c8edcbfc81d8e055")
 z=json.loads(first.read_text())
 if z.get("item_id")!="p18-mmlu-000" or z.get("quality_noninferiority_certified") is not False:
  raise ValueError("Earlier independent MMLU triple source is not one frozen question")
 tasks.insert(0,{"item_id":"p18-mmlu-000","dataset":"mmlu","gold_count":1,
   "BF16_original_mean_NLL":z["teacher_BF16_gold_NLL_nats"],
   "A_gold_mean_NLL":z["policy_A_gold_NLL_nats"],
   "B_gold_mean_NLL":z["policy_B_gold_NLL_nats"],
   "B_minus_A_gold_mean_NLL":z["policy_B_minus_A_gold_NLL_nats"],
   "source_BF16_logits_SHA256":z["teacher_raw_logits_sha256"]})
 avg=sum(t["B_minus_A_gold_mean_NLL"] for t in tasks)/len(tasks)
 result={
  "schema":"beglin-hcp-p18-3-external-source-one-question-each-real-original-27L-BF16-A-B-full-vocab-v1",
  "status":"P18_THREE_SOURCE_REAL_BF16_NATIVE_AB_GOLD_LOGITS_AUDITED_NOT_480_QUALITY",
  "question_count":3,"stratum_question_counts":{"mmlu":1,"arc_challenge":1,"gsm8k":1},
  "gold_token_positions_total":5,
  "BF16_real_full_model_forwards_total":3,
  "native_real_AB_full_vocab_logit_vectors_total":10,
  "source_frozen_480_bank_sha256":FROZEN_SHA,
  "original_AC_GSM_8_A_B_source_receipt_SHA256":AB_RECEIPT_SHA,
  "samples":tasks,
  "three_question_unweighted_B_minus_A_mean_NLL_nats_per_gold_token":avg,
  "single_per_source_not_representative":True,
  "not_480_question_stratified_bootstrap":True,
  "P18_noninferiority_certified":False,"production_touched":False,"automatic_live_promotion":False}
 output=ROOT/"P18_THREE_SOURCE_REAL_BF16_A_B_INDEPENDENT_AUDIT.json"
 if output.exists():raise FileExistsError("Refuse to overwrite source audit result")
 with output.open("x") as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
 print(json.dumps({"status":result["status"],"three_question_mean_B_minus_A":round(avg,7),
   "by_source":[{"dataset":x["dataset"],"gold":x["gold_count"],
    "BF16":round(x["BF16_original_mean_NLL"],7),
    "A":round(x["A_gold_mean_NLL"],7),
    "B":round(x["B_gold_mean_NLL"],7),
    "B_minus_A":round(x["B_minus_A_gold_mean_NLL"],7)} for x in tasks],
   "receipt_sha256":sha(output)},sort_keys=True))
if __name__=="__main__":audit()
