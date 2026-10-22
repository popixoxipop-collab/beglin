#!/usr/bin/env python3
"""Actual original BEGLIN 28-layer Qwen native C RACE48 three-arm benchmark.

A) all Q4G64, B) source-only prior Fisher64 mixed, C) new top2+OTHER
grouped full-vocab mass trust-regularized 28/196 n8 policy.

Same frozen 48 full, untruncated RACE middle articles/question/options,
one question per article, identical A/B/C/D one-token next-label full-vocab
NLL. Original Qv5 audited native C programs remain unchanged. Test inputs
and policy SHA fixed before any evaluation; all raw outputs private 0600.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,re,struct,subprocess,time

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv10_topk_trust_20261023"
TEST=LAB/"RACE_middle_validation_48_locked.json"
PRE=ROOT/"configs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_PRELOCK_20261023.json"
POL=ROOT/"docs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_SOURCE_POLICY_20261023.json"
MASK=ROOT/"configs/research/BEGLIN_QV10_TOP2OTHER_TRUST_28N8_SOURCE_POLICY_20261023.txt"
OLD_MASK=ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt"
ORIGIN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
BIN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin")
SOURCE_MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
R=LAB/"native_RACE48_3arm_original_BEG_C"
PRE_SHA="4ef4d31a4b9d0be2e015ffaac84a8ee86feb86047a8a8aae4a50318144d5d207"
TEST_SHA="acdfd36477df4f440eeb9d03ad68342a34ce2435028eff1b8cef0de4c9f859de"
OLD_MASK_SHA="74fb949e45d13f63a26d412effc39377492ea71bda20ae3690f56fddf6303e08"
BIN_SHA={
"base":"408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c",
"mixed":"f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221"}
MODES=("q4","fisher64","qv10_top2other_trust")
BITMASK={"fisher64":(OLD_MASK,OLD_MASK_SHA),"qv10_top2other_trust":(MASK,None)}
ROWS=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
TOTAL=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def sha(b):return hashlib.sha256(b).hexdigest()
def secured(p):
 if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
  raise RuntimeError("Qv10 original RACE/native private evidence must be 0600 "+str(p))
 return p.read_bytes()
def output(p,b):
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def sources():
 if sha(PRE.read_bytes())!=PRE_SHA or sha(secured(TEST))!=TEST_SHA:
  raise RuntimeError("Qv10 native full model evaluation not pretrained source-lock")
 if not POL.is_file() or not MASK.is_file():raise RuntimeError("Qv10 source-only mask not frozen")
 pol=json.loads(POL.read_text())
 if pol["status"]!="SOURCE_ONLY_REAL_TOPK_OTHER_GROUPED_KL_28N8_MASK_FROZEN_BEFORE_DIAG_OR_RACE" or \
   not pol["new_RACE48_logits_and_labels_not_used_to_tune"] or pol["mask_sha256"]!=sha(MASK.read_bytes()):
  raise RuntimeError("Qv10 grouped precision policy used RACE output labels or modified after selection")
 for label in BIN_SHA:
  path=BIN/f"beglin_qv5_{label}"
  if not path.is_file() or sha(path.read_bytes())!=BIN_SHA[label]:
   raise RuntimeError("audited original native Qwen C binary changed "+label)
 for name,(path,expected) in BITMASK.items():
  dig=expected if expected else pol["mask_sha256"]
  if not path.is_file() or sha(path.read_bytes())!=dig:
   raise RuntimeError("source-only Qv6/Qv10 28n8 mask changed "+name)
  roles=[tuple(line.split()) for line in path.read_text().splitlines()]
  if len(roles)!=28 or len(set(roles))!=28:
   raise RuntimeError("must have 28 real unique 8-bit projections "+name)
 test=json.loads(secured(TEST))
 if test["number_questions"]!=48 or len(test["questions"])!=48 or len({x["passage_original_id"] for x in test["questions"]})!=48:
  raise RuntimeError("RACE48 original one unique article per question constraint failed")
 return test,pol
def prepare():
 test,pol=sources()
 if R.exists():raise FileExistsError("Qv10 native held evaluation already prepared")
 os.umask(0o077)
 R.mkdir(parents=True,mode=0o700)
 infos=[]
 for i,q in enumerate(test["questions"]):
  ids=q["prompt_token_ids"];answer=q["correct"];choices=q["option_ids"]
  if not 18<=len(ids)<=208 or set(choices)!=set("ABCD") or answer not in choices:
   raise RuntimeError("frozen RACE source tokenizer discrepancy")
  prefix=struct.pack("<"+"i"*len(ids),*ids)
  data=(" ".join(str(choices[c]) for c in "ABCD")+" "+str("ABCD".index(answer))+"\n").encode()
  infos.append({"index":i,"id_sha256":sha(q["id"].encode()),
      "article_sha256":q["article_sha"],"prefix_sha256":sha(prefix),
      "fouranswer_sha256":sha(data),"prefix_len":len(ids)})
  for mode in MODES:
   directory=R/mode
   (directory/"eval/ppl_work").mkdir(parents=True,mode=0o700,exist_ok=True)
   (directory/"results").mkdir(mode=0o700,exist_ok=True)
   output(directory/f"eval/ppl_work/w{i}.i32",prefix)
   output(directory/f"eval/ppl_work/w{i}.choices",data)
   if i==0:output(directory/"prompt_ids.i32",prefix)
 obj={"schema":"BEGLIN-Qv10-frozen-original-RACE48-native-3arm-inputs/1",
      "status":"ALL_3_NATIVE_ARM_INPUTS_AND_SOURCE_MASKS_FROZEN_BEFORE_FORWARD",
      "source_RACE48_SHA256":TEST_SHA,
      "GGN_and_topk_source_prelock_sha":PRE_SHA,
      "top2other_source_policy_sha":sha(POL.read_bytes()),
      "new_28n8_frozen_mask_SHA256":pol["mask_sha256"],
      "old_fisher64_mask_SHA256":OLD_MASK_SHA,
      "original_audited_C_binary_SHA256":BIN_SHA,
      "modes":MODES,"all_48_inputs":infos,
      "MMLU_HellaSwag_OpenBookQA_results_not_used_to_select_RACE":True,
      "RACE_source_full_articles_unchanged":True,"automatic_live_promotion":False}
 raw=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
 output(R/"RACE48_native_original_preexecution_lock.json",raw)
 print("QV10_NEW_RACE48_ALL3_NATIVE_ARMS_PREEXECUTION_FROZEN",
     json.dumps({"manifest_sha":sha(raw),"RACE48":48,"distinct_articles":48,
       "new_mask_SHA":pol["mask_sha256"]},sort_keys=True),flush=True)
def parse(log,mode,manifest):
 tiers=ROLE.findall(log)
 expected=(0,0,196) if mode=="q4" else (0,28,168)
 if len(tiers)!=1 or tuple(map(int,tiers[0]))!=expected:
  raise RuntimeError("original native 28-layer mixed precision wrong role count "+mode)
 cases=ROWS.findall(log)
 if len(cases)!=48:raise RuntimeError("native C Qwen 48 RACE prompts not all scored")
 nll=[];correct=[]
 for i,(seq,loss,ans,pred,length) in enumerate(cases):
  val=float(loss)
  if int(seq)!=i or not math.isfinite(val) or val<0 or \
     int(length)!=manifest["all_48_inputs"][i]["prefix_len"] or \
     not 0<=int(ans)<4 or not 0<=int(pred)<4:
   raise RuntimeError("native RACE48 score question incorrect")
  nll.append(val);correct.append(int(ans==pred))
 totals=TOTAL.findall(log)
 if len(totals)!=1:raise RuntimeError("missing RACE native full-vocab score total")
 summed,n,right,mean,ppl=map(float,totals[0])
 if n!=48 or right!=sum(correct) or abs(summed-sum(nll))>1e-7 or \
    abs(mean-summed/48)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
  raise RuntimeError("native Qwen original C RACE48 NLL aggregation inconsistent")
 return {"n":48,"correct":int(right),"accuracy":right/48,
   "correct_letter_full_vocab_mean_NLL":float(mean),
   "per_question_correct_letter_NLL":nll,
   "per_question_four_choice_right":correct}
def evaluate(mode):
 test,policy=sources()
 if mode not in MODES:raise ValueError("unsupported frozen three arm")
 manifest=json.loads(secured(R/"RACE48_native_original_preexecution_lock.json"))
 if manifest["status"]!="ALL_3_NATIVE_ARM_INPUTS_AND_SOURCE_MASKS_FROZEN_BEFORE_FORWARD" or \
   manifest["top2other_source_policy_sha"]!=sha(POL.read_bytes()) or manifest["source_RACE48_SHA256"]!=TEST_SHA:
  raise RuntimeError("policy/race question changed post freeze")
 directory=R/mode
 logfile=directory/"original_Qwen_C_RACE48_native_log.txt"
 if logfile.exists():raise FileExistsError("no replay overwrite original 48 C model")
 for i,meta in enumerate(manifest["all_48_inputs"]):
  if sha(secured(directory/f"eval/ppl_work/w{i}.i32"))!=meta["prefix_sha256"] or \
     sha(secured(directory/f"eval/ppl_work/w{i}.choices"))!=meta["fouranswer_sha256"]:
   raise RuntimeError("original full RACE passage or answer tokens drift")
 tag="base" if mode=="q4" else "mixed"
 exe=BIN/f"beglin_qv5_{tag}"
 env={k:v for k,v in os.environ.items() if not k.startswith((
   "QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV2_","BEGLIN_QV4_","BEGLIN_QV5_","BEGLIN_QV6_","BEGLIN_QV7_","BEGLIN_QV8_","BEGLIN_QV9_","BEGLIN_QV10_"))}
 env.update({"QWEN_SAFETENSORS":str(SOURCE_MODEL/"model.safetensors"),
   "QWEN_HF_CONFIG":str(SOURCE_MODEL/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
   "QWEN_BASE":str(directory.resolve()),"QWEN_PROMPT":str((directory/"prompt_ids.i32").resolve()),
   "BEGLIN_Q_LAB_PRECISION":"q4" if mode=="q4" else "mixed",
   "BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0",
   "VECLIB_MAXIMUM_THREADS":"1"})
 if mode!="q4":env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(BITMASK[mode][0].resolve())
 begun=time.monotonic()
 proc=subprocess.run([str(exe),"masked_choice_qv5","48"],cwd=str(ORIGIN),env=env,
   text=True,capture_output=True,timeout=1800)
 raw=(proc.stdout+"\n"+proc.stderr).encode()
 os.umask(0o077);output(logfile,raw)
 if proc.returncode:
  raise RuntimeError("original BEGLIN C fullmodel RACE48 failed "+mode+" "+str(proc.returncode)+" "+proc.stderr[-600:])
 score=parse(raw.decode(),mode,manifest)
 rec={"schema":"BEGLIN-Qv10-actual-original-Qwen-28layer-native-48-RACE-one-arm/1",
   "status":"ORIGINAL_APPLE_NATIVE_C_FULL_QWEN28_RACE48_COMPLETE",
   "mode":mode,"source_model_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
   "original_audited_C_binary_SHA":sha(exe.read_bytes()),
   "source_only_mask_SHA256":None if mode=="q4" else sha(BITMASK[mode][0].read_bytes()),
   "frozen_input_manifest_SHA256":sha(secured(R/"RACE48_native_original_preexecution_lock.json")),
   "real_original_48_window_C_log_SHA256":sha(raw),
   "measurements":score,
   "elapsed_wall_seconds":round(time.monotonic()-begun,2),
   "not_low_load_sustained_TPS_benchmark":True,"automatic_promotion":False}
 data=(json.dumps(rec,sort_keys=True,indent=2)+"\n").encode()
 output(directory/"original_RACE48_native_receipt.json",data)
 print("QV10_ACTUAL_QWEN_NATIVE_RACE48",mode,json.dumps({
    "correct":score["correct"],"NLL":score["correct_letter_full_vocab_mean_NLL"],
    "receipt_SHA256":sha(data),"wall_s":rec["elapsed_wall_seconds"]},sort_keys=True),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("prepare","eval"))
 parser.add_argument("--mode",choices=MODES)
 args=parser.parse_args()
 if args.stage=="prepare":prepare()
 elif args.mode:evaluate(args.mode)
 else:parser.error("eval needs --mode")
