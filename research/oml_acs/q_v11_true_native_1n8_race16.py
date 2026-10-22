#!/usr/bin/env python3
"""True 1-tensor 8-bit vs allQ4 original BEGLIN 28layer Qwen native C RACEtest16.

7 authentic C runs with 16 identical SHA-frozen article-complete RACE questions:
1 original full Q4 + 6 isolated one-n8 tensor promotions. The six singleton
masks are fixed from prior original BF16 source Fisher high/low role extremes.
Single-n8 C binary is source-pinned only-cardinality modified Qv5 original
native mixed research C; kernel, weights, all other 195 projections unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,re,struct,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv11_single_original_C_20261023"
FREEZE=LAB/"RACE_middle_test16_distinct_article_locked.json"
PRE=ROOT/"configs/research/BEGLIN_QV11_REAL_ONE_N8_INTERVENTION_PRELOCK_20261023.json"
Q5=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
QBASE=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
OLD_BIN=Q5/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base"
NEW_BIN=LAB/"counterfactual_C_1n8/beglin_qv11_singleton_1n8"
BUILD=LAB/"counterfactual_C_1n8/build_evidence.json"
RUN=LAB/"native_7arms_RACEtest16"
PLAN=ROOT/"docs/research/BEGLIN_QV11_RACE_TEST16_PREMODEL_INPUTS_20261023.json"
PRE_SHA="2cd480dc4c72b5383014a1325d350e51a0b47b1335ec0013a6d069773b546a39"
FROZEN_SHA="0de242b608837bd29ed72db99f1e910626753e2b27ea6ada9a24bdf864c96499"
SOURCE_REF="/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv6-fisher-20261022/docs/research/BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json"
ARMS=("q4","down_proj_2","down_proj_10","k_proj_0","k_proj_1","v_proj_20","v_proj_27")
ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
CHOICE=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
TOTAL=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
def sha(b):return hashlib.sha256(b).hexdigest()
def secure(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
  raise RuntimeError("Qv11 real RACE test/private native score was not sealed "+str(p))
 return p.read_bytes()
def output(path,data):
 with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 path.chmod(0o600)
def verify():
 if sha(PRE.read_bytes())!=PRE_SHA or sha(secure(FREEZE))!=FROZEN_SHA:
  raise RuntimeError("Qv11 prereg or distinct articles data changed")
 cfg=json.loads(PRE.read_text())
 data=json.loads(secure(FREEZE))
 if len(data["real_question_source"])!=16 or len({d["article_original_id"] for d in data["real_question_source"]})!=16:
  raise RuntimeError("not 16 original independent RACE test article IDs")
 spec=json.loads(secure(BUILD))
 if spec["status"]!="QWEN28_SINGLE_1N8_195Q4_ORIGINAL_KERNEL_BINARY_BUILT" or \
    sha(NEW_BIN.read_bytes())!=spec["new_binary_sha256"] or \
    sha(OLD_BIN.read_bytes())!="408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c":
  raise RuntimeError("original Qv5 C full Q4 / patched-one-n8 binary not authentic")
 scored=json.loads(pathlib.Path(SOURCE_REF).read_text())["quality_upgrade_by_role_layer_per_size"]["128"]
 for role,x in cfg["role_extremes"].items():
  v=scored[role]
  if (max(range(28),key=lambda i:v[i])!=x["high_layer"] or
      min(range(28),key=lambda i:v[i])!=x["low_layer"]):
   raise RuntimeError("source independent Fisher rank selected against data")
 return cfg,data,spec
def prepare():
 cfg,d,receipt=verify()
 if RUN.exists():raise FileExistsError("Qv11 native source trial already prepared")
 os.umask(0o077)
 RUN.mkdir(parents=True,mode=0o700)
 frozen=[]
 for arm in ARMS:
  folder=RUN/arm
  (folder/"eval/ppl_work").mkdir(parents=True,mode=0o700,exist_ok=True)
  (folder/"results").mkdir(mode=0o700,exist_ok=True)
  if arm!="q4":
   role,layer=arm.rsplit("_",1)
   if role not in cfg["role_extremes"] or int(layer) not in cfg["role_extremes"][role].values():
    raise RuntimeError("native one-n8 role not preselected")
   mask=(role+" "+layer+"\n").encode()
   output(folder/"native_single_1n8_mask.txt",mask)
 for i,q in enumerate(d["real_question_source"]):
  ids=q["prefix_ids"];correct=q["correct"];opts=q["suffix_ids"]
  raw=struct.pack("<"+"i"*len(ids),*ids)
  choice=(" ".join(str(opts[x]) for x in "ABCD")+" "+str("ABCD".index(correct))+"\n").encode()
  frozen.append({"index":i,"article_id_sha":q["id_sha256"],"raw_prompt_sha":sha(raw),
    "answer_option_sha":sha(choice),"prefix_token_count":len(ids)})
  for arm in ARMS:
   folder=RUN/arm
   output(folder/f"eval/ppl_work/w{i}.i32",raw)
   output(folder/f"eval/ppl_work/w{i}.choices",choice)
   if i==0:output(folder/"prompt_ids.i32",raw)
 obj={"schema":"BEGLIN-Qv11-allQ4-and-6-actual-one-n8-original-C-frozen-test16/1",
   "status":"BEFORE_ANY_C_FORWARD_SEVEN_ARMS_WITH_TRUE_SINGLE_TENSOR_INTERVENTIONS_FROZEN",
   "source_prelock_sha256":PRE_SHA,"RACEtest16_original_private_sha256":FROZEN_SHA,
   "Qv5_original_Q4_binary_sha256":sha(OLD_BIN.read_bytes()),
   "Qv11_only_1N8_native_binary_sha256":sha(NEW_BIN.read_bytes()),
   "actual_source_role_mask_SHA256":{x:sha((RUN/x/"native_single_1n8_mask.txt").read_bytes()) for x in ARMS[1:]},
   "all_model_role_count":196,"single_upgrade_role_count":1,"base_q4_role_count":196,
   "arms":list(ARMS),"same_question_SHA_and_candidate_SHA_across_7arms":frozen,
   "no_source_runtime_GPU_or_live_promotion":True,
   "prelock_score_winners_and_losers_selected_without_RACE_outputs":True}
 raw=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
 output(RUN/"Qv11_true_one_tensor_native_all_7arms_prelock.json",raw)
 print("QV11_C_NATIVE_QWEN_ALL7_SINGLETON_ARMS_INPUTS_FROZEN",json.dumps({"receipt_sha":sha(raw),"n":16,"nArms":7},sort_keys=True),flush=True)
def evaluate(arm):
 cfg,d,b=verify()
 if arm not in ARMS:raise ValueError("unsupported native intervention")
 manifest=json.loads(secure(RUN/"Qv11_true_one_tensor_native_all_7arms_prelock.json"))
 if manifest["status"]!="BEFORE_ANY_C_FORWARD_SEVEN_ARMS_WITH_TRUE_SINGLE_TENSOR_INTERVENTIONS_FROZEN" or \
    manifest["RACEtest16_original_private_sha256"]!=FROZEN_SHA:
  raise RuntimeError("native source experiment not frozen")
 dir=RUN/arm;log=dir/"true_28layer_RACE16_original_C_process.log"
 if log.exists():raise FileExistsError("Qv11 genuine native log already exists")
 for i,meta in enumerate(manifest["same_question_SHA_and_candidate_SHA_across_7arms"]):
  if sha(secure(dir/f"eval/ppl_work/w{i}.i32"))!=meta["raw_prompt_sha"] or \
     sha(secure(dir/f"eval/ppl_work/w{i}.choices"))!=meta["answer_option_sha"]:
   raise RuntimeError("test case or original label changed in intervention")
 binary=OLD_BIN if arm=="q4" else NEW_BIN
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV2_","BEGLIN_QV4_","BEGLIN_QV5_","BEGLIN_QV6_","BEGLIN_QV7_","BEGLIN_QV8_","BEGLIN_QV9_","BEGLIN_QV10_","BEGLIN_QV11_"))}
 env.update({"VECLIB_MAXIMUM_THREADS":"1","QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
  "QWEN_HF_CONFIG":str(MODEL/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
  "QWEN_BASE":str(dir.resolve()),"QWEN_PROMPT":str((dir/"prompt_ids.i32").resolve()),
  "BEGLIN_Q_LAB_PRECISION":"q4" if arm=="q4" else "mixed",
  "BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0"})
 if arm!="q4":env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str((dir/"native_single_1n8_mask.txt").resolve())
 started=time.monotonic()
 done=subprocess.run([str(binary),"masked_choice_qv5","16"],cwd=str(QBASE),
   capture_output=True,text=True,timeout=1800,env=env)
 raw=(done.stdout+"\n"+done.stderr).encode()
 os.umask(0o077);output(log,raw)
 if done.returncode!=0:raise RuntimeError("native true singleton source C failed "+arm+str(done.returncode)+" "+done.stderr[-700:])
 roles=ROLE.findall(raw.decode())
 if len(roles)!=1 or tuple(map(int,roles[0]))!=((0,0,196) if arm=="q4" else (0,1,195)):
  raise RuntimeError("experimental arm did not execute exactly 1 n8+195 Q4")
 scores=CHOICE.findall(raw.decode())
 if len(scores)!=16:raise RuntimeError("not all sixteen original RACE prompts scored")
 per=[];hit=[]
 for index,(which,nll,y,pred,length) in enumerate(scores):
  x=d["real_question_source"][index]
  if int(which)!=index or int(length)!=len(x["prefix_ids"]) or \
     int(y)!="ABCD".index(x["correct"]) or not 0<=int(pred)<4:
   raise RuntimeError("original C scorer answers not same exact question")
  v=float(nll)
  if not math.isfinite(v) or v<0:raise RuntimeError("native original C NLL nonfinite")
  per.append(v);hit.append(int(y==pred))
 counts=TOTAL.findall(raw.decode())
 if len(counts)!=1:raise RuntimeError("native C next-token score total missing")
 total,n,right,mean,ppl=map(float,counts[0])
 if n!=16 or right!=sum(hit) or abs(total-sum(per))>1e-7 or abs(mean-sum(per)/16)>1e-7:
  raise RuntimeError("original BEGLIN Qv11 true counterfactual C mean discrepancy")
 result={"schema":"BEGLIN-Qv11-real-one-8bit-tensor-original-C-Qwen-RACEtest16/1",
    "status":"ACTUAL_SINGLE_TENSOR_N8_28LAYER_NATIVE_QWEN_16_RACE_CASES_PASS",
    "arm":arm,"real_original_Qwen_HF_revision":cfg["model_HF_revision"],
    "actual_n8_count":0 if arm=="q4" else 1,
    "actual_q4_count":196 if arm=="q4" else 195,
    "original_singleton_C_binary_sha256":sha(binary.read_bytes()),
    "source_only_singleton_mask_sha256":None if arm=="q4" else sha((dir/"native_single_1n8_mask.txt").read_bytes()),
    "original_7arm_before_forward_SHA256":sha(secure(RUN/"Qv11_true_one_tensor_native_all_7arms_prelock.json")),
    "original_native_CPU_process_log_SHA256":sha(raw),
    "actual_n":16,"correct":int(right),"correct_answer_fullvocab_mean_NLL":mean,
    "per_case_answer_fullvocab_NLL":per,"per_case_choice_right":hit,
    "elapsed_wall_seconds":round(time.monotonic()-started,2),
    "not_actual_live_tokens_per_second_speed_test":True,
    "no_original_model_or_immutable_release_change":True}
 receipt=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode()
 output(dir/"original_singleton_1n8_RACE16_native_result.json",receipt)
 print("QV11_ACTUAL_SINGLE_BIT_TENSOR_FULL_QWEN28_NATIVE_RESULT",json.dumps({
   "arm":arm,"correct":int(right),"NLL":mean,"binary_sha":sha(binary.read_bytes()),
   "private_receipt_sha":sha(receipt),"wall_s":result["elapsed_wall_seconds"]},sort_keys=True),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","eval"))
 p.add_argument("--arm",choices=ARMS)
 args=p.parse_args()
 if args.stage=="prepare":prepare()
 elif args.arm:evaluate(args.arm)
 else:p.error("eval requires arm")
