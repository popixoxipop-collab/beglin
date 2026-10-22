#!/usr/bin/env python3
"""Actual 28-layer BEGLIN C: one Q4 baseline, 3 singleton n8 and 3 pair n8.

All seven arms use one pinned original Qwen2.5 BF16 checkpoint, same 16
new disjoint RACE-middle test articles, original C Q4G64/n8 NEON kernels,
and original exact Qwen tokenizer A/B/C/D single-answer candidates.

Interaction pair means from true NLL:
I_ij(s)=L_ij(s)-L_i(s)-L_j(s)+L_Q4(s), with n8 bits 0/1/2.
Negative interaction is superadditive NLL improvement, positive is
subadditive. Does not imply official 28-n8 budget superiority.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,re,struct,subprocess,time
R=pathlib.Path(__file__).resolve().parents[2]
L=R/"lab_local/qv12_twotensor_20261023"
PRE=R/"configs/research/BEGLIN_QV12_TWO_N8_INTERACTION_PRELOCK_20261023.json"
SOURCE=L/"new_RACE_middle_test16_disjoint_Qv11_frozen.json"
BUILD=L/"native_2n8_original_C/build_original_native_two_n8_receipt.json"
RUN=L/"native_Qv12_7arm_new_RACEtest16"
Q5=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
Q11=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv11-trueintervention-20261023")
ENGINE=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRE_SHA="a565bdec341d675ce52a37132b06355760c2fef96b8e4329c17508a237e4374f"
SOURCE_SHA="0650f791937f0d924706db6462b7f5e370fc58c1819b1228e91402d4937ca63e"
BINARIES={
 "q4":(Q5/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base","408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"),
 "single":(Q11/"lab_local/qv11_single_original_C_20261023/counterfactual_C_1n8/beglin_qv11_singleton_1n8","a054e68789fd6bc20a4bd815da5108b63361b14227b1fc4ab9e68a3b7b9a2711"),
 "pair":(L/"native_2n8_original_C/beglin_qv12_two_tensor_n8","d40fd6d42e08b0bc9d5dd3c7f8e0a4f5bcaabb2d9de7310d89f033e02500e961")}
ARMS=("q4","down_proj_10","v_proj_20","k_proj_0","down_proj_10__v_proj_20","down_proj_10__k_proj_0","v_proj_20__k_proj_0")
ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
WINDOW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUMMARY=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
def sha(b):return hashlib.sha256(b).hexdigest()
def private(p):
 if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
  raise RuntimeError("source/native SHA-locked Qv12 evidence not private 0600 "+str(p))
 return p.read_bytes()
def write(p,b):
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def arm_targets(arm):
 if arm=="q4":return []
 tokens=arm.split("__")
 if len(tokens) not in (1,2):raise RuntimeError("not one or two preregistered target tuples")
 targets=[]
 for t in tokens:
  role,layer=t.rsplit("_",1)
  if role not in ("down_proj","v_proj","k_proj") or not layer.isdecimal():
   raise RuntimeError("not preregistered original projection role")
  targets.append((role,int(layer)))
 if len(set(targets))!=len(tokens):raise RuntimeError("duplicate 8bit target")
 return targets
def verify():
 if sha(PRE.read_bytes())!=PRE_SHA or sha(private(SOURCE))!=SOURCE_SHA:
  raise RuntimeError("Qv12 pair masks or new RACEtest16 SHA mutated")
 cfg=json.loads(PRE.read_text())
 items=json.loads(private(SOURCE))["full_frozen_questions"]
 if tuple(cfg["planned_arms"])!=ARMS or len(items)!=16 or \
    len({x["article_original_id"] for x in items})!=16:
  raise RuntimeError("not preregistered pairs/new article-test protocol")
 selected={(x["role"],x["layer"]) for x in cfg["three_preselected_intervention_tensors"]}
 for name in ARMS[1:]:
  targets=arm_targets(name)
  if set(targets)-selected or len(targets)!=(2 if "__" in name else 1):
   raise RuntimeError("new RACE label-dependent or unregistered precision arm "+name)
 obj=json.loads(private(BUILD))
 if obj["status"]!="ORIGINAL_QV5_28LAYER_2N8_194Q4_REAL_C_BINARY_COMPILED" or \
    obj["prelock_sha256"]!=PRE_SHA or \
    obj["new_Qv12_two_N8_binary_sha256"]!=BINARIES["pair"][1]:
  raise RuntimeError("new 2-n8 C source proof not SHA-authentic")
 for tag,(f,dig) in BINARIES.items():
  if not f.is_file() or sha(f.read_bytes())!=dig:raise RuntimeError("actual C Q4/1n8/2n8 executable changed "+tag)
 return cfg,items,obj
def prepare():
 cfg,items,b=verify()
 if RUN.exists():raise FileExistsError("Qv12 new RACE study already prepared; no re-freeze")
 os.umask(0o077);RUN.mkdir(mode=0o700,parents=True)
 cases=[]
 masks={}
 for mode in ARMS:
  folder=RUN/mode
  (folder/"eval/ppl_work").mkdir(mode=0o700,parents=True)
  (folder/"results").mkdir(mode=0o700)
  targets=arm_targets(mode)
  if targets:
   data="".join(f"{role} {layer}\n" for role,layer in targets).encode()
   write(folder/"native_n8_role_mask.txt",data)
   masks[mode]=sha(data)
 for i,q in enumerate(items):
  ids=q["prefix_ids"];opts=q["suffix_ids"]
  if not 18<=len(ids)<=208 or set(opts)!=set("ABCD"):raise RuntimeError("new test16 original single letter token shapes")
  binary=struct.pack("<"+"i"*len(ids),*ids)
  choice=(" ".join(str(opts[k]) for k in "ABCD")+" "+str("ABCD".index(q["correct"]))+"\n").encode()
  cases.append({"index":i,"new_original_article_SHA256":sha(q["article_original_id"].encode()),
      "prompt_sha256":sha(binary),"answer_sha256":sha(choice),
      "full_RACE_untruncated_original_source_prompt_SHA256":q["original_full_prompt_sha256"],
      "length":len(ids)})
  for mode in ARMS:
   folder=RUN/mode
   write(folder/f"eval/ppl_work/w{i}.i32",binary)
   write(folder/f"eval/ppl_work/w{i}.choices",choice)
   if i==0:write(folder/"prompt_ids.i32",binary)
 manifest={
   "schema":"BEGLIN-Qv12-original-28layer-real-N8-singleton-doubleton-RACEtest16-preexec/1",
   "status":"ALL_SEVEN_ORIGINAL_QWEN_NATIVE_PAIR_AND_SINGLE_ARMS_FROZEN_BEFORE_ANY_FORWARD",
   "prelock_sha256":PRE_SHA,"new_RACEtest16_source_sha256":SOURCE_SHA,
   "original_three_model_binaries_SHA256":{k:v[1] for k,v in BINARIES.items()},
   "original_pairs_and_singletons":list(ARMS),
   "true_native_role_masks_SHA256":masks,
   "input_cases_and_answers_SHA256":cases,
   "previous_Qv11_development_used_for_selection":True,
   "Qv12_test_NLL_or_accuracy_not_seen_by_target_selection":True,
   "production_auto_promotion":False}
 raw=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
 write(RUN/"seven_arm_original_inputs_and_masks_preexec_sha_manifest.json",raw)
 print("QV12_NEW_RACE16_TRUE_7ARM_ALL_C_INPUTS_FROZEN",json.dumps({
    "manifest_sha":sha(raw),"total_cases":len(cases),
    "q4_single_pair_counts":[0,1,2],"num_modes":len(ARMS)},sort_keys=True),flush=True)
def evaluate(mode):
 cfg,items,b=verify()
 if mode not in ARMS:raise ValueError("unsupported Qv12 mode")
 manifest=json.loads(private(RUN/"seven_arm_original_inputs_and_masks_preexec_sha_manifest.json"))
 if manifest["status"]!="ALL_SEVEN_ORIGINAL_QWEN_NATIVE_PAIR_AND_SINGLE_ARMS_FROZEN_BEFORE_ANY_FORWARD" or \
    manifest["prelock_sha256"]!=PRE_SHA or manifest["new_RACEtest16_source_sha256"]!=SOURCE_SHA:
  raise RuntimeError("post-result changed source masks or original evaluation data")
 folder=RUN/mode
 rawlog=folder/"actual_original_Qwen28_C_pair_single_RACEtest16.log"
 if rawlog.exists():raise FileExistsError("real C model evaluation already ran; cannot overwrite")
 for i,original in enumerate(manifest["input_cases_and_answers_SHA256"]):
  if sha(private(folder/f"eval/ppl_work/w{i}.i32"))!=original["prompt_sha256"] or \
     sha(private(folder/f"eval/ppl_work/w{i}.choices"))!=original["answer_sha256"]:
   raise RuntimeError("case prompt or correct next token modified post freeze")
 tag="q4" if mode=="q4" else ("pair" if "__" in mode else "single")
 exe=BINARIES[tag][0]
 env={k:v for k,v in os.environ.items() if not k.startswith((
   "QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV2_","BEGLIN_QV4_","BEGLIN_QV5_","BEGLIN_QV6_","BEGLIN_QV7_","BEGLIN_QV8_","BEGLIN_QV9_","BEGLIN_QV10_","BEGLIN_QV11_","BEGLIN_QV12_"))}
 env.update({"QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
  "QWEN_HF_CONFIG":str(MODEL/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
  "QWEN_BASE":str(folder.resolve()),"QWEN_PROMPT":str((folder/"prompt_ids.i32").resolve()),
  "BEGLIN_Q_LAB_PRECISION":"q4" if tag=="q4" else "mixed",
  "BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0",
  "VECLIB_MAXIMUM_THREADS":"1"})
 if tag!="q4":
  env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str((folder/"native_n8_role_mask.txt").resolve())
 start=time.monotonic()
 done=subprocess.run([str(exe),"masked_choice_qv5","16"],cwd=str(ENGINE),
    env=env,capture_output=True,text=True,timeout=1600)
 result=(done.stdout+"\n"+done.stderr).encode()
 os.umask(0o077);write(rawlog,result)
 if done.returncode!=0:
  raise RuntimeError("true original native Qwen pair intervention failed "+mode+" code "+str(done.returncode)+" "+done.stderr[-850:])
 tiers=ROLE.findall(result.decode())
 expected={"q4":(0,0,196),"single":(0,1,195),"pair":(0,2,194)}[tag]
 if len(tiers)!=1 or tuple(map(int,tiers[0]))!=expected:
  raise RuntimeError("native C quantized projections not exactly 0/1/2 n8 "+mode)
 rows=WINDOW.findall(result.decode())
 if len(rows)!=16:raise RuntimeError("not 16 original full-model Qwen forward windows")
 losses=[];right=[]
 for i,(n,nll,y,prefix,prompt) in enumerate([]):pass
 for i,(n,nll,y,guess,prompt) in enumerate(rows):
  q=items[i]
  if int(n)!=i or int(y)!="ABCD".index(q["correct"]) or \
      not 0<=int(guess)<4 or int(prompt)!=len(q["prefix_ids"]):
   raise RuntimeError("native one/double precision used invalid ground truth/next-token")
  val=float(nll)
  if not math.isfinite(val) or val<0:raise RuntimeError("original C 2-n8 NLL invalid")
  losses.append(val);right.append(int(y==guess))
 totals=SUMMARY.findall(result.decode())
 if len(totals)!=1:raise RuntimeError("native C outcome aggregate not present")
 summed,n,correct,mean,ppl=map(float,totals[0])
 if n!=16 or correct!=sum(right) or abs(summed-sum(losses))>1.e-7 or \
    abs(mean-summed/16)>1.e-7 or abs(math.log(ppl)-mean)>1.e-7:
  raise RuntimeError("real BEGLIN C next-token total arithmetic mismatch")
 record={
  "schema":"BEGLIN-Qv12-one-original-Qwen28-C-single-or-double-n8-16newRACE/1",
  "status":"ACTUAL_QWEN28_ORIGINAL_C_RACE16_PAIRED_PRECISION_EXECUTION_PASS",
  "mode":mode,"n":16,"correct_count":int(correct),
  "correct_label_mean_fullvocab_NLL":float(mean),
  "original_BF16_Qwen_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  "actual_n8_role_count":expected[1],"actual_q4_role_count":expected[2],
  "real_original_C_binary_sha256":sha(exe.read_bytes()),
  "specific_original_role_mask_sha256":None if tag=="q4" else sha((folder/"native_n8_role_mask.txt").read_bytes()),
  "original_RACE16_preexec_manifest_sha256":sha(private(RUN/"seven_arm_original_inputs_and_masks_preexec_sha_manifest.json")),
  "actual_original_C_process_log_sha256":sha(result),
  "per_case_correct_next_token_NLL":losses,
  "per_case_top4_accuracy_correct":right,
  "native_wall_s":round(time.monotonic()-start,2),
  "native_wall_clock_not_quiet_serving_latency":True,
  "operational_release_untouched":True}
 proof=(json.dumps(record,sort_keys=True,indent=2)+"\n").encode()
 write(folder/"original_C_RACE16_arm_receipt.json",proof)
 print("QV12_ACTUAL_QWEN28_ORIGINAL_NATIVE_TWO_TENSOR_INTERACTION_ARM_PASS",json.dumps({
   "mode":mode,"n8":expected[1],"correct":int(correct),"NLL":float(mean),
   "wall_s":record["native_wall_s"],"sha":sha(proof)},sort_keys=True),flush=True)
if __name__=="__main__":
 cli=argparse.ArgumentParser()
 cli.add_argument("action",choices=("prepare","eval"))
 cli.add_argument("--mode",choices=ARMS)
 a=cli.parse_args()
 if a.action=="prepare":prepare()
 elif a.mode:evaluate(a.mode)
 else:cli.error("eval action requires --mode")
