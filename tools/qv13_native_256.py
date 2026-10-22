#!/usr/bin/env python3
"""BEGLIN Qv13: frozen 256-item, role-and-packed-byte-matched original native C audit.
Never select a policy using held-out labels or outputs. All raw cases and native logs remain owner-only.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,os,pathlib,random,re,struct,subprocess,sys,time
from datasets import Dataset
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[1]
PRE=ROOT/"configs/research/BEGLIN_QV13_ACTUAL_NATIVE_EVAL_PRELOCK_20261023.json"
PRE_SHA="5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a"
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023")
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRIOR=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv12-pair-interaction-20261023/research/oml_acs")
CBASE=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin")
EXECUTABLES={"q4":(CBASE/"beglin_qv5_base","408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"),
"mixed":(CBASE/"beglin_qv5_mixed","f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221")}
SOURCE_MASK=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
SOURCE_SHA="9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922"
V2_MASK=ROOT/"configs/research/BEGLIN_Q_V2_SELECTED_28_N8_ROLE_LAYER_POLICY_20261022.txt"
V2_SHA="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
ARROW={
"arc128":(pathlib.Path("/Users/eoe/.cache/huggingface/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-test.arrow"),"e56914301ff03032cabc30b284e0e8d612cc56bac7b6a5ab4e7f35b31d6c3e32"),
"ob128":(pathlib.Path("/Users/eoe/.cache/huggingface/datasets/allenai___openbookqa/main/0.0.0/388097ea7776314e93a529163e0fea805b8a6454/openbookqa-test.arrow"),"586293629974e2ed0fb06f33ef5336a737097cac11e063faf554799caacf791e")
}
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
SEEDS=(73,137,211,389,577)
LOG_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
LOG_TOTAL=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
LOG_ROLES=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
MASK_NAMES=("source_qv8","random73","random137","random211","random389","random577","original_v2")
ARMS=("q4",)+MASK_NAMES
BATCHES=("pilot16","arc128","ob128")
def sha(x):return hashlib.sha256(x).hexdigest()
def file_sha(p):return sha(pathlib.Path(p).read_bytes())
def load_checked(p,h):
 if file_sha(p)!=h:raise RuntimeError("SHA_CONTRACT_VIOLATION "+str(p))
 return json.loads(p.read_text())
def private_write(p,data):
 if p.exists():raise FileExistsError("immutable evidence already exists "+str(p))
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 raw=data if isinstance(data,bytes) else (json.dumps(data,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
 with p.open("xb") as out:out.write(raw);out.flush();os.fsync(out.fileno())
 p.chmod(0o600)
 return sha(raw)
def mask_tuples(path):
 pairs=[]
 for row in path.read_text().splitlines():
  k=row.split()
  if len(k)!=2 or k[0] not in ROLES or not k[1].isdigit() or not 0<=int(k[1])<28:
   raise RuntimeError("BAD_MASK "+str(path))
  pairs.append((k[0],int(k[1])))
 if len(pairs)!=28 or len(set(pairs))!=28:raise RuntimeError("INVALID_N8_CARDINALITY")
 return pairs
def source_lock():
 cfg=load_checked(PRE,PRE_SHA)
 if cfg["status"]!="PRE_EVAL_FROZEN_SOURCE_MASKS_AND_DATASET_SELECTION_RULES":raise RuntimeError("prelock status")
 if file_sha(SOURCE_MASK)!=SOURCE_SHA or file_sha(V2_MASK)!=V2_SHA:
  raise RuntimeError("previous source-only mask drift")
 for p,h in EXECUTABLES.values():
  if file_sha(p)!=h:raise RuntimeError("original C executable drift")
 for p,h in ARROW.values():
  if file_sha(p)!=h:raise RuntimeError("cached original arrow drift")
 if file_sha(SNAP/"tokenizer.json")!="a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf":
  raise RuntimeError("original tokenizer drift")
 sys.path.insert(0,str(PRIOR))
 import q_v2_async_activation_milp as v2
 _,_,_,_,mem=v2.verified_sources()
 q4bytes=sum(28*mem[r]["q4"] for r in ROLES)
 if q4bytes!=736985088:raise RuntimeError("original C packed Q4 ledger drift")
 delta={r:mem[r]["n8"]-mem[r]["q4"] for r in ROLES}
 return cfg,delta,q4bytes
def cost(mask,delta):return sum(delta[r] for r,_ in mask)
def create_masks(delta,q4):
 if (LAB/"qv13_lock_private.json").exists():raise FileExistsError("already frozen, do not select after evaluation")
 source=mask_tuples(SOURCE_MASK)
 base_cost=cost(source,delta)
 if q4+base_cost!=765493248:raise RuntimeError("preknown Qv8 packed ledger different")
 expected=collections.Counter(r for r,_ in source)
 masks={"source_qv8":source,"original_v2":mask_tuples(V2_MASK)}
 for seed in SEEDS:
  rng=random.Random(seed)
  masks["random"+str(seed)]=[(r,l) for r in ROLES for l in sorted(rng.sample(range(28),expected[r]))]
 if len(masks)!=7:raise RuntimeError("policy count")
 info={}
 for name in MASK_NAMES:
  pts=masks[name]
  data="".join(f"{r} {n}\n" for r,n in pts).encode()
  loc=LAB/"masks"/(name+".txt")
  dg=private_write(loc,data)
  bytes_total=q4+cost(pts,delta)
  if name!="original_v2" and (cost(pts,delta)!=base_cost or collections.Counter(r for r,_ in pts)!=expected):
   raise RuntimeError("not equal-byte or same-role random control")
  info[name]={"n8_count":28,"incremental_packed_bytes":cost(pts,delta),
     "projection_packed_bytes":bytes_total,"mask_sha256":dg,
     "role_counts":dict(collections.Counter(r for r,_ in pts))}
 return info
def input_record(dataset,r,tok):
 ch=r.get("choices")
 if not isinstance(ch,dict):return None
 labels=ch["label"]; texts=ch["text"]
 if not isinstance(labels,list) or len(labels)!=4 or len(set(labels))!=4 or len(texts)!=4 or r["answerKey"] not in labels:
  return None
 stem=(r["question"] if dataset=="arc128" else r["question_stem"])
 if not isinstance(stem,str) or not stem.strip() or any(not isinstance(x,str) or not x.strip() for x in texts):
  return None
 text="Choose the correct answer.\nQuestion: "+stem.strip()+"\n"+"\n".join(f"{k}. {v.strip()}" for k,v in zip("ABCD",texts))+"\nAnswer using one letter A, B, C, or D."
 prefix="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
 ids=tok.encode(prefix,add_special_tokens=False).ids
 if not 18<=len(ids)<=208:return None
 suffixes={}
 for c in "ABCD":
  nextids=tok.encode(prefix+c,add_special_tokens=False).ids
  if len(nextids)!=len(ids)+1 or nextids[:len(ids)]!=ids:return None
  suffixes[c]=nextids[-1]
 if len(set(suffixes.values()))!=4:return None
 answer="ABCD"[labels.index(r["answerKey"])]
 if len(r["id"])>200:return None
 return {"id":dataset+":"+r["id"],"source_id":r["id"],"question_sha256":sha(stem.encode()),
   "full_prompt_sha256":sha(text.encode()),"prefix_ids":ids,"suffix_ids":suffixes,
   "correct":answer,"prefix_len":len(ids),"no_truncation":True}
def create_dataset():
 tok=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
 data={}
 for ds,(arrow,h) in ARROW.items():
  raw=Dataset.from_file(str(arrow))
  eligible=[]
  for r in raw:
   v=input_record(ds,r,tok)
   if v is not None:eligible.append(v)
  eligible.sort(key=lambda x:(sha((ds+"/"+x["source_id"]+"/"+x["question_sha256"]).encode()),x["source_id"]))
  if len(eligible)<128 or len({x["source_id"] for x in eligible})<128:
   raise RuntimeError("insufficient unique valid dataset")
  selected=[];ids=set()
  for x in eligible:
   if x["source_id"] in ids:continue
   selected.append(x);ids.add(x["source_id"])
   if len(selected)==128:break
  payload={"schema":"BEGLIN-Qv13-original-256-tokenized-questions-private/1",
    "dataset":ds,"original_arrow_sha256":h,"all_input_options_untruncated":True,
    "selected_count":len(selected),"candidate_count":len(eligible),
    "selection":"sha256 sorted BEFORE any native evaluation","items":selected}
  dg=private_write(LAB/"data"/(ds+".json"),payload)
  data[ds]={"sha256":dg,"selected_count":128,"eligible_count":len(eligible),
   "id_order_sha256":sha(json.dumps([q["source_id"] for q in selected]).encode()),
   "prompt_tokens_min_max":[min(q["prefix_len"] for q in selected),max(q["prefix_len"] for q in selected)]}
 return data
def batch_items(batch):
 data={}
 for name in ("arc128","ob128"):
  p=LAB/"data"/(name+".json")
  raw=json.loads(p.read_text())
  data[name]=raw["items"]
 if batch=="pilot16":return data["arc128"][:8]+data["ob128"][:8]
 return data[batch]
def prepare_inputs(lck):
 batches={}
 for batch in BATCHES:
  cases=batch_items(batch)
  locked=[]
  for i,x in enumerate(cases):
   vals=x["prefix_ids"]
   bytes_=struct.pack("<"+("i"*len(vals)),*vals)
   lab=(" ".join(str(x["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(x["correct"]))+"\n").encode()
   locked.append({"item_id_sha256":sha(x["id"].encode()),"ids_sha256":sha(bytes_),
     "label_sha256":sha(lab),"prefix_len":len(vals),"dataset":x["id"].split(":")[0]})
   for arm in ARMS:
    dest=LAB/"native_runs"/batch/arm/"eval/ppl_work"
    if i==0:
     dest.mkdir(parents=True,exist_ok=True,mode=0o700)
     (dest.parent.parent/"results").mkdir(exist_ok=True,mode=0o700)
     private_write(dest.parent.parent/"prompt_ids.i32",bytes_)
    private_write(dest/("w"+str(i)+".i32"),bytes_)
    private_write(dest/("w"+str(i)+".choices"),lab)
  manifest={"count":len(cases),"original_items":locked,"source_data_shas":lck["datasets"]}
  dg=private_write(LAB/"native_runs"/batch/"input_manifest.json",manifest)
  batches[batch]={"count":len(cases),"manifest_sha256":dg}
 return batches
def freeze():
 os.umask(0o077)
 cfg,delta,q4=source_lock()
 LAB.mkdir(parents=True,exist_ok=True,mode=0o700)
 if (LAB/"qv13_lock_private.json").exists():raise FileExistsError("cannot reevaluate after prelock")
 masks=create_masks(delta,q4)
 datasets=create_dataset()
 lock={"schema":"BEGLIN-Qv13-native-Qwen28-256-eval-preexecution-lock/1",
    "status":"FROZEN_BEFORE_QV13_NATIVE_C_FORWARD",
    "prelock_sha256":PRE_SHA,"original_C_binary_sha256":{k:h for k,(p,h) in EXECUTABLES.items()},
    "source28_prior_Qv8_SHA":SOURCE_SHA,
    "dataset_groups":["arc128","ob128"],
    "datasets":datasets,"masks":masks,
    "original_q4_packed_bytes":q4,
    "source28_packed_bytes":765493248,
    "accuracy_and_endtask_NLL_not_seen":True,
    "holdout_pilot_fixed_first8_each_before_eval":True}
 batches=prepare_inputs(lock)
 lock["batches"]=batches
 dg=private_write(LAB/"qv13_lock_private.json",lock)
 public={"schema":"BEGLIN-Qv13-real-native-Qwen28-pre-eval-256-lock-public/1",
   "status":"SOURCE_POLICY_AND_256_ARC_OPENBOOKQA_SHA_LOCKED_BEFORE_INFERENCE",
   "prelock_sha256":PRE_SHA,
   "private_qv13_full_lock_sha256":dg,
   "selected_domains":datasets,"native_input_manifests":batches,
   "packed_projection_reference_bytes":q4,
   "fixed_source28_packed_bytes":765493248,
   "same_byte_random_control_seed_count":len(SEEDS),
   "frozen_masks_sha_and_bytes":masks,
   "Qv12_RACE_test_not_reused":True,
   "no_per_question_text_or_labels_public":True,
   "no_native_Qv13_inference_before_this_freeze":True,
   "not_operational_promotion":True}
 out=ROOT/"docs/research/BEGLIN_QV13_ORIGINAL_NATIVE_256_PREEXEC_LOCK_PUBLIC_20261023.json"
 if out.exists():raise FileExistsError("public lock already exists")
 raw=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
 out.write_bytes(raw)
 print("QV13_FROZEN",json.dumps({"private_lock_sha256":dg,"public_sha256":sha(raw),
   "source_byte_budget":765493248,"q4_byte_baseline":q4,
   "datasets":datasets,"batches":batches,"random_cost_match_verified":True},sort_keys=True),flush=True)
def validate(batch,arm):
 if batch not in BATCHES or arm not in ARMS:raise ValueError("unregistered native run mode")
 cfg,delta,q4=source_lock()
 lock_path=LAB/"qv13_lock_private.json"
 lock=json.loads(lock_path.read_text())
 if lock["status"]!="FROZEN_BEFORE_QV13_NATIVE_C_FORWARD" or lock["prelock_sha256"]!=PRE_SHA:
  raise RuntimeError("frozen evidence inconsistent")
 manifest_path=LAB/"native_runs"/batch/"input_manifest.json"
 manifest=load_checked(manifest_path,lock["batches"][batch]["manifest_sha256"])
 if manifest["count"]!=lock["batches"][batch]["count"]:raise RuntimeError("input count changed")
 n=manifest["count"]
 dest=LAB/"native_runs"/batch/arm
 for i,meta in enumerate(manifest["original_items"]):
  ids=dest/"eval/ppl_work"/("w"+str(i)+".i32")
  labels=dest/"eval/ppl_work"/("w"+str(i)+".choices")
  if file_sha(ids)!=meta["ids_sha256"] or file_sha(labels)!=meta["label_sha256"]:
   raise RuntimeError("held item input bytes changed")
 if arm!="q4":
  mask=LAB/"masks"/(arm+".txt")
  if file_sha(mask)!=lock["masks"][arm]["mask_sha256"]:raise RuntimeError("native mask drift")
  pts=mask_tuples(mask)
  if cost(pts,delta)!=lock["masks"][arm]["incremental_packed_bytes"]:raise RuntimeError("actual mask byte ledger altered")
 return n,dest,lock,manifest
def evaluate(batch,arm):
 import psutil
 os.umask(0o077)
 n,dest,lock,man=validate(batch,arm)
 result=dest/"native_qv13_original_C_receipt.json"
 logfile=dest/"native_qv13_original_C_run.log"
 if result.exists() or logfile.exists():raise FileExistsError("append-only Qv13 forward run forbids overwrite")
 tag="q4" if arm=="q4" else "mixed"
 exe,h=EXECUTABLES[tag]
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV2_","BEGLIN_QV4_","BEGLIN_QV5_","BEGLIN_QV6_","BEGLIN_QV7_","BEGLIN_QV8_","BEGLIN_QV9_"))}
 env.update({"VECLIB_MAXIMUM_THREADS":"1","QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
 "QWEN_HF_CONFIG":str(SNAP/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
 "QWEN_BASE":str(dest),"QWEN_PROMPT":str(dest/"prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":tag,"BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
 "BEGLIN_QV2_NATIVE_ASYNC":"0"})
 if arm!="q4":env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str((LAB/"masks"/(arm+".txt")).resolve())
 t0=time.monotonic()
 proc=subprocess.Popen([str(exe),"masked_choice_qv5",str(n)],cwd="/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021",env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 peak=0
 while proc.poll() is None:
  try:peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
  except psutil.Error:pass
  time.sleep(.15)
 out,err=proc.communicate()
 log=out+b"\n"+err
 dg=private_write(logfile,log)
 if proc.returncode:raise RuntimeError("original C eval failed "+str(proc.returncode)+": "+err.decode(errors="replace")[-1200:])
 text=log.decode(errors="strict")
 counts=LOG_ROLES.findall(text)
 if len(counts)!=1 or tuple(map(int,counts[0]))!=((0,0,196) if arm=="q4" else (0,28,168)):
  raise RuntimeError("real native 28-layer n8 and Q4 counts mismatch")
 rows=LOG_ROW.findall(text)
 if len(rows)!=n:raise RuntimeError("incomplete original fullvocab NLL rows")
 vals=[];right=[]
 for j,(index,nll,y,pred,tokens) in enumerate(rows):
  if int(index)!=j or int(tokens)!=man["original_items"][j]["prefix_len"] or not 0<=int(y)<=3 or not 0<=int(pred)<=3:
   raise RuntimeError("original C input/native correctness index mismatch")
  loss=float(nll)
  if not math.isfinite(loss) or loss<0:raise RuntimeError("invalid real NLL")
  vals.append(loss);right.append(int(y==pred))
 totals=LOG_TOTAL.findall(text)
 if len(totals)!=1:raise RuntimeError("source C final original NLL total absent")
 t,ct,correct,mean,ppl=map(float,totals[0])
 if int(ct)!=n or int(correct)!=sum(right) or abs(t-sum(vals))>1e-7 or abs(mean-sum(vals)/n)>1e-7:
  raise RuntimeError("original C aggregate/individual loss mismatch")
 receipt={"schema":"BEGLIN-Qv13-authentic-original-C-Qwen28-native-fullvocab-NLL/1",
 "status":"ORIGINAL_NATIVE_28L_REAL_Q4_OR_28N8_SCORING_PASS","arm":arm,"batch":batch,
 "n":n,"native_original_binary_sha256":h,
 "prelock_sha256":PRE_SHA,
 "preexecution_lock_sha256":file_sha(LAB/"qv13_lock_private.json"),
 "input_manifest_sha256":file_sha(LAB/"native_runs"/batch/"input_manifest.json"),
 "mask_sha256":None if arm=="q4" else lock["masks"][arm]["mask_sha256"],
 "native_log_sha256":dg,
 "observed_roles":{"n8":0 if arm=="q4" else 28,"q4":196 if arm=="q4" else 168},
 "total_fullvocab_gold_NLL":t,"mean_fullvocab_gold_NLL":mean,
 "correct_ABCD":int(correct),"article_correct":right,"article_fullvocab_gold_NLL":vals,
 "cpu_peak_rss_observed_bytes":peak,"elapsed_wall_sec":round(time.monotonic()-t0,3),
 "p50_p95_per_token_latency_not_measured":True,
 "same_untruncated_inputs_for_each_arm":True,"no_model_checkpoint_or_kernel_changes":True,
 "no_live_precision_promotion":True}
 evidence_sha=private_write(result,receipt)
 print("QV13_NATIVE_RUN",json.dumps({"arm":arm,"batch":batch,"correct":int(correct),"n":n,"mean_NLL":mean,
 "RSS_peak_MiB":round(peak/1048576,2),"wall_s":receipt["elapsed_wall_sec"],
 "receipt_sha256":evidence_sha},sort_keys=True),flush=True)
def aggregate(batch):
 if batch not in BATCHES:raise ValueError(batch)
 lock=json.loads((LAB/"qv13_lock_private.json").read_text())
 rows={}
 for arm in ARMS:
  receipt=LAB/"native_runs"/batch/arm/"native_qv13_original_C_receipt.json"
  if not receipt.exists():continue
  a=json.loads(receipt.read_text())
  if a["status"]!="ORIGINAL_NATIVE_28L_REAL_Q4_OR_28N8_SCORING_PASS":raise RuntimeError("native receipt not success")
  rows[arm]={"correct":a["correct_ABCD"],"n":a["n"],"mean_NLL":a["mean_fullvocab_gold_NLL"],
  "wall_s":a["elapsed_wall_sec"],"peak_RSS_MiB":round(a["cpu_peak_rss_observed_bytes"]/1048576,2),
  "receipt_sha256":file_sha(receipt)}
 out={"schema":"BEGLIN-Qv13-partial-original-C-Qwen28-public-aggregate/1",
 "status":"PARTIAL_ONLY_NO_256_POLICY_CLAIM" if len(rows)!=len(ARMS) or batch=="pilot16" else "FULL_BATCH_ALL_POLICIES_SCORED_NOT_FINAL_256",
 "batch":batch,"native_real_arms_completed":len(rows),"registered_arms":len(ARMS),
 "n_question_per_arm":lock["batches"][batch]["count"],
 "original_model_identity":lock["original_C_binary_sha256"],
 "source28_reference_packed_bytes":lock["source28_packed_bytes"],
 "comparisons_unblinded_only_descriptive":True,"do_not_use_this_to_reselect_or_retune":True,
 "arms":rows,"no_operational_promotion":True}
 print(json.dumps(out,sort_keys=True,indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("action",choices=("freeze","eval","aggregate","status"))
 p.add_argument("--batch",choices=BATCHES,default="pilot16")
 p.add_argument("--arm",choices=ARMS)
 x=p.parse_args()
 if x.action=="freeze":freeze()
 elif x.action=="eval":
  if not x.arm:p.error("--arm required")
  evaluate(x.batch,x.arm)
 elif x.action=="aggregate":aggregate(x.batch)
 else:
  pth=LAB/"qv13_lock_private.json"
  print(json.dumps({"prelock_sha256":file_sha(PRE),"frozen":pth.is_file(),
  "lock_sha256":file_sha(pth) if pth.is_file() else None,
  "completed":{b:[arm for arm in ARMS if (LAB/"native_runs"/b/arm/"native_qv13_original_C_receipt.json").exists()] for b in BATCHES}},sort_keys=True))
