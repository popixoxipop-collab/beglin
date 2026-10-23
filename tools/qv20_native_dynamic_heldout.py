#!/usr/bin/env python3
"""Qv20 actual Apple C per-request source-only precision selection on untouched
24 ARC-validation or 24 OpenBookQA-validation questions. Frozen ridge policy
computed solely from ARC TRAIN must exist before this can score heldout.
Uses Qv19 model-life mutex and Qv17 original signed Q4/n8 alternate cache.
All per-question scores stay private; publish aggregates only after controls.
"""
import argparse,hashlib,json,math,os,pathlib,psutil,re,struct,subprocess,time
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024")
BUILD=ROOT/"native_prompt_router_Qwen28_Apple_C_v1"
EXE=BUILD/"beglin_qv20_2mask_source_only_ridge"
POLICY=ROOT/"policy/calibration_policy_private.json"
Q17=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023/owner_only_artifacts")
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
QW=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
PINS={"public":"2aae8540fef6bce9c15e4c71becd873f62d69da556056583a41bae810ec5da1a",
 "test_arc_validation":"b0a27e22f0c5769f49c3dc95d2200feef9f06747349abef55bd82c29d13e2119",
 "test_ob_validation":"2f80539f7eca065e8dd5effcc1c320e2fbcf74f7e6af0e1e2816a8fe72aeb204",
 "qv8":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
 "random73":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde",
 "pack":"49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b"}
R_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
R_LEASE=re.compile(r"^\[QV19-LEASE\] window (\d+) worker (\d+) epoch_enter (\d+) epoch_exit (\d+) profile (source_qv8|random73) changed (\d+) wait_ns (\d+) guarded_ns (\d+) snapshot_stable 1$",re.M)
R_SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
R_TOTAL=re.compile(r"^\[QV19-MODEL-LEASE-TOTAL\] requests (\d+) workers 2 worker0 (\d+) worker1 (\d+) max_active (\d+) epoch (\d+) poisoned (\d+) guarded_forward_serial 1$",re.M)
R_CACHE=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
def shaf(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(262144),b""):h.update(b)
 return h.hexdigest()
def need(ok,reason):
 if not ok:raise RuntimeError("BEGLIN_QV20_NATIVE_HELDOUT_FAIL_CLOSED_"+reason)
def dump(o):return (json.dumps(o,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
def save(p,b):
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def profile(ids,p):
 if not p["dynamic_policy_admitted_from_calibration"]:
  return int(p["static_fallback_index"])
 n=len(ids)
 f=[math.log(n/85.0),len(set(ids))/n,
    sum(x>=100000 for x in ids)/n,sum(ids)/(151936.0*n),
    sum(ids[i]==ids[i-1] for i in range(1,n))/(n-1)]
 b=p["coefficients_intercept_and_slopes"];mu=p["feature_means"];sig=p["feature_scales"]
 delta=b[0]+sum(b[i+1]*(f[i]-mu[i])/sig[i] for i in range(5))
 margin=p["profile_Qv8_if_predicted_delta_greater_than"]
 return 0 if delta>margin else 1 if delta<-margin else int(p["static_fallback_index"])
def main(cohort):
 need(cohort in ("test_arc_validation","test_ob_validation"),"allowed only sealed HELDOUT")
 need(shaf(ROOT/"Qv20_PREEXEC_PUBLIC.json")==PINS["public"],"preexecution_lock")
 need(shaf(ROOT/cohort/"frozen_inputs_private.json")==PINS[cohort],"untouched_heldout_cohort")
 need(POLICY.is_file() and (BUILD/"BUILD_RECEIPT.json").is_file(),"trained policy+build must exist")
 build=json.loads((BUILD/"BUILD_RECEIPT.json").read_text())
 need(build["source_calibration_policy_SHA256"]==shaf(POLICY),"native build frozen policy")
 need(build["Qv20_Apple_native_binary_SHA256"]==shaf(EXE),"Qv20 real Apple code SHA")
 pack=Q17/"alternate40_Qv8_random73_qng64_v1.pack"
 need(shaf(pack)==PINS["pack"],"signed_original_Qv17_cache_integrity")
 for name,h in (("static_source",PINS["qv8"]),("static_random",PINS["random73"])):
  need(shaf(ROOT/"masks"/(name+".txt"))==h,"role matched mask")
 p=json.loads(POLICY.read_text())
 rows=json.loads((ROOT/cohort/"frozen_inputs_private.json").read_text())["selected_rows"]
 need(len(rows)==24,"24 heldout")
 policy_profile=[]
 for i,r in enumerate(rows):
  ids=r["prefix_ids"];policy_profile.append(profile(ids,p))
  b=struct.pack("<"+"i"*len(ids),*ids)
  gold=(" ".join(str(r["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(r["correct"]))+"\n").encode()
  base=ROOT/cohort/"dynamic/eval/ppl_work"
  need((base/f"w{i}.i32").read_bytes()==b and (base/f"w{i}.choices").read_bytes()==gold,"pre-exec frozen actual tokens "+str(i))
 folder=ROOT/"native_original_C_dynamic"/cohort
 if folder.exists():raise FileExistsError("Qv20 heldout native run already exists, no rerun")
 os.umask(0o077);folder.mkdir(parents=True,mode=0o700)
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_","BEGLIN_Q_LAB_","BEGLIN_QV"))}
 env.update({
 "VECLIB_MAXIMUM_THREADS":"1","Q4_THREADS":"2","Q4_POOL_SPIN":"0",
 "QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),"QWEN_HF_CONFIG":str(MODEL/"config.json"),
 "QWEN_SAFETENSORS_MAXSEQ":"256",
 "QWEN_BASE":str(ROOT/cohort/"dynamic"),
 "QWEN_PROMPT":str(ROOT/cohort/"dynamic/prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0",
 "BEGLIN_Q_LAB_N8_ROLES_FILE":str(ROOT/"masks/static_source.txt"),
 "BEGLIN_QV14_MASK_SOURCE":str(ROOT/"masks/static_source.txt"),
 "BEGLIN_QV14_MASK_RANDOM":str(ROOT/"masks/static_random.txt"),
 "BEGLIN_QV15_CACHE_MIB":"96","BEGLIN_QV16_PREWARM":"0",
 "BEGLIN_QV17_PACK_LOAD":str(pack),"BEGLIN_QV17_PACK_SHA256":PINS["pack"],
 "BEGLIN_QV19_ENABLE_EXCLUSIVE_MODEL_REQUESTS":"1"})
 t=time.monotonic()
 proc=subprocess.Popen([str(EXE),"qv20_prompt_aware_choice","24"],
   cwd=str(QW),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 rss=0
 while proc.poll() is None:
  try:rss=max(rss,psutil.Process(proc.pid).memory_info().rss)
  except psutil.NoSuchProcess:pass
  time.sleep(.08)
 out,err=proc.communicate()
 logfile=folder/"actual_Qwen28_Qv20_C.log";save(logfile,out+b"\n"+err)
 if proc.returncode:raise RuntimeError("native Qv20 failed exit "+str(proc.returncode)+": "+err.decode(errors="replace")[-1700:])
 text=logfile.read_text()
 cases=R_ROW.findall(text);leases=R_LEASE.findall(text);total=R_TOTAL.findall(text);scores=R_SUM.findall(text);cache=R_CACHE.findall(text)
 need(len(cases)==len(leases)==24 and len(total)==len(scores)==len(cache)==1,"24 actual original C outputs")
 bycase={int(z[0]):z for z in cases};bylease={int(z[0]):z for z in leases}
 need(set(bycase)==set(bylease)==set(range(24)),"unique index")
 losses=[];correct=[];profiles=[];epochs=[];workers=[]
 for i,r in enumerate(rows):
  a=bycase[i];l=bylease[i]
  value=float(a[1]);right=int(a[2]==a[3])
  need(math.isfinite(value) and value>=0 and int(a[2])=="ABCD".index(r["correct"]) and int(a[4])==r["prefix_len"],"gold source case "+str(i))
  actual=0 if l[4]=="source_qv8" else 1
  need(actual==policy_profile[i],"C policy does not match Python calibrated prompt feature rule "+str(i))
  need(int(l[2])==int(l[3]) and int(l[2])>0,"request-held precision version changed")
  losses.append(value);correct.append(right);profiles.append(actual);epochs.append(int(l[2]));workers.append(int(l[1]))
 need(sorted(epochs)==list(range(1,25)) and set(workers)=={0,1},"actual 2 threads stable 24 epochs")
 n,w0,w1,maxin,epoch,poison=map(int,total[0])
 need(n==24 and w0>0 and w1>0 and maxin==1 and poison==0 and epoch==24,"Qv19 lease guarded model and KV")
 total_nll,nt,nt_correct,mean,ppl=map(float,scores[0])
 need(nt==24 and nt_correct==sum(correct) and abs(total_nll-sum(losses))<1e-7 and abs(mean-sum(losses)/24)<1e-8,"native C original scoring")
 hits,miss,drops,bytes_,cachepeak,cap=map(int,cache[0])
 need(miss==drops==0 and cachepeak<=cap==96*1048576,"signed inactive cache budget")
 private={"schema":"BEGLIN-Qv20-real-Qwen28-C-prompt-router-heldout/1",
  "status":"ACTUAL_ORIGINAL_QWEN28_APPLE_C_TWO_MASK_SOURCE_ROUTER_PASS",
  "cohort":cohort,"n":24,"original_gold_full_vocab_NLL":losses,
  "correct_each":correct,"selected_profile_per_case":profiles,
  "correct":sum(correct),"mean_NLL":mean,
  "original_C_binary_SHA256":shaf(EXE),
  "frozen_source_policy_SHA256":shaf(POLICY),"preexecution_input_SHA256":PINS[cohort],
  "native_C_log_SHA256":shaf(logfile),"workers":[w0,w1],"max_concurrent_forward":maxin,
  "all_profile_versions_stable":True,"poison":poison,"cache_hit":hits,"cache_miss":miss,
  "cache_peak_MiB":cachepeak/1048576,"native_peak_RSS_MiB":rss/1048576,
  "native_wall_s":time.monotonic()-t,
  "not_same_total_RAM_as_single_static_Qv8_due_to_extra_inactive_cache":True,
  "not_crossfamily_accuracy_generalization_proven":True}
 receipt=folder/"PRIVATE_PER_CASE.json";save(receipt,dump(private))
 public={k:v for k,v in private.items() if k not in ("original_gold_full_vocab_NLL","correct_each","selected_profile_per_case")}
 public["PRIVATE_PER_CASE_SHA256"]=shaf(receipt)
 pub=folder/"PUBLIC_AGGREGATE.json";save(pub,dump(public))
 print("QV20_REAL_QWEN28_PROMPT_ROUTER_HELDOUT",json.dumps({"cohort":cohort,"n":24,
  "correct":sum(correct),"mean_NLL":round(mean,8),"Qv8_cases":profiles.count(0),
  "random73_cases":profiles.count(1),"cache_misses":miss,"workers":[w0,w1],
  "public_receipt_SHA256":shaf(pub)},sort_keys=True),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--cohort",choices=["test_arc_validation","test_ob_validation"],required=True)
 main(p.parse_args().cohort)
