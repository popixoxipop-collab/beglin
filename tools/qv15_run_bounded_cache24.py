#!/usr/bin/env python3
"""BEGLIN Qv15 measured original-C bounded-cache experiment on Qv14 sealed 24 inputs.
Does not change Qv14 inputs, masks, original C, model, kernel or Tailnet release.
Runs a single native process at exactly cap 0/32/64/96 MiB. Private logs on D50.
Independent correctness: all 24 NLLs must exactly match corresponding ORIGINAL
static source or random73 native C from Qv14.
"""
import argparse,hashlib,json,math,os,pathlib,psutil,re,subprocess,time
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work")
LAB=BASE/"beglin-qv15-bounded-cache-20261023"
Q14=BASE/"beglin-qv14-inprocess-20261023"
FROZEN=Q14/"replication_arc_validation24"
NATIVE=LAB/"cached_original_C_native_build/beglin_qv15_bounded_cache"
SRC=LAB/"cached_original_C_native_build/qv15_from_pinned_original_Qv14_v3.c"
RUN=LAB/"native_24_runs"
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
BIN_SHA="e27695b844559dde4c311f1e5b97e851e56d3f3aeb38836429f1510cbfe77fd6"
SRC_SHA="0b1a069384d042b1b1164eb6ca05499000f21219e5692773476021edfeaafd79"
PRIVATE_SHA="e836a4bb6ea86c43f5a9c28ecbbc235c4c9d7fd306dfc006511027e3bddf3dc6"
PUBLIC_SHA="6cdd0679a8b36512a4f13651a1229bd8ee91b0f979fdd8c193c04dff69ed79ba"
MASK_SHA={"source_qv8":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
"random73":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
P_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
P_TOTAL=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
P_SWITCH=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 28 q4 168 switch_ms ([0-9eE.+-]+)$",re.M)
P_SW_TOTAL=re.compile(r"^\[QV14-DYNAMIC-TOTAL\] switches (\d+) tensors_repacked (\d+) switch_ms ([0-9eE.+-]+)$",re.M)
P_CACHE=re.compile(r"^\[QV15-CACHE\] window (\d+) hits (\d+) misses (\d+) drops_total (\d+) cache_bytes (\d+) peak_cache_bytes (\d+) cache_cap_bytes (\d+)$",re.M)
P_CACHE_TOTAL=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(262144),b""):h.update(c)
 return h.hexdigest()
def require(condition,message):
 if not condition:raise RuntimeError("QV15_FAIL_CLOSED_"+message)
def read(p):return json.loads(p.read_text())
def encode(o):return (json.dumps(o,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
def save(p,raw):
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def validate():
 require(sha(NATIVE)==BIN_SHA,"binary")
 require(sha(SRC)==SRC_SHA,"pinned_derived_C")
 require(sha(FROZEN/"frozen_inputs_private.json")==PRIVATE_SHA,"Qv14_new_val24_input")
 require(sha(FROZEN/"frozen_manifest_PUBLIC.json")==PUBLIC_SHA,"Qv14_val24_frozen_pub")
 for name,digest in MASK_SHA.items():
  require(sha(FROZEN/"masks"/(name+".txt"))==digest,"mask_"+name)
 rows=read(FROZEN/"frozen_inputs_private.json")["exact_input_rows"]
 require(len(rows)==24 and all(rows[i]["prefix_len"]<85 if i%2==0 else rows[i]["prefix_len"]>=85 for i in range(24)),"source_only_mask_order")
 original={}
 for policy in ("source_qv8","random73"):
  original[policy]=read(Q14/"native_replication24_logs"/("static_source.json" if policy=="source_qv8" else "static_random.json"))
  require(original[policy]["n"]==24 and len(original[policy]["gold_full_vocab_NLL"])==24,"same_original_C_static_24")
  name="static_source" if policy=="source_qv8" else "static_random"
  log=Q14/"native_replication24_logs"/(name+".log")
  require(sha(log)==original[policy]["native_original_C_log_SHA256"],"original_C_reference_log_"+name)
 inp=FROZEN/"dynamic/eval/ppl_work"
 for i,r in enumerate(rows):
  import struct
  ids=struct.pack("<"+"i"*len(r["prefix_ids"]),*r["prefix_ids"])
  need=(FROZEN/"dynamic/eval/ppl_work"/f"w{i}.i32").read_bytes()
  require(ids==need,"eval_exact_input_"+str(i))
  lbl=" ".join(str(r["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(r["correct"]))+"\n"
  require((inp/f"w{i}.choices").read_text()==lbl,"eval_exact_choices_"+str(i))
 return rows,original
def run(mib):
 require(mib in [0,32,64,96],"cap")
 rows,original=validate()
 os.umask(0o077);RUN.mkdir(parents=True,exist_ok=True,mode=0o700)
 out=RUN/f"cap{mib}"
 if out.exists():raise FileExistsError("Qv15 existing original run: "+str(out))
 out.mkdir(mode=0o700)
 log=out/"native_C_run.log";receipt=out/"verified_receipt_private.json"
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV","BEGLIN_Q_"))}
 env.update({"VECLIB_MAXIMUM_THREADS":"1","QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
 "QWEN_HF_CONFIG":str(MODEL/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
 "QWEN_BASE":str(FROZEN/"dynamic"),"QWEN_PROMPT":str(FROZEN/"dynamic/prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0",
 "BEGLIN_Q_LAB_N8_ROLES_FILE":str(FROZEN/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_SOURCE":str(FROZEN/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_RANDOM":str(FROZEN/"masks/random73.txt"),
 "BEGLIN_QV15_CACHE_MIB":str(mib)})
 t=time.monotonic()
 p=subprocess.Popen([str(NATIVE),"qv14_dynamic_choice","24"],cwd=str(QWEN),
  env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 peak=0
 while p.poll() is None:
  try:peak=max(peak,psutil.Process(p.pid).memory_info().rss)
  except psutil.NoSuchProcess:pass
  time.sleep(.05)
 out_bytes,err=p.communicate()
 wall=time.monotonic()-t
 data=out_bytes+b"\n"+err
 save(log,data)
 if p.returncode:
  raise RuntimeError("original C derived native returncode "+str(p.returncode)+" log "+err.decode(errors="replace")[-1500:])
 txt=data.decode(errors="strict")
 records=P_ROW.findall(txt)
 switches=P_SWITCH.findall(txt)
 caches=P_CACHE.findall(txt)
 totals=P_TOTAL.findall(txt)
 total_switch=P_SW_TOTAL.findall(txt)
 total_cache=P_CACHE_TOTAL.findall(txt)
 require(len(records)==len(switches)==len(caches)==24,"24_C_logs")
 require(len(totals)==len(total_cache)==len(total_switch)==1,"one_native_summary")
 golds=[];correct=[]
 for i,(idx,loss,ans,pred,n) in enumerate(records):
  require(int(idx)==i and int(ans)=="ABCD".index(rows[i]["correct"]) and int(n)==rows[i]["prefix_len"],"input_source_match_"+str(i))
  loss=float(loss);golds.append(loss);correct.append(int(ans==pred))
  selected="source_qv8" if rows[i]["prefix_len"]<85 else "random73"
  static=original[selected]
  require(loss==static["gold_full_vocab_NLL"][i],"native_static_NLL_bit_exact_"+str(i))
  require(correct[i]==static["correct_each"][i],"native_static_class_"+str(i))
  k,kn,kpol,kchanged,ms=switches[i]
  require(int(k)==i and int(kn)==rows[i]["prefix_len"] and kpol==selected,"switch_route_"+str(i))
  require(int(kchanged)==(0 if i==0 else 40),"real_tensor_change_"+str(i))
  c_idx,hits,misses,drops,live,peakbytes,capbytes=caches[i]
  require(int(c_idx)==i and int(capbytes)==mib*1048576 and int(live)<=int(capbytes) and int(peakbytes)<=int(capbytes),"hard_cap_per_window_"+str(i))
  if mib==96:
   require(int(hits)==(0 if i<=1 else 40) and int(misses)==(40 if i==1 else 0),"warm_steady_cache_hit_"+str(i))
  if mib==0:
   require(int(hits)==0 and int(misses)==(0 if i==0 else 40),"no_cache_equivalence_"+str(i))
 total_s,n,won,mean,ppl=map(float,totals[0])
 require(int(n)==24 and won==sum(correct) and abs(total_s-sum(golds))<1e-7 and abs(mean-total_s/24)<1e-8,"accurate_total")
 sw,n_repack,swms=map(float,total_switch[0])
 require(int(sw)==23 and int(n_repack)==920 and abs(swms-sum(float(x[-1]) for x in switches))<.015,"switch_total")
 hits,miss,drops,live,peakbytes,capbytes=map(int,total_cache[0])
 require(hits+miss==920 and live<=capbytes and peakbytes<=capbytes,"sum_cache_counters")
 require(int(capbytes)==mib*1048576,"memory_limit")
 if mib==96:require(hits==880 and miss==40,"expected_full_cached_hits")
 if mib==0:require(hits==0 and miss==920 and peakbytes==0,"expected_zero_cache")
 receipt_data={"schema":"BEGLIN-Qv15-original-native-Qwen28-CPU-bounded-precision-inactive-cache/1",
 "state":"24_REAL_NATIVE_CPU_DYNAMIC_Q4_N8_SCORE_PARITY_AND_CACHE_BUDGET_PASS",
 "mib_cap":mib,"n":24,"runtime_binary_sha256":BIN_SHA,"derived_source_sha256":SRC_SHA,
 "Qv14_frozen_24_private_sha256":PRIVATE_SHA,"Qv14_public_sha256":PUBLIC_SHA,
 "native_original_log_SHA256":sha(log),
 "actual_switches":int(sw),"actual_tensor_precision_changes":int(n_repack),
 "quantizations_required_cache_miss":miss,"reused_cached_tensor_precisions":hits,
 "actual_cache_drops":drops,"resident_cache_bytes":live,"peak_inactive_cache_bytes":peakbytes,
 "hard_cache_cap_bytes":capbytes,
 "switch_ms":swms,"request_switch_ms":[float(x[-1]) for x in switches],
 "native_full_process_wall_seconds":wall,"native_peak_RSS_bytes":peak,
 "mean_full_vocab_gold_answer_nexttoken_NLL":mean,"correct":sum(correct),
 "static_reference_NLL_identical":True,"static_reference_choice_correctness_identical":True,
 "scores_private_aggregate_only":True,"source_only_not_accuracy_optimized":True,
 "gpu_and_production_untested":True,"original_model_and_operating_release_untouched":True}
 save(receipt,encode(receipt_data))
 public={k:v for k,v in receipt_data.items() if k not in ("request_switch_ms",)}
 public["original_private_receipt_sha256"]=sha(receipt)
 publicpath=out/"PUBLIC_AGGREGATE.json"
 save(publicpath,encode(public))
 print("QV15_VERIFIED_NATIVE_CACHE",json.dumps({
 "capMiB":mib,"cache_peak_MiB":round(peakbytes/1048576,4),
 "cache_hits":hits,"cache_misses":miss,
 "total_switch_ms":round(swms,3),"last_switch_ms":round(float(switches[-1][-1]),4),
 "wall_s":round(wall,3),"correct":sum(correct),"NLL":round(mean,8),
 "RSS_MiB":round(peak/1048576,2),"receipt_SHA256":sha(receipt),
 "public_SHA256":sha(publicpath)},sort_keys=True),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--cache-mib",type=int,choices=[0,32,64,96],required=True)
 args=parser.parse_args();run(args.cache_mib)
