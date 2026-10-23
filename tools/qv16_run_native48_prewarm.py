#!/usr/bin/env python3
"""Real BEGLIN Qv16 48 fresh ARC validation CPU native: prewarm=0 vs 1.
Only scores model original C output and sealed input, not optimizing on labels.
Static original C parity checked separately once both static references exist.
"""
import argparse,hashlib,json,math,os,pathlib,psutil,re,struct,subprocess,time
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
COHORT=LAB/"independent_arc_validation48"
BINARY=LAB/"derived_startup_prewarm_C_v1/beglin_qv16_bounded_cache_prewarm"
RUNS=LAB/"native48"
QW=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
INPUT_SHA="a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296"
PUBLIC_SHA="ac5223bdf25b6cf607cb3a589027d1a406737af7341e8f742231155579c3b26a"
BINARY_SHA="3fc79f4c31af4d70ba36df87f447134a2f6c79b980748c3f4c2048440ee9a44d"
SRC_SHA="b05c2e8aac488f9accba44c61073a72f2103be6284b37aa5de0a500185995a9f"
MASK={"source_qv8":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
"random73":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
REG_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
REG_SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
REG_SW=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 28 q4 168 switch_ms ([0-9eE.+-]+)$",re.M)
REG_SW_TOTAL=re.compile(r"^\[QV14-DYNAMIC-TOTAL\] switches (\d+) tensors_repacked (\d+) switch_ms ([0-9eE.+-]+)$",re.M)
REG_CACHE=re.compile(r"^\[QV15-CACHE\] window (\d+) hits (\d+) misses (\d+) drops_total (\d+) cache_bytes (\d+) peak_cache_bytes (\d+) cache_cap_bytes (\d+)$",re.M)
REG_CACHE_TOTAL=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
REG_PRE=re.compile(r"^\[QV16-PREWARM\] prepared (\d+) startup_ms ([0-9eE.+-]+) inactive_cache_bytes (\d+) cap_bytes (\d+) startup_native_requantizations (\d+) startup_swap_hits (\d+)$",re.M)
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(262144),b""):h.update(chunk)
 return h.hexdigest()
def need(ok,what):
 if not ok:raise RuntimeError("QV16_NATIVE_FAIL_CLOSED_"+what)
def data(x):return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
def write_new(p,b):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def run(prewarm):
 need(prewarm in (0,1),"env_mode")
 need(digest(COHORT/"frozen_inputs_private.json")==INPUT_SHA,"Qv16_original48_private")
 need(digest(COHORT/"frozen_manifest_PUBLIC.json")==PUBLIC_SHA,"Qv16_original48_public")
 need(digest(BINARY)==BINARY_SHA,"original_C_prewarm_binary")
 src=LAB/"derived_startup_prewarm_C_v1/qv16_from_original_Qv15_paired.c"
 need(digest(src)==SRC_SHA,"Qv16_derived_source")
 for k,sha in MASK.items():need(digest(COHORT/"masks"/(k+".txt"))==sha,"source_mask_"+k)
 lock=json.loads((COHORT/"frozen_inputs_private.json").read_text())
 rows=lock["requests"];need(len(rows)==48 and lock["ordered_assigned_profiles"]==["source_qv8","random73"]*24,"48 alternating profiles")
 for i,row in enumerate(rows):
  raw=struct.pack("<"+"i"*len(row["prefix_ids"]),*row["prefix_ids"])
  expected=(COHORT/"dynamic/eval/ppl_work"/f"w{i}.i32").read_bytes()
  need(raw==expected,"Qv16_48_input_binary_"+str(i))
  c=(" ".join(str(row["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(row["correct"]))+"\n").encode()
  need((COHORT/"dynamic/eval/ppl_work"/f"w{i}.choices").read_bytes()==c,"label_"+str(i))
 dest=RUNS/f"prewarm{prewarm}"
 if dest.exists():raise FileExistsError("Qv16 append-only native48 already exists")
 os.umask(0o077);dest.mkdir(parents=True,mode=0o700)
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_QV","BEGLIN_Q_LAB_","BEGLIN_Q_"))}
 env.update({"VECLIB_MAXIMUM_THREADS":"1","QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
 "QWEN_HF_CONFIG":str(SNAP/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
 "QWEN_BASE":str(COHORT/"dynamic"),"QWEN_PROMPT":str(COHORT/"dynamic/prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1","BEGLIN_QV2_NATIVE_ASYNC":"0",
 "BEGLIN_Q_LAB_N8_ROLES_FILE":str(COHORT/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_SOURCE":str(COHORT/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_RANDOM":str(COHORT/"masks/random73.txt"),
 "BEGLIN_QV15_CACHE_MIB":"96","BEGLIN_QV16_PREWARM":str(prewarm)})
 started=time.monotonic()
 proc=subprocess.Popen([str(BINARY),"qv14_dynamic_choice","48"],cwd=str(QW),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 peak=0
 while proc.poll() is None:
  try:peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
  except psutil.NoSuchProcess:pass
  time.sleep(.06)
 stdout,stderr=proc.communicate()
 wall=time.monotonic()-started
 full=stdout+b"\n"+stderr
 log=dest/"native_C_original48.log";write_new(log,full)
 if proc.returncode:raise RuntimeError("derived original C failed "+str(proc.returncode)+": "+stderr.decode(errors="replace")[-1500:])
 txt=full.decode(errors="strict")
 got=REG_ROW.findall(txt);sw=REG_SW.findall(txt);cache=REG_CACHE.findall(txt)
 tot=REG_SUM.findall(txt);tot_sw=REG_SW_TOTAL.findall(txt);tot_c=REG_CACHE_TOTAL.findall(txt);warm=REG_PRE.findall(txt)
 need(len(got)==len(sw)==len(cache)==48 and len(tot)==len(tot_sw)==len(tot_c)==1,"original_C_exact_log_rows")
 need(txt.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"actual_28_Qwen_layer_registered_roles")
 expected=[len(row["prefix_ids"]) for row in rows]
 losses=[];acc=[]
 for i,((k,v,y,p,n),(l,nn,profile,ch,ms),(z,h,miss,drops,bytes_,peakb,cap)) in enumerate(zip(got,sw,cache)):
  need(int(k)==int(l)==int(z)==i and int(n)==int(nn)==expected[i],"input_index_"+str(i))
  need(int(y)=="ABCD".index(rows[i]["correct"]),"answer_"+str(i))
  need(profile==lock["ordered_assigned_profiles"][i],"mask_"+str(i))
  need(int(ch)==(0 if i==0 else 40),"real_WT_count_"+str(i))
  need(int(cap)==96*1048576 and int(bytes_)<=int(cap) and int(peakb)<=int(cap),"inactive_cache_hardcap_"+str(i))
  val=float(v);need(math.isfinite(val) and val>=0,"NLL_finite_"+str(i))
  losses.append(val);acc.append(int(y==p))
  if prewarm:
   need(int(h)==(0 if i==0 else 40) and int(miss)==0,"no_request_cold_misses_"+str(i))
  else:
   need(int(h)==(40 if i>=2 else 0) and int(miss)==(40 if i==1 else 0),"original_no_prewarm_hits_"+str(i))
 nlltot,n,total_correct,nllmean,ppl=map(float,tot[0])
 need(n==48 and total_correct==sum(acc) and abs(sum(losses)-nlltot)<1e-7 and abs(nllmean-nlltot/48)<1e-8,"original_total_scoring")
 switches,weight_events,switch_ms=map(float,tot_sw[0])
 need(int(switches)==47 and int(weight_events)==1880,"real_47_transitions")
 H,M,D,bytes_,peakb,cap=map(int,tot_c[0])
 need(H+M==1880 and D==0 and peakb<=96*1048576,"Cache_bound_total")
 need(H==(1880 if prewarm else 1840) and M==(0 if prewarm else 40),"total_cache_hits")
 if prewarm:
  need(len(warm)==1 and int(warm[0][0])==40 and int(warm[0][4])==40 and int(warm[0][5])==40,"real_native_startup_only_40")
 else:need(not warm,"nonprewarm_but_prewarm_detected")
 startup_ms=float(warm[0][1]) if warm else 0.
 need(startup_ms>=0,"prestartup_finite")
 rec={"schema":"BEGLIN-Qv16-original-28layer-C-native48-startup-prewarm-experiment/1",
 "status":"REAL_NATIVE_48_INPUT_Q4_N8_SWITCH_AND_INACTIVE_CACHE_HARD_CAP_PASS",
 "prewarm_enabled":bool(prewarm),"n":48,
 "source_frozen48_sha256":INPUT_SHA,"public_frozen48_sha256":PUBLIC_SHA,
 "real_derived_C_sha256":SRC_SHA,"real_native_C_binary_sha256":BINARY_SHA,
 "native_log_sha256":digest(log),
 "original_engine_independently_static_numeric_parity_pending":True,
 "input_lengths":expected,
 "gold_fullvocab_token_NLL":losses,"correct_each":acc,
 "correct":sum(acc),"mean_gold_NLL":nllmean,
 "actual_profile_switch_count":int(switches),"real_changed_WT_tensor_events":int(weight_events),
 "cache_request_hits":H,"cache_request_misses":M,"inactive_cache_peak_bytes":peakb,
 "inactive_cache_cap_bytes":cap,"cache_drops":D,
 "explicit_startup_prepare_ms":startup_ms,
 "per_request_switch_ms":[float(s[-1]) for s in sw],
 "total_request_switch_ms":switch_ms,
 "first_request_profile_switch_ms":float(sw[1][-1]),
 "native_process_wall_s":wall,"native_peak_process_RSS_MiB":peak/1048576,
 "startup_cost_is_moved_not_eliminated":True,
 "not_quality_generalization":True,"no_live_release_change":True}
 f=dest/"receipt_private.json";write_new(f,data(rec))
 safe={k:v for k,v in rec.items() if k not in ("input_lengths","gold_fullvocab_token_NLL","correct_each","per_request_switch_ms")}
 safe["private_receipt_sha256"]=digest(f)
 safe["original_binary_log_sha256"]=digest(log)
 public=dest/"PUBLIC_AGGREGATE.json";write_new(public,data(safe))
 print("QV16_REAL_NATIVE48",json.dumps({"prewarm":prewarm,"n":48,
 "start_ms":round(startup_ms,3),"req_switch_total_ms":round(switch_ms,3),
 "cold_first_ms":round(float(sw[1][-1]),3),
 "hits":H,"misses":M,"peak_MiB":round(peakb/1048576,3),
 "native_wall_s":round(wall,3),"process_RSS_MiB":round(peak/1048576,2),
 "correct":sum(acc),"NLL":round(nllmean,8),"public_sha256":digest(public)},sort_keys=True),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--prewarm",type=int,choices=[0,1],required=True)
 args=p.parse_args();run(args.prewarm)
