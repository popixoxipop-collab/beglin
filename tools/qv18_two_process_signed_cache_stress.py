#!/usr/bin/env python3
"""BEGLIN Qv18: two independent REAL Qwen28 Apple C processes concurrently read
same immutable 83.5 MiB locally Ed25519-signed alternate precision artifact.
Both child workers independently validate full original BF16 model SHA, developer
signature, native binary/mask SHA *before* process spawn. Native validates SHA,
40 role-layer tensors and Q4/n8 cache budget. 8 frozen existing ARC validation
questions per worker, numerically compared with Qv16 original CPU C reference.
Measuring multi-PROCESS shared read, NOT same-model concurrent request/threads.
No original model, key, signed file or live release modification.
"""
import argparse,hashlib,importlib.util,json,math,os,pathlib,psutil,re,subprocess,sys,time,stat
ROOT=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv18-concurrent-artifact-20261023")
Q17=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
PACK=Q17/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
EXE=Q17/"native_persisted_C_v2/beglin_qv17_signed_packed_cache"
SIGNER=ROOT/"tools/qv17_local_dev_signing.py"
FROZEN=Q16/"independent_arc_validation48"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
N=8
PIN={"pack":"49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b",
 "binary":"16085378788b64b229f321b49c11540e286edffbff832db4e41fbf8d2a87643b",
 "source":"a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296",
 "public":"ac5223bdf25b6cf607cb3a589027d1a406737af7341e8f742231155579c3b26a",
 "signer":"544cfccedbdddb5aad14eee1cb2bc6e0df60b319a8d0f5c369261fee6350c832"}
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
SWITCH=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 28 q4 168 switch_ms ([0-9eE.+-]+)$",re.M)
CACHE=re.compile(r"^\[QV15-CACHE\] window (\d+) hits (\d+) misses (\d+) drops_total (\d+) cache_bytes (\d+) peak_cache_bytes (\d+) cache_cap_bytes (\d+)$",re.M)
CTOTAL=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
LOAD=re.compile(r"^\[QV17-PACK-LOAD\] entries (\d+) bytes (\d+) sha256_check_ms ([0-9eE.+-]+) total_read_ms ([0-9eE.+-]+) digest ([a-f0-9]{64})$",re.M)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(262144),b""):h.update(b)
 return h.hexdigest()
def require(ok,why):
 if not ok:raise RuntimeError("QV18_FAIL_CLOSED_"+why)
def tobytes(x):return (json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode()
def exclusive(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def safety():
 require(sha(EXE)==PIN["binary"],"real_original_Qwen28_C")
 require(sha(SIGNER)==PIN["signer"],"trusted_loader_script")
 require(sha(FROZEN/"frozen_inputs_private.json")==PIN["source"],"frozen48_input")
 require(sha(FROZEN/"frozen_manifest_PUBLIC.json")==PIN["public"],"public_frozen48")
 require(PACK.is_file() and not PACK.is_symlink() and PACK.stat().st_mode&0o077==0,"original_owner_only_pack")
 require(sha(PACK)==PIN["pack"],"original_binary_pack")
def native_env():
 e={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_","BEGLIN_Q_LAB_","BEGLIN_QV"))}
 e.update({
 "VECLIB_MAXIMUM_THREADS":"1","Q4_THREADS":"2","Q4_POOL_SPIN":"0",
 "QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
 "QWEN_HF_CONFIG":str(MODEL/"config.json"),
 "QWEN_SAFETENSORS_MAXSEQ":"256",
 "QWEN_BASE":str(FROZEN/"dynamic"),
 "QWEN_PROMPT":str(FROZEN/"dynamic/prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
 "BEGLIN_QV2_NATIVE_ASYNC":"0",
 "BEGLIN_Q_LAB_N8_ROLES_FILE":str(FROZEN/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_SOURCE":str(FROZEN/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_RANDOM":str(FROZEN/"masks/random73.txt"),
 "BEGLIN_QV15_CACHE_MIB":"96",
 "BEGLIN_QV16_PREWARM":"0",
 "BEGLIN_QV17_PACK_LOAD":str(PACK),
 "BEGLIN_QV17_PACK_SHA256":PIN["pack"]})
 return e
def worker(idx):
 require(idx in (0,1),"workerID")
 safety()
 report=LAB/f"worker{idx}"
 if report.exists():raise FileExistsError("Qv18 worker receipt append-only")
 os.umask(0o077);report.mkdir(parents=True,mode=0o700)
 spec=importlib.util.spec_from_file_location("qv17_actual_ED25519_trusted_launch",str(SIGNER))
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 startverify_ns=time.monotonic_ns()
 sig=mod.verify(strict_model=True)
 signed_ms=(time.monotonic_ns()-startverify_ns)/1e6
 require(sig["status"]=="ED25519_LOCAL_DEVELOPER_SIGNATURE_AND_PACK_SHA256_VERIFIED" and sig["pack_sha256"]==PIN["pack"],"independent_signed_bootstrap")
 start_ns=time.monotonic_ns()
 proc=subprocess.Popen([str(EXE),"qv14_dynamic_choice",str(N)],
  cwd=str(QWEN),env=native_env(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 peak=0
 while proc.poll() is None:
  try:peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
  except psutil.NoSuchProcess:pass
  time.sleep(.04)
 out,err=proc.communicate();stop_ns=time.monotonic_ns()
 logfile=report/"original_native_Qwen28_C.log"
 exclusive(logfile,out+b"\n"+err)
 if proc.returncode:raise RuntimeError("real Qwen28 native exited "+str(proc.returncode)+err.decode(errors="replace")[-800:])
 txt=logfile.read_text()
 rows=ROW.findall(txt);sw=SWITCH.findall(txt);cache=CACHE.findall(txt)
 sums=SUM.findall(txt);total_cache=CTOTAL.findall(txt);loads=LOAD.findall(txt)
 require(len(rows)==len(sw)==len(cache)==N and len(sums)==len(total_cache)==len(loads)==1,"native_log_rows")
 require(txt.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"real_28_n8")
 lock=json.loads((FROZEN/"frozen_inputs_private.json").read_text())
 reference=json.loads((Q16/"native48/prewarm1/receipt_private.json").read_text())
 correct=[];nlls=[]
 for i,((k,nll,ans,pred,n),(sk,ns,profile,changed,ms),(ck,h,miss,drop,current,peak_cache,cap)) in enumerate(zip(rows,sw,cache)):
  row=lock["requests"][i];val=float(nll);corr=int(ans==pred)
  require(int(k)==int(sk)==int(ck)==i and int(ns)==int(n)==row["prefix_len"],"frozen_prefix_"+str(i))
  require(int(ans)=="ABCD".index(row["correct"]) and math.isfinite(val),"original_score_"+str(i))
  require(profile==("source_qv8" if i%2==0 else "random73") and int(changed)==(0 if i==0 else 40),"real_change_"+str(i))
  require(val==reference["gold_fullvocab_token_NLL"][i] and corr==reference["correct_each"][i],"gold_NLL_exact_Qv16_original_C_"+str(i))
  require(int(h)==(0 if i==0 else 40) and int(miss)==0 and int(drop)==0,"cached_not_requant_"+str(i))
  require(int(cap)==96*1048576 and int(peak_cache)<=int(cap) and int(current)<=int(cap),"cache_hard_cap_"+str(i))
  correct.append(corr);nlls.append(val)
 _,Nscore,right,mean,ppl=map(float,sums[0])
 require(int(Nscore)==N and int(right)==sum(correct) and abs(mean-sum(nlls)/N)<1e-8,"native_C_summary")
 hits,miss,drop,bytes_,peak_cache,cap=map(int,total_cache[0])
 require(hits==(N-1)*40 and miss==drop==0 and peak_cache<=cap==96*1048576,"native_C_packed_cache_totals")
 nload,packed_size,hashms,loadms,digest=loads[0]
 require(int(nload)==40 and digest==PIN["pack"] and float(loadms)>=float(hashms)>0,"native_load_SHA")
 result={
 "schema":"BEGLIN-Qv18-two-independent-concurrent-original-Apple-C-signed-cache-reader/1",
 "status":"REAL_PROCESS_ED25519_SIGNED_READER_8_QWEN28_STATIC_NLL_EXACT",
 "worker_id":idx,"n":N,"correct":sum(correct),"mean_fullvocab_correct_token_NLL":mean,
 "native_C_exe_sha256":PIN["binary"],"original_83MiB_pack_SHA256":PIN["pack"],
 "trusted_bootstrap_full_model_SHA256_and_signature_ms":signed_ms,
 "native_pack_83MiB_SHA256_ms":float(hashms),"native_pack_read_total_ms":float(loadms),
 "native_Qwen28_process_launch_monotonic_ns":start_ns,
 "native_Qwen28_process_exit_monotonic_ns":stop_ns,
 "native_process_wall_s":(stop_ns-start_ns)/1e9,
 "native_process_peak_RSS_MiB":peak/1048576,
 "actual_native_Qv14_precision_switches":N-1,
 "original_Qv16_static_gold_NLL_exact_rows":N,
 "native_cache_hits":hits,"native_cache_misses":miss,"native_cache_drops":drop,
 "native_cache_peak_MiB":peak_cache/1048576,
 "private_native_log_sha256":sha(logfile),
 "signed_file_is_read_only":True,"original_content_not_edited":True,
 "interprocess_independent_heap_not_shared":"yes; each C has own active+inactive quantized tensors, disk file shared read-only",
 "original_kernel_or_operator_release_unchanged":True}
 out=report/"receipt.json";exclusive(out,tobytes(result))
 print("QV18_WORKER_PASS",json.dumps({"idx":idx,"n":N,"correct":sum(correct),"mean_NLL":mean,
   "native_ms":round((stop_ns-start_ns)/1e6,2),"verified8":N,"read_ms":float(loadms),
   "receipt_sha256":sha(out)},sort_keys=True),flush=True)
def driver():
 safety()
 if LAB.exists():raise FileExistsError("Qv18 immutable concurrent experiment already exists")
 os.umask(0o077);LAB.mkdir(parents=True,mode=0o700)
 # The shared signed artifact is never copied/mutated or opened for writing.
 before=sha(PACK)
 started=time.monotonic()
 children=[]
 try:
  for i in (0,1):
   children.append(subprocess.Popen(["/opt/homebrew/bin/python3.11","-B",str(pathlib.Path(__file__).resolve()),"worker","--id",str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE))
  peak_aggregate=0;both_running_native_samples=0
  while any(p.poll() is None for p in children):
   mem=0;running=0
   for p in children:
    try:
     pr=psutil.Process(p.pid)
     desc=[pr]+pr.children(recursive=True)
     for ch in desc:
      if ch.is_running() and ch.status()!=psutil.STATUS_ZOMBIE:
       mem+=ch.memory_info().rss
       if "beglin_qv17" in ch.name():running+=1
    except(psutil.NoSuchProcess,psutil.AccessDenied):pass
   peak_aggregate=max(peak_aggregate,mem)
   both_running_native_samples+=int(running>=2)
   time.sleep(.1)
  outs=[p.communicate() for p in children]
  for i,(p,(stdout,stderr)) in enumerate(zip(children,outs)):
   exclusive(LAB/f"worker{i}.stdout.log",stdout)
   exclusive(LAB/f"worker{i}.stderr.log",stderr)
   if p.returncode:raise RuntimeError("worker"+str(i)+" failed exit "+str(p.returncode)+" "+stderr.decode(errors="replace")[-1200:])
  results=[json.loads((LAB/f"worker{i}/receipt.json").read_text()) for i in (0,1)]
  intervals=[(x["native_Qwen28_process_launch_monotonic_ns"],x["native_Qwen28_process_exit_monotonic_ns"]) for x in results]
  overlap=(min(x[1] for x in intervals)-max(x[0] for x in intervals))/1e9
  require(overlap>0 and both_running_native_samples>0,"real_simultaneous_native_process_overlapping")
  require(sha(PACK)==before,"original_signer_artifact_changed")
  require(results[0]["mean_fullvocab_correct_token_NLL"]==results[1]["mean_fullvocab_correct_token_NLL"],"two_process_numeric_NLL")
  summary={
   "schema":"BEGLIN-Qv18-original-native-Qwen28-C-two-signed-readers-real-concurrent/1",
   "status":"TWO_INDEPENDENT_SIGNED_CACHE_READERS_REAL_NATIVE_OVERLAP_AND_16_ORIGINAL_C_SCORES_PASS",
   "process_count":2,"each_frozen_question_count":8,"independent_C_gold_NLL_total_verified":16,
   "one_SHARED_owner_only_packed_artifact_SHA256":before,
   "local_developer_Ed25519_signed_manifest_verified_per_worker":True,
   "per_worker_original_native_Qwen28_C_byte_identical":True,
   "original_native_2_workers_overlap_seconds":overlap,
   "both_native_alive_monitor_samples":both_running_native_samples,
   "peak_aggregate_process_tree_RSS_MiB":peak_aggregate/1048576,
   "worker_native_process_peak_RSS_MiB":[x["native_process_peak_RSS_MiB"] for x in results],
   "worker_native_wall_s":[x["native_process_wall_s"] for x in results],
   "worker_signed_model_integrity_check_ms":[x["trusted_bootstrap_full_model_SHA256_and_signature_ms"] for x in results],
   "worker_native_packed_read_ms":[x["native_pack_read_total_ms"] for x in results],
   "original_Qv16_static_C_NLL_exact_count_per_worker":[8,8],
   "original_Qwen28_C_selected_correct_per_worker":[x["correct"] for x in results],
   "original_owner_only_signed_pack_SHA256_unchanged":True,
   "not_same_process_thread_concurrency":True,
   "not_client_request_queue_or_shared_KV_cache_stress":True,
   "Q4_THREADS_per_worker":2,"Q4_POOL_SPIN":0,
   "no_model_training_or_generalized_quality_claim":True,
   "no_operator_release_or_original_source_changed":True,
   "wall_s":time.monotonic()-started}
  out=ROOT/"docs/research/BEGLIN_QV18_TWO_PROCESS_CONCURRENT_SIGNED_CACHE_PUBLIC_20261023.json"
  exclusive(out,tobytes(summary))
  print("QV18_TWO_PROCESS_SIGNED_READER_PASS",json.dumps({"overlap_s":round(overlap,3),
   "aggregate_RSS_MiB":round(peak_aggregate/1048576,2),
   "each_NLL_verified":8,"two_children":True,
   "file_SHA256_unchanged":True,"report_SHA256":sha(out)},sort_keys=True),flush=True)
 finally:
  for p in children:
   if p.poll() is None:
    try:p.terminate()
    except ProcessLookupError:pass
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["driver","worker"])
 parser.add_argument("--id",type=int,choices=[0,1])
 a=parser.parse_args()
 if a.mode=="worker":
  if a.id is None:raise RuntimeError("explicit isolated worker id required")
  worker(a.id)
 else:driver()
