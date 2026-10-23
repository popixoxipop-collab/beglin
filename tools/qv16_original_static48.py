#!/usr/bin/env python3
"""Run 24 separate input-length-driven Qv14 C native requests; native genuine in-process re-quantization.
Qv14 is engine mechanism pilot, not full quality/latency generalization.
"""
import argparse,hashlib,json,math,os,pathlib,re,subprocess,time
import psutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
COHORT=LAB/"independent_arc_validation48"
BUILD=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023/native_original_C_dynamic_repack_v3")
EXE={"dynamic":BUILD/"beglin_qv14_inprocess_dynamic",
"static_source":pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed"),
"static_random":pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed")}
BINSHA={"dynamic":"f9ad286158e98e1597ab2f2e97f22d1aa0ce5fb46e7b617ec66a0921a955f7b4",
"static_source":"f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221",
"static_random":"f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221"}
LOCKSHA="a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296"
PUBSHA="ac5223bdf25b6cf607cb3a589027d1a406737af7341e8f742231155579c3b26a"
MASKSHA={"source_qv8":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
"random73":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
RLOG=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
RTOTAL=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
RDYNAMIC=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 (\d+) q4 (\d+) switch_ms ([0-9eE.+-]+)$",re.M)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,what):
 if not ok:raise RuntimeError("qv14 contract fails "+what)
def store(p,data):
 with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def evaluate(mode):
 need(mode in EXE,"unknown native mode")
 need(sha(EXE[mode])==BINSHA[mode],"real Apple binary digest")
 need(sha(COHORT/"frozen_inputs_private.json")==LOCKSHA,"input freeze")
 need(sha(COHORT/"frozen_manifest_PUBLIC.json")==PUBSHA,"public freeze")
 lock=json.loads((COHORT/"frozen_inputs_private.json").read_text())
 rows=lock["requests"];need(len(rows)==48,"n=48")
 for k,sha_val in MASKSHA.items():need(sha(COHORT/"masks"/(k+".txt"))==sha_val,"mask sha "+k)
 src=COHORT/mode
 for i,row in enumerate(rows):
  data=(src/"eval/ppl_work"/f"w{i}.i32").read_bytes()
  import struct
  need(data==struct.pack("<"+"i"*len(row["prefix_ids"]),*row["prefix_ids"]),"real binary ids")
  labels=(src/"eval/ppl_work"/f"w{i}.choices").read_text()
  need(labels==" ".join(str(row["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(row["correct"]))+"\n","score labels")
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV","BEGLIN_Q_"))}
 env.update({"VECLIB_MAXIMUM_THREADS":"1",
 "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
 "QWEN_HF_CONFIG":str(SNAP/"config.json"),
 "QWEN_SAFETENSORS_MAXSEQ":"256",
 "QWEN_BASE":str(src),"QWEN_PROMPT":str(src/"prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":"mixed",
 "BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
 "BEGLIN_QV2_NATIVE_ASYNC":"0",
 "BEGLIN_Q_LAB_N8_ROLES_FILE":str(COHORT/"masks"/("random73.txt" if mode=="static_random" else "source_qv8.txt"))})
 if mode=="dynamic":
  env["BEGLIN_QV14_MASK_SOURCE"]=str(COHORT/"masks/source_qv8.txt")
  env["BEGLIN_QV14_MASK_RANDOM"]=str(COHORT/"masks/random73.txt")
 command=[str(EXE[mode]),"qv14_dynamic_choice" if mode=="dynamic" else "masked_choice_qv5","48"]
 d=LAB/"native_static48_logs";d.mkdir(exist_ok=True,mode=0o700)
 log=d/(mode+".log");rec=d/(mode+".json")
 if log.exists() or rec.exists():raise FileExistsError("Qv14 native mode already run, no rerun")
 t=time.monotonic()
 p=subprocess.Popen(command,cwd=str(QWEN),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 peak=0
 while p.poll() is None:
  try:peak=max(peak,psutil.Process(p.pid).memory_info().rss)
  except psutil.NoSuchProcess:pass
  time.sleep(.1)
 stdout,stderr=p.communicate()
 elapsed=time.monotonic()-t
 logbytes=stdout+b"\n"+stderr
 store(log,logbytes)
 if p.returncode!=0:raise RuntimeError("Qv14 native real process exit "+str(p.returncode)+" logs "+stderr.decode(errors="replace")[-1500:])
 txt=logbytes.decode()
 values=RLOG.findall(txt)
 need(len(values)==48,"native C original gold answer scores")
 losses=[];right=[]
 for i,(idx,nll,ans,pred,n) in enumerate(values):
  row=rows[i]
  need(int(idx)==i and int(n)==row["prefix_len"] and int(ans)=="ABCD".index(row["correct"]),"matched input token")
  v=float(nll);need(math.isfinite(v) and v>=0,"finite NLL")
  losses.append(v);right.append(int(ans==pred))
 totals=RTOTAL.findall(txt)
 need(len(totals)==1,"native total line")
 total,count,correct,mean,ppl=map(float,totals[0])
 need(int(count)==48 and int(correct)==sum(right) and abs(sum(losses)-total)<1e-7 and abs(mean-total/48)<1e-8,"scores match")
 native_roles=re.findall(r"projection_role_tensors: f32=0 n8=28 q4=168",txt)
 need(len(native_roles)==1,"real 28n8 native initial registration")
 switches=[]
 if mode=="dynamic":
  for index,n,policy,changes,c8,c4,ms in RDYNAMIC.findall(txt):
   switches.append({"index":int(index),"len":int(n),"profile":policy,
    "changed":int(changes),"n8":int(c8),"q4":int(c4),"switch_ms":float(ms)})
  need(len(switches)==48,"all real dynamic switch receipts")
  expected=["source_qv8","random73"]*24
  need([r["profile"] for r in switches]==expected,"length based policy routing")
  need(switches[0]["changed"]==0 and all(x["changed"]>0 for x in switches[1:]),"actual live repr changes")
  need(all(x["n8"]==28 and x["q4"]==168 for x in switches),"live original 28 n8 roles after switch")
 else:
  need(not RDYNAMIC.findall(txt),"static engine should not switch")
 result={"schema":"BEGLIN-Qv14-real-original-C-live-precision-switch-pilot/1",
 "status":"REAL_ORIGINAL_QWEN28_NATIVE_C_PROCESS_COMPLETE",
 "mode":mode,"gold_full_vocab_NLL":losses,"correct_each":right,
 "total_gold_NLL":total,"mean_gold_NLL":mean,"correct":sum(right),
 "n":48,"real_derived_or_original_executable_SHA256":BINSHA[mode],
 "private_frozen_input_sha256":LOCKSHA,"public_frozen_manifest_sha256":PUBSHA,
 "native_original_C_log_SHA256":sha(log),"native_process_wall_seconds":elapsed,
 "process_peak_RSS_bytes":peak,
 "actual_dynamic_trace":switches,
 "total_switch_ms":sum(x["switch_ms"] for x in switches),
 "pure_decode_perf_or_GPU_not_measured":True,
 "no_user_facing_model_quality_claim":True,
 "no_operator_release_or_main_change":True}
 store(rec,(json.dumps(result,indent=2,sort_keys=True)+"\n").encode())
 print("QV14_REAL_NATIVE_LAB",json.dumps({"mode":mode,"correct":sum(right),
 "nll":mean,"rss_mib":round(peak/1048576,2),
 "switches":len([x for x in switches if x["changed"]>0]),"switch_ms":result["total_switch_ms"],
 "wall_s":round(elapsed,3),"receipt_sha256":sha(rec)},sort_keys=True),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=EXE,required=True)
 a=parser.parse_args();evaluate(a.mode)
