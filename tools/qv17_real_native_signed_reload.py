#!/usr/bin/env python3
"""BEGLIN Qv17 original native C serialized alternate precision weight-cache.
CREATE: Qv16-calibrated BF16->Q4/n8 prewarm, write 40 alternates owner-only, eval 4
pinned requests. An Ed25519 signature is issued via a separate offline step.
RELOAD: *trusted Python bootstrap* checks local Ed25519 dev signature, 83.5 MiB
artifact digest, exact BF16 3GB file SHA256, binary/masks/model provenance;
native independently checks SHA256 and 40 tensors, then scores 48 requests.
No source model, original C kernel or immutable operator changes.
"""
import argparse,hashlib,importlib.util,json,math,os,pathlib,psutil,re,struct,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
SIGNER=ROOT/"tools/qv17_local_dev_signing.py"
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
COHORT=Q16/"independent_arc_validation48"
NATIVE=LAB/"native_persisted_C_v2/beglin_qv17_signed_packed_cache"
SOURCE=LAB/"native_persisted_C_v2/qv17_original_Qwen28_persisted_cache.c"
PACK=LAB/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
LOG=LAB/"native_executions"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
QW=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
BIN_SHA="16085378788b64b229f321b49c11540e286edffbff832db4e41fbf8d2a87643b"
SRC_SHA="0c9d797c0066fdb1f8ca5972bdf7611ae75431050d0d55c0f0fdef0653f6bdaa"
LOCK_SHA="a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296"
PUB_SHA="ac5223bdf25b6cf607cb3a589027d1a406737af7341e8f742231155579c3b26a"
MASK_SHA={"source_qv8":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
 "random73":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
R_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
R_SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
R_SWITCH=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 28 q4 168 switch_ms ([0-9eE.+-]+)$",re.M)
R_CACHE=re.compile(r"^\[QV15-CACHE\] window (\d+) hits (\d+) misses (\d+) drops_total (\d+) cache_bytes (\d+) peak_cache_bytes (\d+) cache_cap_bytes (\d+)$",re.M)
R_CACHE_TOTAL=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
R_WRITE=re.compile(r"^\[QV17-PACK-WRITE\] entries (\d+) payload_bytes (\d+) total_file_bytes (\d+) write_ms ([0-9eE.+-]+)$",re.M)
R_LOAD=re.compile(r"^\[QV17-PACK-LOAD\] entries (\d+) bytes (\d+) sha256_check_ms ([0-9eE.+-]+) total_read_ms ([0-9eE.+-]+) digest ([a-f0-9]{64})$",re.M)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(262144),b""):h.update(b)
 return h.hexdigest()
def put(p,b):
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def encode(o):return (json.dumps(o,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
def must(x,why):
 if not x:raise RuntimeError("QV17_PROTECTED_RUN_FAIL_CLOSED_"+why)
def eval(mode):
 must(mode in ("create","reload"),"mode")
 n=4 if mode=="create" else 48
 must(sha(NATIVE)==BIN_SHA and sha(SOURCE)==SRC_SHA,"original_C_native")
 must(sha(COHORT/"frozen_inputs_private.json")==LOCK_SHA,"Qv16_frozen_private")
 must(sha(COHORT/"frozen_manifest_PUBLIC.json")==PUB_SHA,"Qv16_preexec_public")
 for name,d in MASK_SHA.items():must(sha(COHORT/"masks"/(name+".txt"))==d,"mask_"+name)
 frozen=json.loads((COHORT/"frozen_inputs_private.json").read_text())
 rows=frozen["requests"][:n]
 must(len(rows)==n and frozen["ordered_assigned_profiles"][:n]==["source_qv8","random73"]*(n//2),"original_input_order")
 for i,row in enumerate(rows):
  ids=struct.pack("<"+"i"*len(row["prefix_ids"]),*row["prefix_ids"])
  expected=(COHORT/"dynamic/eval/ppl_work"/f"w{i}.i32").read_bytes()
  must(ids==expected,"actual_original_ids_"+str(i))
  choices=(" ".join(str(row["suffix_ids"][v]) for v in "ABCD")+" "+str("ABCD".index(row["correct"]))+"\n").encode()
  must((COHORT/"dynamic/eval/ppl_work"/f"w{i}.choices").read_bytes()==choices,"actual_gold_choice_"+str(i))
 spec=importlib.util.spec_from_file_location("Qv17_pinned_Ed25519_bootstrap",str(SIGNER))
 signer=importlib.util.module_from_spec(spec);spec.loader.exec_module(signer)
 must(signer.PUB_SRC.is_file() and signer.sha_file(signer.PUB_SRC)=="5f535c54985ac18257b50c4bd7969ec4ce57aecb766def485c68c8b0a4ad4847","root_pin")
 signer.verify_roots()
 os.umask(0o077)
 LOG.mkdir(parents=True,exist_ok=True,mode=0o700)
 run=LOG/("create4" if mode=="create" else "reload48")
 if run.exists():raise FileExistsError("original protected native run exists; immutable evidence")
 run.mkdir(mode=0o700)
 signer_info=None
 if mode=="create":
  if PACK.exists():raise FileExistsError("native packed artifact previously generated")
  PACK.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 else:
  signer_info=signer.verify(strict_model=True)
  must(signer_info["pack_sha256"]==sha(PACK),"externally_verified_pack_SHA")
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_QV","BEGLIN_Q_","BEGLIN_Q_LAB_"))}
 env.update({"VECLIB_MAXIMUM_THREADS":"1",
  "QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
  "QWEN_HF_CONFIG":str(MODEL/"config.json"),
  "QWEN_SAFETENSORS_MAXSEQ":"256",
  "QWEN_BASE":str(COHORT/"dynamic"),
  "QWEN_PROMPT":str(COHORT/"dynamic/prompt_ids.i32"),
  "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
  "BEGLIN_QV2_NATIVE_ASYNC":"0",
  "BEGLIN_Q_LAB_N8_ROLES_FILE":str(COHORT/"masks/source_qv8.txt"),
  "BEGLIN_QV14_MASK_SOURCE":str(COHORT/"masks/source_qv8.txt"),
  "BEGLIN_QV14_MASK_RANDOM":str(COHORT/"masks/random73.txt"),
  "BEGLIN_QV15_CACHE_MIB":"96",
  "BEGLIN_QV16_PREWARM":"1" if mode=="create" else "0"})
 if mode=="create":env["BEGLIN_QV17_PACK_WRITE"]=str(PACK)
 else:
  env["BEGLIN_QV17_PACK_LOAD"]=str(PACK)
  env["BEGLIN_QV17_PACK_SHA256"]=signer_info["pack_sha256"]
 started=time.monotonic()
 proc=subprocess.Popen([str(NATIVE),"qv14_dynamic_choice",str(n)],cwd=str(QW),env=env,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 peak_rss_bytes=0
 while proc.poll() is None:
  try:peak_rss_bytes=max(peak_rss_bytes,psutil.Process(proc.pid).memory_info().rss)
  except psutil.NoSuchProcess:pass
  time.sleep(.07)
 stdout,stderr=proc.communicate()
 wall=time.monotonic()-started
 logfile=run/"real_original_C_run.log"
 put(logfile,stdout+b"\n"+stderr)
 if proc.returncode:raise RuntimeError("real Qv17 native C exit "+str(proc.returncode)+": "+stderr.decode(errors="replace")[-2100:])
 txt=logfile.read_text()
 vals=R_ROW.findall(txt);summ=R_SUM.findall(txt);sw=R_SWITCH.findall(txt);caches=R_CACHE.findall(txt);cache_tot=R_CACHE_TOTAL.findall(txt)
 must(len(vals)==len(sw)==len(caches)==n and len(summ)==len(cache_tot)==1,"C_native_output_records")
 must(txt.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"real_28layer_Qwen")
 write=R_WRITE.findall(txt);load=R_LOAD.findall(txt)
 if mode=="create":
  must(len(write)==1 and not load,"native_packing_written")
  must(int(write[0][0])==40 and int(write[0][1])<96*1048576 and int(write[0][2])==PACK.stat().st_size,"C_written_payload_bytes")
 else:
  must(len(load)==1 and not write,"native_signed_pack_loaded")
  must(int(load[0][0])==40 and load[0][-1]==signer_info["pack_sha256"],"actual_C_sha_matches_signed")
  must(int(load[0][1])<96*1048576,"loaded_cache_cap")
 reference=json.loads((Q16/"native48/prewarm1/receipt_private.json").read_text())
 must(len(reference["gold_fullvocab_token_NLL"])==48,"original_C_stable_reference")
 losses=[];accuracy=[];total_switch_ms=0
 for i,((index,nll,ans,pred,pref),(index2,pref2,policy,count,ms),(index3,hits,miss,drops,current,peakcache,cap)) in enumerate(zip(vals,sw,caches)):
  row=rows[i];k="source_qv8" if row["prefix_len"]<85 else "random73"
  must(int(index)==int(index2)==int(index3)==i and int(pref)==int(pref2)==row["prefix_len"],"native_index_"+str(i))
  must(policy==k and int(count)==(0 if i==0 else 40),"actual_live_mask_"+str(i))
  must(int(ans)=="ABCD".index(row["correct"]),"gold_label_"+str(i))
  x=float(nll);got=int(ans==pred)
  must(math.isfinite(x) and x>=0 and x==reference["gold_fullvocab_token_NLL"][i],"ORIGINAL_native_nll_bit_identical_"+str(i))
  must(got==reference["correct_each"][i],"correct_choice_"+str(i))
  must(int(cap)==96*1048576 and int(current)<=int(cap) and int(peakcache)<=int(cap),"hard_ram_cap_"+str(i))
  must(int(miss)==0 and int(hits)==(0 if i==0 else 40),"runtime_no_requant_after_ready_"+str(i))
  losses.append(x);accuracy.append(got);total_switch_ms+=float(ms)
 goldtot,N,correct,mean,ppl=map(float,summ[0])
 must(int(N)==n and int(correct)==sum(accuracy) and abs(goldtot-sum(losses))<1e-7 and abs(mean-goldtot/n)<1e-8,"native_scores_totals")
 hits,miss,drops,live,peak,cap=map(int,cache_tot[0])
 must(hits==40*(n-1) and miss==0 and drops==0 and peak<=cap==96*1048576,"native_inactive_cache_account")
 result={
 "schema":"BEGLIN-Qv17-actual-original-C-persistent-precision-cache-create-or-reload/1",
 "status":"REAL_NATIVE_PACK_WRITE_AND_INPUT_PARITY_PASS" if mode=="create" else "ED25519_VERIFIED_NATIVE_SHA_LOADED_48_INPUT_STATIC_PARITY_PASS",
 "mode":mode,"n":n,"correct":sum(accuracy),"mean_gold_fullvocab_NLL":mean,
 "original_native_executable_sha256":BIN_SHA,"original_C_source_sha256":SRC_SHA,
 "frozen_Qv16_inputs_sha256":LOCK_SHA,
 "source_original_Qv16_48_C_per_case_NLL_exact":True,
 "source_original_Qv16_48_C_choice_correctness_exact":True,
 "actual_native_log_SHA256":sha(logfile),
 "artifact_sha256":sha(PACK),"artifact_size":PACK.stat().st_size,
 "native_write":None if not write else {"entries":int(write[0][0]),"payload_bytes":int(write[0][1]),"total_file_bytes":int(write[0][2]),"write_ms":float(write[0][3])},
 "native_read":None if not load else {"entries":int(load[0][0]),"payload_bytes":int(load[0][1]),"SHA256_check_ms":float(load[0][2]),"total_read_ms":float(load[0][3])},
 "local_developer_signed_manifest_validation":signer_info,
 "actual_cache_hits":hits,"actual_cache_requantizations":miss,
 "actual_cache_drops":drops,"actual_inactive_cache_peak_mib":peak/1048576,
 "actual_request_switch_total_ms":total_switch_ms,
 "native_process_wall_seconds":wall,"native_process_peak_RSS_MiB":peak_rss_bytes/1048576,
 "no_original_model_or_operator_live_release_change":True,
 "not_per_request_dynamic_utility_allocator":True,
 "startup_model_snapshot_full_SHA_verification_cost_separate":True}
 outfile=run/"verified_original_C_receipt.json"
 put(outfile,encode(result))
 public={k:v for k,v in result.items() if k not in ("artifact_sha256",)}
 public["private_receipt_sha256"]=sha(outfile)
 pubpath=run/"PUBLIC_AGGREGATE.json";put(pubpath,encode(public))
 print("QV17_REAL_NATIVE_"+mode.upper(),json.dumps({
 "mode":mode,"n":n,"correct":sum(accuracy),"mean_NLL":round(mean,8),
 "asset_mib":round(PACK.stat().st_size/1048576,3),
 "pack_sha":sha(PACK),"native_ms_write":result["native_write"],
 "native_ms_load":result["native_read"],
 "signed_verify":signer_info,
 "switch_total_ms":round(total_switch_ms,3),
 "wall_s":round(wall,3),
 "receipt_sha":sha(outfile)},sort_keys=True),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["create","reload"])
 args=parser.parse_args();eval(args.mode)
