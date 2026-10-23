#!/usr/bin/env python3
"""Independent source-output verification for Qv16 48 new requests.
Read-only: original C raw logs, pinned input IDs, masks, cap counters and static
Qv5 C reference binaries. Neither fits masks nor executes any model.
"""
import argparse,hashlib,json,math,pathlib,re,statistics,struct
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
IN=BASE/"independent_arc_validation48"
RUNS=BASE/"native48"
STATIC=BASE/"native_static48_logs"
BINARY=BASE/"derived_startup_prewarm_C_v1/beglin_qv16_bounded_cache_prewarm"
SRC=BASE/"derived_startup_prewarm_C_v1/qv16_from_original_Qv15_paired.c"
SEALS={
 "private":"a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296",
 "public":"ac5223bdf25b6cf607cb3a589027d1a406737af7341e8f742231155579c3b26a",
 "native":"3fc79f4c31af4d70ba36df87f447134a2f6c79b980748c3f4c2048440ee9a44d",
 "nativeC":"b05c2e8aac488f9accba44c61073a72f2103be6284b37aa5de0a500185995a9f",
 "source":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
 "random":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
REG_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
REG_SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
REG_SW=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 28 q4 168 switch_ms ([0-9eE.+-]+)$",re.M)
REG_PRE=re.compile(r"^\[QV16-PREWARM\] prepared (\d+) startup_ms ([0-9eE.+-]+) inactive_cache_bytes (\d+) cap_bytes (\d+) startup_native_requantizations (\d+) startup_swap_hits (\d+)$",re.M)
REG_CACHE=re.compile(r"^\[QV15-CACHE\] window (\d+) hits (\d+) misses (\d+) drops_total (\d+) cache_bytes (\d+) peak_cache_bytes (\d+) cache_cap_bytes (\d+)$",re.M)
REG_CACHE_TOTAL=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(262144),b""):h.update(c)
 return h.hexdigest()
def check(ok,why):
 if not ok:raise RuntimeError("INDEPENDENT_QV16_FAIL_CLOSED_"+why)
def read(p):return json.loads(p.read_text())
def reparse_C(log,mode,count=48):
 txt=log.read_text();rows=REG_ROW.findall(txt);summ=REG_SUM.findall(txt)
 check(len(rows)==count and len(summ)==1,"native_C_rows_"+mode)
 losses=[];correct=[]
 for i,(k,nll,ans,pred,n) in enumerate(rows):
  check(int(k)==i and 0<=int(ans)<4 and 0<=int(pred)<4 and math.isfinite(float(nll)),"native_rows_"+mode)
  losses.append(float(nll));correct.append(int(ans==pred))
 t,n,won,mean,ppl=map(float,summ[0])
 check(n==count and won==sum(correct) and abs(sum(losses)-t)<1e-7 and abs(mean-t/count)<1e-8,"native_summary_"+mode)
 return txt,rows,losses,correct,mean
def audit():
 check(sha(IN/"frozen_inputs_private.json")==SEALS["private"],"inputs_sha")
 check(sha(IN/"frozen_manifest_PUBLIC.json")==SEALS["public"],"public_sha")
 check(sha(BINARY)==SEALS["native"] and sha(SRC)==SEALS["nativeC"],"native_origin")
 check(sha(IN/"masks/source_qv8.txt")==SEALS["source"],"source_mask")
 check(sha(IN/"masks/random73.txt")==SEALS["random"],"random_mask")
 cases=read(IN/"frozen_inputs_private.json")["requests"]
 check(len(cases)==48,"48_requests")
 for i,case in enumerate(cases):
  expected=struct.pack("<"+"i"*len(case["prefix_ids"]),*case["prefix_ids"])
  check((IN/"dynamic/eval/ppl_work"/f"w{i}.i32").read_bytes()==expected,"real_Qwen_token_ids_"+str(i))
  answer=(" ".join(str(case["suffix_ids"][c]) for c in "ABCD")+" "+str("ABCD".index(case["correct"]))+"\n").encode()
  check((IN/"dynamic/eval/ppl_work"/f"w{i}.choices").read_bytes()==answer,"real_source_option_tokens_"+str(i))
 olds={}
 for name in ("static_source","static_random"):
  rec=STATIC/f"{name}.json";log=STATIC/f"{name}.log"
  check(rec.is_file() and log.is_file(),"original_static_"+name)
  z=read(rec)
  check(z["n"]==48 and z["private_frozen_input_sha256"]==SEALS["private"],"original_reference_inputs")
  check(sha(log)==z["native_original_C_log_SHA256"],"original_reference_C_log")
  txt,rows,losses,correct,mean=reparse_C(log,name)
  check(losses==z["gold_full_vocab_NLL"] and correct==z["correct_each"],"original_static_per_item")
  olds[name]=(losses,correct,mean,sha(log))
 results={}
 for flag in (0,1):
  d=RUNS/f"prewarm{flag}";log=d/"native_C_original48.log";receipt=d/"receipt_private.json"
  check(log.is_file() and receipt.is_file(),"native_result_pre"+str(flag))
  r=read(receipt)
  check(r["prewarm_enabled"]==bool(flag) and r["source_frozen48_sha256"]==SEALS["private"],"source_prewarm_input_contract")
  check(r["native_log_sha256"]==sha(log) and r["real_native_C_binary_sha256"]==SEALS["native"],"C_live_executable")
  txt,rows,losses,correct,mean=reparse_C(log,"prewarm"+str(flag))
  check(losses==r["gold_fullvocab_token_NLL"] and correct==r["correct_each"],"receipt_actual_raw_C_"+str(flag))
  check(txt.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"real_Qwen28_C_runtime")
  sw=REG_SW.findall(txt);cache=REG_CACHE.findall(txt);pre=REG_PRE.findall(txt);cs=REG_CACHE_TOTAL.findall(txt)
  check(len(sw)==len(cache)==48 and len(cs)==1,"real_live_prof_switch_48")
  check(len(pre)==flag,"prewarm_startup_count")
  chosen=[]
  for i,(w,entry) in enumerate(zip(rows,sw)):
   idx,nll,ans,pred,n=w
   j,m,profile,num,ms=entry
   target="static_source" if int(n)<85 else "static_random"
   needprofile="source_qv8" if target=="static_source" else "random73"
   check(int(j)==int(idx)==i and int(m)==int(n)==cases[i]["prefix_len"],"each_exact_locked_prefix")
   check(profile==needprofile and int(num)==(0 if i==0 else 40),"actual_cache_switch")
   check(int(ans)=="ABCD".index(cases[i]["correct"]),"source_correct_answer")
   reference=olds[target]
   check(float(nll)==reference[0][i] and int(ans==pred)==reference[1][i],"original_real_C_numeric_static_exact")
   ix,hit,miss,drop,current,peak,cap=cache[i]
   check(int(ix)==i and int(cap)==96*1048576 and int(current)<=int(cap) and int(peak)<=int(cap),"hard_bound")
   chosen.append((int(hit),int(miss),float(ms)))
  H,M,D,live,peak,cap=map(int,cs[0])
  check(H+M==47*40 and D==0 and peak<=cap==96*1048576,"hard_cap_and_counters")
  expected_H=1880 if flag else 1840
  check(H==expected_H and M==(0 if flag else 40),"startup_vs_request_hit_split")
  if flag:
   check(int(pre[0][0])==40 and int(pre[0][4])==40 and int(pre[0][5])==40,"startup_40_exact")
  results[str(flag)]={
   "n":48,"raw_C_rows_verified":48,"origin_native_C_log_SHA256":sha(log),
   "owner_only_receipt_SHA256":sha(receipt),
   "mean_fullvocab_correct_token_NLL":mean,"correct":sum(correct),
   "all_48_selected_original_static_C_scores_bit_exact":True,
   "cache_cap_bytes":cap,"cache_peak_bytes":peak,
   "cached_precision_reuse_hits":H,"request_requantization_misses":M,
   "cache_drops":D,"startup_precompute_ms":float(pre[0][1]) if pre else 0.,
   "request_switch_ms_total":r["total_request_switch_ms"],
   "first_long_request_switch_ms":chosen[1][2],
   "median_warm_request_switch_ms":statistics.median(x[2] for x in chosen[2:]),
   "process_wall_s":r["native_process_wall_s"],
   "process_peak_RSS_MiB":r["native_peak_process_RSS_MiB"]}
 zero,one=results["0"],results["1"]
 return {
  "schema":"BEGLIN-Qv16-second-original-native-C-48-new-requests-explicit-startup-packing-audit/1",
  "status":"ORIGINAL_C_48_NEW_INPUTS_PER_MODE_STATIC_NUMERIC_PARITY_AND_96_MIB_PASS",
  "Qv16_private_source_input_sha256":SEALS["private"],
  "original_C_derived_executable_sha256":SEALS["native"],
  "original_output_before_cache_compare_preserved":True,
  "Qv14_28_old_validation_questions_excluded":True,
  "same_original_static_reference_gold_NLL_for_48_of_48_requests":True,
  "cases_by_preparation":results,
  "first_long_request_switch_reduction_ms":zero["first_long_request_switch_ms"]-one["first_long_request_switch_ms"],
  "total_request_switch_reduction_ms":zero["request_switch_ms_total"]-one["request_switch_ms_total"],
  "explicit_startup_cost_ms":one["startup_precompute_ms"],
  "net_wall_time_difference_s_uncontrolled_host":zero["process_wall_s"]-one["process_wall_s"],
  "limits":["Explicit prewarming moves initial 40 tensor quantizations to process startup; it does not remove the cold CPU work",
    "96 MiB is an ADDITIONAL inactive packed/scales cap and not same-RAM as static Q4",
    "The 48 evaluation items are fresh vs Qv14 validation4/24 but same ARC validation source domain",
    "The router still uses arbitrary prefix length; this experiment is about start-request latency, not trained quality selection",
    "Total wall mixes model load and uncontrolled host load; directly instrumented startup/request-switch times are stronger",
    "GPU Metal, multi-client concurrency and cold reboot service SLA remain untested",
    "No main merge or operator production release change"],
  "no_per_item_question_token_or_loss_exported":True}
if __name__=="__main__":
 args=argparse.ArgumentParser();args.add_argument("--output",type=pathlib.Path)
 a=args.parse_args();result=audit()
 content=json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n"
 if a.output:
  if a.output.exists():raise FileExistsError("append-only audit output")
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open("x") as f:f.write(content)
  print("QV16_INDEPENDENT48_AUDITED",a.output,sha(a.output))
 else:print(content)
