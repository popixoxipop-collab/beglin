#!/usr/bin/env python3
"""Independent Qv15 original C bounded cache check; no inference, no writes.
Validates exact source/compiled binary SHA, Qv14 immutable 24 inputs,
each C record's NLL/answer to original STATIC native C, switched WT roles,
cache hard cap including scales, hits/misses, cold vs warm costs and actual RSS.
Never outputs individual tokens/questions/NLL. stdout is public aggregate.
"""
import argparse,hashlib,json,math,pathlib,re,statistics,struct
R=pathlib.Path(__file__).resolve().parents[1]
Q14=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023")
Q15=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv15-bounded-cache-20261023")
SOURCE=Q15/"cached_original_C_native_build_paired_v2/qv15_original_C_paired_cache_v2.c"
EXE=Q15/"cached_original_C_native_build_paired_v2/beglin_qv15_bounded_cache_paired_v2"
INPUTS=Q14/"replication_arc_validation24"
SRC_SHA="cf4a3bafa1e20887d97f44b66bcc29dccaef6b93027dbb9471f873e0ba219d7f"
EXE_SHA="daba57fd5bd747d3e6923de05114f9d313c9dac4765d63b5fa2f84060f69af67"
Q14_LOCK_SHA="e836a4bb6ea86c43f5a9c28ecbbc235c4c9d7fd306dfc006511027e3bddf3dc6"
Q14_PUB_SHA="6cdd0679a8b36512a4f13651a1229bd8ee91b0f979fdd8c193c04dff69ed79ba"
LOG_REC=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
LOG_TOT=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
LOG_SWITCH=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 28 q4 168 switch_ms ([0-9eE.+-]+)$",re.M)
LOG_SUM_SW=re.compile(r"^\[QV14-DYNAMIC-TOTAL\] switches (\d+) tensors_repacked (\d+) switch_ms ([0-9eE.+-]+)$",re.M)
LOG_CACHE=re.compile(r"^\[QV15-CACHE\] window (\d+) hits (\d+) misses (\d+) drops_total (\d+) cache_bytes (\d+) peak_cache_bytes (\d+) cache_cap_bytes (\d+)$",re.M)
LOG_SUM_CACHE=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for data in iter(lambda:f.read(262144),b""):h.update(data)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise RuntimeError("SECONDARY_QV15_FAIL_CLOSED_"+msg)
def audit(mib):
 need(mib in (0,32,64,96),"cap")
 need(sha(SOURCE)==SRC_SHA and sha(EXE)==EXE_SHA,"original_derived_executable")
 need(sha(INPUTS/"frozen_inputs_private.json")==Q14_LOCK_SHA,"inputs")
 need(sha(INPUTS/"frozen_manifest_PUBLIC.json")==Q14_PUB_SHA,"public_freeze")
 data=json.loads((INPUTS/"frozen_inputs_private.json").read_text())
 cases=data["exact_input_rows"]
 need(len(cases)==24,"no_frozen_24")
 old={}
 for policy in ("source_qv8","random73"):
  which="static_source" if policy=="source_qv8" else "static_random"
  orig=Q14/"native_replication24_logs"/(which+".json")
  d=json.loads(orig.read_text())
  need(sha(Q14/"native_replication24_logs"/(which+".log"))==d["native_original_C_log_SHA256"],"original_C_reference_"+policy)
  old[policy]=d
 run=Q15/"native_24_runs_paired_v2"/f"cap{mib}"
 log=run/"native_C_run.log"
 receipt=run/"verified_receipt_private.json"
 need(log.is_file() and receipt.is_file(),"original_C_and_receipt")
 doc=json.loads(receipt.read_text())
 need(doc["runtime_binary_sha256"]==EXE_SHA and doc["derived_source_sha256"]==SRC_SHA,"receipt_code")
 need(doc["native_original_log_SHA256"]==sha(log),"raw_C_log")
 need(doc["Qv14_frozen_24_private_sha256"]==Q14_LOCK_SHA,"receipt_inputs")
 need(doc["hard_cache_cap_bytes"]==mib*1048576,"cap_receipt")
 text=log.read_text()
 rows=LOG_REC.findall(text);sw=LOG_SWITCH.findall(text);cache=LOG_CACHE.findall(text)
 tsum=LOG_TOT.findall(text);swsum=LOG_SUM_SW.findall(text);csum=LOG_SUM_CACHE.findall(text)
 need(len(rows)==len(sw)==len(cache)==24 and len(tsum)==len(swsum)==len(csum)==1,"native_c_record_count")
 need(text.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"original_Qwen28")
 differences=[];correct=[];ms=[];dcache=[]
 h=miss=0
 for i,(r,s,c) in enumerate(zip(rows,sw,cache)):
  j,v,ans,pred,n=r
  sj,sp,mask,changed,t=s
  cj,hit,missing,drops,live,peak,cap=c
  expected=cases[i]
  need(int(j)==int(sj)==int(cj)==i and int(n)==int(sp)==expected["prefix_len"],"case_index_"+str(i))
  need(int(ans)=="ABCD".index(expected["correct"]),"gold_choice_"+str(i))
  want=data["sequence_of_selected_masks"][i]
  need(mask==want and mask==("source_qv8" if int(n)<85 else "random73"),"length_based_mask_"+str(i))
  need(int(changed)==(0 if i==0 else 40),"precision_change_count_"+str(i))
  need(int(live)<=int(cap)==mib*1048576 and int(peak)<=int(cap),"hardcap_"+str(i))
  checkfloat=float(v)
  need(math.isfinite(checkfloat) and checkfloat>=0,"NLL_finite_"+str(i))
  ref=old[want]
  need(checkfloat==ref["gold_full_vocab_NLL"][i],"static_original_C_NLL_BIT_EXACT_"+str(i))
  thisright=int(ans==pred)
  need(thisright==ref["correct_each"][i],"static_C_choice_"+str(i))
  correct.append(thisright);differences.append(abs(checkfloat-ref["gold_full_vocab_NLL"][i]))
  ms.append(float(t));dcache.append(int(live));h+=int(hit);miss+=int(missing)
  import struct
  ids=struct.pack("<"+"i"*len(expected["prefix_ids"]),*expected["prefix_ids"])
  need((INPUTS/"dynamic/eval/ppl_work"/f"w{i}.i32").read_bytes()==ids,"original_locked_input_"+str(i))
 total,n,won,mean,ppl=map(float,tsum[0])
 need(n==24 and won==sum(correct),"native_accuracy")
 switches,changes,total_ms=map(float,swsum[0])
 need(switches==23 and changes==920 and abs(sum(ms)-total_ms)<.015,"native_switch_counter")
 H,M,D,cache_bytes,peak_cache,capbytes=map(int,csum[0])
 need(H==h and M==miss and H+M==920 and capbytes==mib*1048576 and peak_cache<=capbytes,"cache_accounting")
 if mib==0:need(H==0 and M==920 and cache_bytes==peak_cache==0,"zero_cache_case")
 if mib==96:need(H==880 and M==40 and D==0,"warm_cache_case")
 return {"schema":"BEGLIN-Qv15-second-independent-original-C-paired-role-bounded-cache-reparse/2",
 "status":"REAL_NATIVE_Q4_N8_24_REQUEST_WT_CACHE_CORRECT_AND_HARD_LIMIT_ENFORCED",
 "sample":"Qv14 source-frozen independent 24 ARC Challenge validation inputs (mechanics reuse)",
 "n_requests":24,"cap_MiB":mib,"cache_cap_bytes":capbytes,"peak_inactive_quantized_cache_MiB":peak_cache/1048576,
 "cache_hits":H,"cold_quantization_misses":M,"discards_due_to_hard_cap":D,
 "actual_tensor_role_bit_switches":int(changes),"actual_precision_profile_switches":int(switches),
 "cold_first_switch_ms":ms[1],"median_warm_switch_ms":statistics.median(ms[2:]),
 "warm_switch_time_sum_ms":sum(ms[2:]),"native_total_switch_ms":total_ms,
 "actual_native_original_C_24_logs_verified":24,"gold_fullvocab_correct_NLL_mean":mean,
 "correct_choices":int(won),"max_abs_correct_token_NLL_to_original_static_reference":max(differences),
 "number_exact_numeric_parity":sum(x==0 for x in differences),
 "original_C_binary_sha256":EXE_SHA,"original_C_source_sha256":SRC_SHA,
 "owner_only_native_log_sha256":sha(log),"owner_only_result_receipt_sha256":sha(receipt),
 "native_process_wall_s":doc["native_full_process_wall_seconds"],
 "native_process_peak_RSS_MiB":doc["native_peak_RSS_bytes"]/1048576,
 "prior_uncached_v3_C24_wall_s":186.41042608302087,
 "prior_static_Qv8_C24_wall_s":131.65761866699904,
 "research_not_quality_generalization":True,
 "not_measure_of_metal_gpu_TTFT_or_decoding_tokens_per_second":True,
 "inactive_cache_budget_is_ADDITIONAL_not_same_weight_budget_as_Qv13":True,
 "no_original_model_kernel_or_operating_release_changes":True}
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--mib",type=int,choices=[0,32,64,96],required=True)
 args=parser.parse_args();print(json.dumps(audit(args.mib),sort_keys=True,indent=2))
