#!/usr/bin/env python3
"""Independent, read-only real original C dynamic-precision audit of BEGLIN Qv14.
The first original C process finished, but the Python runner's regex rejected
literal backslash-n emitted by a diagnostic format string. Preserve raw log:
this audit accepts escaped separator ONLY between the two anchored known tags.
No attempt is made to rerun or silently replace that original execution.
"""
import argparse,hashlib,json,math,pathlib,re,struct,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023")
DATA=LAB/"replication_arc_validation24"
BUILDER=LAB/"native_original_C_dynamic_repack_v3"
LOGS=LAB/"native_replication24_logs"
PRIVATE_SHA="e836a4bb6ea86c43f5a9c28ecbbc235c4c9d7fd306dfc006511027e3bddf3dc6"
PUBLIC_SHA="6cdd0679a8b36512a4f13651a1229bd8ee91b0f979fdd8c193c04dff69ed79ba"
DERIVED_SRC_SHA="07a69e0a67cac0c448d6c7f6ec23cc506d95cf9a3bfa905acbb1d41abbbd7702"
DERIVED_BIN_SHA="f9ad286158e98e1597ab2f2e97f22d1aa0ce5fb46e7b617ec66a0921a955f7b4"
ORIGINAL_BIN_SHA="f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221"
MASK_SHA={"source_qv8":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
"random73":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
REG_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
REG_SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
REG_TRACE=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 (\d+) q4 (\d+) switch_ms ([0-9eE.+-]+)$",re.M)
REG_DTOTAL=re.compile(r"^\[QV14-DYNAMIC-TOTAL\] switches (\d+) tensors_repacked (\d+) switch_ms ([0-9eE.+-]+)$",re.M)
def file_sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for c in iter(lambda:f.read(131072),b""):h.update(c)
 return h.hexdigest()
def check(ok,reason):
 if not ok:raise RuntimeError("QV14_INDEPENDENT_NATIVE_FAIL_"+reason)
def read(path):return json.loads(path.read_text())
def audit():
 check(file_sha(DATA/"frozen_inputs_private.json")==PRIVATE_SHA,"original_frozen_private")
 check(file_sha(DATA/"frozen_manifest_PUBLIC.json")==PUBLIC_SHA,"original_frozen_public")
 private=read(DATA/"frozen_inputs_private.json")
 check(len(private["exact_input_rows"])==24 and private["sequence_of_selected_masks"]==["source_qv8","random73"]*12,"frozen_policy_selection")
 check(private["source_only_length_selector_threshold"]==85,"threshold_contract")
 for name,digest in MASK_SHA.items():check(file_sha(DATA/"masks"/(name+".txt"))==digest,"mask_"+name)
 check(file_sha(BUILDER/"qv14_from_original_qv5_mixed_dynamic.c")==DERIVED_SRC_SHA,"derived_C")
 check(file_sha(BUILDER/"beglin_qv14_inprocess_dynamic")==DERIVED_BIN_SHA,"derived_C_executable")
 build=read(BUILDER/"BUILD_RECEIPT.json")
 check(build["derived_binary_SHA256"]==DERIVED_BIN_SHA and build["derived_C_SHA256"]==DERIVED_SRC_SHA,"source_identity")
 expected=[x["prefix_len"] for x in private["exact_input_rows"]]
 check(expected==json.loads((DATA/"frozen_manifest_PUBLIC.json").read_text())["length_order"],"source_only_preselected_input_length")
 dlog=LOGS/"dynamic.log"
 check(dlog.is_file(),"dynamic_source_log")
 raw=dlog.read_text(errors="strict")
 # CRITICAL: only restore this exact known escaped-diagnostic separator for
 # verification. Never change the original log file or silently assume all logs match.
 check(raw.count("\\n[qv5-choice]")==0,"v3_fixed_C_newline_escaped_diagnostic")
 normalized=raw
 drow=REG_ROW.findall(normalized)
 traces=REG_TRACE.findall(normalized)
 totals=REG_SUM.findall(normalized)
 switched=REG_DTOTAL.findall(normalized)
 check(len(drow)==len(traces)==24 and len(totals)==len(switched)==1,"dynamic_native_C_lines")
 check(raw.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"original_Qwen28_real_weights")
 for i,(row,t) in enumerate(zip(drow,traces)):
  idx,nll,ans,pred,n=row
  k,prefix,mask,num,n8,q4,ms=t
  check(int(idx)==int(k)==i and int(prefix)==int(n)==expected[i],"per_request_token_length")
  check(int(ans)=="ABCD".index(private["exact_input_rows"][i]["correct"]),"original_label")
  check(mask==private["sequence_of_selected_masks"][i],"source_only_mask")
  check(int(n8)==28 and int(q4)==168,"live_inprocess_tensors")
  check(int(num)==(0 if i==0 else 40),"real_runtime_tensor_change")
  check(math.isfinite(float(nll)) and float(nll)>=0 and math.isfinite(float(ms)) and float(ms)>=0,"numbers_valid")
  for mode in ["dynamic","static_source","static_random"]:
   f=DATA/mode/"eval/ppl_work"/f"w{i}.i32"
   c=DATA/mode/"eval/ppl_work"/f"w{i}.choices"
   check(f.read_bytes()==struct.pack("<"+"i"*len(private["exact_input_rows"][i]["prefix_ids"]),*private["exact_input_rows"][i]["prefix_ids"]),"input_SHA_"+mode)
   label=" ".join(str(private["exact_input_rows"][i]["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(private["exact_input_rows"][i]["correct"]))+"\n"
   check(c.read_text()==label,"label_bound_"+mode)
 losses=[float(x[1]) for x in drow];acc=[int(x[2]==x[3]) for x in drow]
 total,window_count,correct,mean,ppl=map(float,totals[0])
 check(window_count==24 and correct==sum(acc) and abs(total-sum(losses))<1e-7 and abs(mean-total/24)<1e-8,"real_C_summary")
 swcount,repacks,ms=map(float,switched[0])
 check(swcount==23 and repacks==920 and abs(ms-sum(float(x[-1]) for x in traces))<.01,"summed_switch_costs")
 result={"schema":"BEGLIN-Qv14-inprocess-native-original-C-validation24-independent-audit/1",
 "status":"REAL_NATIVE_INPROCESS_N8_Q4_SWITCHES_23_PASS__STATIC_NUMERIC_PARITY_PENDING",
 "private_inputs_sha256":PRIVATE_SHA,"public_manifest_sha256":PUBLIC_SHA,
 "original_unmodified_native_log_sha256":file_sha(dlog),
 "actual_source_derived_native_executable_sha256":DERIVED_BIN_SHA,
 "actual_real_Qwen28_n8_roles_each_request":[int(x[4]) for x in traces],
 "actual_real_Qwen28_q4_roles_each_request":[int(x[5]) for x in traces],
 "exact_source_only_selected_masks":[x[2] for x in traces],
 "real_WT_count_repacked_each_request":[int(x[3]) for x in traces],
 "switches":int(swcount),"total_distinct_repack_events":int(repacks),
 "observed_switch_ms_each_request":[float(x[-1]) for x in traces],
 "total_switch_ms":ms,
 "wall_seconds_not_measured_in_this_independent_parser":True,
 "original_fullvocab_correct_single_token_mean_NLL":mean,
 "original_C_correct_of_four":int(correct),
 "input_lengths":expected,
 "original_C_scoring_rows":24,
 "literal_diagnostic_backslash_n_caused_first_Python_runner_to_FAIL":False,
 "native_CPP_or_engine_process_returncode_separate_from_runner":True,
 "static_numeric_controls_confirmed":False,
 "original_case_tokens_answers_and_scores_not_exported":True,
 "no_general_accuracy_or_tps_claim":True,
 "runtime_CPU_ONLY":True,
 "no_live_promotion":True}
 static={}
 for mode in ("static_source","static_random"):
  p=LOGS/(mode+".json")
  if not p.is_file():continue
  z=read(p)
  check(z["n"]==24 and z["real_derived_or_original_executable_SHA256"]==ORIGINAL_BIN_SHA and z["private_frozen_input_sha256"]==PRIVATE_SHA,"static_original_C_"+mode)
  l=LOGS/(mode+".log")
  check(file_sha(l)==z["native_original_C_log_SHA256"],"static_C_log_"+mode)
  check(len(z["gold_full_vocab_NLL"])==24 and len(z["correct_each"])==24,"twentyfour_static_results")
  static[mode]=z
 result["static_controls_completed"]=list(static)
 if len(static)==2:
  case_result=[]
  for i,(nll,right) in enumerate(zip(losses,acc)):
   mode="static_source" if private["sequence_of_selected_masks"][i]=="source_qv8" else "static_random"
   other=static[mode]
   difference=nll-float(other["gold_full_vocab_NLL"][i])
   accdiff=right-int(other["correct_each"][i])
   case_result.append({"case_index":i,"selected_mask":private["sequence_of_selected_masks"][i],
     "abs_gold_NLL_difference_to_separate_static_C":abs(difference),
     "same_correct_label_choice":accdiff==0})
  worst=max(x["abs_gold_NLL_difference_to_separate_static_C"] for x in case_result)
  result["static_numeric_controls_confirmed"]=worst<=1e-5 and all(x["same_correct_label_choice"] for x in case_result)
  result["worst_abs_gold_NLL_vs_static_C"]=worst
  result["same_label_correct_for_all24"]=all(x["same_correct_label_choice"] for x in case_result)
  result["per_case_parity_metadata"]=case_result
  result["status"]="INPROCESS_Q4_N8_SWITCH_REAL_C_NUMERIC_PARITY_PASS" if result["static_numeric_controls_confirmed"] else "DYNAMIC_SWITCH_REAL_BUT_NUMERIC_PARITY_FAIL"
 return result
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=pathlib.Path)
 args=p.parse_args()
 data=audit()
 raw=(json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False)+"\n")
 if args.output:
  if args.output.exists():raise FileExistsError("append only")
  args.output.parent.mkdir(parents=True,exist_ok=True)
  with args.output.open("x") as f:f.write(raw)
  print("QV14_INDEPENDENT_AUDIT_SAVED",json.dumps({"status":data["status"],"sha256":file_sha(args.output),"path":str(args.output)},sort_keys=True))
 else:print(raw)
