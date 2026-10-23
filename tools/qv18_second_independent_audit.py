#!/usr/bin/env python3
"""Qv18 second independent read-only replay of the real two simultaneous C logs,
the native format-version/role denial logs, and offline dev Ed25519 key policy.
Prints and saves aggregate only: NEVER raw case text/token/C per-case NLL,
key material, native pack bytes, or private model. No native inference rerun.
"""
import hashlib,json,math,pathlib,re,struct
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
R=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv18-concurrent-artifact-20261023")
NEG=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv18-native-negative-schema-20261023")
Q17=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
FROZEN=Q16/"independent_arc_validation48/frozen_inputs_private.json"
PACK=Q17/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
MAN=Q17/"owner_only_artifacts/alternate40_Qv8_random73_manifest.json"
SIGN=Q17/"owner_only_artifacts/alternate40_Qv8_random73_manifest.sig"
PUB=R/"docs/research/BEGLIN_QV17_ED25519_DEV_TRUST_ROOT_PUBLIC_20261023.txt"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/model.safetensors")
EXE=Q17/"native_persisted_C_v2/beglin_qv17_signed_packed_cache"
REF=Q16/"native48/prewarm1/receipt_private.json"
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
SUMMARY=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
SWITCH=re.compile(r"^\[QV14-DYNAMIC-SWITCH\] window (\d+) prefix (\d+) profile (source_qv8|random73) tensors_repacked (\d+) n8 28 q4 168 switch_ms ([0-9eE.+-]+)$",re.M)
CACHE=re.compile(r"^\[QV15-CACHE\] window (\d+) hits (\d+) misses (\d+) drops_total (\d+) cache_bytes (\d+) peak_cache_bytes (\d+) cache_cap_bytes (\d+)$",re.M)
TOTALCACHE=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
LOAD=re.compile(r"^\[QV17-PACK-LOAD\] entries (\d+) bytes (\d+) sha256_check_ms ([0-9eE.+-]+) total_read_ms ([0-9eE.+-]+) digest ([a-f0-9]{64})$",re.M)
S={
 "model":"073400b311d049b6379b8f3f7ad5a0258268031dd009b99389cfd4030144850a",
 "pack":"49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b",
 "exe":"16085378788b64b229f321b49c11540e286edffbff832db4e41fbf8d2a87643b",
 "public":"5f535c54985ac18257b50c4bd7969ec4ce57aecb766def485c68c8b0a4ad4847",
 "manifest":"491deb61dbc30fad08b4a5d819b270b64167d5551b422823ae0b431f4a5c6824",
 "frozen":"a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296",
 "main":"7f39e581a1fa492ea358af5603cea628194f8592e38b89dcc98b91107eb42605",
 "negative":"095a9701ff8a02b5268d7e3ea953d72df752031742fdbec173f008504a1c8720",
 "rotationV2":"8daa0f695dadb5964e9873f4f1cc17bceb6494d1c9864bfc4797508a5564c948"}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(262144),b""):h.update(x)
 return h.hexdigest()
def need(cond,msg):
 if not cond:raise RuntimeError("QV18_SECOND_INDEPENDENT_AUDIT_FAILED_"+msg)
def obj(p):return json.loads(p.read_text())
def audit():
 for name,p in [("model",MODEL),("pack",PACK),("exe",EXE),("public",PUB),("manifest",MAN),("frozen",FROZEN),
  ("main",R/"docs/research/BEGLIN_QV18_TWO_PROCESS_CONCURRENT_SIGNED_CACHE_PUBLIC_20261023.json"),
  ("negative",R/"docs/research/BEGLIN_QV18_NATIVE_VERSION_ROLE_FAIL_CLOSED_PUBLIC_20261023.json"),
  ("rotationV2",R/"docs/research/BEGLIN_QV18_KEY_ROTATION_ANTIROLLBACK_V2_PUBLIC_20261023.json")]:
  need(p.is_file() and sha(p)==S[name],"authoritative_source_"+name)
 pubkey=bytes.fromhex(PUB.read_text().strip().split(":",1)[1])
 Ed25519PublicKey.from_public_bytes(pubkey).verify(SIGN.read_bytes(),MAN.read_bytes())
 signed=obj(MAN)
 need(signed["BF16_original_model_file_sha256"]==S["model"] and signed["packed_file_sha256"]==S["pack"],"real_signed_model_and_pack")
 lock=obj(FROZEN)["requests"];ref=obj(REF)
 need(len(lock)==len(ref["gold_fullvocab_token_NLL"])==48,"real_source_test_baseline")
 summary=obj(R/"docs/research/BEGLIN_QV18_TWO_PROCESS_CONCURRENT_SIGNED_CACHE_PUBLIC_20261023.json")
 need(summary["process_count"]==2 and summary["independent_C_gold_NLL_total_verified"]==16 and summary["original_owner_only_signed_pack_SHA256_unchanged"],"original_concurrent_summary")
 readings=[]
 for idx in (0,1):
  folder=LAB/f"worker{idx}";receipt=obj(folder/"receipt.json")
  log=folder/"original_native_Qwen28_C.log";text=log.read_text()
  need(receipt["worker_id"]==idx and receipt["private_native_log_sha256"]==sha(log),"worker_source_"+str(idx))
  need(receipt["native_C_exe_sha256"]==S["exe"] and receipt["original_83MiB_pack_SHA256"]==S["pack"],"worker_engine_identity")
  need(receipt["n"]==8 and receipt["original_Qv16_static_gold_NLL_exact_rows"]==8,"worker_claim")
  need("safetensors Q4_THREADS=2" in text and text.count("projection_role_tensors: f32=0 n8=28 q4=168")==1,"actual_original_C_Qwen28_quantization")
  vals=ROW.findall(text);sw=SWITCH.findall(text);cache=CACHE.findall(text);tot=SUMMARY.findall(text);caches=TOTALCACHE.findall(text);load=LOAD.findall(text)
  need(len(vals)==len(sw)==len(cache)==8 and len(tot)==len(caches)==len(load)==1,"original_C_eight_logs")
  ncorrect=0;losses=[]
  for i,((k,nll,ans,pred,n),(j,nj,profile,changes,ms),(cachei,hits,miss,drops,current,peak,cap)) in enumerate(zip(vals,sw,cache)):
   row=lock[i];v=float(nll);correct=int(ans==pred)
   need(int(k)==int(j)==int(cachei)==i and int(n)==int(nj)==row["prefix_len"],"source_idx_"+str(idx)+"_"+str(i))
   need(int(ans)=="ABCD".index(row["correct"]) and math.isfinite(v) and v>=0,"gold_letter_"+str(idx)+"_"+str(i))
   need(v==ref["gold_fullvocab_token_NLL"][i] and correct==ref["correct_each"][i],"actual_native_C_exact_"+str(idx)+"_"+str(i))
   need(profile==("source_qv8" if i%2==0 else "random73") and int(changes)==(0 if i==0 else 40),"native_Qv14_alternate_state_"+str(i))
   need(int(hits)==(0 if i==0 else 40) and int(miss)==0 and int(drops)==0,"native_warm_switch_"+str(i))
   need(int(peak)<=int(cap)==96*1048576 and int(current)<=int(cap),"native_cache_hard_cap_"+str(i))
   ncorrect+=correct;losses.append(v)
  _,number,correct,mean,_=map(float,tot[0])
  need(int(number)==8 and int(correct)==ncorrect and abs(sum(losses)/8-mean)<1e-8,"native_score_total_"+str(idx))
  hits,miss,drop,live,peak,cap=map(int,caches[0])
  need(hits==280 and miss==drop==0 and peak<=cap==96*1048576,"actual_C_cache_account_"+str(idx))
  need(int(load[0][0])==40 and load[0][-1]==S["pack"],"native_file_source_load_"+str(idx))
  begin=receipt["native_Qwen28_process_launch_monotonic_ns"]
  end=receipt["native_Qwen28_process_exit_monotonic_ns"]
  need(end>begin,"measured_process_interval_"+str(idx))
  readings.append({"idx":idx,"correct":ncorrect,"mean_NLL":mean,
   "start_ns":begin,"end_ns":end,"process_RSS_MiB":receipt["native_process_peak_RSS_MiB"],
   "real_C_log_sha256":sha(log)})
 overlap=(min(x["end_ns"] for x in readings)-max(x["start_ns"] for x in readings))/1e9
 need(overlap>1 and abs(overlap-summary["original_native_2_workers_overlap_seconds"])<.01,"actual_overlap_wall")
 need(readings[0]["mean_NLL"]==readings[1]["mean_NLL"],"independent_process_same_numeric")
 negative=obj(R/"docs/research/BEGLIN_QV18_NATIVE_VERSION_ROLE_FAIL_CLOSED_PUBLIC_20261023.json")
 need(negative["n"]==2 and negative["status"]=="BOTH_ACTUAL_NATIVE_FORMAT_GUARDS_FAIL_CLOSED_WITH_MATCHING_UNTRUSTED_SHA","native_negative_summary")
 for test in negative["negative_tests"]:
  log=NEG/(test["scenario"]+".original_C.log")
  need(sha(log)==test["original_C_negative_log_SHA256"],"native_negative_log_SHA_"+test["scenario"])
  txt=log.read_text()
  need(test["native_C_error"] in txt and "[qv5-choice]" not in txt and "qv5_choice NLL" not in txt,"native_denial_before_scoring_"+test["scenario"])
 v=obj(R/"docs/research/BEGLIN_QV18_KEY_ROTATION_ANTIROLLBACK_V2_PUBLIC_20261023.json")
 need(v["status"]=="ALL_10_ED25519_ROTATION_REVOCATION_ROLLBACK_BINDING_TESTS_PASS" and len(v["test_cases"])==10 and all(v["test_cases"].values()),"ten_ephemeral_trust_lifecycle")
 return {
 "schema":"BEGLIN-Qv18-second-independent-Qwen28-C-signed-artifact-multiprocess-and-negative-cases/1",
 "status":"ORIGINAL_C_TWO_SIGNED_READERS_AND_NATIVE_VERSION_ROLE_DENIALS_INDEPENDENT_PASS",
 "native_process_count":2,"native_Qwen28_actual_8_case_rows_each":8,"Qv16_original_C_static_gold_NLL_exact_total":16,
 "simultaneous_original_C_process_overlap_s":overlap,
 "same_83_5MiB_local_Ed25519_dev_signed_pack_SHA256":S["pack"],
 "original_full_3GB_BF16_model_SHA256":S["model"],
 "native_actual_exe_SHA256":S["exe"],
 "independent_C_log_SHA256s":[x["real_C_log_sha256"] for x in readings],
 "original_C_correct_per_worker":[x["correct"] for x in readings],
 "independent_2_native_format_denials_fail_closed":True,
 "second_generator_exact_matching_untrusted_sha_cannot_disable_layout_check":True,
 "offline_ephemeral_dev_signing_key_rotation_and_stale_version_cases":10,
 "original_Qv17_signed_asset_and_model_unchanged":True,
 "max_aggregate_process_tree_RSS_MiB_from_first_sampler":summary["peak_aggregate_process_tree_RSS_MiB"],
 "limits":[
 "Two separate C processes share one disk packed asset; this does NOT validate two threads in one model process or shared KV-cache",
 "Q4_THREADS=2 for each worker: numerical parity was rechecked but peak throughput/p95 is not a production benchmark",
 "Developer-only signature and an IN-MEMORY simulated trust-root rotation; production organizational keys and tamper-resistant rollback state do not exist in this test",
 "Native C verifies SHA256 and binary layout, but Ed25519 is in the trusted launcher, not native",
 "Physical power-off cold-boot and GPU Metal residency are not tested",
 "No new calibrated quality-aware precision selector nor cross-model accuracy advantage proven"],
 "no_original_secrets_or_raw_question_rows_published":True,"no_operator_release_change":True}
if __name__=="__main__":
 result=audit()
 out=R/"docs/research/BEGLIN_QV18_SECOND_INDEPENDENT_SIGNED_MULTIPROCESS_AUDIT_PUBLIC_20261023.json"
 b=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
 with out.open("xb") as f:f.write(b)
 print("QV18_SECOND_INDEPENDENT_AUDIT_PASS",json.dumps({"native_gold_case_rows":16,
 "actual_original_C_overlap_s":round(result["simultaneous_original_C_process_overlap_s"],3),
 "native_malformed_format_rejections":2,"offline_trust_cases":10,
 "public_report_sha256":hashlib.sha256(b).hexdigest()},sort_keys=True),flush=True)
