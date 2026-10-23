#!/usr/bin/env python3
"""Independent, read-only validation of REAL Qwen2.5-1.5B 28-layer Apple C Qv19.
Checks source/23 kernel SHA, Ed25519-signed new native executable, Qv17 signed
packed Q4/n8 cache, source-frozen 8 exact inputs, per-case original C gold NLL,
both real pthread worker receipts, exclusive-model epoch/KV contract.
Does not invoke inference or expose original case text/tokens/gold scores.
"""
import hashlib,json,math,pathlib,re,statistics
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023")
BUILD=LAB/"build_v1"
RUN=LAB/"native_first8"
Q17=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
NATIVE=BUILD/"beglin_qv19_original_C_exclusive_lease"
SRC=BUILD/"qv19_original_Qwen28_derived.c"
LOG=RUN/"real_original_C_Qv19.log"
PUB=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv13-github-commit-20261023/docs/research/BEGLIN_QV17_ED25519_DEV_TRUST_ROOT_PUBLIC_20261023.txt")
ORIG=Q17/"native_persisted_C_v2/qv17_original_Qwen28_persisted_cache.c"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/model.safetensors")
PACK=Q17/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
NATIVE_SHA="c333c5a95aca340e59352c7e0e716521b3f57028002a53e307a9f962a2c654c2"
DERIVED_SHA="b3a2535c28cb276df66f0fc5ae677c0c0c4f996d34de3b3032907f3a5337fe32"
ORIG_SHA="0c9d797c0066fdb1f8ca5972bdf7611ae75431050d0d55c0f0fdef0653f6bdaa"
SCORE_SHA="10c84007027d15a9327c15ec687c2327e19390de65f03bf9d95aee4bd4c9fe72"
PACK_SHA="49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b"
INPUT_SHA="a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296"
PUBROOT_SHA="5f535c54985ac18257b50c4bd7969ec4ce57aecb766def485c68c8b0a4ad4847"
SIGN_SHA="55a9f6573dcbd1cdbc1daf248f76de71af05e40fb4b3e4c2f192f855abaa7297"
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
LEASE=re.compile(r"^\[QV19-LEASE\] window (\d+) worker (\d+) epoch_enter (\d+) epoch_exit (\d+) profile (source_qv8|random73) changed (\d+) wait_ns (\d+) guarded_ns (\d+) snapshot_stable 1$",re.M)
TOTAL=re.compile(r"^\[QV19-MODEL-LEASE-TOTAL\] requests (\d+) workers 2 worker0 (\d+) worker1 (\d+) max_active (\d+) epoch (\d+) poisoned (\d+) guarded_forward_serial 1$",re.M)
SUMMARY=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
CACHE=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for data in iter(lambda:f.read(262144),b""):h.update(data)
 return h.hexdigest()
def need(cond,msg):
 if not cond:raise RuntimeError("INDEPENDENT_QV19_REAL_NATIVE_REPARSE_FAILED_"+msg)
def audit():
 for name,p,want in (("derived_executable",NATIVE,NATIVE_SHA),("derived_C",SRC,DERIVED_SHA),
  ("parent_Qv17_original_C",ORIG,ORIG_SHA),("original_packed_cache",PACK,PACK_SHA),
  ("dev_trust_root",PUB,PUBROOT_SHA),
  ("frozen_input",Q16/"independent_arc_validation48/frozen_inputs_private.json",INPUT_SHA),
  ("Qv19_developer_signed_envelope",BUILD/"QV19_DEV_SIGNED_BUILD.json",SIGN_SHA)):
  need(p.is_file() and sha(p)==want,"SHA_"+name)
 with NATIVE.open("rb") as f:need(f.read(8)==bytes.fromhex("cffaedfe0c000001"),"genuine_Apple_ARM64_MachO")
 trust_key=bytes.fromhex(PUB.read_text().split(":",1)[1])
 Ed25519PublicKey.from_public_bytes(trust_key).verify(
  (BUILD/"QV19_DEV_SIGNED_BUILD.sig").read_bytes(),
  (BUILD/"QV19_DEV_SIGNED_BUILD.json").read_bytes())
 envelope=json.loads((BUILD/"QV19_DEV_SIGNED_BUILD.json").read_text())
 need(envelope["native_sha256"]==NATIVE_SHA and
      envelope["parent_Qv17_C_SHA256"]==ORIG_SHA and
      envelope["original_Qwen_gold_NLL_scorer_SHA256"]==SCORE_SHA and
      envelope["Qv17_signed_alternate_pack_SHA256"]==PACK_SHA and
      envelope["Qv16_frozen_source_only_case_SHA256"]==INPUT_SHA,
      "Ed25519_signed_source_model_binary_binding")
 original_src=ORIG.read_text()
 a=original_src.index('            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);')
 b=original_src.index('        if(qv14_active)fprintf(stderr,"[QV14-DYNAMIC-TOTAL]',a)
 need(hashlib.sha256(original_src[a:b].encode()).hexdigest()==SCORE_SHA,
      "unaltered_original_C_gold_NLL")
 derived_src=SRC.read_text()
 aa=derived_src.index('            for(int p=0;p<n;p++)forward_token(ids[p],p,xtmp,0);')
 bb=derived_src.index('        if(qv14_active)fprintf(stderr,"[QV14-DYNAMIC-TOTAL]',aa)
 need(derived_src[aa:bb]==original_src[a:b],"derived_C_scorer_exact")
 baseline=json.loads((Q17/"native_persisted_C_v2/BUILD_RECEIPT.json").read_text())
 kernel=baseline["original_23_kernel_objects"]
 need(len(kernel)==23 and all(sha(pathlib.Path(i["file"]))==i["sha256"] for i in kernel),
      "23_original_kernel_objects_unmodified")
 # Re-read entire BF16 file (not merely cache/filename).
 need(sha(MODEL)=="073400b311d049b6379b8f3f7ad5a0258268031dd009b99389cfd4030144850a",
      "real_Qwen_original_BF16_full_file_hash")
 frozen=json.loads((Q16/"independent_arc_validation48/frozen_inputs_private.json").read_text())["requests"]
 original=json.loads((Q16/"native48/prewarm1/receipt_private.json").read_text())
 text=LOG.read_text()
 rows=ROW.findall(text);leases=LEASE.findall(text);summ=SUMMARY.findall(text);totals=TOTAL.findall(text);caches=CACHE.findall(text)
 need(len(rows)==len(leases)==8 and len(summ)==len(totals)==len(caches)==1,
      "actual_C_8_case_and_lease_log_count")
 byrow={int(r[0]):r for r in rows};byl={int(r[0]):r for r in leases}
 need(set(byrow)==set(byl)==set(range(8)),"eight_unique_original_C_cases")
 waits=[];held=[];workers=[];correct=[];loss=[];epochs=[]
 for i in range(8):
  q=byrow[i];z=byl[i];inp=frozen[i]
  v=float(q[1]);right=int(q[2]==q[3]);n=int(q[4])
  need(math.isfinite(v) and v>=0 and n==inp["prefix_len"] and
       int(q[2])=="ABCD".index(inp["correct"]),"native_gold_input_"+str(i))
  need(v==original["gold_fullvocab_token_NLL"][i] and right==original["correct_each"][i],
       "original_C_gold_NLL_bit_exact_"+str(i))
  need(int(z[2])==int(z[3]) and int(z[2])>=1 and
       z[4]==("source_qv8" if n<85 else "random73"),"selected_precision_inflight_stability_"+str(i))
  need(int(z[7])>0 and int(z[6])>=0 and int(z[1]) in (0,1),"native_worker_receipt_"+str(i))
  epochs.append(int(z[2]));workers.append(int(z[1]));waits.append(int(z[6]));held.append(int(z[7]))
  correct.append(right);loss.append(v)
 need(sorted(epochs)==list(range(1,9)) and sorted(set(workers))==[0,1],
      "two_actual_pthreads_and_unique_epoch")
 n,w0,w1,peak_active,epoch,poison=map(int,totals[0])
 need(n==8 and w0==workers.count(0)>0 and w1==workers.count(1)>0 and
      peak_active==1 and epoch==8 and poison==0,"whole_model_single_active_KV_lifetime")
 _,nitems,won,mean,ppl=map(float,summ[0])
 need(nitems==8 and won==sum(correct) and abs(mean-sum(loss)/8)<1e-8,"native_summary")
 hits,miss,drop,current,peak,cap=map(int,caches[0])
 need(miss==drop==0 and peak<=cap==96*1048576,"native_signed_cache_cap")
 receipt=json.loads((RUN/"PUBLIC_AGGREGATE.json").read_text())
 need(receipt["original_C_log_sha256"]==sha(LOG) and
      receipt["signed_dev_binary_envelope_sha256"]==SIGN_SHA and
      receipt["original_gold_NLL_exact_cases"]==8,
      "first_Qv19_main_receipt")
 return {"schema":"BEGLIN-Qv19-second-independent-original-Qwen28-native-C-two-pthread-lease-audit/1",
 "status":"ORIGINAL_APPLE_QWEN28_TWO_PTHREAD_MODEL_LIFETIME_LEASE_NLL_AND_SECURITY_PASS",
 "original_Qwen_28_layer_CPU_C_native_rows_reparsed":8,
 "original_source_frozen_Qv16_static_gold_NLL_exact":8,
 "original_2_worker_request_counts":{"worker0":w0,"worker1":w1},
 "exclusive_inflight_global_KV_and_WT_peak":peak_active,
 "no_precision_profile_epoch_mutation_mid_request":True,
 "unique_sequential_precision_epochs":len(set(epochs)),
 "cache_misses":miss,"cache_evictions":drop,
 "native_inactive_cache_peak_MiB":peak/1048576,
 "guard_lock_wait_max_ms":max(waits)/1e6,
 "guard_lock_wait_median_ms":statistics.median(waits)/1e6,
 "guarded_forward_median_ms":statistics.median(held)/1e6,
 "Qv19_model_poisoned":bool(poison),
 "model_original_source_SHA256":ORIG_SHA,
 "derived_real_Apple_arm64_C_binary_SHA256":NATIVE_SHA,
 "Ed25519_local_dev_signed_derived_binary_envelope_SHA256":SIGN_SHA,
 "signed_original_Qv17_inactive_packed_cache_SHA256":PACK_SHA,
 "original_gold_NLL_scorer_unchanged_SHA256":SCORE_SHA,
 "real_C_log_SHA256":sha(LOG),
 "first_original_C_main_receipt_SHA256":sha(RUN/"PUBLIC_AGGREGATE.json"),
 "local_dev_signing_only_not_production_CA":True,
 "conservative_serial_forward_only_not_true_parallel_logits":True,
 "no_shared_KV_race_detected_in_serialized_8_case_scope":True,
 "no_other_unguarded_command_entrypoints_certified":True,
 "no_model_checkpoint_source_or_immutable_operator_release_changed":True,
 "original_case_text_tokens_individual_NLL_not_exported":True}
if __name__=="__main__":
 result=audit()
 out=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023/INDEPENDENT_PUBLIC_AUDIT.json")
 if out.exists():raise FileExistsError("append-only independent Qv19 audit receipt exists")
 data=(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
 with out.open("xb") as f:f.write(data)
 print("QV19_SECOND_INDEPENDENT_NATIVE_AUDIT_PASS",json.dumps({
  "native_original_C_rows":8,"both_workers":result["original_2_worker_request_counts"],
  "global_KV_peak":result["exclusive_inflight_global_KV_and_WT_peak"],
  "stable_epochs":result["unique_sequential_precision_epochs"],
  "source_static_NLL_parity":8,"audit_SHA256":hashlib.sha256(data).hexdigest()},sort_keys=True),flush=True)
