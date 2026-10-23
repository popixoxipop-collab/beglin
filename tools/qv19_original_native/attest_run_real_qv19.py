#!/usr/bin/env python3
"""Qv19 research ONLY: independently pin a newly built real original Apple C,
sign derived executable with existing LOCAL DEVELOPMENT Ed25519 key, then run
8 SHA-frozen Qv16 questions through two real pthreads and audit original NLL.
No model/checkpoint, Qv17 packed asset, GitHub main or G6 release is changed.
"""
import argparse,hashlib,importlib.util,json,math,os,pathlib,re,struct,subprocess,time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey,Ed25519PublicKey
REPO=pathlib.Path(__file__).resolve().parents[2]
SRC=REPO/"tools/qv19_original_native"
OUT=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023/build_v1")
RUN=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023/native_first8")
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work")
Q17=BASE/"beglin-qv17-signed-packed-cache-20261023"
Q16=BASE/"beglin-qv16-cost-aware-prewarm-20261023"
NATIVE=OUT/"beglin_qv19_original_C_exclusive_lease"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
ORIGINAL=Q17/"native_persisted_C_v2/qv17_original_Qwen28_persisted_cache.c"
SIGNER=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv13-github-commit-20261023/tools/qv17_local_dev_signing.py")
ENVELOPE=OUT/"QV19_DEV_SIGNED_BUILD.json"
DETACHED=OUT/"QV19_DEV_SIGNED_BUILD.sig"
PIN={
 "parent":"0c9d797c0066fdb1f8ca5972bdf7611ae75431050d0d55c0f0fdef0653f6bdaa",
 "original_NLL":"10c84007027d15a9327c15ec687c2327e19390de65f03bf9d95aee4bd4c9fe72",
 "lease_c":"5788e0f4d2afbf592f93e7ca2bc6e1cbe0b3f5ecdfcaace3742b157842ad3b93",
 "lease_h":"5fc29c9e25bf6148166f823eb1587c90ebe41d6cfe6d7a6e8e5f4d711398ec8d",
 "bridge":"aab974fa09b4ff65d2559aeaa1aff2bb9f03dcfb7743d51dbc11b41dd3363cd7",
 "native":"c333c5a95aca340e59352c7e0e716521b3f57028002a53e307a9f962a2c654c2",
 "signer":"544cfccedbdddb5aad14eee1cb2bc6e0df60b319a8d0f5c369261fee6350c832",
 "pack":"49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b",
 "source_lock":"a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296",
 "public_lock":"ac5223bdf25b6cf607cb3a589027d1a406737af7341e8f742231155579c3b26a"}
ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
LEASE=re.compile(r"^\[QV19-LEASE\] window (\d+) worker (\d+) epoch_enter (\d+) epoch_exit (\d+) profile (source_qv8|random73) changed (\d+) wait_ns (\d+) guarded_ns (\d+) snapshot_stable 1$",re.M)
TOTAL=re.compile(r"^\[QV19-MODEL-LEASE-TOTAL\] requests (\d+) workers 2 worker0 (\d+) worker1 (\d+) max_active (\d+) epoch (\d+) poisoned (\d+) guarded_forward_serial 1$",re.M)
SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
CACHE=re.compile(r"^\[QV15-CACHE-TOTAL\] hits (\d+) misses (\d+) drops (\d+) bytes (\d+) peak_bytes (\d+) cap_bytes (\d+)$",re.M)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(262144),b""):h.update(c)
 return h.hexdigest()
def must(c,why):
 if not c:raise RuntimeError("QV19_NATIVE_SIGN_OR_RUN_FAIL_CLOSED_"+why)
def canonical(x):return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
def new(p,contents):
 with p.open("xb") as f:f.write(contents);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def get_signer():
 must(sha(SIGNER)==PIN["signer"],"dev_signer_code")
 spec=importlib.util.spec_from_file_location("Qv19_pinned_local_legacy_dev_signer",str(SIGNER))
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 mod.verify_roots()
 return mod
def provenance():
 must(sha(ORIGINAL)==PIN["parent"],"parent_native_C_original")
 must(sha(SRC/"qv19_request_lease.c")==PIN["lease_c"] and
      sha(SRC/"qv19_request_lease.h")==PIN["lease_h"] and
      sha(SRC/"qv19_original_c_bridge.inc")==PIN["bridge"],"Qv19_lease_bridge_original_exact")
 receipt=json.loads((OUT/"BUILD_RECEIPT.json").read_text())
 must(receipt["status"]=="REAL_APPLE_NATIVE_DERIVED_C_BINARY_COMPILED_NOT_EVALUATED","unapproved_build_status")
 must(sha(NATIVE)==PIN["native"]==receipt["native_executable_sha256"],"signed_Qv19_native_binary")
 with NATIVE.open("rb") as f: must(f.read(8)==bytes.fromhex("cffaedfe0c000001"),"Apple_arm64_MachO")
 must(sha(OUT/"qv19_original_Qwen28_derived.c")==receipt["derived_C_sha256"],"derived_C")
 must(receipt["original_scoring_source_sha256"]==PIN["original_NLL"],"unchanged_native_NLL")
 baseline=json.loads((Q17/"native_persisted_C_v2/BUILD_RECEIPT.json").read_text())
 objs=baseline["original_23_kernel_objects"]
 must(len(objs)==23 and all(sha(pathlib.Path(o["file"]))==o["sha256"] for o in objs),
      "23_native_Q4_n8_original_objects")
 must(sha(Q16/"independent_arc_validation48/frozen_inputs_private.json")==PIN["source_lock"] and
      sha(Q16/"independent_arc_validation48/frozen_manifest_PUBLIC.json")==PIN["public_lock"],
      "frozen_48_source_only_case_contract")
 must(sha(Q17/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack")==PIN["pack"],
      "original_Qv17_signed_cache_bytes")
 return receipt
def trusted_binding():
 receipt=provenance()
 return {"schema":"BEGLIN-Qv19-REAL-Apple-Qwen28-local-development-Ed25519-model-lease-binding/1",
    "native_sha256":PIN["native"],"derived_C_sha256":receipt["derived_C_sha256"],
    "parent_Qv17_C_SHA256":PIN["parent"],"original_Qwen_gold_NLL_scorer_SHA256":PIN["original_NLL"],
    "lease_implementation_sha256":PIN["lease_c"],"lease_header_sha256":PIN["lease_h"],
    "native_bridge_sha256":PIN["bridge"],
    "Qv17_signed_alternate_pack_SHA256":PIN["pack"],
    "Qv16_frozen_source_only_case_SHA256":PIN["source_lock"],
    "unchanged_original_23_native_objects":True,
    "attestation_scope":"local developer only; not production release signature"}
def sign():
 signer=get_signer();envelope=trusted_binding()
 if ENVELOPE.exists() or DETACHED.exists():raise FileExistsError("Qv19 developer envelope sealed")
 must(signer.PRIVATE.is_file() and not signer.PRIVATE.is_symlink() and
      signer.PRIVATE.stat().st_mode&0o077==0,"owner_only_dev_private")
 key=serialization.load_pem_private_key(signer.PRIVATE.read_bytes(),password=None)
 must(isinstance(key,Ed25519PrivateKey),"Ed25519_private_only")
 pub=key.public_key().public_bytes(encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)
 must(pub==signer.root_public(),"Qv17_local_root_pinned_for_new_binary")
 payload=canonical(envelope);sig=key.sign(payload)
 new(ENVELOPE,payload);new(DETACHED,sig)
 print("QV19_LOCAL_DEV_ATTESTED",json.dumps({"binary_sha":PIN["native"],
    "signed_envelope_sha256":sha(ENVELOPE),"detached_signature_sha256":sha(DETACHED),
    "local_development_only":True},sort_keys=True))
def verify_signed():
 signer=get_signer()
 must(ENVELOPE.is_file() and DETACHED.is_file() and
      not ENVELOPE.is_symlink() and not DETACHED.is_symlink(),"sealed_signature_files")
 Ed25519PublicKey.from_public_bytes(signer.root_public()).verify(DETACHED.read_bytes(),ENVELOPE.read_bytes())
 must(json.loads(ENVELOPE.read_text())==trusted_binding(),"signed_native_source_binary_context")
 trusted=signer.verify(strict_model=True)
 must(trusted["pack_sha256"]==PIN["pack"],"original_Qv17_ed25519_and_original_BF16")
 return {"dev_signed_envelope_sha256":sha(ENVELOPE),"source_and_weight_verified":True}
def run():
 attest=verify_signed()
 if RUN.exists():raise FileExistsError("Qv19 real model receipt immutable")
 os.umask(0o077);RUN.mkdir(parents=True,mode=0o700)
 inputs=Q16/"independent_arc_validation48"
 original=json.loads((inputs/"frozen_inputs_private.json").read_text())["requests"]
 for i,row in enumerate(original[:8]):
  ids=struct.pack("<"+"i"*len(row["prefix_ids"]),*row["prefix_ids"])
  must((inputs/"dynamic/eval/ppl_work"/f"w{i}.i32").read_bytes()==ids,"frozen_input_binary_"+str(i))
  opts=(" ".join(str(row["suffix_ids"][c]) for c in "ABCD")+" "+str("ABCD".index(row["correct"]))+"\n").encode()
  must((inputs/"dynamic/eval/ppl_work"/f"w{i}.choices").read_bytes()==opts,"frozen_gold_options_"+str(i))
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_","BEGLIN_QV","BEGLIN_Q_LAB_"))}
 env.update({
  "VECLIB_MAXIMUM_THREADS":"1","Q4_THREADS":"2","Q4_POOL_SPIN":"0",
  "QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),"QWEN_HF_CONFIG":str(MODEL/"config.json"),
  "QWEN_SAFETENSORS_MAXSEQ":"256","QWEN_BASE":str(inputs/"dynamic"),
  "QWEN_PROMPT":str(inputs/"dynamic/prompt_ids.i32"),
  "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
  "BEGLIN_QV2_NATIVE_ASYNC":"0","BEGLIN_Q_LAB_N8_ROLES_FILE":str(inputs/"masks/source_qv8.txt"),
  "BEGLIN_QV14_MASK_SOURCE":str(inputs/"masks/source_qv8.txt"),
  "BEGLIN_QV14_MASK_RANDOM":str(inputs/"masks/random73.txt"),
  "BEGLIN_QV15_CACHE_MIB":"96","BEGLIN_QV16_PREWARM":"0",
  "BEGLIN_QV17_PACK_LOAD":str(Q17/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"),
  "BEGLIN_QV17_PACK_SHA256":PIN["pack"],
  "BEGLIN_QV19_ENABLE_EXCLUSIVE_MODEL_REQUESTS":"1"})
 t=time.monotonic()
 proc=subprocess.run([str(NATIVE),"qv19_guarded_choice","8"],
        cwd=str(QWEN),env=env,capture_output=True,text=True,timeout=750)
 wall=time.monotonic()-t
 logfile=RUN/"real_original_C_Qv19.log";new(logfile,(proc.stdout+"\n"+proc.stderr).encode())
 if proc.returncode:raise RuntimeError("real Qv19 native exited "+str(proc.returncode)+"; "+proc.stderr[-2200:])
 rows=ROW.findall(proc.stderr);leases=LEASE.findall(proc.stderr)
 totals=TOTAL.findall(proc.stderr);summ=SUM.findall(proc.stdout);cache=CACHE.findall(proc.stderr)
 must(len(rows)==len(leases)==8 and len(totals)==len(summ)==len(cache)==1,"all_real_native_logs")
 ref=json.loads((Q16/"native48/prewarm1/receipt_private.json").read_text())
 byrow={int(r[0]):r for r in rows};bylease={int(r[0]):r for r in leases}
 must(set(byrow)==set(bylease)==set(range(8)),"unique_original_8_inputs")
 epochs=set();workers=set();correct=0;loss=[]
 for i in range(8):
  row=byrow[i];z=bylease[i];r=original[i]
  nll=float(row[1]);right=int(int(row[2])==int(row[3]))
  must(nll==ref["gold_fullvocab_token_NLL"][i] and right==ref["correct_each"][i],
       "original_C_static_gold_NLL_bit_exact_"+str(i))
  must(int(row[4])==r["prefix_len"] and int(row[2])=="ABCD".index(r["correct"]),
       "input_and_label_index_"+str(i))
  must(z[4]==("source_qv8" if r["prefix_len"]<85 else "random73"),"real_selected_precision_"+str(i))
  must(int(z[2])==int(z[3]) and int(z[2])>0 and int(z[7])>0,"epoch_and_guarded_duration")
  epochs.add(int(z[2]));workers.add(int(z[1]));correct+=right;loss.append(nll)
 must(epochs==set(range(1,9)) and workers=={0,1},"both_real_native_pthreads_and_stable_epochs")
 n,worker0,worker1,maxactive,epoch,poison=map(int,totals[0])
 must(n==8 and worker0>0 and worker1>0 and maxactive==1 and epoch==8 and poison==0,
      "one_inflight_native_Qwen28_forward_and_KV_owner")
 h,miss,drops,now,peak,cap=map(int,cache[0])
 must(miss==drops==0 and peak<=cap==96*1048576,"persistent_cache_hardlimit")
 must(int(summ[0][1])==8 and int(summ[0][2])==correct and abs(float(summ[0][3])-sum(loss)/8)<1e-8,
      "gold_NLL_totals")
 receipt={"schema":"BEGLIN-Qv19-real-Apple-Qwen28-2-pthread-exclusive-model-lifetime/1",
   "status":"REAL_ORIGINAL_NATIVE_QWEN28_MUTEX_LEASE_8_OF_8_SOURCE_STATIC_EXACT",
   "real_native_EOE_workers":2,"original_gold_NLL_exact_cases":8,"correct":correct,
   "mean_gold_NLL":sum(loss)/8,
   "full_precision_switch_forward_logit_exclusive":True,
   "shared_KV_forward_inflight_max":maxactive,"unique_stable_epoch_count":len(epochs),
   "poison":poison,"cache_misses":miss,"cache_drops":drops,"peak_cache_mib":peak/1048576,
   "source_original_Qv17_C_sha256":PIN["parent"],
   "derived_native_exe_sha256":PIN["native"],
   "signed_dev_binary_envelope_sha256":attest["dev_signed_envelope_sha256"],
   "original_C_log_sha256":sha(logfile),"actual_native_wall_s":wall,
   "same_process_actual_simultaneous_forwards":False,
   "model_source_Qv17_signed_asset_and_operating_release_unchanged":True}
 evidence=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode()
 new(RUN/"PUBLIC_AGGREGATE.json",evidence)
 print("QV19_REAL_NATIVE_APPLE_PTHREAD_LEASE_PASS",json.dumps({
    "n":8,"correct":correct,"mean_gold_NLL":sum(loss)/8,
    "workers":2,"peak_simultaneous_forward":maxactive,
    "epochs":sorted(epochs),"wall_s":round(wall,3),"evidence_SHA256":sha(RUN/"PUBLIC_AGGREGATE.json")},sort_keys=True),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["sign","verify","run"])
 a=p.parse_args()
 if a.mode=="sign":sign()
 elif a.mode=="verify":print("QV19_DEV_SIGNED_VERIFY_PASS",json.dumps(verify_signed(),sort_keys=True))
 else:run()
