#!/usr/bin/env python3
"""Qv19 real original Apple Qwen28 C safety negative test.
After independently verifying the LOCAL DEVELOPMENT signature, intentionally
omit BEGLIN_QV19_ENABLE_EXCLUSIVE_MODEL_REQUESTS. The new native mode MUST
fail closed before a single gold score; originals and operator are unchanged.
"""
import hashlib,importlib.util,json,os,pathlib,subprocess,time
REPO=pathlib.Path(__file__).resolve().parents[2]
SCRIPT=REPO/"tools/qv19_original_native/attest_run_real_qv19.py"
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv19-native-20261023")
OUT=LAB/"negative_missing_gate"
BIN=LAB/"build_v1/beglin_qv19_original_C_exclusive_lease"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023/independent_arc_validation48")
Q17=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023/owner_only_artifacts")
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
BIN_SHA="c333c5a95aca340e59352c7e0e716521b3f57028002a53e307a9f962a2c654c2"
PACK_SHA="49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(262144),b""):h.update(x)
 return h.hexdigest()
def save(p,data):
 with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def main():
 if OUT.exists():raise FileExistsError("Qv19 missing-gate native evidence immutable")
 if sha(BIN)!=BIN_SHA:raise RuntimeError("different Qv19 original C native")
 spec=importlib.util.spec_from_file_location("qv19_signed_native_verifier",str(SCRIPT))
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 attested=mod.verify_signed()
 pack=Q17/"alternate40_Qv8_random73_qng64_v1.pack"
 if sha(pack)!=PACK_SHA:raise RuntimeError("old signed pack altered")
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_","BEGLIN_Q_LAB_","BEGLIN_QV"))}
 env.update({
 "VECLIB_MAXIMUM_THREADS":"1","Q4_THREADS":"2","Q4_POOL_SPIN":"0",
 "QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
 "QWEN_HF_CONFIG":str(MODEL/"config.json"),
 "QWEN_SAFETENSORS_MAXSEQ":"256",
 "QWEN_BASE":str(Q16/"dynamic"),
 "QWEN_PROMPT":str(Q16/"dynamic/prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
 "BEGLIN_QV2_NATIVE_ASYNC":"0",
 "BEGLIN_Q_LAB_N8_ROLES_FILE":str(Q16/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_SOURCE":str(Q16/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_RANDOM":str(Q16/"masks/random73.txt"),
 "BEGLIN_QV15_CACHE_MIB":"96","BEGLIN_QV16_PREWARM":"0",
 "BEGLIN_QV17_PACK_LOAD":str(pack),
 "BEGLIN_QV17_PACK_SHA256":PACK_SHA})
 if "BEGLIN_QV19_ENABLE_EXCLUSIVE_MODEL_REQUESTS" in env:
  raise RuntimeError("missing opt-in case was incorrectly enabled")
 os.umask(0o077);OUT.mkdir(parents=True,mode=0o700)
 start=time.monotonic()
 p=subprocess.run([str(BIN),"qv19_guarded_choice","1"],cwd=str(QWEN),
  env=env,capture_output=True,text=True,timeout=300)
 elapsed=time.monotonic()-start
 log=OUT/"real_Qwen28_C_native_missing_optin.log"
 save(log,(p.stdout+"\n"+p.stderr).encode())
 reason="FATAL: Qv19 guarded original C requires explicit original C research gate"
 if p.returncode==0 or reason not in p.stderr or "[qv5-choice]" in p.stderr or "qv5_choice NLL" in p.stdout:
  raise RuntimeError("Qv19 original C didn't reject missing model lease gate: exit"+str(p.returncode)+" stderr="+p.stderr[-800:])
 if sha(pack)!=PACK_SHA:raise RuntimeError("original pack was changed")
 doc={"schema":"BEGLIN-Qv19-actual-original-C-missing-request-lifetime-optin-negative/1",
 "status":"REAL_APPLE_QWEN28_QV19_MISSING_EXCLUSIVE_LEASE_GATE_DENIED_BEFORE_ANY_MODEL_SCORING",
 "native_Qv19_derived_binary_SHA256":BIN_SHA,
 "local_dev_signed_new_binary_verified":True,
 "local_dev_binary_envelope_SHA256":attested["dev_signed_envelope_sha256"],
 "native_process_exit_code":p.returncode,
 "native_gold_scoring_completed":0,
 "model_lease_missing_gate_rejected":True,
 "original_Qv17_signed_pack_SHA256_unchanged":PACK_SHA,
 "Qv19_negative_native_C_log_SHA256":sha(log),
 "native_denial_wall_s":elapsed,
 "no_main_model_or_immutable_operator_release_changed":True}
 b=(json.dumps(doc,sort_keys=True,indent=2)+"\n").encode()
 save(OUT/"PUBLIC_AGGREGATE.json",b)
 print("QV19_REAL_NATIVE_FAIL_CLOSED_GATE_PASS",json.dumps({"exit":p.returncode,
  "no_gold_scoring":True,"elapsed_s":round(elapsed,3),
  "receipt_sha256":sha(OUT/"PUBLIC_AGGREGATE.json")},sort_keys=True),flush=True)
if __name__=="__main__":main()
