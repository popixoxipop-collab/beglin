#!/usr/bin/env python3
"""Qv17 direct native checksum fail-closed regression. Intentionally calls the
research C binary OUTSIDE the trusted Ed25519 bootstrap with a wrong SHA-256.
The binary must still refuse the 83MiB pack before inference; no model output.
Owner-only original artifact, signer, native code and Tailnet release untouched.
"""
import hashlib,json,os,pathlib,psutil,subprocess,time
Q17=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
SOURCE=Q16/"independent_arc_validation48"
BIN=Q17/"native_persisted_C_v2/beglin_qv17_signed_packed_cache"
PACK=Q17/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
QW=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
REPORT=Q17/"negative_native_sha_gate"
PIN_BIN="16085378788b64b229f321b49c11540e286edffbff832db4e41fbf8d2a87643b"
PIN_PACK="49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b"
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(262144),b""):h.update(chunk)
 return h.hexdigest()
def put(path,raw):
 path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with path.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 path.chmod(0o600)
def run():
 if REPORT.exists():raise FileExistsError("native negative gate evidence already exists")
 if sha(BIN)!=PIN_BIN or sha(PACK)!=PIN_PACK:raise RuntimeError("original untouched source drift")
 os.umask(0o077);REPORT.mkdir(mode=0o700)
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_","BEGLIN_Q_LAB_","BEGLIN_QV"))}
 env.update({
 "VECLIB_MAXIMUM_THREADS":"1",
 "QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
 "QWEN_HF_CONFIG":str(MODEL/"config.json"),
 "QWEN_SAFETENSORS_MAXSEQ":"256",
 "QWEN_BASE":str(SOURCE/"dynamic"),
 "QWEN_PROMPT":str(SOURCE/"dynamic/prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":"mixed",
 "BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
 "BEGLIN_QV2_NATIVE_ASYNC":"0",
 "BEGLIN_Q_LAB_N8_ROLES_FILE":str(SOURCE/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_SOURCE":str(SOURCE/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_RANDOM":str(SOURCE/"masks/random73.txt"),
 "BEGLIN_QV15_CACHE_MIB":"96",
 "BEGLIN_QV16_PREWARM":"0",
 "BEGLIN_QV17_PACK_LOAD":str(PACK),
 "BEGLIN_QV17_PACK_SHA256":"0"*64
 })
 t=time.monotonic()
 cp=subprocess.run([str(BIN),"qv14_dynamic_choice","1"],
  cwd=str(QW),env=env,capture_output=True,timeout=600,text=True)
 elapsed=time.monotonic()-t
 log=REPORT/"native_original_C_wrong_SHA_gate.log"
 put(log,(cp.stdout+"\n"+cp.stderr).encode())
 reason="FATAL: Qv17 packed cache pinned packed SHA256 mismatch"
 if cp.returncode==0 or reason not in cp.stderr or "[qv5-choice]" in cp.stderr or "qv5_choice NLL" in cp.stdout:
  raise RuntimeError("FAIL_CLOSED_DENIAL_NOT_OBSERVED exit="+str(cp.returncode)+" stderr="+cp.stderr[-1000:])
 if sha(PACK)!=PIN_PACK:raise RuntimeError("native rejection modified original packed asset")
 result={"schema":"BEGLIN-Qv17-native-Apple-C-wrong-SHA256-direct-invocation-denial/1",
 "status":"REAL_NATIVE_WRONG_EXPECTED_SHA256_FAIL_CLOSED_BEFORE_ANY_GOLD_SCORE",
 "run_exit_code":cp.returncode,
 "intentional_wrong_expected_pack_SHA256":"0"*64,
 "original_pack_SHA256":PIN_PACK,
 "derived_original_Qwen28_C_executable_SHA256":PIN_BIN,
 "fatal_diagnostic":reason,
 "original_gold_C_evaluations_completed":0,
 "native_wrong_digest_rejected":True,
 "native_process_wall_s":elapsed,
 "native_private_log_SHA256":sha(log),
 "artifact_digest_unchanged":True,
 "test_direct_native_bypass_not_a_signature_verification_test":True,
 "local_dev_signer_private_key_never_exposed":True,
 "no_operating_release_change":True}
 p=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv13-github-commit-20261023/docs/research/BEGLIN_QV17_NATIVE_WRONG_SHA_FAIL_CLOSED_PUBLIC_20261023.json")
 content=(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
 put(p,content)
 print("QV17_NEGATIVE_REAL_NATIVE_SHA_GUARD_PASS",json.dumps({"exit":cp.returncode,
 "elapsed_s":round(elapsed,3),"log_SHA256":sha(log),
 "report_SHA256":hashlib.sha256(content).hexdigest(),
 "zero_native_gold_scoring":True},sort_keys=True),flush=True)
if __name__=="__main__":run()
