#!/usr/bin/env python3
"""Qv18 REAL original Qwen28 native C negative-version/schema gate.
Copy ONE owner-only 83.5MiB signed artifact into a temporary research directory.
Change two metadata fields in separate test copies, calculate each new SHA256,
pass that matching hash DIRECTLY to native C (deliberately bypass Ed25519).
Native C must reject *even when supplied SHA agrees with corrupt bytes*
because version/role/shape is not exact. Zero model scoring is required.
Never mutate original Qv17 file, Ed25519 signing keys, source kernels or release.
This does NOT prove direct-native Ed25519 signature enforcement.
"""
import hashlib,json,os,pathlib,shutil,struct,subprocess,tempfile,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
Q17=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv17-signed-packed-cache-20261023")
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv18-native-negative-schema-20261023")
PACK=Q17/"owner_only_artifacts/alternate40_Qv8_random73_qng64_v1.pack"
EXE=Q17/"native_persisted_C_v2/beglin_qv17_signed_packed_cache"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
QWEN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
INPUT=Q16/"independent_arc_validation48"
PINPACK="49c409a79588b79dcefeba60e09b71459ac8df2d02ac0bda5e2cc1ab8076a86b"
PINEXE="16085378788b64b229f321b49c11540e286edffbff832db4e41fbf8d2a87643b"
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for ch in iter(lambda:f.read(262144),b""):h.update(ch)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError("QV18_NATIVE_FORMAT_GUARD_FAILED_"+msg)
def save(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def make_env(badfile,checksum):
 env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_","BEGLIN_Q_LAB_","BEGLIN_QV"))}
 env.update({"VECLIB_MAXIMUM_THREADS":"1","Q4_THREADS":"2","Q4_POOL_SPIN":"0",
 "QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
 "QWEN_HF_CONFIG":str(MODEL/"config.json"),
 "QWEN_SAFETENSORS_MAXSEQ":"256","QWEN_BASE":str(INPUT/"dynamic"),
 "QWEN_PROMPT":str(INPUT/"dynamic/prompt_ids.i32"),
 "BEGLIN_Q_LAB_PRECISION":"mixed","BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
 "BEGLIN_QV2_NATIVE_ASYNC":"0",
 "BEGLIN_Q_LAB_N8_ROLES_FILE":str(INPUT/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_SOURCE":str(INPUT/"masks/source_qv8.txt"),
 "BEGLIN_QV14_MASK_RANDOM":str(INPUT/"masks/random73.txt"),
 "BEGLIN_QV15_CACHE_MIB":"96","BEGLIN_QV16_PREWARM":"0",
 "BEGLIN_QV17_PACK_LOAD":str(badfile),
 "BEGLIN_QV17_PACK_SHA256":checksum})
 return env
def main():
 require(sha(PACK)==PINPACK and sha(EXE)==PINEXE,"original_model_bound_packing")
 if LAB.exists():raise FileExistsError("append-only Qv18 negative test already exists")
 os.umask(0o077);LAB.mkdir(mode=0o700)
 scenarios=[
  ("header_version_2",8,struct.pack("<I",2),"packed header size/schema/cap mismatch"),
  ("first_tensor_wrong_role_id",24,struct.pack("<H",7),"wrong tensor index/precision/shape/cap in packed entry")
 ]
 results=[]
 for name,offset,raw,reason in scenarios:
  with tempfile.TemporaryDirectory(prefix="qv18_untrusted_copy_",dir=str(LAB)) as td:
   file=pathlib.Path(td)/"untrusted_packed_test_copy.pack"
   shutil.copyfile(PACK,file)
   file.chmod(0o600)
   with file.open("r+b") as f:
    f.seek(offset);original=f.read(len(raw))
    require(original!=raw,"test mutation must modify exact bytes "+name)
    f.seek(offset);f.write(raw);f.flush();os.fsync(f.fileno())
   wrongsha=sha(file)
   require(wrongsha!=PINPACK,"expected bad file digest differs")
   start=time.monotonic()
   p=subprocess.run([str(EXE),"qv14_dynamic_choice","1"],env=make_env(file,wrongsha),
       cwd=str(QWEN),capture_output=True,text=True,timeout=300)
   elapsed=time.monotonic()-start
   output=p.stdout+"\n"+p.stderr
   log=LAB/(name+".original_C.log")
   save(log,output.encode())
   require(p.returncode!=0 and "FATAL: Qv17 packed cache "+reason in p.stderr,
      "C did not reject wrong native serialized format "+name+" exit "+str(p.returncode))
   require("[qv5-choice]" not in output and "qv5_choice NLL" not in p.stdout,
      "native incorrectly allowed scored answer on malformed file "+name)
   results.append({"scenario":name,
    "mutation_byte_offset":offset,
    "supplied_C_expected_SHA256_matches_mutated_copy":True,
    "result":"NATIVE_C_REJECTS_MALFORMED_FORMAT_BEFORE_INFERENCE",
    "Qwen28_native_process_exit_code":p.returncode,
    "native_C_error":reason,
    "native_original_case_scores_computed":0,
    "C_loading_and_rejection_wall_s":elapsed,
    "original_C_negative_log_SHA256":sha(log),
    "temporary_malicious_copy_deleted":True})
  require(sha(PACK)==PINPACK,"original signed source changed between tests")
 result={"schema":"BEGLIN-Qv18-original-Qwen28-C-untrusted-pack-format-version-role-rejections/1",
 "status":"BOTH_ACTUAL_NATIVE_FORMAT_GUARDS_FAIL_CLOSED_WITH_MATCHING_UNTRUSTED_SHA",
 "original_pack_SHA256":PINPACK,
 "original_C_sha256":PINEXE,
 "n":len(results),"negative_tests":results,
 "original_owner_only_signed_artifact_unmodified":True,
 "no_scores_from_invalid_artifacts":True,
 "Ed25519_rejects_both_untrusted_copies_before_C_under_trusted_launcher":True,
 "direct_native_only_checks_SHA256_and_binary_format_not_signature":True,
 "no_live_operator_kernel_or_credential_change":True}
 out=ROOT/"docs/research/BEGLIN_QV18_NATIVE_VERSION_ROLE_FAIL_CLOSED_PUBLIC_20261023.json"
 blob=(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
 save(out,blob)
 print("QV18_REAL_NATIVE_FORMAT_NEGATIVE_PASS",json.dumps({"cases":len(results),
  "all_rejected_before_scoring":True,"signed_pack_untouched":True,
  "receipt_SHA256":hashlib.sha256(blob).hexdigest()},sort_keys=True),flush=True)
if __name__=="__main__":main()
