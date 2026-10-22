#!/usr/bin/env python3
"""Original BEGLIN Qwen28 C with exactly TWO n8 tensors instead of 28.

Copy SHA-authenticated Qv5 original mixed research C. Only semantic change
is if(total_n8!=28) -> !=2; two log messages also adapted. All original
NEON and Q4/n8 encoding kernels and model loader weights are untouched.
Singleton and baseline original binary remain separately source-pinned.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess
R=pathlib.Path(__file__).resolve().parents[2]
PRE=R/"configs/research/BEGLIN_QV12_TWO_N8_INTERACTION_PRELOCK_20261023.json"
Q5=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
E=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
Q11=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv11-trueintervention-20261023")
OLD=Q5/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/qv5_mixed.c"
BUILD=Q5/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/mixed_build_receipt.json"
OLD_BASE=Q5/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base"
OLD_SINGLE=Q11/"lab_local/qv11_single_original_C_20261023/counterfactual_C_1n8/beglin_qv11_singleton_1n8"
PAIR=R/"lab_local/qv12_twotensor_20261023/native_2n8_original_C"
PRE_SHA="a565bdec341d675ce52a37132b06355760c2fef96b8e4329c17508a237e4374f"
BASE_SHA="408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"
SINGLE_SHA="a054e68789fd6bc20a4bd815da5108b63361b14227b1fc4ab9e68a3b7b9a2711"
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,data):
 with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def main():
 if PAIR.exists():raise FileExistsError("Qv12 pair native C already built; preserve immutable receipt")
 if sha(PRE.read_bytes())!=PRE_SHA:raise RuntimeError("Qv12 prereg changed")
 original=OLD.read_bytes()
 cfg=json.loads(BUILD.read_text())
 if sha(original)!=cfg["new_C_SHA256"] or cfg["actual_engine_binary_SHA256"]!="f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221":
  raise RuntimeError("audited original Qv5 mixed native C source authority changed")
 if sha(OLD_BASE.read_bytes())!=BASE_SHA or sha(OLD_SINGLE.read_bytes())!=SINGLE_SHA:
  raise RuntimeError("original 0n8/1n8 C research binary source identity drift")
 s=original.decode()
 for old,new in (
  ("if(total_n8!=28)","if(total_n8!=2)"),
  ("expected exactly 28 n8 tensor roles","expected exactly 2 n8 tensor roles"),
  ("validated mixed 28/196 n8 tensor role-layer mask","validated mixed 2/196 n8 tensor role-layer mask")):
  if s.count(old)!=1:raise RuntimeError("unambiguous cardinality native source contract missing "+old)
  s=s.replace(old,new,1)
 if "BEGLIN_Q_LAB_N8_ROLES_FILE" not in s or "masked_choice_qv5" not in s:
  raise RuntimeError("real native Qwen original C input/masked choice kernel not present")
 os.umask(0o077);PAIR.mkdir(parents=True,mode=0o700)
 cp=PAIR/"qv12_original_qv5_mixed_only2n8.c"
 save(cp,s.encode())
 obj=PAIR/"qv12_source_cardinality2.o";exe=PAIR/"beglin_qv12_two_tensor_n8"
 subprocess.run(["clang","-O3","-w","-I",str(E),"-c",str(cp),"-o",str(obj)],check=True,timeout=150)
 parts=[E/"lab_local/qng64_n8_repacked_full_20261022"/x for x in
         ("qng64_original_aliased.o","n8_repacked_simd.o","n8_repacked_dispatch.o")]
 extra=sorted(x for x in (E/".build").glob("*.o") if x.name not in ("qwen_infer.o","qng64_dense.o"))
 if len(extra)!=20 or not all(x.is_file() for x in parts):
  raise RuntimeError("original Apple NEON/Q4/n8 kernel object closure missing")
 subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in parts+extra]+
    ["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],check=True,timeout=150)
 exe.chmod(0o700)
 evidence={"schema":"BEGLIN-Qv12-authentic-Qwen28-native-C-two-single-bit-projections/1",
   "status":"ORIGINAL_QV5_28LAYER_2N8_194Q4_REAL_C_BINARY_COMPILED",
   "prelock_sha256":PRE_SHA,
   "original_Qv5_source_C_sha256":sha(original),
   "original_Qv5_full_Q4_binary_sha256":BASE_SHA,
   "original_Qv11_one_N8_binary_sha256":SINGLE_SHA,
   "new_Qv12_two_N8_source_C_sha256":sha(s.encode()),
   "new_Qv12_two_N8_binary_sha256":sha(exe.read_bytes()),
   "semantic_C_change":"cardinality mask check total_n8 28 -> 2",
   "only_2_C_diagnostic_message_changes":True,
   "native_original_NEON_8bit_and_Q4_kernel_objects_unchanged":True,
   "real_full_model_decoder_layers":28,"projection_roles_total":196,
   "actual_Q4_if_two_N8":194,"both_N8_required":2,
   "source_HF_Qwen_rev":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
   "original_model_weights_not_changed":True,
   "main_or_operator_runtime_not_changed":True}
 data=(json.dumps(evidence,sort_keys=True,indent=2)+"\n").encode()
 save(PAIR/"build_original_native_two_n8_receipt.json",data)
 print("QV12_TRUE_2N8_ORIGINAL_C_QWEN_NATIVE_ENGINE_COMPILED",json.dumps({
    "binary_sha":evidence["new_Qv12_two_N8_binary_sha256"],
    "base_sha":BASE_SHA,"one_sha":SINGLE_SHA,
    "receipt_sha":sha(data)},sort_keys=True),flush=True)
if __name__=="__main__":main()
