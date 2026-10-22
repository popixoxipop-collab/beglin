#!/usr/bin/env python3
"""Isolated native Qwen28 counterfactual: original mixed C, one n8 instead of 28."""
import hashlib,json,os,pathlib,subprocess
R=pathlib.Path(__file__).resolve().parents[2]
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
E=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
P=R/"configs/research/BEGLIN_QV11_REAL_ONE_N8_INTERVENTION_PRELOCK_20261023.json"
C=OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/qv5_mixed.c"
B=OLD/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/mixed_build_receipt.json"
OUT=R/"lab_local/qv11_single_original_C_20261023/counterfactual_C_1n8"
def sha(x):return hashlib.sha256(x).hexdigest()
def save(p,b):
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def main():
 if OUT.exists():raise FileExistsError("already built, refuse overwrite")
 if sha(P.read_bytes())!="2cd480dc4c72b5383014a1325d350e51a0b47b1335ec0013a6d069773b546a39":raise RuntimeError("prelock drift")
 old=C.read_bytes(); rec=json.loads(B.read_text())
 if sha(old)!=rec["new_C_SHA256"]:raise RuntimeError("original C source SHA changed")
 s=old.decode()
 for a,b in (("if(total_n8!=28)","if(total_n8!=1)"),("expected exactly 28 n8 tensor roles","expected exactly 1 n8 tensor role"),("validated mixed 28/196 n8 tensor role-layer mask","validated mixed 1/196 n8 tensor role-layer mask")):
  if s.count(a)!=1:raise RuntimeError("unambiguous native C source patch absent "+a)
  s=s.replace(a,b,1)
 os.umask(0o077);OUT.mkdir(parents=True,mode=0o700)
 source=OUT/"qv11_native_1n8_original_mixed_copy.c"
 save(source,s.encode())
 obj=OUT/"qv11_native_singleton.o";exe=OUT/"beglin_qv11_singleton_1n8"
 subprocess.run(["clang","-O3","-w","-I",str(E),"-c",str(source),"-o",str(obj)],check=True,timeout=150)
 links=[E/"lab_local/qng64_n8_repacked_full_20261022"/x for x in ("qng64_original_aliased.o","n8_repacked_simd.o","n8_repacked_dispatch.o")]
 extra=sorted(x for x in (E/".build").glob("*.o") if x.name not in ("qwen_infer.o","qng64_dense.o"))
 if len(extra)!=20 or any(not x.is_file() for x in links):raise RuntimeError("original native C object closure changed")
 subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in links+extra]+["-o",str(exe),"-framework","Accelerate","-lpthread","-lm"],check=True,timeout=150)
 exe.chmod(0o700)
 proof={"schema":"BEGLIN-Qv11-original-native-28layers-1N8-counterfactual-build/1","status":"QWEN28_SINGLE_1N8_195Q4_ORIGINAL_KERNEL_BINARY_BUILT",
  "original_Qv5_C_sha256":sha(old),"original_Qv5_binary_sha256":rec["actual_engine_binary_SHA256"],
  "source_patch_sha256":sha(s.encode()),"new_binary_sha256":sha(exe.read_bytes()),
  "one_semantic_source_cardinality_modification":True,"two_log_message_changes":True,
  "same_original_Apple_NEON_quantized_weight_kernel_object_files":True,"production_untouched":True}
 raw=(json.dumps(proof,sort_keys=True,indent=2)+"\n").encode()
 save(OUT/"build_evidence.json",raw)
 print("QV11_ORIGINAL_28LAYER_REAL_SINGLETON_C_BUILT",json.dumps({"binary_sha":proof["new_binary_sha256"],"public_parent_binary_sha":proof["original_Qv5_binary_sha256"],"receipt_sha":sha(raw)}),flush=True)
if __name__=="__main__":main()
