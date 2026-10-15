#!/usr/bin/env python3
"""P18 D50 model-copy authority: byte-exact original source, SHA verified.

Strictly READS XOX original bf16 checkpoint and quantized GGUF/flat MoE.
Use --inspect to report copy completeness WITHOUT hashing big files.
Use --seal ONLY after authorized rsync process has EXITED 0 on both transfers.
Writes two immutable D50 research receipts, no host deletion or promotion.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,re,subprocess
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
BF16=ROOT/"teacher_checkpoint_source_copy"
MOE=ROOT/"native_moe_base_source_copy"
BF16_XOX=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
MOE_XOX=pathlib.Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
BF16_SHARDS={
 "model-00001-of-000004.safetensors":8594887408,
 "model-00002-of-000004.safetensors":8591757448,
 "model-00003-of-000004.safetensors":8590718520,
 "model-00004-of-000004.safetensors":5636263200}
BF16_SMALL=("model.safetensors.index.json","config.json",
 "configuration_deepseek.py","modeling_deepseek.py",
 "tokenizer.json","tokenizer_config.json",
 "tokenization_deepseek_fast.py","generation_config.json")
MOE_FILES={
 "weights_moe/deepseek_moe_f32.bin":14137344,
 "weights_moe/deepseek_moe_af.bin":9814343680,
 "weights_moe/arch_config_moe.txt":372,
 "weights_moe/layout_f32.txt":6695,
 "weights_moe/layout_af.txt":25625}
SOURCE_MODEL_INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
SOURCE_MODEL_CONFIG_SHA="f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"
SOURCE_MODEL_TORCH_SHA="7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c"
def sha(path):
 p=pathlib.Path(path)
 if not p.is_file() or p.is_symlink():raise ValueError("Missing or untrusted source file: "+str(p))
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def remote_sha(root,name):
 p=str(root/name)
 cmd=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8","xox",
      "shasum","-a","256",p]
 out=subprocess.check_output(cmd,text=True,timeout=300).strip()
 match=re.match(r"^([0-9a-f]{64})\s+",out)
 if not match:raise ValueError("XOX remote sha256 output missing for "+name)
 return match.group(1)
def check_state():
 incomplete=[];files=[]
 for name,size in BF16_SHARDS.items():
  p=BF16/name
  if not p.is_file() or p.is_symlink() or p.stat().st_size!=size:
   incomplete.append({"source":"BF16","name":name,"required_bytes":size,
                      "existing_bytes":p.stat().st_size if p.exists() else 0})
  else:files.append(str(p))
 for name in BF16_SMALL:
  p=BF16/name
  if not p.is_file() or p.is_symlink() or p.stat().st_size<50:
   incomplete.append({"source":"BF16","name":name,"existing_bytes":p.stat().st_size if p.exists() else 0})
 for name,size in MOE_FILES.items():
  p=MOE/name
  if not p.is_file() or p.is_symlink() or p.stat().st_size!=size:
   incomplete.append({"source":"MOE","name":name,"required_bytes":size,
                      "existing_bytes":p.stat().st_size if p.exists() else 0})
  else:files.append(str(p))
 return incomplete
def source_digests(parent,remote,names):
 found={}
 for name in names:
  relative=pathlib.Path(name)
  if relative.is_absolute() or ".." in relative.parts:
   raise ValueError("Untrusted source-relative shard path")
  local=parent/relative
  remote_h=remote_sha(remote,name)
  local_h=sha(local)
  if local_h!=remote_h:raise RuntimeError("COPY_SOURCE_SHA_MISMATCH: "+name)
  found[name]={"bytes":local.stat().st_size,"source_XOX_SHA256":remote_h,"copied_D50_SHA256":local_h}
 return found
def seal():
 if not ROOT.is_dir() or ROOT.is_symlink() or BF16.is_symlink() or MOE.is_symlink():
  raise ValueError("Trusted D50 canonical model copy paths missing")
 blocked=check_state()
 if blocked:raise RuntimeError("RESEARCH_COPY_PARTIAL: "+json.dumps(blocked[:3]))
 # Proof beyond file size: exact original model graph and original torch class SHA.
 if sha(BF16/"model.safetensors.index.json")!=SOURCE_MODEL_INDEX_SHA or \
    sha(BF16/"config.json")!=SOURCE_MODEL_CONFIG_SHA or \
    sha(BF16/"modeling_deepseek.py")!=SOURCE_MODEL_TORCH_SHA:
  raise ValueError("D50 copied model source/index differs from P17 original")
 outcomes=(
  ("EOE_P18_BF16_MODEL_COPY_RECEIPT.json",
   "beglin-hcp-p18-sha-verified-d50-original-BF16-model-copy-v1",
   "ORIGINAL_BF16_FOUR_SHARDS_COPY_SHA_VERIFIED",
   BF16,BF16_XOX,tuple(BF16_SHARDS)+BF16_SMALL),
  ("EOE_P18_MOE_BASE_COPY_RECEIPT.json",
   "beglin-hcp-p18-sha-verified-d50-native-MOE-copy-v1",
   "ORIGINAL_MOE_NATIVE_BASE_COPY_SHA_VERIFIED",
   MOE,MOE_XOX,tuple(MOE_FILES)))
 for dest,schema,status,root,remote,names in outcomes:
  path=ROOT/dest
  if path.exists():raise FileExistsError("P18 immutable source model receipt already exists")
  hashes=source_digests(root,remote,names)
  payload={"schema":schema,"status":status,
   "source_host":"XOX","original_source_read_only":True,
   "original_source_root":str(remote),"D50_copied_source_root":str(root),
   "original_source_index_sha256":SOURCE_MODEL_INDEX_SHA if root==BF16 else None,
   "exact_source_relative_file_shas":hashes,
   "model_relative_file_count":len(hashes),
   "all_four_shards_source_digests_verified":root==BF16,
   "all_source_files_sha_verified":True,
   "source_model_arch_original_27_layer":True,
   "local_D50_reread_all_source_blobs":True,
   "production_touched":False,"automatic_live_promotion":False}
  with path.open("x") as f:json.dump(payload,f,sort_keys=True,indent=2);f.write("\n")
  print("P18_D50_ORIGINAL_MODEL_COPY_SHA_SEALED",
   json.dumps({"status":status,"source_count":len(hashes),
    "receipt_sha256":sha(path)},sort_keys=True),flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument("--inspect",action="store_true")
 a.add_argument("--seal",action="store_true")
 args=a.parse_args()
 if args.inspect and not args.seal:
  v=check_state()
  print(json.dumps({"status":"P18_D50_COPY_READY_TO_SEAL" if not v else "P18_D50_MODEL_TRANSFER_INCOMPLETE",
   "incomplete_file_count":len(v),"missing_or_partial":v[:6],"production_touched":False},sort_keys=True))
  return
 if args.seal and not args.inspect:seal();return
 raise ValueError("Choose --inspect or --seal explicitly")
if __name__=="__main__":main()
