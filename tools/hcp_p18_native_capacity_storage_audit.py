#!/usr/bin/env python3
"""P18 fail-closed read-only native-context + disk admission audit.

Reads pinned XOX P17 source macros through batch SSH; executes no native model.
The guard never authorizes a GPU job, route switch, or release activation.
"""
from __future__ import annotations
import hashlib,json,pathlib,re,subprocess
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
XOX_SRC="/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015/native_src/"
XOX_VDSP="/Users/xox/mcp-sandbox/tailnet-commander"
C_SHA="d724af1eb97ee715e206188fa120547f4cc1f18f55a3c37938b6b5a4ce9986d2"
MLX_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
COHORT_SHA="19e7080b7208b8c7dd6fd06a656bb400dd2a9f72ca5f5674a80f5cc9a654cf65"
PROTO_SHA="77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c"
DISK_RESERVE_BYTES=20*1024**3

def sha(p):
 p=pathlib.Path(p)
 if p.is_symlink() or not p.is_file():raise ValueError("No read of nonregular source")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def ssh(*args):
 cmd=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8","xox",*args]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=16)
 if p.returncode!=0:raise RuntimeError("XOX read-only SSH failed: "+p.stderr[:400])
 return p.stdout
def read_macro(path,symbol):
 raw=ssh("grep","-n",symbol,path)
 exact=re.findall(r"(?m)^\d+:#define\s+"+re.escape(symbol)+r"\s+(\d+)\s*$",raw)
 if len(exact)!=1:raise ValueError("missing/duplicate live native constant "+symbol)
 return int(exact[0])
def main():
 if not ROOT.is_dir() or ROOT.is_symlink():raise RuntimeError("P18 D50 missing")
 if sha(ROOT/"p18_gate_cohort_PRIVATE.json")!=COHORT_SHA or sha(ROOT/"p18_protocol_FROZEN.json")!=PROTO_SHA:
  raise ValueError("preregistered cohort or protocol drift")
 values=ssh("shasum","-a","256",XOX_SRC+"qwen_infer.c",XOX_SRC+"mlx_moe.cpp")
 source_lines=values.strip().splitlines()
 parsed=[row.split()[0] for row in source_lines]
 if parsed!=[C_SHA,MLX_SHA]:raise RuntimeError("P17 exact XOX C/MLX source identity drift")
 macros={"MOE_CBATCH_MAXPOS":read_macro(XOX_SRC+"qwen_infer.c","MOE_CBATCH_MAXPOS"),
         "MOE_MAXPOS":read_macro(XOX_SRC+"qwen_infer.c","MOE_MAXPOS"),
         "MLA_L0_MAXPOS":read_macro(XOX_SRC+"mlx_moe.cpp","MLA_L0_MAXPOS")}
 if macros!={"MOE_CBATCH_MAXPOS":128,"MOE_MAXPOS":160,"MLA_L0_MAXPOS":128}:
  raise RuntimeError("P17 bound changed: research provenance drift")
 df=ssh("df","-k",XOX_VDSP).strip().splitlines()
 if len(df)<2:raise RuntimeError("XOX disk guard unavailable")
 free_kib=int(df[-1].split()[3]);free_bytes=free_kib*1024
 cohort=json.loads((ROOT/"p18_gate_cohort_PRIVATE.json").read_text())
 rows=cohort["items"]
 if len(rows)!=480:raise ValueError("P18 cohort is not 480")
 caps=[r["prompt_token_count"]+r["max_new_tokens"] for r in rows]
 exact_native=max(macros["MOE_CBATCH_MAXPOS"],macros["MLA_L0_MAXPOS"])
 fit_128=sum(n<=exact_native for n in caps)
 fit_160=sum(n<=macros["MOE_MAXPOS"] for n in caps)
 fit_4096=sum(n<=4096 for n in caps)
 required=max(caps)
 report={"schema":"beglin-hcp-p18-native-capacity-storage-preflight-v1",
   "status":"P18_NATIVE_CONTEXT_AND_XOX_SSD_GUARD_BLOCKED",
   "source_C_sha256":C_SHA,"source_MLX_sha256":MLX_SHA,
   "P18_protocol_sha256":PROTO_SHA,"P18_cohort_sha256":COHORT_SHA,
   "observed_source_macros":macros,"source_capability_4096_verified":False,
   "proposed_4096_vs_existing_KV_positional_dimension_multiplier":32,
   "required_max_total_positions":required,
   "frozen_question_count":len(caps),
   "fits_existing_cb_and_mlx_128_positions":fit_128,
   "fits_single_sequence_MOE_MAXPOS_160":fit_160,
   "fits_proposed_4096_positions":fit_4096,
   "xox_disk_free_bytes":free_bytes,"xox_disk_protected_reserve_bytes":DISK_RESERVE_BYTES,
   "xox_disk_deficit_bytes":max(0,DISK_RESERVE_BYTES-free_bytes),
   "full_480_admissible":False,
   "native_model_forward_count":0,"production_touched":False,
   "automatic_live_promotion":False}
 dest=ROOT/"p18_native_capacity_storage_audit.json"
 if dest.exists():raise FileExistsError("Native capability audit already exists")
 with dest.open("x") as f:json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps({"status":report["status"],"native_max_positions":exact_native,
   "required_max_positions":required,"fits128":fit_128,"fits4096":fit_4096,
   "xox_free_GiB":round(free_bytes/1024**3,3),
   "below_reserve_GiB":round(max(0,DISK_RESERVE_BYTES-free_bytes)/1024**3,3),
   "evidence_sha256":sha(dest)},sort_keys=True))
if __name__=="__main__":main()
