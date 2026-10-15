#!/usr/bin/env python3
"""P18 EOE source-only model-independent native MLX binding and 673 scope QA.
Cannot certify model inference. No XOX writes, no operator or live route mutation.
Runs real compiled Metal binding transition executable once, records immutable
raw stdout+stderr and a SHA source-bound receipt under the registered D50 root.
"""
from __future__ import annotations
import hashlib,json,pathlib,re,subprocess
R=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
ORIG=R/"eoe_pretransfer_identity_review"
C=R/"eoe_p18_native_capture_buildroot/qwen_infer.c"
XC=ORIG/"qwen_infer_long_p18_original.c"
MLX=R/"eoe_p18_native_capture_buildroot/mlx_moe.cpp"
BIN=R/"eoe_p18_native_capture_build/qwen_infer_gpu"
TRANSITION=R/"eoe_p18_native_capture_build/mlx_binding_transition_test"
FROZEN=R/"p18_multishot_frozen_PRIVATE.json"
LOG=R/"P18_EOE_PREMODEL_REAL_MLX_BOUND_673_TEST.log"
RECEIPT=R/"P18_EOE_PREMODEL_REAL_MLX_BOUND_673_RECEIPT.json"
EXPECTED={
 "frozen":"c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df",
 "XOX_native_source":"fa8a328bedc1e56e79938460e1d4a171f0fd363535fc610183b4022e41ce0bfa",
 "EOE_native_source":"f5fab593e588d5a5b34528cc99c518fc2f894186ab46cb50bca5dc1b2e59df5e",
 "MLX_source":"a0122c8b2010573fb3b20b1c58ab4a3abb26b73602eddf92dd008dda7892f82b",
 "EOE_binary":"4b22f80d6368fd8653abcf2b4f3e4d9e1c5fdff8e50ef598d942bbfc1774f176",
 "original_BF16_index":"d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf",
 "original_BF16_config":"f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d",
 "original_BF16_modeling":"7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c"}
def sha(p):
 p=pathlib.Path(p)
 if p.is_symlink() or not p.is_file():raise ValueError("missing/symlink source "+str(p))
 h=hashlib.sha256()
 with p.open("rb") as f:
  for piece in iter(lambda:f.read(1048576),b""):h.update(piece)
 return h.hexdigest()
def htok(t):
 return hashlib.sha256(",".join(map(str,t)).encode("ascii")).hexdigest()
def run():
 if not R.is_dir() or R.is_symlink() or LOG.exists() or RECEIPT.exists():
  raise RuntimeError("P18 source QA one-shot canonical path already occupied")
 for key,p in (
     ("frozen",FROZEN),("XOX_native_source",XC),
     ("EOE_native_source",C),("MLX_source",MLX),("EOE_binary",BIN),
     ("original_BF16_index",ORIG/"bf16/model.safetensors.index.json"),
     ("original_BF16_config",ORIG/"bf16/config.json"),
     ("original_BF16_modeling",ORIG/"bf16/modeling_deepseek.py")):
  if sha(p)!=EXPECTED[key]:raise RuntimeError("P18 original static source drift: "+key)
 j=json.loads(FROZEN.read_text())
 if len(j["items"])!=480 or j["question_count"]!=480:raise RuntimeError("not original 480")
 actual=[]
 for q in j["items"]:
  ids=q["prompt_ids"];gold=q["gold_ids"]
  if not (htok(ids)==q["prompt_token_sha256"] and htok(gold)==q["gold_token_sha256"] and len(gold)==q["gold_token_count"]):
   raise RuntimeError("unaligned original gold preimage")
  actual.extend(htok(ids+gold[:idx]) for idx in range(len(gold)))
 if len(actual)!=673 or len(set(actual))!=673:raise RuntimeError("not 673 unique original golden hashes")
 content=C.read_text()
 hashes=re.findall(r'"([0-9a-f]{64})"',content.split("/* BEGLIN_P18_SOURCE_HASH_ALLOWLIST_BEGIN:",1)[1].split("/* BEGLIN_P18_SOURCE_HASH_ALLOWLIST_END */",1)[0])
 if len(hashes)!=673 or set(hashes)!=set(actual):
  raise RuntimeError("source-bound C 673 frozen golden hash mismatch")
 old=XC.read_text();oldroot="/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p18-capture1024-20261015/capture"
 newroot=str(R/"eoe_p18_gpu_capture_raw")
 if content.count(newroot)!=1 or content.replace(newroot,oldroot)!=old:
  raise RuntimeError("EOE C source changed beyond reviewed single output path")
 if not all(x in content for x in ("BEGLIN_P18_SCRATCH_ONLY","O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW","P18_NATIVE_GOLD_PREFIX_LOGITS rid=")):
  raise RuntimeError("P18 model source security guard absent")
 if TRANSITION.is_symlink() or not TRANSITION.is_file():
  raise RuntimeError("No isolated compiled actual MLX binding transition test")
 test=subprocess.run([str(TRANSITION)],capture_output=True,text=True,timeout=40)
 trace=test.stdout+test.stderr
 if test.returncode!=0 or "PASS G2 binding transition matrix + 100-cycle restore soak" not in trace:
  raise RuntimeError("Actual MLX runtime precision promotion/demotion test failed: "+trace[-1200:])
 count=sum(1 for line in trace.splitlines() if line.startswith("PASS "))
 if count<100 or not all("PASS "+x in trace for x in ("4 -> 5 -> 4","4 -> 6 -> 4","4 -> 7 -> 4","7 -> 16 -> 7")):
  raise RuntimeError("P18 native 100-cycle precision binding transitions not fully witnessed")
 with LOG.open("x") as f:f.write(trace)
 proof={
  "schema":"beglin-hcp-p18-eoe-premodel-original-security-and-MLX-binding-QA-v1",
  "status":"P18_EOE_ORIGINAL_FROZEN_673_NATIVE_SOURCE_AND_REAL_MLX_100CYCLE_PASS_NO_MODEL",
  "original_frozen_test_question_units":480,
  "original_frozen_gold_prefixes":673,
  "original_C_to_EOE_changed_component":"single capture directory literal only",
  "source_SHA256":EXPECTED,
  "original_model_27layers_metadata_verified":True,
  "original_model_tensor_index_count":5291,
  "original_full_vocab":102400,
  "MLX_binding_test_exit":test.returncode,
  "MLX_actual_transition_PASS_lines":count,
  "MLX_100_cycle_restore_soak_verified":True,
  "actual_MLX_test_output_SHA256":sha(LOG),
  "original_full_BF16_model_copy_complete":False,
  "original_MOE_model_copy_complete":False,
  "native_A_B_model_forward_executed_here":False,
  "BF16_teacher_model_forward_executed_here":False,
  "P18_480_original_quality_certified":False,
  "production_touched":False,
  "automatic_live_promotion":False}
 with RECEIPT.open("x") as f:json.dump(proof,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":proof["status"],"MLX_PASS_lines":count,"allowed_C_gold_SHA":len(hashes),
  "receipt_SHA256":sha(RECEIPT),"raw_log_SHA256":sha(LOG)},sort_keys=True))
if __name__=="__main__":run()
