#!/usr/bin/env python3
"""Independent P12 sealed BF16-local and native startup-journal receipt auditor."""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,re
import gpu_runtime_control as grc
from hcp_p12_prestart_journal import (
 MODEL_SHA,SOURCE_SHA,BINARY_SHA,parse_txn,verify_archive_folder
)
P6_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
TRANS="7371cc352226e912dbf42762e5e63a70e91acb5c90691d90e32401e18af7ec5f"
DENSE="cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956"
DYNAMIC="1bb29f8c73abdf6afc157495b14efcd93d82f5cc61c6416b8359fc9e7259fe6e"
def sh(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def audit(root,repo,quality):
 root,repo,quality=map(pathlib.Path,(root,repo,quality))
 if not root.is_dir() or root.is_symlink() or sh(quality)!=P6_SHA:
  raise ValueError("P12 evidence root or P6 baseline cannot be verified")
 l=root/"p12_local_BF16_oracle.json";b=root/"P12_native_prestart.json"
 if any(not x.is_file() or x.is_symlink() for x in (l,b)):
  raise ValueError("P12 BF16/native evidence missing")
 local=json.loads(l.read_text())
 native=json.loads(b.read_text())
 if (local.get("schema")!="beglin-hcp-p12-real-native-BF16-local-two-tensor-projections-v1"
     or local.get("status")!="TWO_ORIGINAL_BF16_MATRICES_EXACT_C_QNG64_LOCAL_ORACLE_PASS"
     or local.get("source_commit")!="e2271d6fe758c6d09877130637dccb193427ea15"
     or local.get("model_index_sha256")!=MODEL_SHA
     or local.get("source_local_tensors")!=2 or local.get("bits_tested")!=[5,6]
     or local.get("independent_projection_vectors")!=8 or
     local.get("full_BF16_teacher_forward_executed") is not False or
     local.get("model_BF16_logits_reference_measured") is not False or
     local.get("task_accuracy_noninferiority_proven") is not False or
     local.get("whole_model_quality_certified") is not False or
     local.get("production_touched") is not False or
     local.get("max_RSS_bytes",2**40)>=1536*(1024**2)):
  raise ValueError("P12 BF16 local scope/performance data misrepresented")
 if (local.get("compiled_source_sha256")!={"gguf_transcode.c":TRANS,
                                          "qng64_dense.c":DENSE} or
     local.get("C_compiled_library_sha256")!=DYNAMIC):
  raise ValueError("P12 native quantization algorithm not the audited source")
 for source_name,source_sha in (("gguf_transcode.c",TRANS),("qng64_dense.c",DENSE)):
  if sh(repo/source_name)!=source_sha:raise ValueError("quantizer source file changed")
 by={}
 for x in local["records"]:
  key=x["target"]
  if key in by:raise ValueError("duplicate BF16 target")
  if (key not in ("shared_up_L3","shared_down_L26") or
      x["original_dtype"]!="BF16" or
      x["vector_count"]!=8 or x["input_rng_seed"]<20261014):
   raise ValueError("BF16 local source or random vectors unexpectedly changed")
  if (set(x["quantization_bits"])!={"5","6"} or
     x["weight_shape"] not in ([2816,2048],[2048,2816])):
   raise ValueError("P12 not the two frozen local tensor dimensions")
  for n in ("5","6"):
   d=x["quantization_bits"][n]
   if (d.get("bits")!=int(n) or
      not d.get("native_quantizer","").startswith("gguf_quantize_qNg64 original C")):
    raise ValueError("P12 quantization C semantic identity changed")
   metrics=("weight_relative_l2","projection_relative_l2","weight_rmse",
    "native_C_vs_independent_unpack_projection_max_abs","projection_rmse")
   if any(not math.isfinite(float(d[m])) or float(d[m])<0 for m in metrics):
    raise ValueError("P12 local BF16 numeric error invalid")
   if d["native_C_vs_independent_unpack_projection_max_abs"]>0.0003:
    raise ValueError("P12 native C qNg64 and independent unpack disagree")
  by[key]=x
 if set(by)!={"shared_up_L3","shared_down_L26"}:
  raise ValueError("P12 expected original BF16 target tensors absent")

 if (native.get("schema")!="beglin-hcp-p12-real-native-acknowledged-prestart-journal-v1"
     or native.get("status")!="XOX_SAME_PINNED_GPU_BINARY_TERMINAL_ARCHIVED_AND_FRESH_RESTART_PASS"
     or native.get("real_native_p9_binary_sha256")!=BINARY_SHA
     or native.get("model_index_sha256")!=MODEL_SHA
     or native.get("p9_mlx_source_sha256")!=SOURCE_SHA
     or native.get("old_terminal_transactions_escrowed")!=2
     or native.get("new_P6_matched_generations")!=4 or
     native.get("restart_new_pid") is not True or
     native.get("archived_old_transaction_was_never_replayed") is not True or
     native.get("production_touched") is not False or
     native.get("automatic_promotion") is not False or
     native.get("P12_gate_never_installed_in_production") is not True):
  raise ValueError("P12 real native startup journal result missing")
 live=native["live_before"]
 if native["live_after"]!=live or live!={"stage_sha":STAGE_SHA,"route_sha":ROUTE_SHA,"PIDs":[35481,35493]}:
  raise ValueError("P12 original live stage/workers not preserved")
 references={row["prompt_id"]:row for row in json.loads(quality.read_text())["records"]}
 truth=references["p000"]
 outputs=native["responses"]
 if [r["policy"] for r in outputs]!=["B","B","A","B"] or len(outputs)!=4:
  raise ValueError("P12 native policy sequence not the certified replay")
 for i,x in enumerate(outputs):
  if (x["seq_sha256"]!=truth[x["policy"]]["sequence_sha256"] or
      x["native_ACK_status"]!="REBIND_SET_APPLIED" or
      x["transitioned"] is not True):
   raise ValueError("P12 real generated output differs from original DeepSeek P6 reference")
 if outputs[0]["PID"]==outputs[1]["PID"] or len({x["PID"] for x in outputs[1:]})!=1:
  raise ValueError("P12 native restart did not change PID exactly once")

 folder=root/"p12_journal"
 if folder.is_symlink() or not folder.is_dir():raise ValueError("missing P12 owned fsync archive")
 recorded={}
 dirs=list(folder.iterdir())
 if len(dirs)!=2:raise ValueError("P12 archive not two authentic B commits")
 for f in dirs:
  verify=verify_archive_folder(f)
  cmd=parse_txn((f/"original.cmd").read_bytes())
  ack=grc.read_runtime_ack(f/"original_ack.json")
  if (verify["decision"]!="TERMINAL_COMMITTED" or
      verify["no_auto_replay"] is not True or
      ack["status"]!="REBIND_SET_APPLIED" or
      ack["txn_id"]!=cmd["txn_id"] or
      ack["expected_epoch"]!=cmd["epoch"] or
      ack["weight_epoch"]!=cmd["epoch"]+1 or
      ack["active_policy_hash"]!=cmd["after_hash"] or
      cmd["after_hash"]!=truth["B"]["runtime_policy_hash"]):
   raise ValueError("P12 archived original command / terminal ACK contradict")
  recorded[sh(f/"original.cmd")]=sh(f/"VERIFIED.json")
 if (set(recorded)!=set(native["distinct_escrow_command_sha256"]) or
    set(recorded.values())!=set(native["verified_receipt_sha256"])):
  raise ValueError("P12 saved real terminal receipts do not match reported SHA")
 owner=json.loads((root/"P12_OWNER_IDENTITY.json").read_text())
 if (owner.get("P9_binary_sha256")!=BINARY_SHA or
    owner.get("P9_mlx_source_sha256")!=SOURCE_SHA or
    owner.get("model_index_sha256")!=MODEL_SHA):
  raise ValueError("P12 owner identity drift")
 pf=json.loads((root/"preflight.json").read_text())
 if (pf.get("P9_MLX_KV128_SHA256")!=SOURCE_SHA or
    pf.get("original_stage_sha256")!=STAGE_SHA or
    pf.get("original_route_sha256")!=ROUTE_SHA):
  raise ValueError("P12 pinned source preflight differs")
 return {"schema":"beglin-hcp-p12-independent-double-study-audit-v1",
 "status":"NATIVE_PRESTART_JOURNAL_AND_ORIGINAL_BF16_LOCAL_ORACLE_INDEPENDENTLY_VERIFIED",
 "BF16_local_sha256":sh(l),"real_startup_sha256":sh(b),
 "local_BF16_tensors_checked":2,"local_qNg64_bitwidths":[5,6],
 "source_C_sha256":{"gguf_transcode.c":TRANS,"qng64_dense.c":DENSE},
 "native_terminal_transactions_verified":2,
 "new_generated_sequences_verified":4,
 "native_reboot_after_terminal_journal":True,
 "crash_ambiguous_transactions_not_replayed":True,
 "whole_BF16_model_forward_executed":False,
 "LLM_task_quality_noninferiority_certified":False,
 "main_modified":False,"promotion_allowed":False}

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--evidence",type=pathlib.Path,required=True)
 ap.add_argument("--repository",type=pathlib.Path,required=True)
 ap.add_argument("--quality",type=pathlib.Path,required=True)
 ap.add_argument("--output",type=pathlib.Path,required=True)
 a=ap.parse_args();doc=audit(a.evidence,a.repository,a.quality)
 with a.output.open("x") as f:json.dump(doc,f,sort_keys=True,indent=2);f.write("\n")
 print("P12_INDEPENDENT_AUDIT_PASS",json.dumps({"bf16_local":2,"real_transactions":2,
  "real_generations":4,"whole_model":False,
  "SHA256":sh(a.output)},sort_keys=True),flush=True)
