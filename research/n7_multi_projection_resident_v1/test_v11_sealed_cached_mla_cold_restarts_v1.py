#!/usr/bin/env python3
"""BEGLIN v11 research-only original-model sealed-fused-MLA cold-process QA.

Launch exact same 27-layer source-attested Metal cached two-greedy test in
additional fresh OS processes; compare selected token IDs, both post-prefill
102,400-vocabulary logit hashes, role/call counts, 27 source resident kv_b
handles, 217,728-B latent-KV after 7 tokens, and explicit handle cleanup.

Does not claim long-session leak-free, production TTFT or energy efficiency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def need(ok,msg):
    if not ok:
        raise RuntimeError("BEGLIN_V11_COLD_RESTART_SOURCE_FAIL_CLOSED:"+msg)


def read_receipt(path):
    p=Path(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<10*1024*1024,
         "source receipt not regular bounded local file")
    raw=p.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def comparable(r):
    need(r.get("status")=="PASS_REAL_FULL27_TWO_GREEDY_SEALED_N7_FUSED_MLA_KV" and
         r.get("model_layers")==27 and
         r.get("predictions")==[429,4281,6230] and
         r.get("greedy_input_tokens")==[429,4281] and
         r.get("device_resident_fused_kvb_weight_layers")==27 and
         r.get("persistent_MLA_KV_total_bytes")==217728 and
         r.get("fused_mla_kvb_resident_GPU_packed_bytes")==53084160 and
         r.get("CPU_original_qNg64_requant_during_model_generation")==0 and
         r.get("all_GPU_weight_handles_fully_released") is True and
         r.get("expanded_MLA_KV_persistent") is False and
         r.get("production_promotion_authorized") is False,
         "source original model cached Metal two-greedy receipt not PASS")
    stages=r.get("stage_receipts")
    need(isinstance(stages,list) and len(stages)==3 and
         all(stage.get("source_layer_count")==27 for stage in stages),
         "three full 27-layer source stages required")
    need(stages[1].get("prediction")==4281 and stages[2].get("prediction")==6230 and
         isinstance(stages[1].get("logits_SHA256"),str) and
         isinstance(stages[2].get("logits_SHA256"),str),
         "both actually decoded post-prefill source logits SHA needed")
    return {
        "predictions":r["predictions"],
        "greedy_input_tokens":r["greedy_input_tokens"],
        "source_prompt_sha256":r["source_prompt_sha256"],
        "original_policy_sha256":r["original_source_policy_sha256"],
        "sealed_manifest_sha256":r["full_sealed_prequant_manifest_sha256"],
        "fused_GPU_queries":r["fused_mla_GPU_attention_query_calls"],
        "source_selected_GPU_role_loads":r["source_new_model_role_compressed_file_loads"],
        "normal_GPU_forward_calls":r["GPU_normal_qng64_model_forward_calls"],
        "GPU_weight_evictions":r["GPU_normal_qng64_evictions"],
        "GPU_weight_hits":r["GPU_normal_qng64_reuse_hits"],
        "active_latent_BF16_KV_bytes":r["persistent_MLA_KV_total_bytes"],
        "step1_logits_SHA256":stages[1]["logits_SHA256"],
        "step2_logits_SHA256":stages[2]["logits_SHA256"],
        "prefill_uniform7_relative_L2":
            stages[0]["logits_compare_original_uniform7"]["relative_L2"],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-receipt",required=True)
    ap.add_argument("--checkpoint-dir",required=True)
    ap.add_argument("--sealed-root",required=True)
    ap.add_argument("--manifest-sha",required=True)
    ap.add_argument("--native-c-dylib",required=True)
    ap.add_argument("--native-gpu-dylib",required=True)
    ap.add_argument("--receipt-dir",required=True)
    ap.add_argument("--additional-runs",type=int,choices=(1,2,3),required=True)
    args=ap.parse_args()
    root=Path(args.receipt_dir)
    need(root.is_dir() and not root.is_symlink(),"append-only evidence location unavailable")
    baseline,bsha=read_receipt(args.baseline_receipt)
    need(bsha=="165460894349ed00efbaa7ce96a276a6772d7e9bfa82c89e90c61bc244f21081",
         "known first full original v11 cached source digest changed")
    reference=comparable(baseline)
    need(reference["sealed_manifest_sha256"]==args.manifest_sha,
         "selected model sealed source authority changed")
    driver=Path(__file__).resolve().parent/"run_v11_sealed_full27_cached_mla_v1.py"
    need(driver.is_file() and not driver.is_symlink(),
         "pinned source model research driver absent")
    outcomes=[{
        "run":0,
        "receipt_sha256":bsha,
        "wall_seconds":baseline["wall_seconds"],
        "process_peak_RSS_bytes":baseline["peak_process_RSS_bytes"],
    }]
    for trial in range(1,args.additional_runs+1):
        cmd=[sys.executable,"-u",str(driver),
             "--checkpoint-dir",args.checkpoint_dir,
             "--sealed-root",args.sealed_root,
             "--manifest-sha",args.manifest_sha,
             "--native-c-dylib",args.native_c_dylib,
             "--native-gpu-dylib",args.native_gpu_dylib,
             "--receipt-dir",args.receipt_dir,
             "--max-layers","27",
             "--greedy-tokens","2"]
        completed=subprocess.run(
            cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            timeout=900,check=False)
        need(completed.returncode==0,
             "fresh process "+str(trial)+" failed: "+completed.stderr[-1200:])
        last=completed.stdout.splitlines()
        need(len(last)>=82,
             "fresh real source 27L cached Metal lacks layer-by-layer proof")
        summary=json.loads(last[-1])
        need(summary.get("status")=="PASS_REAL_FULL27_TWO_GREEDY_SEALED_N7_FUSED_MLA_KV"
             and summary.get("all_GPU_handles_released") is True,
             "fresh-process native C++ Metal GPU register teardown failed")
        name=summary.get("receipt_basename","")
        need(isinstance(name,str) and
             name.startswith("BEGLIN_v11_real_27L_sealed_fused_mla_cached_27L_2tokens_")
             and name.endswith(".json") and "/" not in name,
             "untrusted research receipt name")
        new,new_sha=read_receipt(root/name)
        need(new_sha==summary["receipt_SHA256"],
             "fresh experiment output digest/receipt mismatch")
        rec=comparable(new)
        need(rec==reference,
             "independent cold-start model-source selected roles/logits parity drift")
        outcomes.append({
            "run":trial,"receipt_sha256":new_sha,
            "wall_seconds":new["wall_seconds"],
            "process_peak_RSS_bytes":new["peak_process_RSS_bytes"],
        })
        print(json.dumps({
            "phase":"BEGLIN_V11_SEALED_SOURCE_COLD_RESTART_PASS",
            "run":trial,"unchanged_greedy_inputs":[429,4281],
            "unchanged_logits_hashes":True,
            "CPU_dynamic_requantizations":0,
            "all_GPU_handles_released":True,
            "receipt_SHA256":new_sha,
        },sort_keys=True),flush=True)
    summary={
        "schema":"beglin.v11.real-27L-sealed-prequant-fused-mla-cold-process/1",
        "status":"PASS_REAL_27L_SEALED_FUSED_MLA_INDEPENDENT_COLD_RESTARTS",
        "original_model_hash_and_frozen_quantized_authority_kept":True,
        "total_independent_model_processes":len(outcomes),
        "actual_greedy_input_tokens":[429,4281],
        "next_prediction":6230,
        "all_normal_device_roles_and_fused_MLA_logits_hashes_repeatable":True,
        "persistent_compact_BF16_KV_bytes":217728,
        "all_native_gpu_weights_released_after_each_process":True,
        "CPU_n7_requantize_during_generation":0,
        "runs":outcomes,
        "model_wide_long_duration_soak_verified":False,
        "genuine_serving_TTFT_TPOT_verified":False,
        "production_promotion_authorized":False,
    }
    data=(json.dumps(summary,sort_keys=True,indent=2)+"\n").encode()
    p=root/("BEGLIN_v11_real_cached_cold_start_"+str(time.time_ns())+".json")
    with p.open("xb") as fd:
        fd.write(data)
        fd.flush()
        os.fsync(fd.fileno())
    print(json.dumps({
        "status":summary["status"],
        "cold_processes":len(outcomes),
        "all_full_vocab_hashes_equal":True,
        "receipt_basename":p.name,
        "receipt_sha256":hashlib.sha256(data).hexdigest(),
        "promotion_allowed":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
