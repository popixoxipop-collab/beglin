#!/usr/bin/env python3
"""Source-pinned, read-only two-cold-process DeepSeek-v11 holdout1 replay audit.

Independently checks the same frozen 7-token prompt and real 27-layer/2-greedy
GPU Metal fused-MLA results across two separately launched processes. Requires
private full-vocabulary F32 logits numpy SHA match at *each* of three stages.
No source tensor payload is exported or attached to GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

INDEX="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
SEALED="c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8"
RUNNER_SHA="f5129ad58a841e982cfdd94ddbf3c29797b9ff3e6534eb909c75c71d94c08c7a"


def gate(ok,reason):
    if not ok:
        raise ValueError("BEGLIN_COLD_HOLDOUT_REPLAY_FAIL_CLOSED: "+reason)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def load_receipt(path):
    p=Path(path)
    gate(not p.is_symlink() and p.is_file() and 0<p.stat().st_size<1024*1024,
         "private immutable model receipt not a bounded regular file")
    return json.loads(p.read_text()),sha(p)


def inspect_private_logits(root,stage):
    basename=stage.get("private_logits_basename")
    gate(isinstance(basename,str) and
         basename==Path(basename).name and basename.endswith(".npy") and
         basename.startswith("BEGLIN_v11_holdout1_cached_logits_"),
         "frozen private logits filename malformed")
    path=root/basename
    gate(path.is_file() and not path.is_symlink() and
         0<path.stat().st_size<=1024*1024,"missing compact private frozen logits")
    digest=sha(path)
    gate(digest==stage.get("private_logits_npy_sha256"),
         "private logit evidence SHA mismatch")
    return digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runner-source",required=True)
    ap.add_argument("--a",required=True)
    ap.add_argument("--b",required=True)
    ap.add_argument("--output",required=True)
    x=ap.parse_args()
    output=Path(x.output)
    gate(not output.exists(),"append-only independent receipt already exists")
    runner=Path(x.runner_source)
    gate(runner.is_file() and not runner.is_symlink() and sha(runner)==RUNNER_SHA,
         "new runtime source SHA changed between cold processes")
    a,ah=load_receipt(x.a)
    b,bh=load_receipt(x.b)
    gate(ah!=bh,"cold runs must have separate independent append-only receipts")
    expected="PASS_REAL_FULL27_HOLDOUT1_TWO_GREEDY_SEALED_N7_FUSED_MLA_KV"
    for d in (a,b):
        gate(d.get("status")==expected and
             d.get("schema")=="beglin.v11.original-deepseek-full27-sealed-n7-fused-MLA-cached/1" and
             d.get("checkpoint_index_sha256")==INDEX and
             d.get("full_sealed_prequant_manifest_sha256")==SEALED and
             d.get("source_prompt_id")==1 and
             d.get("source_prompt_token_count")==7 and
             d.get("greedy_input_tokens")==[185,185] and
             d.get("predictions")==[185,185,1679] and
             d.get("model_layers")==27,
             "independent real 27-layer holdout 2-greedy source/model identity drift")
        gate(d.get("CPU_original_qNg64_requant_during_model_generation")==0 and
             d.get("source_model_true_MoE_gate_experts") is True and
             d.get("per_layer_fused_kvb_weight_registered_once") is True and
             d.get("device_resident_fused_kvb_weight_layers")==27 and
             d.get("persistent_MLA_KV_total_bytes")==279936 and
             d.get("all_GPU_weight_handles_fully_released") is True and
             d.get("all_MLA_KV_layers_detached_from_original_attention") is True and
             d.get("production_promotion_authorized") is False,
             "source GPU fused MLA or cleanup boundary changed")
    stable=["checkpoint_index_sha256","full_sealed_prequant_manifest_sha256",
            "original_source_policy_sha256","source_prompt_sha256",
            "source_prompt_id","source_prompt_token_count","model_layers",
            "greedy_input_tokens","predictions","device_resident_fused_kvb_weight_layers",
            "fused_mla_GPU_attention_query_calls","fused_mla_kvb_resident_GPU_packed_bytes",
            "GPU_normal_qng64_model_forward_calls","GPU_normal_qng64_reuse_hits",
            "GPU_normal_qng64_evictions","source_new_model_role_compressed_file_loads",
            "source_gpu_C_oracle_checked_role_modules",
            "CPU_original_qNg64_requant_during_model_generation",
            "BF16_neighbor_ULP_transition_count",
            "FP32_C_vs_Metal_max_relative_L2_on_BF16_ambiguity",
            "persistent_MLA_KV_total_bytes"]
    for k in stable:
        gate(a.get(k)==b.get(k),"separate cold-process source/model/GPU counters drifted: "+k)
    aa=a.get("stage_receipts")
    bb=b.get("stage_receipts")
    gate(isinstance(aa,list) and isinstance(bb,list) and len(aa)==len(bb)==3,
         "three genuine decoder phases required")
    logits=[]
    for i,(u,v) in enumerate(zip(aa,bb)):
        for k in ("source_phase","source_layer_count","new_context_position",
                  "latent_KV_size_bytes","source_gpu_packed_module_instances_this_stage",
                  "source_full_model_C_gpu_oracle_bound_module_instances",
                  "MoE_experts_used_sum","prediction","logits_SHA256"):
            gate(u.get(k)==v.get(k),"replay stage numerical/source field drift step="+str(i)+":"+k)
        pu=inspect_private_logits(Path(x.a).parent,u)
        pv=inspect_private_logits(Path(x.b).parent,v)
        gate(pu==pv,"private 102400-vocab cached logits bytes changed between cold processes")
        logits.append({"step":i,"full_vocab_logits_npy_sha256":pu,
                       "decoded_token_prediction":u["prediction"],
                       "source_kv_bytes":u["latent_KV_size_bytes"]})
        if i==0:
            gate(u.get("logits_compare_original_uniform7")==
                 v.get("logits_compare_original_uniform7"),
                 "original independent n7 prefill reference diverged")
    result={
        "schema":"beglin.v11.sealed-original-holdout1-two-cold-cached-mla-replay-audit/1",
        "status":"PASS_REAL_HOLDOUT1_TWO_COLD_CACHED_MLA_IDENTICAL_MODEL_LOGITS_AND_TOKEN_STREAM",
        "frozen_7token_source_sha256":a["source_prompt_sha256"],
        "runner_sha256":RUNNER_SHA,
        "original_checkpoint_index_sha256":INDEX,
        "model_wide_sealed_n7_manifest_sha256":SEALED,
        "source_true_model_tokens":[185,185,1679],
        "cold_receipt_SHA256":[ah,bh],
        "all_three_exact_logits_numpy_payloads_match":True,
        "stage_logit_sha256":logits,
        "normal_Metal_n7_model_calls":a["GPU_normal_qng64_model_forward_calls"],
        "fused_Metal_MLA_queries":a["fused_mla_GPU_attention_query_calls"],
        "resident_MLA_kv_B":279936,
        "process_times_seconds":[a["wall_seconds"],b["wall_seconds"]],
        "process_peak_RSS_bytes":[a["peak_process_RSS_bytes"],b["peak_process_RSS_bytes"]],
        "inference_C_requantization_count":0,
        "all_GPU_weight_handles_released":True,
        "multi_hour_stability_verified":False,
        "end_to_end_serving_TTFT_TPOT_verified":False,
        "production_promotion_authorized":False,
    }
    blob=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode("utf8")
    with output.open("xb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status":result["status"],"token_stream":result["source_true_model_tokens"],
                      "full_logit_hashes":[z["full_vocab_logits_npy_sha256"] for z in logits],
                      "run_times_s":result["process_times_seconds"],
                      "audit_SHA256":hashlib.sha256(blob).hexdigest(),
                      "promotion_allowed":False},sort_keys=True))


if __name__=="__main__":
    main()
