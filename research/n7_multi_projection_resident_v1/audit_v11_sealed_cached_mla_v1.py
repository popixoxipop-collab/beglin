#!/usr/bin/env python3
"""Independent original v11 cached vs sealed-prequant fused MLA research audit.

Cross-checks prior exact source n7 27-layer GPU-owned MLA KV cached execution,
new all-5181 sealed-prequant original source 27-layer cached execution, and a
separately executed full-27 sealed-prequant NO-KV full-context recomputation.

This is a multi-source research numerical/ownership audit, NOT a serving
throughput qualification. The old v11 and new v11 do not use identical test
harnesses for CPU source references or GPU API timing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

BASELINE_SHA="8c36149cd9b062724efe604495b5f56d14aede59d9b86883ea93757cb8676745"
UNCACHED_SHA="ae29603fec2814c5164bd7598006117a7ba776cd154243e1a215cf5b30d8263e"
MANIFEST_SHA="c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8"
AUTHORITY_SHA="3b13bc1d98c5843f146afa428b6dfff45352f7def698d6a5a156d32d0fae136f"
INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
POLICY_SHA="39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612"


def need(ok,why):
    if not ok:
        raise RuntimeError("BEGLIN_V11_SEALED_MLA_KV_INDEPENDENT_FAIL_CLOSED:"+why)


def read(path,expected_sha=None):
    p=Path(path)
    need(p.is_file() and not p.is_symlink() and 0<p.stat().st_size<12*1024*1024,
         "missing/non-regular independent source fixture")
    b=p.read_bytes()
    d=hashlib.sha256(b).hexdigest()
    need(expected_sha is None or expected_sha==d,
         "externally pinned source receipt SHA drift")
    return b,d


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--previous-cached",required=True)
    a.add_argument("--sealed-cached",required=True)
    a.add_argument("--sealed-uncached",required=True)
    a.add_argument("--prequant-manifest",required=True)
    a.add_argument("--prequant-run-authority",required=True)
    a.add_argument("--output",required=True)
    x=a.parse_args()
    out=Path(x.output)
    need(not out.exists(),"independent append-only receipt already exists")
    oldraw,_=read(x.previous_cached,BASELINE_SHA)
    uncraw,_=read(x.sealed_uncached,UNCACHED_SHA)
    nraw,nsha=read(x.sealed_cached)
    read(x.prequant_manifest,MANIFEST_SHA)
    read(x.prequant_run_authority,AUTHORITY_SHA)
    old=json.loads(oldraw);new=json.loads(nraw);unc=json.loads(uncraw)

    need(old.get("status")=="PASS_TWO_GREEDY_TOKENS_DUAL_GPU_V11_GPUOWNED_AND_HOST_SHA_MEMO"
         and old.get("source_checkpoint_sha256")==INDEX_SHA
         and old.get("fixed_uniform7_policy_sha256")==POLICY_SHA
         and old.get("model_layers_done")==27
         and old.get("BF16_KV_cache_bytes_real")==217728,
         "original v11 cached source and K/V contract invalid")
    need(old.get("greedy_next_token_id")==429
         and old.get("model_selected_second_token_id")==4281
         and old.get("greedy_next_prediction_after_second_real_token")==6230,
         "old source original autoregressive greedy contract")
    old_mla=old.get("real_device_GPU_resident_source_KVB",{})
    need(old_mla.get("actual_device_Metal_layer_count")==27
         and old_mla.get("resident_device_bitplanes_F32scales_total_bytes")==53084160
         and old_mla.get("GPU_native_resident_handles_all_explicitly_released") is True,
         "old v11 native GPU-resident kv_b 27-layer authority")

    need(unc.get("status")=="PASS_REAL_FULL27_PREQUANT_TWO_GREEDY_UNCACHED"
         and unc.get("checkpoint_index_sha256")==INDEX_SHA
         and unc.get("packed_source_manifest_sha256")==MANIFEST_SHA
         and unc.get("generated_token_inputs")==[429,4281]
         and unc.get("predicted_token_after_last_input")==6230
         and unc.get("dynamic_C_quantizations_during_inference")==0
         and unc.get("all_GPU_handles_released") is True,
         "separate full-context original source recomputation absent")

    need(new.get("schema")=="beglin.v11.original-deepseek-full27-sealed-n7-fused-MLA-cached/1"
         and new.get("status")=="PASS_REAL_FULL27_TWO_GREEDY_SEALED_N7_FUSED_MLA_KV"
         and new.get("checkpoint_index_sha256")==INDEX_SHA
         and new.get("full_sealed_prequant_manifest_sha256")==MANIFEST_SHA
         and new.get("original_source_policy_sha256")==POLICY_SHA
         and new.get("model_layers")==27,
         "full source packed and fused cached challenger identity")
    need(new.get("source_prompt_sha256")==old.get("source_prompt_sha256")
         and new.get("greedy_input_tokens")==[429,4281]
         and new.get("predictions")==[429,4281,6230],
         "actual source prompt or model-selected greedy sequence not identical")

    rec=new.get("stage_receipts")
    need(isinstance(rec,list) and len(rec)==3,
         "three full27 actual model cached stages unavailable")
    for i,r in enumerate(rec):
        need(r.get("source_layer_count")==27
             and r.get("new_context_position")==(0 if i==0 else 4+i)
             and r.get("latent_KV_size_bytes")==27*(5+i)*1152
             and r.get("source_gpu_packed_module_instances_this_stage",0)>
                 (1000 if i==0 else 500),
             "missing 27-layer sealed prequant source/cached KV stage: "+str(i))
    need(new.get("persistent_MLA_KV_total_bytes")==217728
         and new.get("persistent_KV_source_storage") ==
             "BF16 512D normalized latent + 64D rotated RoPE key"
         and new.get("expanded_MLA_KV_persistent") is False,
         "fused latent KV compact storage/source shape mismatch")
    need(new.get("device_resident_fused_kvb_weight_layers")==27
         and new.get("per_layer_fused_kvb_weight_registered_once") is True
         and new.get("fused_mla_kvb_resident_GPU_packed_bytes")==53084160
         and new.get("fused_mla_GPU_attention_query_calls")==
             old_mla["physical_fused_GPU_source_query_calls"],
         "actual 27 source-attested GPU resident kv_b lifetime/parity drift")

    need(type(new.get("GPU_normal_qng64_model_forward_calls")) is int and
         new["GPU_normal_qng64_model_forward_calls"]>0
         and new.get("GPU_normal_qng64_evictions",0)>0
         and new.get("GPU_normal_qng64_reuse_hits",0)>0
         and new.get("source_gpu_C_oracle_checked_role_modules",0)>0
         and new.get("source_new_model_role_compressed_file_loads",0)>0
         and new.get("CPU_original_qNg64_requant_during_model_generation")==0,
         "source model not genuinely consuming sealed prequant native C++ Metal")
    need(math.isfinite(new.get("FP32_C_vs_Metal_max_relative_L2_on_BF16_ambiguity",float("inf")))
         and new["FP32_C_vs_Metal_max_relative_L2_on_BF16_ambiguity"]<1e-5,
         "model source GPU/CPU native C error exceeds original numerical gate")
    pref=rec[0].get("logits_compare_original_uniform7")
    need(isinstance(pref,dict) and 0<=pref.get("relative_L2",float("inf"))<0.025,
         "original v11 BF16 full-vocab reference parity not verified")
    old_pref=old.get("prefix_logits_cpp_vs_previous_independent_model",{})
    need(isinstance(old_pref,dict) and
         abs(pref["relative_L2"]-old_pref.get("relative_L2",-2))<1e-3,
         "source original MLA fused prefill numerical fidelity changed")

    need(new.get("all_GPU_weight_handles_fully_released") is True
         and new.get("all_MLA_KV_layers_detached_from_original_attention") is True
         and new.get("completed_without_original_checkpoint_changes") is True
         and new.get("BF16_original_source_layer_and_reference_weight_reads_still_present") is True
         and new.get("full_model_GPU_exclusively_resident") is False
         and new.get("genuine_serving_TTFT_TPOT_certified") is False
         and new.get("original_source_modified") is False
         and new.get("production_promotion_authorized") is False,
         "unsafe scope inflation or missing GPU/model-handle cleanups")
    result={
        "schema":"beglin.v11.27L-sealed-prequant-fused-MLA-independent/1",
        "status":"PASS_REAL_FULL27_TWO_GREEDY_PREQUANT_FUSED_MLA_KV_INDEPENDENT",
        "original_BF16_index_SHA256":INDEX_SHA,
        "full_qng64_n7_5181_manifest_SHA256":MANIFEST_SHA,
        "full_source_export_authority_SHA256":AUTHORITY_SHA,
        "prior_v11_cached_receipt_SHA256":BASELINE_SHA,
        "sealed_v11_uncached_reference_receipt_SHA256":UNCACHED_SHA,
        "sealed_v11_fused_cached_receipt_SHA256":nsha,
        "source_greedy_token_inputs":[429,4281],
        "source_predicted_next_token":6230,
        "27_layer_passes":81,
        "GPU_packed_model_role_forward_calls":new["GPU_normal_qng64_model_forward_calls"],
        "true_fused_MLA_attention_queries":new["fused_mla_GPU_attention_query_calls"],
        "KV_gpu_resident_kvb_layer_weights":27,
        "KV_gpu_resident_kvb_n7_bytes":53084160,
        "persistent_BF16_latent_KV_bytes_after_two_generated_inputs":217728,
        "original_native_C_n7_runtime_requantize_calls":0,
        "prior_v11_cached_source_fidelity_relative_L2":old_pref["relative_L2"],
        "new_cached_vs_independent_previous_uniform7_relative_L2":pref["relative_L2"],
        "all_GPU_handles_released":True,
        "MLA_KV_cached_real_model_verified":True,
        "all_5181_model_weights_simultaneously_gpu_resident":False,
        "bitwise_identical_full_logit_arrays_across_arms_verified":False,
        "genuine_serving_TTFT_TPOT_verified":False,
        "production_promotion_authorized":False,
    }
    blob=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode()
    with out.open("xb") as fd:
        fd.write(blob)
        fd.flush()
        os.fsync(fd.fileno())
    print(json.dumps({
        "status":result["status"],
        "new_model_receipt_sha256":nsha,
        "audit_receipt_sha256":hashlib.sha256(blob).hexdigest(),
        "KV_BF16_bytes":217728,
        "predictions":[429,4281,6230],
        "all_handles_released":True,
        "promotion_allowed":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
