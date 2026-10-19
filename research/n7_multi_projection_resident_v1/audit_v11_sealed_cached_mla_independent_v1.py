#!/usr/bin/env python3
"""Read-only independent numerical/authority audit of sealed v11 cached MLA.

Compare original DeepSeek-V2-Lite v11 SHA-pinned cached reference, independent
uncached 27-layer sealed run, and NEW direct sealed n7 + fused MLA actual Metal.
Never change, download, store or expose private model payloads, logits arrays,
prompts or GPU resident weights. Source identity and all receipts are local.

This certifies original-source/packed-MLA integration and token sequence only:
not bitwise identity of entire cached logits arrays, TTFT/TPOT or robustness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import math

ORIGINAL_INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
ORIGINAL_POLICY_SHA="39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612"
ORIGINAL_V11_BASELINE_SHA="8c36149cd9b062724efe604495b5f56d14aede59d9b86883ea93757cb8676745"
SEALED_MANIFEST_SHA="c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8"
ORIGINAL_RUN_AUTHORITY_SHA="3b13bc1d98c5843f146afa428b6dfff45352f7def698d6a5a156d32d0fae136f"
UNCACHED_TWO_GREEDY_SHA="ae29603fec2814c5164bd7598006117a7ba776cd154243e1a215cf5b30d8263e"
MLA_WEIGHT_PERSISTENT_PER_LAYER=1966080
KV_BF16_PER_TOKEN_PER_LAYER=(512+64)*2
GENERATED=[429,4281]
PREDICTED=[429,4281,6230]


def gate(condition,why):
    if not condition:
        raise RuntimeError("BEGLIN_SEALED_CACHED_MLA_INDEPENDENT_FAIL_CLOSED: "+why)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, expected_sha=None):
    p=Path(path)
    gate(p.is_file() and not p.is_symlink() and 0<p.stat().st_size<32*1024*1024,
         "bounded local source receipt missing/untrusted")
    raw=p.read_bytes()
    if expected_sha is not None:
        gate(digest(raw)==expected_sha,
             "original immutable authority/receipt SHA mismatch: "+p.name)
    return json.loads(raw),digest(raw)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-v11",required=True)
    p.add_argument("--sealed-manifest",required=True)
    p.add_argument("--run-authority",required=True)
    p.add_argument("--uncached-sealed",required=True)
    p.add_argument("--cached-sealed",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    output=Path(args.output)
    gate(not output.exists() and not output.is_symlink(),
         "append-only independent audit receipt path must be new")
    baseline,baseline_sha=read_json(args.original_v11,ORIGINAL_V11_BASELINE_SHA)
    manifest,manifest_sha=read_json(args.sealed_manifest,SEALED_MANIFEST_SHA)
    authority,authority_sha=read_json(args.run_authority,ORIGINAL_RUN_AUTHORITY_SHA)
    uncached,uncached_sha=read_json(args.uncached_sealed,UNCACHED_TWO_GREEDY_SHA)
    cached,cached_sha=read_json(args.cached_sealed)

    gate(baseline.get("status")==
         "PASS_TWO_GREEDY_TOKENS_DUAL_GPU_V11_GPUOWNED_AND_HOST_SHA_MEMO" and
         baseline.get("source_checkpoint_sha256")==ORIGINAL_INDEX_SHA and
         baseline.get("fixed_uniform7_policy_sha256")==ORIGINAL_POLICY_SHA and
         baseline.get("model_layers_done")==27 and
         [baseline.get("greedy_next_token_id"),
          baseline.get("model_selected_second_token_id"),
          baseline.get("greedy_next_prediction_after_second_real_token")]==PREDICTED,
         "independent original v11 reference model authority/token mismatch")
    gate(manifest.get("schema")=="beglin.qng64.n7.prequant-sealed/1" and
         manifest.get("source_checkpoint_index_sha256")==ORIGINAL_INDEX_SHA and
         len(manifest.get("roles",[]))==5181 and
         len({entry["role"] for entry in manifest["roles"]})==5181,
         "sealed role manifest missing full original 5181 policy")
    gate(authority.get("index_sha256")==ORIGINAL_INDEX_SHA and
         authority.get("qng64_role_count")==5181 and
         authority.get("BF16_exclusion_count")==110 and
         authority.get("quantized_expected_bytes")==14328299520,
         "sealed original source/exporter integrity")
    gate(uncached.get("status")=="PASS_REAL_FULL27_PREQUANT_TWO_GREEDY_UNCACHED" and
         uncached.get("checkpoint_index_sha256")==ORIGINAL_INDEX_SHA and
         uncached.get("packed_source_manifest_sha256")==SEALED_MANIFEST_SHA and
         uncached.get("generated_token_inputs")==GENERATED and
         uncached.get("predicted_token_after_last_input")==6230 and
         uncached.get("all_GPU_handles_released") is True,
         "independent uncached whole model model-source candidate mismatched")

    gate(cached.get("schema")=="beglin.v11.original-deepseek-full27-sealed-n7-fused-MLA-cached/1" and
         cached.get("status")=="PASS_REAL_FULL27_TWO_GREEDY_SEALED_N7_FUSED_MLA_KV",
         "new direct real source fused MLA all27/two greedy run failed")
    gate(cached.get("checkpoint_index_sha256")==ORIGINAL_INDEX_SHA and
         cached.get("full_sealed_prequant_manifest_sha256")==SEALED_MANIFEST_SHA and
         cached.get("original_source_policy_sha256")==ORIGINAL_POLICY_SHA and
         cached.get("source_prompt_sha256")==baseline.get("source_prompt_sha256") ==
            uncached.get("source_prompt_sha256"),
         "original/model/source/prompt/precision change")
    gate(cached.get("model_layers")==27 and
         cached.get("greedy_input_tokens")==GENERATED and
         cached.get("predictions")==PREDICTED,
         "actual original autoregressive greedy embedded inputs differ")
    gate(cached.get("source_model_true_MoE_gate_experts") is True and
         cached.get("device_resident_fused_kvb_weight_layers")==27 and
         cached.get("per_layer_fused_kvb_weight_registered_once") is True and
         cached.get("fused_mla_kvb_resident_GPU_packed_bytes")==
             27*MLA_WEIGHT_PERSISTENT_PER_LAYER and
         cached.get("fused_mla_GPU_attention_query_calls")>=27*7,
         "real source resident kv_b GPU handles / original fused MLA kernel invocation")
    gate(cached.get("persistent_MLA_KV_total_bytes")==
             27*7*KV_BF16_PER_TOKEN_PER_LAYER and
         cached.get("persistent_KV_source_storage")==
             "BF16 512D normalized latent + 64D rotated RoPE key" and
         cached.get("expanded_MLA_KV_persistent") is False,
         "original small persistent BF16 latent KV source format drift")
    gate(cached.get("CPU_original_qNg64_requant_during_model_generation")==0 and
         cached.get("GPU_normal_qng64_model_forward_calls",0)>0 and
         cached.get("source_new_model_role_compressed_file_loads",0)>2000 and
         cached.get("source_gpu_C_oracle_checked_role_modules",0)>2000,
         "real model compressed source load / native Metal+C oracle was bypassed")
    gate(cached.get("BF16_original_source_layer_and_reference_weight_reads_still_present") is True and
         cached.get("full_model_GPU_exclusively_resident") is False and
         cached.get("genuine_serving_TTFT_TPOT_certified") is False and
         cached.get("original_source_modified") is False and
         cached.get("production_promotion_authorized") is False and
         cached.get("all_GPU_weight_handles_fully_released") is True and
         cached.get("all_MLA_KV_layers_detached_from_original_attention") is True and
         cached.get("completed_without_original_checkpoint_changes") is True,
         "experimental vs serving boundary or GPU cleanup claim drift")

    stages=cached.get("stage_receipts")
    gate(isinstance(stages,list) and len(stages)==3,
         "prefill plus two real model-selected KV decode stages missing")
    for i,stage in enumerate(stages):
        gate(stage.get("source_phase")==("PREFILL" if i==0 else "DECODE_"+str(i)) and
             stage.get("source_layer_count")==27 and
             stage.get("latent_KV_size_bytes")==27*(5+i)*KV_BF16_PER_TOKEN_PER_LAYER and
             stage.get("source_gpu_packed_module_instances_this_stage",0)>0,
             "source 27-layer KV history/tensor lifetime drift in step "+str(i))
    before=stages[0].get("logits_compare_original_uniform7")
    gate(isinstance(before,dict) and
         isinstance(before.get("relative_L2"),(int,float)) and
         math.isfinite(before["relative_L2"]) and
         before["relative_L2"]<0.025,
         "original numeric prefill C++ source didn't match historical logits")

    result={
        "schema":"beglin.v11.real-model-full27-sealed-n7-fused-mla-independent-audit/1",
        "status":"PASS_REAL_FULL27_SEALED_PREQUANT_GPU_FUSED_MLA_TWO_GREEDY_SOURCE_PARITY",
        "original_model_index_sha256":ORIGINAL_INDEX_SHA,
        "uniform_source_quant_policy_sha256":ORIGINAL_POLICY_SHA,
        "whole_model_sealed_manifest_sha256":SEALED_MANIFEST_SHA,
        "source_checkpoint_and_exporter_authority_sha256":ORIGINAL_RUN_AUTHORITY_SHA,
        "original_v11_baseline_receipt_sha256":baseline_sha,
        "uncached_sealed_27L_receipt_sha256":uncached_sha,
        "new_cached_sealed_27L_receipt_sha256":cached_sha,
        "true_model_greedy_selected_inputs":GENERATED,
        "actual_next_predictions":PREDICTED,
        "original_source_27_layers_each_phase":3,
        "fused_MLA_GPU_resident_KVB_packed_bytes":27*MLA_WEIGHT_PERSISTENT_PER_LAYER,
        "BF16_latent_MLA_persistent_KV_bytes":27*7*KV_BF16_PER_TOKEN_PER_LAYER,
        "physical_Metal_normal_GEMV":cached["GPU_normal_qng64_model_forward_calls"],
        "physical_Metal_fused_attention_queries":cached["fused_mla_GPU_attention_query_calls"],
        "actual_prequant_model_role_load_instances":cached["source_new_model_role_compressed_file_loads"],
        "on_the_fly_native_C_quantizations":0,
        "all_compressed_GPU_handles_cleaned":True,
        "prior_prefix_logits_relL2":before["relative_L2"],
        "whole_logits_byte_identity_confirmed":False,
        "fully_GPU_resident_model_certified":False,
        "throughput_TTFT_TPOT_certified":False,
        "concurrent_model_requests_verified":False,
        "production_promotion_authorized":False,
    }
    content=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode("utf8")
    with output.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status":result["status"],
                      "tokens":result["actual_next_predictions"],
                      "model_GEMVs":result["physical_Metal_normal_GEMV"],
                      "fused_queries":result["physical_Metal_fused_attention_queries"],
                      "KV_bytes":result["BF16_latent_MLA_persistent_KV_bytes"],
                      "independent_receipt_SHA256":digest(content),
                      "new_inference_receipt_SHA256":cached_sha,
                      "production_authorized":False},sort_keys=True))


if __name__=="__main__":
    main()
