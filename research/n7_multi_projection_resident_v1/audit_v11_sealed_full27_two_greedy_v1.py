#!/usr/bin/env python3
"""Research-only independent v11 host-memo vs sealed-prequant 27L metrics audit.

Compares receipt files; does not read private logits, original prompts or
checkpoint weight payload. Source and model-policy hashes must match; token
sequence, cache parity metrics and actual Metal dispatch/cleanup are checked.

Identical aggregate scalar errors are NOT proof of bitwise identical logits.
This script deliberately does not call it an E2E serving TTFT/TPOT benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def need(ok,message):
    if not ok:
        raise RuntimeError("BEGLIN_V11_SEALED_27L_INDEPENDENT_FAIL_CLOSED:"+message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-receipt",required=True)
    ap.add_argument("--sealed-receipt",required=True)
    ap.add_argument("--full-pack-manifest",required=True)
    ap.add_argument("--full-pack-manifest-sha",required=True)
    ap.add_argument("--pack-artifact-audit",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    old=Path(args.baseline_receipt)
    new=Path(args.sealed_receipt)
    output=Path(args.output)
    gate_file=Path(args.full_pack_manifest)
    pack_audit=Path(args.pack_artifact_audit)
    need(not output.exists(),"append-only independent receipt must not exist")
    need(all(p.is_file() and not p.is_symlink()
             for p in (old,new,gate_file,pack_audit)),
         "one of pinned local research audit receipts is missing/symlink")
    baseline=json.loads(old.read_text())
    sealed=json.loads(new.read_text())
    independently=json.loads(pack_audit.read_text())
    need(digest(gate_file)==args.full_pack_manifest_sha and
         independently["sealed_manifest_sha256"]==args.full_pack_manifest_sha and
         independently["status"]=="PASS_REAL_FULL27_5181_SEALED_ALL_HASHES_27_INDEPENDENT_C_REQUANT" and
         independently["independently_source_requantized_qproj_layers"]==27 and
         independently["all_qng64_group64_bitwidth7_roles"]==5181,
         "whole-model artifact independent authority not certified")

    need(baseline["status"]=="PASS_TWO_GREEDY_TOKENS_DUAL_GPU_V11_GPUOWNED_AND_HOST_SHA_MEMO" and
         sealed["status"]=="PASS_TWO_GREEDY_TOKENS_DUAL_GPU_V11_PREQUANT_SEALED_5181" and
         sealed["schema"]=="beglin.v11.two-greedy-27layer-sealed-n7-prequant/1",
         "baseline/challenger actual 27L two greedy implementation status")
    fixed=["source_checkpoint_sha256","source_config_sha256",
           "source_model_code_sha256","fixed_uniform7_policy_sha256",
           "cpp_source_sha256","cpp_compiled_library_sha256",
           "source_prompt_sha256","model_layers_done"]
    for key in fixed:
        need(baseline.get(key)==sealed.get(key),
             "actual source/code/prompt/precision policy changed: "+key)

    gate_layers = baseline["model_layers_done"]==sealed["model_layers_done"]==27
    need(gate_layers, "full 27-layer coverage missing")
    expected_token_fields={
        "greedy_next_token_id":429,
        "model_selected_second_token_id":4281,
        "greedy_next_prediction_after_second_real_token":6230,
        "final_cached_next_token_id":4281,
        "final_uncached_next_token_id":4281,
    }
    for field, expected in expected_token_fields.items():
        need(baseline.get(field)==sealed.get(field)==expected,
             "actual model-selected token stream differs: "+field)
    for field in ("second_real_token_cached_vs_full_logits",
                  "cached_vs_uncached_final_logits",
                  "prefix_logits_cpp_vs_previous_independent_model"):
        a,b=baseline[field],sealed[field]
        need(isinstance(a,dict) and isinstance(b,dict),
             "original/prequant aggregate logits scope absent: "+field)
        for metric in ("relative_L2","max_abs"):
            aa,bb=a[metric],b[metric]
            need(math.isfinite(aa) and math.isfinite(bb) and
                 abs(aa-bb)<1e-10,
                 "paired aggregate logit error differs: "+field+"."+metric)
    need(baseline["BF16_KV_cache_bytes_real"]==
         sealed["BF16_KV_cache_bytes_real"]==217728,
         "original latent MLA KV cache bytes diverged")
    for field in ("Cpp_Metal_quantized_matrix_modules_created_and_executed",
                  "actual_CPP_gather_called_projected_modules"):
        need(baseline[field]==sealed[field] and baseline[field]>0,
             "actual packed decoder model role coverage differs: "+field)

    old_gpu=baseline["general_Metal_qNg64_bounded_weight_cache"]
    new_gpu=sealed["general_Metal_qNg64_bounded_weight_cache"]
    for field in ("GPU_budget_B","GPU_handle_cap","actual_GPU_forward_calls",
                  "GPU_reuse_hits","GPU_LRU_evictions","GPU_source_roles_ever_registered"):
        need(old_gpu[field]==new_gpu[field],
             "different frozen GPU cache role behavior: "+field)
    need(old_gpu["all_GPU_n7_handles_cleanly_released"] is True and
         new_gpu["all_GPU_n7_handles_cleanly_released"] is True and
         sealed["real_device_GPU_resident_source_KVB"][
           "GPU_native_resident_handles_all_explicitly_released"] is True,
         "GPU weight handles not cleaned up")

    old_memo=baseline["original_C_n7_host_packed_weight_memo_research"]
    new_memo=sealed["original_C_n7_host_packed_weight_memo_research"]
    need(old_memo["original_native_C_quantize_executions"]>0 and
         new_memo["original_native_C_quantize_executions"]==0 and
         new_memo["source_dense_matrix_SHA256_validations"]==0 and
         new_memo["source_quantization_avoided_by_content_hash_hit"]==0 and
         new_memo["resident_host_compact_n7_bytes"]==0,
         "on-the-fly quantization or extra host compressed cache retained")

    package=sealed["v11_sealed_model_loader"]
    need(package["sealed_manifest_sha256"]==args.full_pack_manifest_sha and
         package["frozen_qng64_roles_in_manifest"]==5181 and
         package["on_demand_model_role_mmap_loads"]>0 and
         package["native_C_quantize_calls_during_decode"]==0 and
         package["all_source_roles_from_sealed_manifest"] is True,
         "real whole frozen source prequant adapter role authority not proven")
    need(sealed["v11_native_C_quantize_during_inference"]==0 and
         sealed["original_model_BF16_reference_reads_not_eliminated"] is True and
         sealed["benchmark_original_BF16_source_reloads_and_no_cache_reference_included"] is True and
         sealed["GPU_v11_production_promotion_authorized"] is False and
         sealed["real_end_to_end_decode_tokens_per_second_certified"] is False and
         sealed["production_workload_tokens_per_second_certified"] is False,
         "research vs serving boundary mislabeled")

    result={
        "schema":"beglin.v11.real-full27-two-greedy-sealed-comparison-audit/1",
        "status":"PASS_REAL_FULL27_V11_PREQUANT_GREEDY_TWO_TOKENS_AND_AGGREGATE_LOGITS_MATCH_V10_V11",
        "original_v11_receipt_sha256":digest(old),
        "new_v11_sealed_receipt_sha256":digest(new),
        "prequant_manifest_sha256":args.full_pack_manifest_sha,
        "full_5181_file_SHA_independent_audit_sha256":digest(pack_audit),
        "original_checkpoint_and_model_source_equal":True,
        "original_frozen_bitwidth_policy_equal":True,
        "all_27_decoder_layers_real_Metal":True,
        "real_model_greedy_input_tokens":[429,4281],
        "predicted_token_after_second_real_decode":6230,
        "BF16_KV_bytes":217728,
        "old_requant_executions":old_memo["original_native_C_quantize_executions"],
        "new_requant_executions":0,
        "old_host_compressed_memo_peak_bytes":old_memo.get("max_observed_host_compact_n7_bytes"),
        "new_host_compressed_memo_current_bytes":new_memo["resident_host_compact_n7_bytes"],
        "gpu_weight_forward_calls":new_gpu["actual_GPU_forward_calls"],
        "gpu_reuse_hits":new_gpu["GPU_reuse_hits"],
        "gpu_evictions":new_gpu["GPU_LRU_evictions"],
        "old_process_peak_RSS_bytes":baseline["process_peak_rss_bytes"],
        "new_process_peak_RSS_bytes":sealed["process_peak_rss_bytes"],
        "old_full_research_wall_seconds":baseline["program_wall_seconds"],
        "new_full_research_wall_seconds":sealed["program_wall_seconds"],
        "total_original_Cpp_packed_model_role_modules":
           sealed["Cpp_Metal_quantized_matrix_modules_created_and_executed"],
        "all_final_GPU_handles_released":True,
        "aggregate_logit_error_metrics_match":True,
        "bitwise_identical_full_logit_arrays_verified":False,
        "end_to_end_TTFT_TPOT_verified":False,
        "no_original_BF16_checkpoint_reads_in_reference":False,
        "production_promotion_allowed":False,
    }
    raw=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode("utf-8")
    with output.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({
        "status":result["status"],
        "tokens":result["real_model_greedy_input_tokens"],
        "next_prediction":6230,
        "old_C_quantize":result["old_requant_executions"],
        "new_C_quantize":0,
        "old_RSS":result["old_process_peak_RSS_bytes"],
        "new_RSS":result["new_process_peak_RSS_bytes"],
        "new_GPU_calls":result["gpu_weight_forward_calls"],
        "receipt_sha256":hashlib.sha256(raw).hexdigest(),
    },sort_keys=True))


if __name__=="__main__":
    main()
