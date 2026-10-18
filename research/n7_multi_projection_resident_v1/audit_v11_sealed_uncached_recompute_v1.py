#!/usr/bin/env python3
"""Independent research-only v11 sealed-prequant vs v11 baseline TWO greedy audit.

The old baseline uses fused MLA KV caching, SHA CPU compressed memo and GPU
computed ownership. The challenger recomputes the entire source-defined
DeepSeek-V2-Lite decoder for each token from the sealed all-5181 qNg64 source
and physically executes each selected packed role on Metal. They are NOT
serving-latency / memory equivalent arms.

We require exact original source/prompt/quant policy identities, three complete
27-layer model steps, actual generated inputs 429,4281, next predicted 6230,
legacy source logits parity on the *first* step, both original C and Metal
numerical gates, and all GPU handles released. Neither test proves identical
full logits arrays, KV cache integration, or production TTFT/TPOT.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

INDEX="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
MANIFEST="c82361f7fbd73884b633bb595334805c0b7c0cf2f82ab8de26311ca78a0142b8"
BASELINE="8c36149cd9b062724efe604495b5f56d14aede59d9b86883ea93757cb8676745"
POLICY="39b98921c30a992f93ea896283336a6b5fe1810c5737443132f2410b474e9612"
SOURCE_AUTHORITY="3b13bc1d98c5843f146afa428b6dfff45352f7def698d6a5a156d32d0fae136f"


def gate(ok,why):
    if not ok:
        raise RuntimeError("BEGLIN_V11_SEALED_RECOMPUTE_AUDIT_FAIL_CLOSED: "+why)


def raw(path):
    p=Path(path)
    gate(p.is_file() and not p.is_symlink() and 0<p.stat().st_size<12*1024*1024,
         "bounded regular audit source missing")
    return p.read_bytes()


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",required=True)
    ap.add_argument("--challenger",required=True)
    ap.add_argument("--sealed-manifest",required=True)
    ap.add_argument("--source-authority",required=True)
    ap.add_argument("--original-policy",required=True)
    ap.add_argument("--output",required=True)
    x=ap.parse_args()
    gate(not Path(x.output).exists(),"audit output must be append-only")
    baseline_raw=raw(x.baseline)
    challenger_raw=raw(x.challenger)
    gate(sha(baseline_raw)==BASELINE,"unchanged original v11 source receipt SHA")
    gate(sha(raw(x.sealed_manifest))==MANIFEST,"sealed full 5181 role manifest")
    gate(sha(raw(x.source_authority))==SOURCE_AUTHORITY,"original shard/exporter authority")
    gate(sha(raw(x.original_policy))==POLICY,"original uniform7 policy SHA")
    old=json.loads(baseline_raw)
    new=json.loads(challenger_raw)
    gate(old.get("status")=="PASS_TWO_GREEDY_TOKENS_DUAL_GPU_V11_GPUOWNED_AND_HOST_SHA_MEMO"
         and old.get("source_checkpoint_sha256")==INDEX
         and old.get("fixed_uniform7_policy_sha256")==POLICY
         and old.get("model_layers_done")==27
         and old.get("greedy_next_token_id")==429
         and old.get("model_selected_second_token_id")==4281
         and old.get("greedy_next_prediction_after_second_real_token")==6230
         and old.get("BF16_KV_cache_bytes_real")==217728,
         "original v11 27L cached baseline not independently established")
    gate(new.get("schema")=="beglin.v11.real-deepseek-full27-sealed-n7-metal-uncached-recompute/1"
         and new.get("status")=="PASS_REAL_FULL27_PREQUANT_TWO_GREEDY_UNCACHED"
         and new.get("checkpoint_index_sha256")==INDEX
         and new.get("packed_source_manifest_sha256")==MANIFEST
         and new.get("model_layers")==27,
         "new sealed full27 physical Metal source contract")
    gate(old["source_prompt_sha256"]==new.get("source_prompt_sha256"),
         "old/new source prompt digest changed")
    stages=new.get("stage_receipts")
    gate(type(stages) is list and len(stages)==3,
         "exactly three original model forward paths required")
    for j,step in enumerate(stages):
        gate(step.get("context_length")==5+j
             and step.get("model_layers")==27
             and step.get("full_model_logits_computed") is True
             and type(step.get("packed_source_module_instances")) is int
             and step.get("packed_source_module_instances")>27
             and type(step.get("vocab_logits_sha256")) is str
             and len(step["vocab_logits_sha256"])==64,
             "complete 27L original MoE/packed GPU logits missing at step "+str(j))
    predicted=[s["next_prediction"] for s in stages]
    gate(predicted==[429,4281,6230]
         and new.get("generated_token_inputs")==[429,4281]
         and new.get("predicted_token_after_last_input")==6230,
         "model-selected autoregressive sequence differs")
    rel=stages[0].get("prefill_logits_relative_L2_vs_prior_uniform7")
    gate(type(rel) is float and math.isfinite(rel) and rel<0.025,
         "fresh original logits changed from independent previous full27 uniform n7")
    gate(new.get("actual_model_source_expert_routing") is True
         and new.get("native_C_cpp_Metal_v11_GPU_compressed_all_matmuls") is True
         and new.get("dynamic_C_quantizations_during_inference")==0
         and new.get("compressed_disk_role_load_instances") ==
             new.get("GPU_original_C_oracle_checked_instances")
         and new.get("compressed_disk_role_load_instances")>2000
         and new.get("Metal_GPU_registration_count")>0
         and new.get("normal_Model_GPU_forward_calls") >=
             new.get("compressed_disk_role_load_instances")
         and new.get("physical_Metal_GPU_forward_calls") ==
             new["normal_Model_GPU_forward_calls"] +
             new.get("extra_native_Metal_F32_oracle_validation_calls",0),
         "all selected packed matrices did not use actual C-tested Metal dispatch")
    gate(type(new.get("GPU_vs_C_max_BF16_relative_L2")) is float
         and new["GPU_vs_C_max_BF16_relative_L2"]<0.005
         and new.get("GPU_vs_native_C_max_F32_relL2_on_BF16_ambiguity",0)<1e-5,
         "GPU/CPU original C arithmetic or BF16 rounding contract drift")
    gate(new.get("all_GPU_handles_released") is True
         and new.get("MLA_cached_KV_tested") is False
         and new.get("recomputed_full_context_each_step") is True
         and new.get("BF16_original_source_layer_and_expert_loads_still_present") is True
         and new.get("real_serving_TTFT_TPOT_validated") is False
         and new.get("source_model_weights_gpu_full_resident") is False
         and new.get("original_checkpoint_modified") is False
         and new.get("production_promotion_allowed") is False,
         "research-only recompute and cleanup scope not preserved")
    receipt={
        "schema":"beglin.v11.real-full27-sealed-n7-recompute-vs-cached-independent-audit/1",
        "status":"PASS_REAL_FULL27_TWO_GREEDY_SEALED_N7_METAL_RECOMPUTE_SOURCE_AND_TOKEN_PARITY",
        "source_original_index_sha256":INDEX,
        "source_original_policy_sha256":POLICY,
        "sealed_manifest_sha256":MANIFEST,
        "source_authority_sha256":SOURCE_AUTHORITY,
        "old_v11_cached_receipt_sha256":BASELINE,
        "challenger_uncached_receipt_sha256":sha(challenger_raw),
        "original_model_selected_prefix_and_two_token_predictions":[429,4281,6230],
        "full_model_decoder_layer_passes":3*27,
        "actual_selected_source_gpu_model_module_instances":
            new["compressed_disk_role_load_instances"],
        "actual_native_metal_model_forward_calls":
            new["normal_Model_GPU_forward_calls"],
        "extra_native_Metal_F32_oracle_validation_calls":
            new["extra_native_Metal_F32_oracle_validation_calls"],
        "source_native_C_requantization_during_inference":0,
        "original_model_BF16_reference_loader_reads_present":True,
        "all_gpu_handles_cleaned":True,
        "prior_prefix_logits_relative_L2":rel,
        "bitwise_equal_entire_full_logits_arrays_verified":False,
        "MLA_KV_cached_candidate_verified":False,
        "TTFT_TPOT_comparable_to_old_v11":False,
        "production_promotion_allowed":False,
    }
    data=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
    out=Path(x.output)
    with out.open("xb") as fd:
        fd.write(data)
        fd.flush()
        os.fsync(fd.fileno())
    print(json.dumps({
        "status":receipt["status"],
        "modules":receipt["actual_selected_source_gpu_model_module_instances"],
        "real_token_inputs":[429,4281],
        "next_prediction":6230,
        "new_run_receipt_sha256":receipt["challenger_uncached_receipt_sha256"],
        "independent_audit_receipt_sha256":sha(data),
        "production_allowed":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
