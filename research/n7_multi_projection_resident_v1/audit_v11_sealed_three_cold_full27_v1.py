#!/usr/bin/env python3
"""Three independent cold-process 27-layer prequant v11 repeat receipt auditor.

Checks frozen checkpoint/policy/binary, two true greedy input tokens and
next-token prediction, exact scalar cached/uncached logit error parity,
C qNg64 re-quantization zero, all true GPU calls & teardown. Reports raw
per-process research wall/RSS and median/range, never serving TTFT/TPOT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


def required(ok, message):
    if not ok:
        raise ValueError("BEGLIN_THREE_V11_SEALED_COLD_FAIL_CLOSED: "+message)


def read(path):
    p=Path(path)
    required(p.is_file() and not p.is_symlink(),"cold process receipt missing/symlink")
    raw=p.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def close_values(a,b, tolerance=1e-11):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--primary",required=True)
    ap.add_argument("--repeat-a",required=True)
    ap.add_argument("--repeat-b",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    output=Path(args.output)
    required(not output.exists() and output.parent.is_dir(),
             "append-only independent audit path invalid")
    sources=[]
    hashes=[]
    for file in (args.primary,args.repeat_a,args.repeat_b):
        result,digest=read(file)
        sources.append(result)
        hashes.append(digest)
    first=sources[0]
    required(first["status"]=="PASS_TWO_GREEDY_TOKENS_DUAL_GPU_V11_PREQUANT_SEALED_5181",
             "not a real complete-27 decoder baseline")
    keys=("source_checkpoint_sha256","source_config_sha256",
          "source_model_code_sha256","fixed_uniform7_policy_sha256",
          "cpp_source_sha256","cpp_compiled_library_sha256",
          "source_prompt_sha256","model_layers_done")
    for j,model in enumerate(sources):
        required(model["status"]==first["status"] and
                 model["schema"]==first["schema"] and
                 model["model_layers_done"]==27,
                 "repetition not successful full27 source model "+str(j))
        for key in keys:
            required(model[key]==first[key], "repetition identity changed: "+key)
        for key,value in (
            ("greedy_next_token_id",429),
            ("model_selected_second_token_id",4281),
            ("greedy_next_prediction_after_second_real_token",6230),
            ("BF16_KV_cache_bytes_real",217728),
        ):
            required(model[key]==value, "repeated true sampled tokens/KV mismatch: "+key)
        for name in ("second_real_token_cached_vs_full_logits",
                     "cached_vs_uncached_final_logits",
                     "prefix_logits_cpp_vs_previous_independent_model"):
            for field in ("relative_L2","max_abs"):
                required(close_values(model[name][field],first[name][field]),
                         "logits scalar comparison varied across process: "+name+"/"+field)
        full=model["v11_sealed_model_loader"]
        required(full["frozen_qng64_roles_in_manifest"]==5181 and
                 full["sealed_manifest_sha256"]==
                 first["v11_sealed_model_loader"]["sealed_manifest_sha256"] and
                 full["native_C_quantize_calls_during_decode"]==0 and
                 model["original_C_n7_host_packed_weight_memo_research"]
                   ["original_native_C_quantize_executions"]==0 and
                 model["original_C_n7_host_packed_weight_memo_research"]
                   ["resident_host_compact_n7_bytes"]==0,
                 "source role no-requantization/manifest contract changed")
        a=model["general_Metal_qNg64_bounded_weight_cache"]
        b=first["general_Metal_qNg64_bounded_weight_cache"]
        for k in ("actual_GPU_forward_calls","GPU_LRU_evictions","GPU_reuse_hits",
                  "GPU_source_roles_ever_registered","GPU_budget_B"):
            required(a[k]==b[k],"actual physical GPU role/dispatch changed: "+k)
        required(a["all_GPU_n7_handles_cleanly_released"] is True and
                 model["real_device_GPU_resident_source_KVB"]
                   ["GPU_native_resident_handles_all_explicitly_released"] is True,
                 "GPU handles left alive after independent process")
        required(model["real_end_to_end_decode_tokens_per_second_certified"] is False and
                 model["production_workload_tokens_per_second_certified"] is False and
                 model["GPU_v11_production_promotion_authorized"] is False,
                 "research performance misrepresented as serving")
    times=[x["program_wall_seconds"] for x in sources]
    rss=[x["process_peak_rss_bytes"] for x in sources]
    result={
        "schema":"beglin.v11.sealed-prequant-three-cold-process-repeats/1",
        "status":"PASS_THREE_COLD_FULL27_PREQUANT_REAL_TWO_GREEDY_IDENTICAL_METRICS",
        "original_27_layer_model":True,
        "independent_cold_processes":3,
        "original_checkpoint_attested_and_same":True,
        "full_prequant_model_roles":5181,
        "two_real_source_greedy_inputs":[429,4281],
        "next_token_predicted":6230,
        "all_27_decoder_layers_thrice":True,
        "raw_receipt_SHA256s":hashes,
        "observed_full_protocol_seconds":times,
        "observed_full_protocol_peak_RSS_bytes":rss,
        "protocol_seconds_median":statistics.median(times),
        "protocol_seconds_min":min(times),
        "protocol_seconds_max":max(times),
        "peak_RSS_bytes_median":statistics.median(rss),
        "peak_RSS_bytes_min":min(rss),
        "peak_RSS_bytes_max":max(rss),
        "all_processes_zero_requantization":True,
        "all_processes_GPU_handles_released":True,
        "GPU_actual_forward_calls_per_process":
            first["general_Metal_qNg64_bounded_weight_cache"]["actual_GPU_forward_calls"],
        "full_logits_bytewise_identity_across_processes_verified":False,
        "three_sample_p95_not_reported_due_to_small_n":True,
        "sustained_real_request_concurrency_verified":False,
        "pure_serving_TTFT_TPOT_certified":False,
        "production_promotion_allowed":False,
    }
    raw=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode()
    with output.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({
        "status":result["status"],
        "processes":3,"median_seconds":result["protocol_seconds_median"],
        "time_range_seconds":[min(times),max(times)],
        "median_peak_RSS_bytes":result["peak_RSS_bytes_median"],
        "receipt_sha256":hashlib.sha256(raw).hexdigest(),
    },sort_keys=True))


if __name__=="__main__":
    main()
