#!/usr/bin/env python3
"""BEGLIN Qv2 native whole-decoder CPU async execution: fail-closed audit.

Seven short model paths (serial, both-async, QKV-only, FFN-only, repeated
same-executable controls), original 28-n8 + 168-Q4 precision mask, and
original eight public 131-position evaluation windows. Recompute integrity
and report NEGATIVE/CONFLICTING speed rather than promoting a cherry-picked
short best-of-three latency result. No model inference in this script.

Only reads private original experimental receipts and raw *logits*, not
historical final holds, model checkpoint or private training labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "lab_local"
ACTUAL = DATA / "q_v2_actual_full_cpu_async_20261022"
HYBRID = DATA / "q_v2_hybrid_full_cpu_async_20261022"
ORIGINAL = DATA / "q_v2_activation_async_native_eval_20261022"
EVIDENCE = ROOT / "docs/research/BEGLIN_QV2_ACTIVATION_ASYNC_6POLICY_REAL_NATIVE_20261022.json"
AUDIT_OUT = HYBRID / "Q_V2_FULL_CPU_ASYNC_FAIL_CLOSED_AUDIT_20261022.json"

EXPECTED = {
    "actual_audit": "af142423b54012ca28bdcf3b17c8be6824df1b4e51abbca6eb6a70426314c391",
    "actual_build": "effe1a219234d2b751cd0f7f995d258785caa43322a71c0f0afc5a850b285f12",
    "hybrid_build": "4169bec8a80b4bc624a0a8c7df721817be77e0b81d9fffa24c6e242fec45d46a",
    "repeat": "b13abf47bd2369a95a041741d65dfb9ee55e86a6a233da7742319165382564b2",
}
RUN_SHA = {
    ("qkv_only", "dump"): "b1382f96b2f53deb9c51734d2586450ea44511dfbac303706c3ff08f11ec7c76",
    ("qkv_only", "bench"): "bc4a1c44e7668d51c04a1e608c2ede3d60eeee9325ede0641258d4e304c613e5",
    ("qkv_only", "ppl"): "38aa022fdc4bba9d0f99dbcc4b9b39dd908bbdc6f78742eb4dca47f6da899819",
    ("ffn_only", "dump"): "39bcf1c77d49423f9d0df514fbee6d9a568646c44d452f38424d985a4625b953",
    ("ffn_only", "bench"): "8b90073a0ec4862b7d9a23df4d7707017f4f9fb411f6c72178eb8dbf64a95044",
}
ORIGINAL_MASK_SHA = "cb9268c9177d47705e548c2a90a6b22772569b3f2bb54dfef2767ca53ffa3e76"
PRIOR_LLM_OUTPUT_SHA = "a9ba83b59a03faa84956bb5fc5e4f3ad2867a5b3e53c18455ddde11608d18653"
ACTUAL_BIN = "17a7df7487963ea5b7e30073f046408b022559eca6d5a51b954114a08d53e1e1"
HYBRID_BIN = "4cfc89d2ee4137745d7830a32b799beab20fe681d7c3c87581ce8cd67fffe718"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def private_bytes(path: Path, expected_sha: str | None = None) -> bytes:
    if (
        not path.is_file()
        or path.is_symlink()
        or (path.stat().st_mode & 0o777) != 0o600
    ):
        raise RuntimeError("missing/nonprivate experimental file: " + str(path))
    content = path.read_bytes()
    if expected_sha is not None and sha(content) != expected_sha:
        raise RuntimeError("SHA-bound experimental receipt drift: " + str(path))
    return content


def private_json(path: Path, expected_sha: str | None = None):
    return json.loads(private_bytes(path, expected_sha))


def verify_original_source_and_mask():
    native = private_json(ACTUAL / "build_receipt.json", EXPECTED["actual_build"])
    hybrid = private_json(HYBRID / "build_receipt.json", EXPECTED["hybrid_build"])
    if not native["source_protocol_SHA256"] or not native["output_original_main_unchanged"]:
        raise RuntimeError("source native root not locked")
    if not native["native_q4pool_per_background_role_separate"]:
        raise RuntimeError("actual async Q4 worker pool may race through shared job")
    if native["selected_role_mask_SHA256"] != ORIGINAL_MASK_SHA or hybrid["role_policy_SHA256"] != ORIGINAL_MASK_SHA:
        raise RuntimeError("original quality-failed 28/196 bit policy mutated")
    for path in (ACTUAL / "policy.txt", HYBRID / "policy.txt"):
        if sha(private_bytes(path)) != ORIGINAL_MASK_SHA:
            raise RuntimeError("native mixed model mask changed")
    if sha((ACTUAL / "beglin_qwen_actual_async").read_bytes()) != ACTUAL_BIN:
        raise RuntimeError("actual native C executable changed")
    if sha((HYBRID / "beglin_qwen_stage_selective_native").read_bytes()) != HYBRID_BIN:
        raise RuntimeError("hybrid native C executable changed")
    if (
        native["native_binary_sha256"] != ACTUAL_BIN
        or hybrid["native_hybrid_bin_sha256"] != HYBRID_BIN
        or hybrid["source_prior_async_C_sha256"] != native["actual_async_source_copy_sha256"]
    ):
        raise RuntimeError("source provenance chain broken")
    orig_src = ORIGINAL / "v2_async_activation"
    inputs = ["prompt_ids.i32"] + [f"eval/ppl_work/w{i}.i32" for i in range(8)]
    for rel in inputs:
        standard = private_bytes(orig_src / rel)
        for mode in ("serial", "async"):
            if private_bytes(ACTUAL / mode / rel) != standard:
                raise RuntimeError("same public eval input mismatch in actual CPU async")
        for mode in ("qkv_only", "ffn_only"):
            if private_bytes(HYBRID / mode / rel) != standard:
                raise RuntimeError("same public eval input mismatch in CPU hybrid")
        for mode in ("serial_control", "qkv_repeat", "both_control"):
            if private_bytes(HYBRID / mode / rel) != standard:
                raise RuntimeError("same public eval input mismatch in repeat timing control")
    evidence = json.loads(EVIDENCE.read_text())
    if (
        evidence["private_native_evaluation_summary_sha256"] != "1dfb72b95b11262bbde56e2e47fa52965c96a4fe6c21f089b999ba1c816c9185"
        or evidence["selected_tensors"]["bits8"] != 28
        or evidence["selected_tensors"]["bits4"] != 168
        or evidence["actual_v2"]["actual_PPL"] != 807.363
    ):
        raise RuntimeError("previous frozen Qv2 quality failure evidence changed")
    return native, hybrid, evidence


def compare_logit_files():
    p = private_bytes(ORIGINAL / "v2_async_activation/results/c_logits_last.f32", PRIOR_LLM_OUTPUT_SHA)
    if len(p) != 151936 * 4:
        raise RuntimeError("incorrect original native Qwen vocabulary logits count")
    comparisons = {
        "serial": ACTUAL / "serial/results/c_logits_last.f32",
        "qkv_and_ffn_async": ACTUAL / "async/results/c_logits_last.f32",
        "qkv_only": HYBRID / "qkv_only/results/c_logits_last.f32",
        "ffn_only": HYBRID / "ffn_only/results/c_logits_last.f32",
    }
    outputs = {}
    for label, path in comparisons.items():
        current = private_bytes(path)
        if len(current) != len(p) or current != p:
            raise RuntimeError("real native CPU logits lost bitwise identity: " + label)
        arr = np.frombuffer(current, dtype="<f4")
        if not np.isfinite(arr).all() or int(arr.argmax()) != 9707:
            raise RuntimeError("incorrect fixed native model final logit: " + label)
        outputs[label] = {"bitwise_equal_to_original": True, "logit_sha256": sha(current), "top1_id": 9707}
    return outputs


def audit_receipts():
    original = private_json(ACTUAL / "ACTUAL_NATIVE_CPU_ASYNC_AUDITED_NEGATIVE_SPEED_20261022.json", EXPECTED["actual_audit"])
    if (
        original["status"] != "PASS_ASYNC_CORRECTNESS__NEGATIVE_SPEED_FALLBACK_SERIAL"
        or original["evidence"]["full_decoder_parallel_faster_than_serial"]
        or original["evidence"]["recommended_research_dispatch"] != "serial"
        or original["evidence"]["public_PPL_131_original"] != 807.363
        or original["evidence"]["public_PPL_131_async"] != 807.363
    ):
        raise RuntimeError("original actual async NEGATIVE finding silently changed")
    short = original["evidence"]
    selective = {}
    for (mode, phase), sha_expected in RUN_SHA.items():
        receipt = private_json(HYBRID / f"{mode}_{phase}_execution_receipt.json", sha_expected)
        transcript = private_bytes(HYBRID / f"{mode}_{phase}_native_log.txt")
        if receipt["condition"] != mode or receipt["phase"] != phase or receipt["native_process_log_sha256"] != sha(transcript):
            raise RuntimeError("hybrid real native executable log SHA mismatch")
        if receipt["status"] != "PASS_NATIVE_EXPERIMENTAL_SINGLE_STAGE_CONCURRENCY" or receipt["original_native_binary_sha256"] != HYBRID_BIN:
            raise RuntimeError("different executable or failed experiment")
        q, f = receipt["QKV_stage_async_token_pairs"], receipt["FFN_stage_async_token_pairs"]
        if mode == "qkv_only" and not (q > 0 and f == 0):
            raise RuntimeError("QKV-only actual dependency split not executed")
        if mode == "ffn_only" and not (q == 0 and f > 0):
            raise RuntimeError("FFN-only actual dependency split not executed")
        if phase == "ppl" and (mode != "qkv_only" or receipt.get("PPL") != 807.363):
            raise RuntimeError("same 131-position QKV quality not verified")
        selective.setdefault(mode, {})[phase] = receipt
    samples = private_json(HYBRID / "same_binary_three_trial_results.json", EXPECTED["repeat"])
    rows = {x["name"]: x for x in samples["candidates"]}
    if set(rows) != {"serial_control", "qkv_repeat", "both_control"}:
        raise RuntimeError("three same-executable short latency controls missing")
    for name, r in rows.items():
        if r["binary_sha256"] != HYBRID_BIN:
            raise RuntimeError("not truly same native executable")
        log = private_bytes(HYBRID / f"{name}_same_binary_native_log.txt")
        if sha(log) != r["native_log_sha256"]:
            raise RuntimeError("same executable actual benchmark transcript drift")
        if r["frozen_public_eval_inputs_sha256"] != "62a0b508a74e166e0e4ef56edc684a0bd3365569abf3d4aa4d08376a6b0997f3":
            raise RuntimeError("same-executable prompt bytes different")
    first = {
        "serial": short["native_CPU_best_of_three_serial_ms"],
        "both_parallel": short["native_CPU_best_of_three_async_ms"],
        "qkv_only": selective["qkv_only"]["bench"]["best_of_three_one_decode_ms"],
        "ffn_only": selective["ffn_only"]["bench"]["best_of_three_one_decode_ms"],
    }
    repeat = {name: r["observed_native_single_token_best_of_three_ms"] for name, r in rows.items()}
    if not (
        first["qkv_only"] < first["serial"]
        and first["both_parallel"] > first["serial"]
        and first["ffn_only"] > first["serial"]
        and repeat["qkv_repeat"] > repeat["serial_control"]
        and repeat["both_control"] > repeat["serial_control"]
    ):
        raise RuntimeError("short CPU async performance uncertainty evidence changed")
    if not samples["source_evaluation_quality_failure_unmodified"]:
        raise RuntimeError("bad v2 quality must not be promoted")
    return {
        "same_executable_repeat_receipt_sha256": EXPECTED["repeat"],
        "actual_initial_native_bench_best_of_three_ms": first,
        "same_binary_repeat_bench_best_of_three_ms": repeat,
        "QKV_initial_speed_gain_fraction": first["serial"] / first["qkv_only"] - 1,
        "QKV_repeat_speed_gain_fraction": repeat["serial_control"] / repeat["qkv_repeat"] - 1,
        "best_of_three_short_bench_nonreproducible_QKV_speedup": True,
        "FFN_only_PPL_measured": False,
        "QKV_only_PPL_131_public_positions": 807.363,
        "real_QKV_worker_joins": {
            phase: selective["qkv_only"][phase]["QKV_stage_async_token_pairs"] for phase in ("dump", "bench", "ppl")
        },
        "real_FFN_only_worker_joins": {
            phase: selective["ffn_only"][phase]["FFN_stage_async_token_pairs"] for phase in ("dump", "bench")
        },
        "all_same_source_precision_and_eval_inputs": True,
        "PPL_131_serial_and_both_parallel": 807.363,
    }


def evaluate():
    native, hybrid, old_evidence = verify_original_source_and_mask()
    logits = compare_logit_files()
    trials = audit_receipts()
    reasons = [
        "v2_offline_precision_policy_fails_original_development_131_position_NLL_ceiling",
        "v2_precision_policy_dominated_by_all_Q4_on_original_short_development_quality_memory_speed",
        "full_QKV_and_FFN_parallel_native_benchmark_slower_than_serial",
        "QKV_only_short_best_of_three_speedup_not_reproduced_on_same_binary_repeat",
        "no_repeated_sustained_decode_latency_confidence_interval_or_concurrent_requests",
        "native_pthread_worker_sanitizer_and_oversubscription_safety_not_certified",
        "no_heldout_real_task_AUROC_or_original_V6_24_contamination_audit",
        "no_Metal_GPU_async_or_multi_batch_forward_path_validation",
    ]
    return {
        "schema": "BEGLIN-Qv2-actual-native-CPU-QKV-FFN-async-fail-closed/1",
        "date": "2026-10-22",
        "status": "NOT_READY_DEFAULT_SERIAL",
        "default_runtime_decision": "SERIAL",
        "source_model": "Qwen2.5-1.5B-Instruct",
        "model_source_BF16_rev": "4ae77cb209f06199b8df1c94e21ff341332a3a89",
        "original_preselected_precision_mask_sha256": ORIGINAL_MASK_SHA,
        "native_source_metadata": {
            "whole_model_serial_and_both_async_same_binary": ACTUAL_BIN,
            "whole_model_hybrid_and_repeat_same_binary": HYBRID_BIN,
            "native_model_Q4pool_private_per_pthread": True,
            "native_thread_count_background": 3,
            "original_source_model_and_operational_engine_unchanged": True,
        },
        "evidence_source_shas": dict(EXPECTED),
        "actual_151936_final_logits_bitwise_equal_for_all_four_conditions": logits,
        "actual_native_best_of_three_short_timing": trials,
        "PPL_131_evaluation_dev_positions": 807.363,
        "PPL_131_dev_allQ4_reference": old_evidence["controls"]["all_q4"]["PPL_public_131_tokens"],
        "PPL_131_dev_matched_random_reference": old_evidence["controls"]["role_balanced_random"]["PPL_public_131_tokens"],
        "PPL_dev_reference_F32_from_source_BF16": old_evidence["controls"]["source_f32"]["PPL_public_131_tokens"],
        "quality_gain_established": False,
        "whole_decoder_actual_qkv_and_ffn_async_executed": True,
        "whole_decoder_async_speed_gain_established": False,
        "promotion_blocking_reasons": reasons,
        "production_changed": False,
        "main_changed": False,
        "tailnet_immutable_release_changed": False,
        "original_V6_24_used_holdout_opened": False,
        "prior_security_blocked_per_tensor_causal_logits_intervention_retried": False,
        "scientific_Q_precision_research_verdict": "NO_VERDICT",
        "notes": [
            "Initial short QKV-only speed benefit was contradicted by one controlled same-executable repeat; neither result is a stable p95/CI.",
            "Native QKV/FFN worker barriers are real and produce bit-identical logits, but data-race sanitization was not run.",
            "FFN-only short decode and logit parity measured; FFN-only PPL was not measured because it was already slower.",
            "No automagic live optimization, no background task left running and no production release activation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    os.umask(0o077)
    if not args.verify_only and not AUDIT_OUT.exists():
        with AUDIT_OUT.open("xb") as f:
            f.write((json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
        AUDIT_OUT.chmod(0o600)
    if AUDIT_OUT.exists() and json.loads(private_bytes(AUDIT_OUT)) != result:
        raise RuntimeError("immutable fail-closed audit receipt changed")
    print("BEGLIN_QV2_FULL_CPU_ASYNC_FAIL_CLOSED_AUDIT_PASS",
          json.dumps({
              "status": result["status"],
              "recommendation": result["default_runtime_decision"],
              "repeat_best_of_three_ms": result["actual_native_best_of_three_short_timing"]["same_binary_repeat_bench_best_of_three_ms"],
              "full_native_logits_byte_equal": True,
              "131_position_PPL": result["PPL_131_evaluation_dev_positions"],
              "receipt_sha256": sha(private_bytes(AUDIT_OUT)) if AUDIT_OUT.exists() else None
          }, sort_keys=True))
    if args.require_ready and result["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
