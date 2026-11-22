#!/usr/bin/env python3
"""JE01 true original27 activation-driven JOINT SwiGLU FFN replay.

This is *not* a 27-layer language-model NLL or Fisher/Hessian measurement.
Computes complete nonlinear gate/up -> SiLU -> down FFN for original AF4
and one already-frozen clean TRAIN240 SSE-DP rolewise precision challenger,
relative to exact same Chat BF16 source FFN at actual original AF4 h2.
Includes per-(case,layer) co-routed Expert aggregation (including cross terms).
No heldout/OOD labels; never modify production.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

WORK = Path("/Volumes/D50/TailnetCommander/work")
ACT = WORK / "je01_real_original256_activation_20261122_v1"
NATIVE = WORK / "je01_real_router256_native_weight_dp_20261122_v1"
CAL_SCRIPT = ACT / "je01_cal240_native_activation_dp_source_20261122.py"
CAL_SCRIPT_SHA = "863fb9c1d5a17fc810564610739bd1c53f727addee99a7966f65404b8725a414"
CAL_LEDGER_NAME = "JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP.json"
CAL_LEDGER_SIGNED_SHA = "35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb"
TRACE_SIGNED_SHA = "b911aef00007b1f9b44214996353f93c7218fe51c82daf91966e450ebba460b6"
AF_SIGNED_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
ROUTER_SIGNED_SHA = "62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb"
COST = {"AF4": 1802240, "Q4": 1622016, "Q5": 1982464}
FROZEN = ((12, 13), (13, 22), (13, 28), (14, 63),
          (20, 17), (21, 61), (24, 26), (24, 60))
ROLES = ("gate_proj", "up_proj", "down_proj")
SCHEMA = "beglin-je01-true-original27-clean240-joint-nonlinear-swiglu-composition-ffn-coexpert-source-replay/1"


def need(c, msg):
    if not c:
        raise ValueError(msg)


def sha_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def canonical(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def silu(x):
    """Stable float32 SiLU, matching model's SiLU gate math (not exact Metal parity)."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    good = x >= 0
    out[good] = x[good] / (1.0 + np.exp(-x[good]))
    e = np.exp(x[~good])
    out[~good] = x[~good] * (e / (1.0 + e))
    return out


def find_ledger():
    found = [p for p in WORK.glob("je01*/**/" + CAL_LEDGER_NAME)
             if p.is_file() and not p.is_symlink() and
             len(p.relative_to(WORK).parts) <= 5]
    need(len(found) == 1, "JE01_FFN_CAL240_ROLE_LEDGER_DISCOVERY_" + str(len(found)))
    p = found[0]
    d = json.loads(p.read_text())
    digest = d.get("receipt_sha256")
    body = {k: v for k, v in d.items() if k != "receipt_sha256"}
    need(digest == CAL_LEDGER_SIGNED_SHA and canonical(body) == digest and
         d.get("status") == "JE01_ACTUAL_ORIGINAL27_CAL240_UNSEEN_FFN_H2_SWIGLU_24ROLE_ACTIVATION_DP_COMPLETE",
         "JE01_FFN_ROLE_LEDGER_DIGEST_OR_STATUS_INVALID")
    need(d.get("training_first16_prior_pilot_excluded_from_precision_selection") is True and
         d.get("training_unseen240_prompt_count") == 240 and
         d.get("observed_total_original256_routed_activation_records") == 682 and
         d.get("source_actual_training_original27_256_h2_sw_trace_receipt_sha256") == TRACE_SIGNED_SHA and
         d.get("source_full_original_AF4_sha256") == AF_SIGNED_SHA and
         d.get("source_original_router256_receipt_sha256") == ROUTER_SIGNED_SHA and
         d.get("source_heldout_OOD_labels_used") is False and
         d.get("production_modified") is False, "JE01_FFN_ROLE_LEDGER_SCOPE_DRIFT")
    best = d["separate_exactbyte_dp_objective_results"]["route_weighted_output_SSE"]["best_overall"]
    mapping = {(l, e, role): tier for l, e, role, tier in best["source_assignment"]}
    need(len(mapping) == 24 and all((l, e, r) in mapping for l, e in FROZEN for r in ROLES) and
         sum(t == "Q4" for t in mapping.values()) == 6 and
         sum(t == "Q5" for t in mapping.values()) == 6 and
         sum(COST[t] for t in mapping.values()) == 43253760 and
         math.isclose(best["risk_difference"], -1607.2109554533818,
                      rel_tol=1e-10, abs_tol=1e-5), "JE01_FFN_ROLE_POLICY_NOT_FROZEN")
    role_receipts = {(r["layer"], r["expert"], r["role"]): r for r in d["per_role"]}
    need(len(role_receipts) == 24, "JE01_ROLE_SOURCE_PROVENANCE_NOT_COMPLETE")
    return mapping, role_receipts, str(p)


def load_cal_source():
    need(CAL_SCRIPT.is_file() and not CAL_SCRIPT.is_symlink() and
         sha_file(CAL_SCRIPT) == CAL_SCRIPT_SHA, "JE01_SOURCE_CAL240_SCRIPT_SHA_CHANGED")
    spec = importlib.util.spec_from_file_location("je01_cal240_preverified", CAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ffn_output(weights, X):
    """All gate/up/down weights from the *same* tier policy and same h2."""
    gate, up, down = (weights[r] for r in ROLES)
    G = (gate @ X).astype(np.float32)
    U = (up @ X).astype(np.float32)
    SW = silu(G) * U
    Y = (down @ SW).astype(np.float32)
    need(np.isfinite(SW).all() and np.isfinite(Y).all(), "JE01_FFN_NAN_OR_INF")
    return Y, SW


def weighted_sse(delta, probs):
    delta = np.asarray(delta, dtype=np.float64)
    return float(np.sum(probs * np.sum(delta * delta, axis=0)))


def run(output):
    need(not output.exists() and not output.is_symlink() and
         output.parent == WORK, "JE01_JOINT_FFN_NEW_RESEARCH_ROOT_ONLY")
    policy, role_receipts, policy_path = find_ledger()
    mod = load_cal_source()
    data, trace_sha, trace = mod.load()
    need(trace_sha == TRACE_SIGNED_SHA and
         len(data) == 8 and sum(len(v) for v in data.values()) == 637,
         "JE01_NATIVE_CAL240_H2_SOURCES_NOT_CORRECT")
    native, helper = mod.original()
    af = native.safe_file(native.AF_ROOT, "deepseek_moe_af.bin",
                          native.AF_FILE_SHA, 9814343680, rehash=True)
    layout = native.read_af_layout()
    weightmap, shards, headers = native.read_bf16_headers()
    dylib = NATIVE / "result" / "je01_native_q4q5.dylib"
    need(dylib.is_file() and not dylib.is_symlink(), "JE01_NATIVE_CODEC_MISSING")
    lib = ctypes.CDLL(str(dylib))
    f32 = ctypes.POINTER(ctypes.c_float)
    lib.gguf_quantize_q4g64_error_feedback.argtypes = [
        f32, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.gguf_quantize_qNg64.argtypes = [
        f32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]

    per_expert = []
    # (training prompt index, layer) -> sum over frozen co-routed Experts p * delta FFN output
    coexpert = {}
    for layer, expert in FROZEN:
        selected = data[(layer, expert)]
        n = len(selected)
        need(n >= 32, "JE01_REAL_CLEAN_EXPERT_ROUTING_TOO_SMALL")
        cases = [s[0] for s in selected]
        probs = np.array([s[1] for s in selected], dtype=np.float64)
        X = np.stack([s[2] for s in selected], axis=1).astype(np.float32)
        actual_sw = np.stack([s[3] for s in selected], axis=1).astype(np.float32)
        need(X.shape == (2048, n) and actual_sw.shape == (1408, n) and
             all(16 <= i < 256 for i in cases) and
             len(set(cases)) == n, "JE01_CAL240_SAMPLES_SCOPE_INVALID")

        weight_ref, weight_af, weight_candidate = {}, {}, {}
        for role in ROLES:
            af_w, _, _, _, af_src = native.affine4_original(af, layout, layer, expert, role)
            bf_w, bf_src = native.bf16_selected(weightmap, shards, headers,
                                                 layer, expert, role)
            original = role_receipts[(layer, expert, role)]
            need(original["actual_original_BF16_tensor_sha256"] ==
                 bf_src["original_bf16_tensor_sha256"] and
                 original["actual_original_AF4_packed_sha256"] == af_src["af_packed_sha256"],
                 "JE01_ROLE_NATIVE_WEIGHT_SOURCE_CHANGED")
            weight_ref[role], weight_af[role] = bf_w, af_w
            tier = policy[(layer, expert, role)]
            weight_candidate[role] = (af_w if tier == "AF4" else
                                       helper.quant(lib, bf_w, 4 if tier == "Q4" else 5)[0])

        y_ref, _ = ffn_output(weight_ref, X)
        y_af, sw_af = ffn_output(weight_af, X)
        y_candidate, _ = ffn_output(weight_candidate, X)

        parity_diff = sw_af.astype(np.float64) - actual_sw.astype(np.float64)
        parity_maxabs = float(np.max(np.abs(parity_diff)))
        parity_relrmse = math.sqrt(float(np.sum(parity_diff ** 2)) /
                                  max(1e-20, float(np.sum(actual_sw.astype(np.float64) ** 2))))
        # Independently check that the reconstructed AF4 gate/up SwiGLU matches
        # the actual native h2-to-SwiGLU trace before drawing any scientific conclusion.
        print(json.dumps({"layer": layer, "expert": expert, "clean_routes": n,
                          "swiglu_parity_relrmse": parity_relrmse,
                          "swiglu_parity_maxabs": parity_maxabs}, sort_keys=True), flush=True)
        need(parity_relrmse < 0.005 and parity_maxabs < 0.25,
             "JE01_RECONSTRUCTED_SILU_VS_NATIVE_ACTUAL_PARITY_FAIL")

        off_diff = y_af.astype(np.float64) - y_ref.astype(np.float64)
        on_diff = y_candidate.astype(np.float64) - y_ref.astype(np.float64)
        sse_off = weighted_sse(off_diff, probs)
        sse_on = weighted_sse(on_diff, probs)
        energy = float(np.sum(probs * np.sum(y_ref.astype(np.float64)**2, axis=0)))
        need(energy > 0 and np.isfinite(energy), "JE01_REAL_FFN_BF16_ENERGY_ZERO")
        per_expert.append({
            "layer": layer, "expert": expert, "training_clean_prompt_events": n,
            "weighted_true_BF16_reference_FFN_energy": energy,
            "full_swiglu_AF4_baseline_weighted_SSE": sse_off,
            "full_swiglu_frozen_SSE_policy_weighted_SSE": sse_on,
            "full_swiglu_candidate_minus_original_SSE": sse_on - sse_off,
            "actual_original_swiglu_replay_parity_maxabs": parity_maxabs,
            "actual_original_swiglu_replay_parity_relative_rmse": parity_relrmse,
            "tiers": {role: policy[(layer, expert, role)] for role in ROLES}
        })
        for k, case in enumerate(cases):
            key = (case, layer)
            if key not in coexpert:
                coexpert[key] = [np.zeros((2048,), dtype=np.float64),
                                 np.zeros((2048,), dtype=np.float64)]
            coexpert[key][0] += probs[k] * off_diff[:, k]
            coexpert[key][1] += probs[k] * on_diff[:, k]

    off_total = sum(r["full_swiglu_AF4_baseline_weighted_SSE"] for r in per_expert)
    on_total = sum(r["full_swiglu_frozen_SSE_policy_weighted_SSE"] for r in per_expert)
    joint_off = sum(float(np.dot(v[0], v[0])) for v in coexpert.values())
    joint_on = sum(float(np.dot(v[1], v[1])) for v in coexpert.values())
    paired = [{"case": case, "layer": layer,
               "baseline_squared": float(np.dot(v[0], v[0])),
               "candidate_squared": float(np.dot(v[1], v[1]))}
              for (case, layer), v in sorted(coexpert.items())]
    d = {
        "schema": SCHEMA,
        "status": "JE01_CAL240_FROZEN_SSE_DP_JOINT_NONLINEAR_FFN_REPLAY_COMPLETE",
        "original_actual_cal256_activation_receipt_sha256": trace_sha,
        "original_cal240_rolewise_DP_receipt_sha256": CAL_LEDGER_SIGNED_SHA,
        "original_cal240_policy_ledger_readback_path": policy_path,
        "original_AF4_checkpoint_sha256": AF_SIGNED_SHA,
        "native_source_original_BF16_verified": True,
        "training_first16_prior_pilot_excluded": True,
        "clean_training_prompts": 240, "clean_routed_expert_records": 637,
        "frozen_selected_expert_pairs": 8, "FFN_roles": 24,
        "physical_selected_budget_bytes": sum(COST[t] for t in policy.values()),
        "frozen_Q4_roles": sum(t == "Q4" for t in policy.values()),
        "frozen_Q5_roles": sum(t == "Q5" for t in policy.values()),
        "frozen_original_AF4_roles": sum(t == "AF4" for t in policy.values()),
        "per_expert_original_activations_full_joint_FFN": per_expert,
        "native_full_FFN_weighted_SSE_AF4_baseline": off_total,
        "native_full_FFN_weighted_SSE_SSE_DP_candidate": on_total,
        "native_full_FFN_weighted_SSE_relative_change": (on_total - off_total)/off_total,
        "native_same_layer_corouted_experts_squared_error_baseline": joint_off,
        "native_same_layer_corouted_experts_squared_error_candidate": joint_on,
        "native_same_layer_corouted_experts_squared_error_relative_change":
            (joint_on-joint_off)/joint_off,
        "joint_layer_prompt_pairs": len(coexpert),
        "paired_training_prompt_layer_metrics": paired,
        "true_loss_Fisher": "NOT_MEASURED",
        "true_loss_Hessian": "NOT_MEASURED",
        "real_full_native_27layer_NLL": "NOT_RUN",
        "untouched_validation_OOD_gold": "NOT_USED",
        "production_modified": False,
        "scientific_scope": "TRAIN240_FROZEN_ORIGINAL_H2_FFN_LOCAL_NONLINEAR_AND_SAME_LAYER_COEXPERT_REPLAY_NOT_COMPLETE_MODEL"
    }
    need(off_total > 0 and joint_off > 0 and
         d["physical_selected_budget_bytes"] == 43_253_760 and
         d["frozen_Q4_roles"] == d["frozen_Q5_roles"] == 6 and
         len(per_expert) == 8 and len(coexpert) > 0, "JE01_FFN_JOINT_NO_VALID_OBJECTIVE")
    d["receipt_sha256"] = canonical(d)
    output.mkdir(mode=0o700)
    out = output / "JE01_CAL240_TRUE_SOURCE_JOINT_NONLINEAR_FFN_REPLAY.json"
    out.write_text(json.dumps(d, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": d["status"], "receipt_sha256": d["receipt_sha256"],
                      "fullffn_weighted_error_relative_change":
                      d["native_full_FFN_weighted_SSE_relative_change"],
                      "corouted_same_layer_error_relative_change":
                      d["native_same_layer_corouted_experts_squared_error_relative_change"],
                      "joint_pairs": len(coexpert), "output": str(out)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.output)
