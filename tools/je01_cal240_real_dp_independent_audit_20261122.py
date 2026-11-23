#!/usr/bin/env python3
"""JE01 CAL240 independent signed-ledger and exact-byte DP audit (stdlib only).

Audits the finished actual FFN output-error ledger from original 27-layer
TRAIN240, not the upstream matrix multiplications. No checkpoint or held-out
labels loaded. Fisher/Hessian or real NLL is NOT measured here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_RECEIPT = "35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb"
TRACE_RECEIPT = "b911aef00007b1f9b44214996353f93c7218fe51c82daf91966e450ebba460b6"
AF_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
ROUTER_SHA = "62d0a86956c1a07fa8143efcb52b0b5dea6808767716cd160e05ceb1c0c8a8eb"
FROZEN = ((12, 13), (13, 22), (13, 28), (14, 63),
          (20, 17), (21, 61), (24, 26), (24, 60))
CLEAN_COUNT = dict(zip(FROZEN, (75, 75, 82, 74, 79, 89, 87, 76)))
ROLE_ORDER = ("gate_proj", "up_proj", "down_proj")
TIERS = ("AF4", "Q4", "Q5")
METRICS = ("route_weighted_output_SSE", "route_weighted_output_relative_MSE")
COST = {"AF4": 1_802_240, "Q4": 1_622_016, "Q5": 1_982_464}
BUDGET = 43_253_760
FILENAME = "JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP.json"


def canonical(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def assert_valid(condition, reason):
    if not condition:
        raise ValueError(reason)


def independent_dp(rows, metric):
    """Independent objective replay, enumerate tier choices by budget-balance DP."""
    states = {(0, False): (0.0, ())}
    for row in rows:
        nxt = {}
        for (balance, changed), (score, choices) in states.items():
            for tier in TIERS:
                bdelta = (COST[tier] - COST["AF4"]) // 180224
                key = (balance + bdelta, changed or tier != "AF4")
                candidate = (score + row[metric + "_" + tier] -
                             row[metric + "_AF4"], choices +
                             ((row["layer"], row["expert"], row["role"], tier),))
                if key not in nxt or candidate < nxt[key]:
                    nxt[key] = candidate
        states = nxt

    def serialize(record):
        risk, assignment = record
        return {"risk_difference": risk,
                "source_assignment": [list(t) for t in assignment],
                "q4_roles": sum(t[3] == "Q4" for t in assignment),
                "q5_roles": sum(t[3] == "Q5" for t in assignment),
                "unchanged_AF4_roles": sum(t[3] == "AF4" for t in assignment)}

    assert_valid((0, False) in states and (0, True) in states, "BALANCED_STATES_MISSING")
    return {"best_overall": serialize(min(states[(0, False)], states[(0, True)])),
            "best_nonbaseline_exactbyte": serialize(states[(0, True)]),
            "number_final_dynamic_states": len(states)}


def audit(data, expected_receipt=EXPECTED_RECEIPT, npz_path=None):
    assert_valid(isinstance(data, dict), "INPUT_NOT_OBJECT")
    signature = data.get("receipt_sha256")
    body = {k: v for k, v in data.items() if k != "receipt_sha256"}
    assert_valid(signature == expected_receipt and canonical(body) == signature,
                 "SIGNED_CAL240_RECEIPT_SHA_MISMATCH")
    assert_valid(data.get("status") ==
                 "JE01_ACTUAL_ORIGINAL27_CAL240_UNSEEN_FFN_H2_SWIGLU_24ROLE_ACTIVATION_DP_COMPLETE",
                 "STATUS_NOT_COMPLETE")
    for key, value in {
        "source_actual_training_original27_256_h2_sw_trace_receipt_sha256": TRACE_RECEIPT,
        "source_original_router256_receipt_sha256": ROUTER_SHA,
        "source_full_original_AF4_sha256": AF_SHA,
        "training_unseen240_prompt_count": 240,
        "observed_total_original256_routed_activation_records": 682,
        "role_count": 24,
        "selected_frozen_expert_count": 8,
        "observed_FFN_input_dimensions": {"gate_up": 2048, "down_SwiGLU": 1408},
        "scientific_scope":
        "ACTUAL_TRAINED_ORIGINAL_ROUTE_WEIGHTED_GEMV_DIAGONAL_OUTPUT_METRIC_NOT_TRUE_LOSS_FISHER_OR_HESSIAN",
        "true_logprob_gradients": "NOT_MEASURED",
        "full_native_27layer_new_NLL": "NOT_RUN",
        "source_heldout_OOD_labels_used": False,
        "production_modified": False,
        "true_original_Chat_BF16_four_shard_SHA_reverified": True,
        "actual_original_256_training_prompts_all_capture": True,
        "training_first16_prior_pilot_excluded_from_precision_selection": True,
    }.items():
        assert_valid(data.get(key) == value, "CONTRACT_DRIFT_" + key)

    path_sha = data.get("observed_source_input_moment_npz_sha256")
    assert_valid(isinstance(path_sha, str) and len(path_sha) == 64, "MOMENT_DIGEST_MISSING")
    if npz_path is not None:
        assert_valid(npz_path.is_file() and not npz_path.is_symlink() and
                     file_sha(npz_path) == path_sha, "MOMENT_NPZ_DIGEST_MISMATCH")

    rows = data.get("per_role")
    assert_valid(isinstance(rows, list) and len(rows) == 24, "ROLE_COUNT_INVALID")
    observed = set()
    for row in rows:
        assert_valid(isinstance(row, dict), "ROLE_RECORD_NOT_OBJECT")
        key = (row["layer"], row["expert"])
        role = row["role"]
        assert_valid(key in FROZEN and role in ROLE_ORDER and
                     (key, role) not in observed, "ROLE_DUPLICATE_OR_UNKNOWN")
        observed.add((key, role))
        assert_valid(row["actual_clean_unseen_training_router_observations"] ==
                     CLEAN_COUNT[key], "CLEAN_ROUTING_COUNT_DRIFT")
        assert_valid(row["actual_input_dimension"] == (1408 if role == "down_proj" else 2048) and
                     row["actual_output_dimension"] == (2048 if role == "down_proj" else 1408),
                     "ROLE_DIMENSION_INVALID")
        energy = row["original_BF16_activation_output_energy"]
        prob_mass = row["routed_probability_mass"]
        assert_valid(math.isfinite(energy) and energy > 0 and
                     math.isfinite(prob_mass) and prob_mass > 0, "ENERGY_PROB_INVALID")
        for field in ("actual_original_BF16_tensor_sha256", "actual_original_AF4_packed_sha256"):
            v = row.get(field)
            assert_valid(isinstance(v, str) and len(v) == 64, "WEIGHT_SOURCE_SHA_MISSING")
        for tier in TIERS:
            sse = row["route_weighted_output_SSE_" + tier]
            ratio = row["route_weighted_output_relative_MSE_" + tier]
            assert_valid(math.isfinite(sse) and math.isfinite(ratio) and sse >= 0 and ratio >= 0,
                         "NONFINITE_OR_NEGATIVE_ERROR")
            assert_valid(math.isclose(sse / energy, ratio, rel_tol=1e-10, abs_tol=1e-12),
                         "RELATIVE_MSE_NOT_MATCHING_SSE")

    assert_valid(observed == {(key, role) for key in FROZEN for role in ROLE_ORDER},
                 "MISSING_ROLE")
    expected = [(l, e, role) for l, e in sorted(FROZEN) for role in ROLE_ORDER]
    assert_valid([(r["layer"], r["expert"], r["role"]) for r in rows] == expected,
                 "ROLE_ORDER_DRIFT")
    published = data.get("separate_exactbyte_dp_objective_results", {})
    results = {}
    for metric in METRICS:
        result = independent_dp(rows, metric)
        saved = published.get(metric)
        assert_valid(isinstance(saved, dict), "MISSING_DP_RESULT_" + metric)
        assert_valid(saved.get("number_final_dynamic_states") ==
                     result["number_final_dynamic_states"], "DP_STATE_COUNT_DRIFT")
        for target in ("best_overall", "best_nonbaseline_exactbyte"):
            a, b = result[target], saved.get(target, {})
            assert_valid(a["source_assignment"] == b.get("source_assignment") and
                         a["q4_roles"] == b.get("q4_roles") and
                         a["q5_roles"] == b.get("q5_roles") and
                         a["unchanged_AF4_roles"] == b.get("unchanged_AF4_roles") and
                         math.isclose(a["risk_difference"], b.get("risk_difference", math.inf),
                                      rel_tol=1e-10, abs_tol=1e-7), "DP_REPLAY_MISMATCH")
            assert_valid(a["q4_roles"] == a["q5_roles"] and
                         sum(COST[t[3]] for t in a["source_assignment"]) == BUDGET,
                         "EXACT_PHYSICAL_BUDGET_DRIFT")
        results[metric] = result
    assert_valid(math.isclose(results[METRICS[0]]["best_overall"]["risk_difference"],
                              -1607.2109554533818, rel_tol=1e-10, abs_tol=1e-5) and
                 results[METRICS[0]]["best_overall"]["q4_roles"] == 6 and
                 results[METRICS[1]]["best_overall"]["risk_difference"] == 0.0 and
                 math.isclose(results[METRICS[1]]["best_nonbaseline_exactbyte"]["risk_difference"],
                              0.0024733855809832266, rel_tol=1e-10, abs_tol=1e-10),
                 "SERVED_RESULT_DIFFERS_FROM_FINISHED_JOB_RECEIPT")
    return {"status": "JE01_CAL240_LEDGER_AND_INDEPENDENT_DP_REPLAY_PASS",
            "receipt_sha256": signature,
            "role_count": len(rows), "clean_actual_route_events": sum(CLEAN_COUNT.values()),
            "budget_bytes": BUDGET, "results": {m: {
                "best_risk_delta": r["best_overall"]["risk_difference"],
                "best_q4_roles": r["best_overall"]["q4_roles"],
                "best_q5_roles": r["best_overall"]["q5_roles"],
                "nonbaseline_risk_delta": r["best_nonbaseline_exactbyte"]["risk_difference"]}
                for m, r in results.items()},
            "scope": "SIGNED_METADATA_AND_DP_REPLAY_ONLY__NOT_FISHER_HESSIAN_OR_NATIVE_NLL"}


def discover(root):
    assert_valid(root.is_dir() and not root.is_symlink(), "WORK_ROOT_INVALID")
    found = []
    for candidate in root.glob("je01*/**/" + FILENAME):
        if candidate.is_file() and not candidate.is_symlink() and len(candidate.relative_to(root).parts) <= 5:
            found.append(candidate)
    assert_valid(len(found) == 1, "JE01_ACTUAL_CAL240_RECEIPT_DISCOVERY_COUNT_" + str(len(found)))
    return found[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", type=Path)
    p.add_argument("--discover-root", type=Path)
    args = p.parse_args()
    assert_valid((args.receipt is None) != (args.discover_root is None),
                 "SPECIFY_EXACTLY_ONE_SOURCE")
    path = args.receipt if args.receipt is not None else discover(args.discover_root)
    assert_valid(path.is_file() and not path.is_symlink(), "RECEIPT_FILE_MISSING")
    npz = path.parent / "JE01_CAL240_ACTUAL_ORIGINAL_FFN_INPUT_DIAGONAL_MOMENTS.npz"
    result = audit(json.loads(path.read_text()), npz_path=npz)
    result["readback_path"] = str(path)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
