#!/usr/bin/env python3
"""JE01: offline, exact-byte, independent-Expert Bayesian MoE pilot planner.

Real model and per-expert NLL deltas must be measured externally. This module
does NOT run inference, edit checkpoints, touch serving, infer cross-expert
covariance or certify predicted multi-expert gain. Result is SHADOW_ONLY.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

SCHEMA = "BEGLIN-JE01-independent-expert-exact-byte-Bayesian-shadow/1"
ALLOWED_BITS = frozenset((2, 3, 4, 5, 6, 7, 8, 16, 32))
MIN_NATIVE_CALIBRATION = 32
MAX_PILOT_EXPERTS = 8
RISK_MULTIPLIER = 2.0
MIN_PREDICTED_GAIN = 0.15  # research screen, never a deployment criterion
ALPHA0 = 2.0
BETA0 = 1.0
KAPPA0 = 1.0


def digest(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _real(value):
    return type(value) in (int, float) and math.isfinite(value)


def nig_predict(samples):
    """NIG(0,1,2,1) scalar posterior predictive Student-t location/scale."""
    if not isinstance(samples, list) or len(samples) < MIN_NATIVE_CALIBRATION or not all(
        _real(x) for x in samples
    ):
        raise ValueError("JE01_EXPERT_OBSERVATIONS_MISSING_OR_NONFINITE")
    n = len(samples)
    xbar = sum(samples) / n
    ss = sum((x - xbar) ** 2 for x in samples)
    k = KAPPA0 + n
    mu = n * xbar / k
    alpha = ALPHA0 + n / 2
    beta = BETA0 + ss / 2 + KAPPA0*n*xbar*xbar / (2*k)
    scale = math.sqrt(beta * (k + 1) / (alpha * k))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("JE01_EXPERT_SCALE_NONFINITE")
    return {"posterior_mean": mu, "student_t_df": 2*alpha,
            "predictive_scale": scale,
            "conservative_2scale_upper": mu + RISK_MULTIPLIER * scale,
            "n_calibration": n}


def _check_profile(profile):
    if not isinstance(profile, Mapping):
        raise ValueError("JE01_PROFILE_INVALID")
    keys = {"checkpoint_sha256", "calibration_manifest_sha256", "backend",
            "slots"}
    if set(profile) != keys:
        raise ValueError("JE01_PROFILE_FIELDS")
    for key in ("checkpoint_sha256", "calibration_manifest_sha256"):
        value = profile[key]
        if not isinstance(value, str) or len(value) != 64 or any(
            x not in "0123456789abcdef" for x in value
        ):
            raise ValueError("JE01_SOURCE_SHA_MISSING")
    if profile["backend"] not in ("metal", "cpu"):
        raise ValueError("JE01_BACKEND_INVALID")
    slots = profile["slots"]
    if not isinstance(slots, list) or not 1 <= len(slots) <= MAX_PILOT_EXPERTS:
        raise ValueError("JE01_PILOT_EXPERT_COUNT")
    seen = set()
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {
            "layer", "expert_id", "baseline_bits", "options"
        }:
            raise ValueError("JE01_EXPERT_SLOT_FIELDS")
        layer, e = slot["layer"], slot["expert_id"]
        if type(layer) is not int or type(e) is not int or layer < 1 or e < 0 or e >= 64:
            raise ValueError("JE01_EXPERT_ID")
        if (layer, e) in seen:
            raise ValueError("JE01_DUPLICATE_EXPERT_SLOT")
        seen.add((layer, e))
        if slot["baseline_bits"] not in ALLOWED_BITS or type(slot["baseline_bits"]) is not int:
            raise ValueError("JE01_BASELINE_BITS")
        opts = slot["options"]
        if not isinstance(opts, list) or not opts:
            raise ValueError("JE01_EXPERT_OPTIONS")
        tiers = set()
        for opt in opts:
            if not isinstance(opt, dict) or set(opt) != {
                "bits", "packed_bytes", "paired_delta_nll", "router_flip_count",
                "router_checks"
            }:
                raise ValueError("JE01_OPTION_FIELDS")
            bit, cost = opt["bits"], opt["packed_bytes"]
            if type(bit) is not int or bit not in ALLOWED_BITS or bit in tiers:
                raise ValueError("JE01_BITS_DUPLICATE_OR_UNSUPPORTED")
            tiers.add(bit)
            if type(cost) is not int or cost <= 0:
                raise ValueError("JE01_PACKED_BYTES_INVALID")
            if type(opt["router_flip_count"]) is not int or type(opt["router_checks"]) is not int:
                raise ValueError("JE01_ROUTER_FLIP_COUNTS")
            if opt["router_checks"] < 0 or not 0 <= opt["router_flip_count"] <= opt["router_checks"]:
                raise ValueError("JE01_ROUTER_FLIP_RANGE")
            if bit == slot["baseline_bits"]:
                if opt["paired_delta_nll"] != [] or opt["router_flip_count"] != 0:
                    raise ValueError("JE01_BASELINE_MUST_BE_ZERO_DELTA")
            elif opt["paired_delta_nll"] and (
                not isinstance(opt["paired_delta_nll"], list) or
                not all(_real(z) for z in opt["paired_delta_nll"])
            ):
                raise ValueError("JE01_OPTION_NONFINITE")
        if slot["baseline_bits"] not in tiers:
            raise ValueError("JE01_BASELINE_TIER_MISSING")


def optimize_exact_bytes(profile):
    """DP over exact byte differences. Zero candidate = unchanged base map.

    Every nonbaseline tier requires >=32 independent paired NLL observations and
    zero measured router flips; these are NECESSARY, not sufficient.
    """
    _check_profile(profile)
    states = {0: (0.0, ())}
    baseline_bytes = 0
    audited_alternatives = 0
    for slot in sorted(profile["slots"], key=lambda v: (v["layer"], v["expert_id"])):
        base = next(o for o in slot["options"] if o["bits"] == slot["baseline_bits"])
        baseline_bytes += base["packed_bytes"]
        cand = []
        for opt in sorted(slot["options"], key=lambda o: o["bits"]):
            bits = opt["bits"]
            if bits == slot["baseline_bits"]:
                cand.append((0, 0.0, bits, base["packed_bytes"]))
                continue
            if len(opt["paired_delta_nll"]) < MIN_NATIVE_CALIBRATION:
                continue
            if opt["router_checks"] < MIN_NATIVE_CALIBRATION or opt["router_flip_count"] != 0:
                continue
            posterior = nig_predict(opt["paired_delta_nll"])
            cand.append((opt["packed_bytes"] - base["packed_bytes"],
                         posterior["conservative_2scale_upper"], bits,
                         opt["packed_bytes"]))
            audited_alternatives += 1
        new_states = {}
        for cost_so_far, (upper, choices) in states.items():
            for delta_cost, contribution, bits, measured_bytes in cand:
                cost = cost_so_far + delta_cost
                value = upper + contribution
                candidate = choices + ((slot["layer"], slot["expert_id"],
                                        bits, measured_bytes),)
                prior = new_states.get(cost)
                if prior is None or value < prior[0] - 1e-12 or (
                    abs(value - prior[0]) <= 1e-12 and candidate < prior[1]
                ):
                    new_states[cost] = (value, candidate)
        states = new_states
        if len(states) > 100000:
            raise ValueError("JE01_DP_STATE_LIMIT_NO_APPROXIMATION")
    baseline_choice = tuple((s["layer"], s["expert_id"], s["baseline_bits"],
                            next(o["packed_bytes"] for o in s["options"]
                                 if o["bits"] == s["baseline_bits"]))
                            for s in sorted(profile["slots"], key=lambda x: (
                                x["layer"], x["expert_id"])))
    objective, choice = states.get(0, (0.0, baseline_choice))
    improved = bool(objective < -MIN_PREDICTED_GAIN)
    if not improved:
        objective, choice = 0.0, baseline_choice
    assert sum(c[3] for c in choice) == baseline_bytes
    state = {"schema": SCHEMA, "status": "SHADOW_ONLY_NO_WEIGHT_EDIT",
             "checkpoint_sha256": profile["checkpoint_sha256"],
             "calibration_manifest_sha256": profile["calibration_manifest_sha256"],
             "backend": profile["backend"], "baseline_packed_bytes": baseline_bytes,
             "selected_packed_bytes": sum(c[3] for c in choice),
             "selected": [{"layer": l, "expert_id": e, "bits": bits,
                           "measured_packed_bytes": n} for l, e, bits, n in choice],
             "predicted_conservative_loss_upper_delta": objective,
             "pilot_prediction_improves": improved,
             "audited_alternative_tiers_with_samples": audited_alternatives,
             "cross_expert_covariance": "NOT_ESTIMATED",
             "full_moe_forward_additivity": "NOT_VERIFIED",
             "production_promotion_allowed": False}
    state["sha256"] = digest(state)
    return state
