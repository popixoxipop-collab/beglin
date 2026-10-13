#!/usr/bin/env python3
"""Offline supervised micro-expert router + residual corrector; NO serving mutation.

Training uses real quantized-Beglin h2 vectors and an actual macro Top-K trace,
but BF16 checkpoint expert weights. Dense teacher labels are computed only for
calibration positions (including counterfactual expert activations).
Holdout positions are never used in training or hyperparameter choice.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib
import numpy as np
from real_activation_micro_router import read_capture, read_route, channels
from real_checkpoint_reader import RealExpertTileReader
from permutation_alignment_probe import read_rows
from hierarchical_moe_probe import tupper_encode, tupper_decode

SCHEMA = "beglin-trained-micro-router-offline-v1"

def features(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("nonfinite input features")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.e-12)

def fit_kernel(xtrain, ytrain, lam):
    x = features(xtrain)
    y = np.asarray(ytrain, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != len(x) or lam <= 0:
        raise ValueError("invalid ridge training")
    mu = np.mean(y, axis=0)
    a = np.linalg.solve(x @ x.T + lam * np.eye(len(x)), y - mu)
    return (x, mu, a)

def predict(model, xtest):
    x, mu, a = model
    return mu + features(xtest) @ x.T @ a

def mask_for(scores, keep):
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or keep < 1 or keep > s.shape[1] or not np.all(np.isfinite(s)):
        raise ValueError("invalid channel scores")
    chosen = np.argsort(s, axis=1, kind="stable")[:, -keep:]
    mask = np.zeros(s.shape, dtype=bool)
    np.put_along_axis(mask, chosen, True, axis=1)
    return mask

def masked_output(contrib, mask):
    if contrib.ndim != 3 or mask.shape != contrib.shape[:2]:
        raise ValueError("mask/contribution shape mismatch")
    return np.einsum("pgh,pg->ph", contrib, mask.astype(np.float64), optimize=True)

def fit_one_expert(xcal, cal, xval, val, fraction, score_lam=.25, residual_lam=2.0):
    if cal.ndim != 3 or val.ndim != 3 or cal.shape[1:] != val.shape[1:]:
        raise ValueError("cal/test channel mismatch")
    if xcal.shape[0] < 3 or xcal.shape[0] != cal.shape[0] or xval.shape[0] != val.shape[0]:
        raise ValueError("invalid calibration split")
    n, ng, hidden = cal.shape
    k = max(1, math.ceil(fraction * ng))
    magnitudes = np.linalg.norm(cal, axis=2)
    salience = magnitudes / np.maximum(magnitudes.mean(axis=1, keepdims=True), 1.e-12)
    static_scores = salience.mean(axis=0)
    fixed_cal = np.broadcast_to(mask_for(static_scores[None, :], k), (n, ng))
    fixed_test = np.broadcast_to(mask_for(static_scores[None, :], k), (len(xval), ng)).copy()
    gate_model = fit_kernel(xcal, salience, score_lam)
    gate_cal = mask_for(predict(gate_model, xcal), k)
    gate_test = mask_for(predict(gate_model, xval), k)
    ycal_full = np.sum(cal, axis=1, dtype=np.float64)
    static_offset = np.mean(ycal_full - masked_output(cal, fixed_cal), axis=0)
    missing = ycal_full - masked_output(cal, gate_cal)
    offset = missing.mean(axis=0)
    residual_model = fit_kernel(xcal, missing, residual_lam)
    static_y = masked_output(val, fixed_test)
    dynamic_y = masked_output(val, gate_test)
    return {
        "full": np.sum(val, axis=1, dtype=np.float64),
        "static": static_y,
        "static_mean_residual": static_y + static_offset,
        "dynamic": dynamic_y,
        "dynamic_mean_residual": dynamic_y + offset,
        "dynamic_ridge_residual": dynamic_y + predict(residual_model, xval),
        "oracle_magnitude": masked_output(val, mask_for(np.linalg.norm(val, axis=2), k)),
        "static_mask": fixed_test, "dynamic_mask": gate_test,
        "stored_router_floats_per_expert": int(n * (ng + hidden * 2) + ng + hidden * 2),
    }

def evaluate(index, capture, routing, layer=1, hidden=2048, intermediate=1408,
             widths=(16, 64), fractions=(.25, .5, .75, 1.)):
    h = read_capture(capture, layer, hidden)
    if len(h) < 6:
        raise ValueError("need >= 6 real h2 positions; holdout is not reliable with fewer")
    macro = read_route(routing, layer, set(h))
    pos = sorted(h)
    split = len(pos) // 2
    tr, te = pos[:split], pos[split:]
    xc, xt = np.stack([h[p] for p in tr]), np.stack([h[p] for p in te])
    experts = sorted({e for v in macro.values() for e, _ in v})
    if any(e < 0 or e >= 64 for e in experts):
        raise ValueError("unexpected macro expert")
    if len({len(x) for x in macro.values()}) != 1:
        raise ValueError("inconsistent macro routing Top-K")
    rdr = RealExpertTileReader(index, layer=layer, expert_ids=experts,
                               hidden=hidden, intermediate=intermediate)
    methods = ("static", "static_mean_residual", "dynamic",
               "dynamic_mean_residual", "dynamic_ridge_residual", "oracle_magnitude")
    result = {}
    for width in widths:
        if not (width > 0 and intermediate % width == 0):
            raise ValueError("bad micro width")
        ng = intermediate // width
        cached = {}
        for expert in experts:
            rows = read_rows(rdr, expert, np.arange(intermediate))
            cal = np.stack([channels(rows, h[p], width) for p in tr])
            test = np.stack([channels(rows, h[p], width) for p in te])
            cached[expert] = (cal, test)
            del rows
        result[str(width)] = {}
        for fraction in fractions:
            target = np.zeros((len(te), hidden), np.float64)
            guesses = {method: np.zeros_like(target) for method in methods}
            fixed_grid = np.zeros((ng, 64), np.uint8)
            gate_grid_by_token = np.zeros((len(te), ng, 64), np.uint8)
            resident_floats = 0
            for expert, (cal, test) in cached.items():
                model = fit_one_expert(xc, cal, xt, test, fraction)
                resident_floats += model["stored_router_floats_per_expert"]
                fixed_grid[:, expert] = model["static_mask"][0].astype(np.uint8)
                gate_grid_by_token[:, :, expert] = model["dynamic_mask"].astype(np.uint8)
                for i, p in enumerate(te):
                    weight = next((v for e, v in macro[p] if e == expert), 0.)
                    if weight == 0: continue
                    target[i] += weight * model["full"][i]
                    for method in methods:
                        guesses[method][i] += weight * model[method][i]
            packed = tupper_encode(fixed_grid)
            if not np.array_equal(tupper_decode(packed["k"], ng, 64), fixed_grid):
                raise AssertionError("generalized Tupper mask roundtrip failed")
            for mask in gate_grid_by_token:
                pkg = tupper_encode(mask)
                if not np.array_equal(tupper_decode(pkg["k"], ng, 64), mask):
                    raise AssertionError("dynamic Tupper mask roundtrip failed")
            denominator = max(float(np.sum(target**2)), 1.e-25)
            metric = {}
            for name in methods:
                err = guesses[name] - target
                metric[name] = {
                    "heldout_macro_output_relative_l2": float(np.sqrt(np.sum(err**2) / denominator)),
                    "heldout_worst_position_relative_l2": float(max(
                        np.linalg.norm(err[i]) / max(np.linalg.norm(target[i]), 1.e-25)
                        for i in range(len(te))))}
            result[str(width)][str(fraction)] = {
                "methods": metric,
                "tupper_grid": [ng, 64], "tupper_roundtrip": True,
                "trained_resident_floats_approx": resident_floats,
                "full_expert_compute_fraction_idealized": math.ceil(ng*fraction) / ng,
                "raw_activations_uploaded": False}
        del cached
    return {
        "schema": SCHEMA, "status": "EVALUATED_NOT_PROMOTABLE",
        "real_activations": True, "real_macro_routes": True,
        "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
        "macro_route_sha256": hashlib.sha256(routing.read_bytes()).hexdigest(),
        "checkpoint_index_sha256": rdr.provenance()["index_sha256"],
        "train_token_positions": len(tr), "heldout_token_positions": len(te),
        "split": "within_single_fixed_eight_position_verification_sequence",
        "training": "per-expert linear kernel ridge fitted to dense channel salience",
        "corrector": "per-expert ridge prediction of omitted output trained on calibration only",
        "ridge_score_lambda": .25, "ridge_residual_lambda": 2.0,
        "counterfactual_teacher_calculation_for_unrouted_experts": True,
        "input_source_mismatch": "quantized AF hidden state and BF16 expert FFN weights",
        "actual_task_metrics": False, "runtime_speed_measured": False,
        "production_touched": False, "automatic_live_promotion": False,
        "results": result}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-index", required=True, type=pathlib.Path)
    ap.add_argument("--capture", required=True, type=pathlib.Path)
    ap.add_argument("--routing", required=True, type=pathlib.Path)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=1408)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    a = ap.parse_args()
    out = evaluate(a.checkpoint_index, a.capture, a.routing, a.layer, a.hidden, a.intermediate)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    print("TRAINABLE_MICRO_RESULTS", json.dumps({
        "train": out["train_token_positions"], "holdout": out["heldout_token_positions"],
        "results": {w: {f: {k: round(v["heldout_macro_output_relative_l2"], 6)
                  for k, v in m["methods"].items()} for f, m in a.items()}
                  for w, a in out["results"].items()}}, sort_keys=True))
if __name__ == "__main__": main()
