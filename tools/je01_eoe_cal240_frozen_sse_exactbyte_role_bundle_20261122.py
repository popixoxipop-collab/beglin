#!/usr/bin/env python3
"""JE01 physical SAME-BYTE frozen TRAIN240 SSE role bundle (research-only).

Copies actual signed original affine AF4 packed/scale/bias bytes for unchanged
roles. For changed roles it invokes the original pinned native 64-group Q4/Q5
codec against true DeepSeek-V2-Lite-Chat BF16 weights. Never adjusts policy,
reads labels, touches production, or claims NLL/Fisher quality improvement.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import numpy as np

WORK = Path("/Volumes/D50/TailnetCommander/work")
ACT = WORK / "je01_real_original256_activation_20261122_v1"
NATIVE = WORK / "je01_real_router256_native_weight_dp_20261122_v1"
CAL_SRC = ACT / "je01_cal240_native_activation_dp_source_20261122.py"
CAL_SRC_SHA = "863fb9c1d5a17fc810564610739bd1c53f727addee99a7966f65404b8725a414"
LEDGER = WORK / "je01_real_cal240_ffn_act_dp_20261122_v1" / "JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP.json"
LEDGER_SHA = "35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb"
AF_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
TRACE_SHA = "b911aef00007b1f9b44214996353f93c7218fe51c82daf91966e450ebba460b6"
BUDGET = 43_253_760
COST = {"AF4": 1_802_240, "Q4": 1_622_016, "Q5": 1_982_464}
ROLES = ("gate_proj", "up_proj", "down_proj")
FROZEN = ((12, 13), (13, 22), (13, 28), (14, 63), (20, 17), (21, 61), (24, 26), (24, 60))
Q4_SET = frozenset((l, e, r) for l, e in ((12, 13), (13, 22), (13, 28)) for r in ROLES[:2])
Q5_SET = frozenset([(20, 17, "down_proj"), (21, 61, "down_proj"),
                    (24, 60, "down_proj")] +
                   [(24, 26, r) for r in ROLES])
SCHEMA = "beglin-je01-original27-frozen-actual-cal240-sse-role-specific-af4-q4-q5-physical43mb/1"


def need(cond, code):
    if not cond:
        raise ValueError(code)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(payload):
    return sha_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode())


def tier_for(layer, expert, role):
    ident = (layer, expert, role)
    return "Q4" if ident in Q4_SET else ("Q5" if ident in Q5_SET else "AF4")


def load_ledger():
    need(LEDGER.is_file() and not LEDGER.is_symlink(), "JE01_REAL_CAL240_LEDGER_MISSING")
    d = json.loads(LEDGER.read_text())
    claim = d.get("receipt_sha256")
    body = {k: v for k, v in d.items() if k != "receipt_sha256"}
    need(claim == LEDGER_SHA == canonical(body) and
         d["status"] == "JE01_ACTUAL_ORIGINAL27_CAL240_UNSEEN_FFN_H2_SWIGLU_24ROLE_ACTIVATION_DP_COMPLETE" and
         d["source_full_original_AF4_sha256"] == AF_SHA and
         d["source_actual_training_original27_256_h2_sw_trace_receipt_sha256"] == TRACE_SHA and
         d["training_first16_prior_pilot_excluded_from_precision_selection"] is True and
         d["training_unseen240_prompt_count"] == 240 and
         d["observed_total_original256_routed_activation_records"] == 682 and
         d["source_heldout_OOD_labels_used"] is False and
         d["production_modified"] is False, "JE01_CAL240_SOURCE_RECEIPT_OR_POLICY_LEAK")
    best = d["separate_exactbyte_dp_objective_results"]["route_weighted_output_SSE"]["best_overall"]
    selected = {(l, e, r): tier for l, e, r, tier in best["source_assignment"]}
    expected = {(l, e, r): tier_for(l, e, r) for l, e in FROZEN for r in ROLES}
    need(len(selected) == 24 and selected == expected and
         best["q4_roles"] == best["q5_roles"] == 6 and
         best["unchanged_AF4_roles"] == 12 and
         math.isclose(best["risk_difference"], -1607.2109554533818,
                      rel_tol=1e-12, abs_tol=1e-5) and
         sum(COST[x] for x in selected.values()) == BUDGET,
         "JE01_CAL240_ROLE_POLICY_NOT_EXACTLY_FROZEN")
    per = {(r["layer"], r["expert"], r["role"]): r for r in d["per_role"]}
    need(len(per) == 24 and set(per) == set(expected), "JE01_ACTUAL_SOURCE_24ROLE_MISSING")
    return expected, per


def load_original_sources():
    need(CAL_SRC.is_file() and not CAL_SRC.is_symlink() and
         sha_file(CAL_SRC) == CAL_SRC_SHA, "JE01_ORIGINAL_CAL240_SOURCE_SHA_CHANGED")
    spec = importlib.util.spec_from_file_location("je01_cal240_source_pinned", CAL_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    native, _ = mod.original()  # independently checks original codec Git blob
    af = native.safe_file(native.AF_ROOT, "deepseek_moe_af.bin",
                          native.AF_FILE_SHA, 9814343680, rehash=True)
    layout = native.read_af_layout()
    weightmap, shards, headers = native.read_bf16_headers()
    dylib = NATIVE / "result" / "je01_native_q4q5.dylib"
    need(dylib.is_file() and not dylib.is_symlink(), "JE01_NATIVE_Q4Q5_CODEC_DYLIB_MISSING")
    lib = ctypes.CDLL(str(dylib))
    f32 = ctypes.POINTER(ctypes.c_float)
    lib.gguf_quantize_q4g64_error_feedback.argtypes = [
        f32, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.gguf_quantize_qNg64.argtypes = [
        f32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p]
    return native, af, layout, weightmap, shards, headers, lib


def original_af4_parts(af, layout, layer, expert, role):
    key = f"model.layers.{layer}.mlp.switch_mlp.{role}"
    need(key in layout, "JE01_ORIGINAL_AFFINE_ROLE_NOT_FOUND")
    E, out_dim, in_dim, groups, po, plen, so, bo = layout[key]
    need(E == 64 and groups == in_dim // 64 and expert in range(64) and
         (out_dim, in_dim) == ((2048, 1408) if role == "down_proj"
                              else (1408, 2048)) and
         plen == E * out_dim * in_dim // 2, "JE01_AFFINE_SOURCE_LAYOUT_CHANGED")
    n_codes = out_dim * in_dim // 2
    n_groups = out_dim * groups * 4
    blocks = {}
    with af.open("rb") as f:
        for name, offset, size in (("packed", po + expert * n_codes, n_codes),
                                   ("scale", so + expert * n_groups, n_groups),
                                   ("bias", bo + expert * n_groups, n_groups)):
            need(offset >= 0 and offset + size <= af.stat().st_size,
                 "JE01_AF4_SOURCE_REGION_INVALID")
            f.seek(offset)
            data = f.read(size)
            need(len(data) == size, "JE01_AF4_SOURCE_REGION_SHORT")
            blocks[name] = data
    need(sum(map(len, blocks.values())) == COST["AF4"],
         "JE01_AFFINE_PHYSICAL_SOURCE_BYTES_MISMATCH")
    return blocks


def native_symmetric_parts(lib, bf16, bits):
    need(bits in (4, 5), "JE01_Q_BITS_WRONG")
    W = np.ascontiguousarray(bf16, dtype=np.float32)
    rows, cols = W.shape
    need((rows, cols) in ((1408, 2048), (2048, 1408)) and
         cols % 64 == 0 and np.isfinite(W).all(), "JE01_NATIVE_ORIGINAL_BF16_SHAPE_OR_VALUES")
    packed = np.zeros(rows * cols * bits // 8, dtype=np.uint8)
    scales = np.zeros((rows, cols // 64), dtype=np.float32)
    src = W.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if bits == 4:
        lib.gguf_quantize_q4g64_error_feedback(src, rows, cols,
                                                packed.ctypes.data, scales.ctypes.data)
    else:
        lib.gguf_quantize_qNg64(src, rows, cols, bits,
                                 packed.ctypes.data, scales.ctypes.data)
    need(np.isfinite(scales).all() and np.all(scales > 0),
         "JE01_NATIVE_SYMMETRIC_SCALE_VALUES_INVALID")
    return {"packed": packed.tobytes(), "scale": scales.tobytes()}


def build(output):
    need(output.parent == WORK and not output.exists() and not output.is_symlink(),
         "JE01_NEW_ISOLATED_D50_RESEARCH_OUTPUT_REQUIRED")
    assignment, source_roles = load_ledger()
    native, af, layout, weightmap, shards, headers, lib = load_original_sources()
    output.mkdir(mode=0o700)
    binary = output / "JE01_CAL240_FROZEN_SSE_12AF4_6Q4_6Q5_EXACT43MB.bin"
    meta = output / "JE01_CAL240_FROZEN_SSE_24ROLE_SOURCE.tsv"
    rows, spans, policy_counts = [], [], {"AF4": 0, "Q4": 0, "Q5": 0}
    with binary.open("xb") as sink:
        for layer, expert in FROZEN:
            for role in ROLES:
                tier = assignment[(layer, expert, role)]
                bf, ref = native.bf16_selected(weightmap, shards, headers,
                                                layer, expert, role)
                origin = source_roles[(layer, expert, role)]
                need(ref["original_bf16_tensor_sha256"] ==
                     origin["actual_original_BF16_tensor_sha256"],
                     "JE01_ROLE_ORIGINAL_BF16_SHA_CHANGED")
                if tier == "AF4":
                    parts = original_af4_parts(af, layout, layer, expert, role)
                    need(sha_bytes(parts["packed"]) ==
                         origin["actual_original_AF4_packed_sha256"],
                         "JE01_NATIVE_ORIGINAL_AFFINE4_PACKED_SHA_CHANGED")
                else:
                    parts = native_symmetric_parts(lib, bf, 4 if tier == "Q4" else 5)
                regions = {}
                for kind in ("packed", "scale", "bias"):
                    if kind in parts:
                        beg = sink.tell()
                        sink.write(parts[kind])
                        end = sink.tell()
                        spans.append((beg, end))
                        regions[kind] = (beg, end)
                size = sum(len(x) for x in parts.values())
                need(size == COST[tier], "JE01_ROLE_PHYSICAL_GROUP_BYTES_CHANGED")
                shape = (2048, 1408) if role == "down_proj" else (1408, 2048)
                row = {"layer": layer, "expert": expert, "role": role,
                       "tier": tier, "bits": 5 if tier == "Q5" else 4,
                       "output": shape[0], "input": shape[1],
                       "regions": {k: list(v) for k, v in regions.items()},
                       "part_sha256": {k: sha_bytes(v) for k, v in parts.items()},
                       "source_original_bf16_tensor_sha256":
                           ref["original_bf16_tensor_sha256"],
                       "source_original_affine4_packed_sha256":
                           origin["actual_original_AF4_packed_sha256"],
                       "physical_bytes": size}
                rows.append(row)
                policy_counts[tier] += 1
                print(json.dumps({"layer": layer, "expert": expert,
                                  "role": role, "tier": tier, "roles_done": len(rows)},
                                 sort_keys=True), flush=True)
        sink.flush()
        os.fsync(sink.fileno())
    spans.sort()
    need(len(rows) == 24 and policy_counts == {"AF4": 12, "Q4": 6, "Q5": 6} and
         len(spans) == 60 and spans[0][0] == 0 and spans[-1][1] == BUDGET and
         all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1)) and
         binary.stat().st_size == BUDGET, "JE01_FROZEN_PHYSICAL_BUNDLE_NOT_EXACT_43253760")
    lines = []
    for r in rows:
        b = r["regions"].get("bias", [0, 0])
        pp, ss = r["regions"]["packed"], r["regions"]["scale"]
        lines.append("\t".join(map(str, (r["layer"], r["expert"], r["role"],
               r["bits"], r["output"], r["input"], *pp, *ss, *b, r["tier"]))))
    meta.write_text("\n".join(lines) + "\n")
    receipt = {"schema": SCHEMA,
               "status": "JE01_REAL_ORIGINAL_AF4_Q4Q5_CAL240_FROZEN_ROLEWISE_43MB_CREATED",
               "source_original_AF4_sha256": AF_SHA,
               "source_actual_cal240_SSE_DP_receipt_sha256": LEDGER_SHA,
               "source_training_actual_h2_SwiGLU_receipt_sha256": TRACE_SHA,
               "first16_train_pilot_excluded": True,
               "calibration_training_prompt_count": 240,
               "calibration_routed_expert_records": 637,
               "physical_file_sha256": sha_file(binary),
               "physical_file_size_bytes": binary.stat().st_size,
               "role_manifest_sha256": sha_file(meta),
               "source_cal240_python_sha256": CAL_SRC_SHA,
               "role_count": 24, "counts": policy_counts,
               "contiguous_file_spans": len(spans),
               "per_role": rows,
               "true_full_model_gold_NLL": "NOT_MEASURED",
               "true_loss_fisher_hessian": "NOT_MEASURED",
               "native_full_model_overlay": "NOT_YET_BOUND",
               "heldout_OOD_gold_accessed": False, "production_modified": False}
    receipt["receipt_sha256"] = canonical(receipt)
    out = output / "JE01_CAL240_FROZEN_SSE_PHYSICAL_SOURCE_RECEIPT.json"
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"],
                      "signed_receipt_sha256": receipt["receipt_sha256"],
                      "binary_sha256": receipt["physical_file_sha256"],
                      "bytes": receipt["physical_file_size_bytes"],
                      "roles": policy_counts, "manifest": str(out)},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
