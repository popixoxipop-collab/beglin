#!/usr/bin/env python3
"""Read-only, independent physical-byte audit for the frozen JE01 TRAIN240 bundle.

Does NOT run/verify the quantizer's arithmetic, native Metal inference,
loss gradients, Fisher, Hessian, gold-token NLL, or heldout performance.
Requires the original signed CAL240 role ledger. No network or writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BUDGET = 43_253_760
ROLE_ORDER = ("gate_proj", "up_proj", "down_proj")
EXPERT_ORDER = ((12, 13), (13, 22), (13, 28), (14, 63),
                (20, 17), (21, 61), (24, 26), (24, 60))
COST = {"AF4": 1_802_240, "Q4": 1_622_016, "Q5": 1_982_464}
LEDGER_FILE_SHA = "479c8548f74de0a8efab51547559be7db5fcd0e34e3e2722192c67c10c8f1dbd"
LEDGER_RECEIPT_SHA = "35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb"
ORIGINAL_AF4_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
TRACE_RECEIPT_SHA = "b911aef00007b1f9b44214996353f93c7218fe51c82daf91966e450ebba460b6"
BINARY = "JE01_CAL240_FROZEN_SSE_12AF4_6Q4_6Q5_EXACT43MB.bin"
TSV = "JE01_CAL240_FROZEN_SSE_24ROLE_SOURCE.tsv"
RECEIPT = "JE01_CAL240_FROZEN_SSE_PHYSICAL_SOURCE_RECEIPT.json"


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(d):
    body = {k: v for k, v in d.items() if k != "receipt_sha256"}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def tier(layer, expert, role):
    if (layer, expert) in ((12, 13), (13, 22), (13, 28)) and role in ROLE_ORDER[:2]:
        return "Q4"
    if ((layer, expert, role) in ((20, 17, "down_proj"),
                                  (21, 61, "down_proj"),
                                  (24, 60, "down_proj"))
            or (layer, expert) == (24, 26)):
        return "Q5"
    return "AF4"


def audit(folder, ledger_path):
    check(folder.is_dir() and not folder.is_symlink(), "JE01_BUNDLE_FOLDER_MISSING_OR_SYMLINK")
    paths = [folder / name for name in (BINARY, TSV, RECEIPT)]
    check(all(p.is_file() and not p.is_symlink() for p in paths),
          "JE01_BUNDLE_COMPONENT_MISSING_OR_SYMLINK")
    check(ledger_path.is_file() and not ledger_path.is_symlink() and
          file_sha(ledger_path) == LEDGER_FILE_SHA, "JE01_ORIGINAL_LEDGER_FILE_SHA_CHANGED")
    ledger = json.loads(ledger_path.read_bytes())
    check(ledger.get("receipt_sha256") == LEDGER_RECEIPT_SHA and
          canonical(ledger) == LEDGER_RECEIPT_SHA and
          ledger.get("training_unseen240_prompt_count") == 240 and
          ledger.get("source_heldout_OOD_labels_used") is False,
          "JE01_ORIGINAL_LEDGER_RECEIPT_OR_SCOPE_CHANGED")
    ledger_rows = {(r["layer"], r["expert"], r["role"]): r for r in ledger["per_role"]}
    check(len(ledger_rows) == 24, "JE01_ORIGINAL_LEDGER_24ROLE_MISSING")

    binary, tsv, receipt_path = paths
    check(binary.stat().st_size == BUDGET, "JE01_PHYSICAL_FILE_SIZE_WRONG")
    receipt = json.loads(receipt_path.read_bytes())
    check(receipt.get("receipt_sha256") == canonical(receipt),
          "JE01_PHYSICAL_RECEIPT_CANONICAL_SHA_INVALID")
    check(receipt.get("status") ==
          "JE01_REAL_ORIGINAL_AF4_Q4Q5_CAL240_FROZEN_ROLEWISE_43MB_CREATED" and
          receipt.get("source_original_AF4_sha256") == ORIGINAL_AF4_SHA and
          receipt.get("source_actual_cal240_SSE_DP_receipt_sha256") == LEDGER_RECEIPT_SHA and
          receipt.get("source_training_actual_h2_SwiGLU_receipt_sha256") == TRACE_RECEIPT_SHA and
          receipt.get("first16_train_pilot_excluded") is True and
          receipt.get("calibration_training_prompt_count") == 240 and
          receipt.get("calibration_routed_expert_records") == 637 and
          receipt.get("heldout_OOD_gold_accessed") is False and
          receipt.get("production_modified") is False and
          receipt.get("true_full_model_gold_NLL") == "NOT_MEASURED" and
          receipt.get("native_full_model_overlay") == "NOT_YET_BOUND",
          "JE01_PHYSICAL_RECEIPT_SCOPE_CHANGED")
    check(receipt.get("physical_file_size_bytes") == BUDGET and
          receipt.get("role_count") == 24 and
          receipt.get("counts") == {"AF4": 12, "Q4": 6, "Q5": 6} and
          receipt.get("contiguous_file_spans") == 60,
          "JE01_PHYSICAL_RECEIPT_ALLOCATION_CHANGED")
    check(receipt.get("physical_file_sha256") == file_sha(binary) and
          receipt.get("role_manifest_sha256") == file_sha(tsv),
          "JE01_PHYSICAL_BINARY_OR_TSV_SHA_MISMATCH")

    rows = receipt.get("per_role")
    check(isinstance(rows, list) and len(rows) == 24, "JE01_PHYSICAL_ROLES_MISSING")
    expected_order = [(l, e, role) for l, e in EXPERT_ORDER for role in ROLE_ORDER]
    check([(r["layer"], r["expert"], r["role"]) for r in rows] == expected_order,
          "JE01_PHYSICAL_ROLE_ORDER_MISMATCH")
    spans, lines = [], []
    counts = {"AF4": 0, "Q4": 0, "Q5": 0}
    with binary.open("rb") as data:
        for row in rows:
            layer, expert, role = row["layer"], row["expert"], row["role"]
            t = tier(layer, expert, role)
            check(row.get("tier") == t and row.get("bits") == (5 if t == "Q5" else 4),
                  "JE01_PHYSICAL_TIER_POLICY_DRIFT")
            outdim, indim = (2048, 1408) if role == "down_proj" else (1408, 2048)
            check(row.get("output") == outdim and row.get("input") == indim and
                  row.get("physical_bytes") == COST[t],
                  "JE01_PHYSICAL_ROLE_SHAPE_OR_COST_DRIFT")
            expected_parts = ("packed", "scale", "bias") if t == "AF4" else ("packed", "scale")
            part_sizes = {
                "packed": outdim * indim * row["bits"] // 8,
                "scale": outdim * (indim // 64) * 4,
                "bias": outdim * (indim // 64) * 4,
            }
            regs, digests = row["regions"], row["part_sha256"]
            check(set(regs) == set(expected_parts) and set(digests) == set(expected_parts),
                  "JE01_PHYSICAL_REGION_FIELDS_WRONG")
            check(row.get("source_original_bf16_tensor_sha256") ==
                  ledger_rows[(layer, expert, role)]["actual_original_BF16_tensor_sha256"] and
                  row.get("source_original_affine4_packed_sha256") ==
                  ledger_rows[(layer, expert, role)]["actual_original_AF4_packed_sha256"],
                  "JE01_PHYSICAL_ROLE_BF16_OR_AF4_PROVENANCE_DRIFT")
            size = 0
            for name in expected_parts:
                region = regs[name]
                check(isinstance(region, list) and len(region) == 2 and
                      all(type(v) is int for v in region), "JE01_REGION_OFFSET_TYPE_INVALID")
                start, end = region
                check(0 <= start < end <= BUDGET and end - start == part_sizes[name],
                      "JE01_REGION_OFFSET_OR_LENGTH_INVALID")
                data.seek(start)
                raw = data.read(end - start)
                check(len(raw) == end - start and
                      hashlib.sha256(raw).hexdigest() == digests[name],
                      "JE01_REGION_BYTES_SHA_MISMATCH")
                spans.append((start, end))
                size += end - start
            check(size == COST[t], "JE01_PHYSICAL_ROLE_BYTES_NOT_EXACT")
            if t == "AF4":
                check(digests["packed"] ==
                      ledger_rows[(layer, expert, role)]["actual_original_AF4_packed_sha256"],
                      "JE01_AF4_PRESERVED_PACKED_BYTES_NOT_ORIGINAL")
            counts[t] += 1
            p, s = regs["packed"], regs["scale"]
            b = regs.get("bias", [0, 0])
            lines.append("\t".join(map(str, (
                layer, expert, role, row["bits"], outdim, indim, *p, *s, *b, t
            ))))
    spans.sort()
    check(len(spans) == 60 and spans[0][0] == 0 and spans[-1][1] == BUDGET and
          all(a[1] == b[0] for a, b in zip(spans, spans[1:])),
          "JE01_PHYSICAL_SPANS_GAP_OR_OVERLAP")
    check(counts == {"AF4": 12, "Q4": 6, "Q5": 6}, "JE01_PHYSICAL_ROLE_COUNTS_MISMATCH")
    check(tsv.read_bytes() == ("\n".join(lines) + "\n").encode(),
          "JE01_PHYSICAL_TSV_ROW_BYTES_MISMATCH")
    return {
        "status": "JE01_CAL240_PHYSICAL_SOURCE_BYTE_AUDIT_PASS",
        "bundle_sha256": receipt["physical_file_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "role_count": 24, "region_count": len(spans),
        "bytes": BUDGET, "tiers": counts,
        "native_27layer_NLL": "NOT_MEASURED",
        "native_Metal_q5_numerical_parity": "NOT_MEASURED",
        "production_modified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.ledger), sort_keys=True))
