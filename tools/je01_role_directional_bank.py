#!/usr/bin/env python3
"""Build the source-pinned AF4/Q4/Q5 bank for native JE01 role directions.

This is a 129,761,280-byte research catalogue, not a same-budget candidate.
Every original AF4 role is copied verbatim; Q4/Q5 use the pinned native codec
on the same SHA-attested BF16 tensor. No labels or model forwards are read.
The small packing helpers are also used by tests with explicitly synthetic
payloads; only ``build`` can produce a real-source bank receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/je01_eoe_cal240_frozen_sse_exactbyte_role_bundle_20261122.py"
SOURCE_BLOB = "89f05ad396b3face37da585b024f23e31ab706e6"
LEDGER = ROOT / "docs/research/JE01_ACTUAL_CAL240_FFN_H2_SWIGLU_NATIVE_OUTPUT_DP_20261122.json"
LEDGER_RECEIPT = "35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb"
CAL_SOURCE_SHA = "863fb9c1d5a17fc810564610739bd1c53f727addee99a7966f65404b8725a414"
WORK = Path("/Volumes/D50/TailnetCommander/work")
AF_SHA = "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
FROZEN = ((12, 13), (13, 22), (13, 28), (14, 63),
          (20, 17), (21, 61), (24, 26), (24, 60))
ROLES = ("gate_proj", "up_proj", "down_proj")
TIERS = ("AF4", "Q4", "Q5")
ROLE_KEYS = tuple((l, e, r) for l, e in FROZEN for r in ROLES)
COST = {"AF4": 1_802_240, "Q4": 1_622_016, "Q5": 1_982_464}
BANK_BYTES = 129_761_280
BUDGET = 43_253_760
BINARY = "JE01_ROLE_DIRECTION_BANK.bin"
MANIFEST = "JE01_ROLE_DIRECTION_BANK.tsv"
RECEIPT = "JE01_ROLE_DIRECTION_BANK_RECEIPT.json"
SCHEMA = "beglin-je01-original27-native-role-direction-bank/1"
STATUS = "JE01_NATIVE_AF4_Q4_Q5_72_ENTRY_DIRECTION_BANK_BUILT"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(2 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode())


def gitblob(path):
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def shape_for(role):
    need(role in ROLES, "JE01_DIRECTION_UNKNOWN_ROLE")
    return (2048, 1408) if role == "down_proj" else (1408, 2048)


def part_sizes(tier, out, inp):
    need(tier in TIERS and type(out) is int and type(inp) is int and
         out > 0 and inp > 0 and inp % 64 == 0,
         "JE01_DIRECTION_INVALID_CODEC_SHAPE")
    result = {"packed": out * inp * (5 if tier == "Q5" else 4) // 8,
              "scale": out * (inp // 64) * 4}
    if tier == "AF4":
        result["bias"] = result["scale"]
    return result


def append_payload(sink, metadata, parts):
    """Write one role's parts and return exact offsets plus content hashes."""
    row = dict(metadata)
    expected = part_sizes(row["tier"], row["output"], row["input"])
    need(set(parts) == set(expected), "JE01_DIRECTION_PART_SET_MISMATCH")
    need(all(isinstance(parts[k], bytes) and len(parts[k]) == n
             for k, n in expected.items()), "JE01_DIRECTION_PART_LENGTH_MISMATCH")
    row["bits"] = 5 if row["tier"] == "Q5" else 4
    row["regions"] = {}
    row["part_sha256"] = {}
    combined = hashlib.sha256()
    for name in ("packed", "scale", "bias"):
        if name not in parts:
            continue
        begin = sink.tell()
        need(sink.write(parts[name]) == len(parts[name]), "JE01_DIRECTION_SHORT_WRITE")
        row["regions"][name] = [begin, sink.tell()]
        row["part_sha256"][name] = digest(parts[name])
        combined.update(parts[name])
    row["physical_bytes"] = sum(expected.values())
    row["payload_sha256"] = combined.hexdigest()
    return row


def validate_layout(rows, total_bytes):
    """Validate order, exact lengths and contiguous coverage; no source claim."""
    need(rows and type(total_bytes) is int and total_bytes > 0,
         "JE01_DIRECTION_EMPTY_LAYOUT")
    cursor = 0
    seen = set()
    for row in rows:
        ident = (row["layer"], row["expert"], row["role"], row["tier"])
        need(type(row["layer"]) is int and type(row["expert"]) is int and
             row["layer"] >= 0 and row["expert"] >= 0 and
             row["role"] in ROLES and ident not in seen,
             "JE01_DIRECTION_DUPLICATE_OR_INVALID_ID")
        seen.add(ident)
        expected = part_sizes(row["tier"], row["output"], row["input"])
        need(row["bits"] == (5 if row["tier"] == "Q5" else 4) and
             row["physical_bytes"] == sum(expected.values()) and
             set(row["regions"]) == set(expected) == set(row["part_sha256"]),
             "JE01_DIRECTION_LAYOUT_CODEC_MISMATCH")
        for part, size in expected.items():
            span = row["regions"][part]
            need(isinstance(span, list) and len(span) == 2 and
                 all(type(v) is int for v in span) and
                 span == [cursor, cursor + size],
                 "JE01_DIRECTION_NONCONTIGUOUS_OR_INVALID_SPAN")
            need(isinstance(row["part_sha256"][part], str) and
                 re.fullmatch(r"[0-9a-f]{64}", row["part_sha256"][part]),
                 "JE01_DIRECTION_INVALID_PART_HASH")
            cursor += size
        need(isinstance(row["payload_sha256"], str) and
             re.fullmatch(r"[0-9a-f]{64}", row["payload_sha256"]),
             "JE01_DIRECTION_INVALID_PAYLOAD_HASH")
    need(cursor == total_bytes, "JE01_DIRECTION_TOTAL_BYTES_MISMATCH")


def manifest_text(rows):
    lines = []
    for row in rows:
        packed, scale = row["regions"]["packed"], row["regions"]["scale"]
        bias = row["regions"].get("bias", [0, 0])
        values = (row["layer"], row["expert"], row["role"], row["bits"],
                  row["output"], row["input"], *packed, *scale, *bias, row["tier"])
        lines.append("\t".join(map(str, values)))
    return "\n".join(lines) + "\n"


def copy_payload(source, sink, row):
    """Copy a selected bank entry, checking every part before writing."""
    parts = {}
    for name in ("packed", "scale", "bias"):
        if name not in row["regions"]:
            continue
        begin, end = row["regions"][name]
        source.seek(begin)
        raw = source.read(end - begin)
        need(len(raw) == end - begin and digest(raw) == row["part_sha256"][name],
             "JE01_DIRECTION_SOURCE_PAYLOAD_HASH_MISMATCH")
        parts[name] = raw
    need(digest(b"".join(parts.values())) == row["payload_sha256"],
         "JE01_DIRECTION_SOURCE_COMBINED_HASH_MISMATCH")
    return append_payload(sink, row, parts)


def original_module():
    need(SOURCE.is_file() and not SOURCE.is_symlink() and
         gitblob(SOURCE) == SOURCE_BLOB, "JE01_DIRECTION_PINNED_BUILDER_DRIFT")
    spec = importlib.util.spec_from_file_location("je01_direction_original_builder", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(module.AF_SHA == AF_SHA and tuple(module.FROZEN) == FROZEN and
         tuple(module.ROLES) == ROLES and module.COST == COST,
         "JE01_DIRECTION_SOURCE_CONTRACT_DRIFT")
    return module


def pinned_source_roles():
    """Use repository source evidence to check role labels independently."""
    need(LEDGER.is_file() and not LEDGER.is_symlink(), "JE01_DIRECTION_PINNED_LEDGER_MISSING")
    data = json.loads(LEDGER.read_text())
    need(data.get("receipt_sha256") == LEDGER_RECEIPT ==
         canonical({k: v for k, v in data.items() if k != "receipt_sha256"}) and
         data["source_full_original_AF4_sha256"] == AF_SHA,
         "JE01_DIRECTION_PINNED_LEDGER_HASH_MISMATCH")
    roles = {(r["layer"], r["expert"], r["role"]): r for r in data["per_role"]}
    need(len(data["per_role"]) == 24 and set(roles) == set(ROLE_KEYS),
         "JE01_DIRECTION_PINNED_LEDGER_ROLE_COVERAGE")
    return roles


def fresh_output(output):
    output = Path(output)
    need(output.parent == WORK and "je01" in output.name.lower() and
         not output.exists() and not output.is_symlink() and
         WORK.is_dir() and not WORK.is_symlink(),
         "JE01_DIRECTION_FRESH_D50_RESEARCH_OUTPUT_REQUIRED")
    return output


def write_receipt(path, record):
    record["receipt_sha256"] = canonical(record)
    with path.open("x") as target:
        target.write(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def build(output):
    output = fresh_output(output)
    original = original_module()
    baseline, source_roles = original.load_ledger()
    need(set(baseline) == set(ROLE_KEYS) == set(source_roles),
         "JE01_DIRECTION_ORIGINAL_ROLE_UNIVERSE_DRIFT")
    native, af, layout, weightmap, shards, headers, lib = original.load_original_sources()
    # load_original_sources rehashes the complete 9.8 GB original AF4 file.
    need(native.AF_FILE_SHA == AF_SHA and af.stat().st_size == 9_814_343_680,
         "JE01_DIRECTION_ORIGINAL_AF_IDENTITY")
    output.mkdir(mode=0o700)
    rows = []
    with (output / BINARY).open("xb") as sink:
        for role_index, (layer, expert, role) in enumerate(ROLE_KEYS):
            bf16, source = native.bf16_selected(weightmap, shards, headers, layer, expert, role)
            origin = source_roles[(layer, expert, role)]
            need(source["original_bf16_tensor_sha256"] ==
                 origin["actual_original_BF16_tensor_sha256"],
                 "JE01_DIRECTION_BF16_TENSOR_SOURCE_DRIFT")
            out, inp = shape_for(role)
            for tier in TIERS:
                if tier == "AF4":
                    parts = original.original_af4_parts(af, layout, layer, expert, role)
                    need(digest(parts["packed"]) == origin["actual_original_AF4_packed_sha256"],
                         "JE01_DIRECTION_ORIGINAL_AF4_PACKED_DRIFT")
                else:
                    parts = original.native_symmetric_parts(lib, bf16, 4 if tier == "Q4" else 5)
                row = append_payload(sink, {
                    "role_index": role_index, "layer": layer, "expert": expert,
                    "role": role, "tier": tier, "output": out, "input": inp,
                    "source_original_bf16_tensor_sha256": source["original_bf16_tensor_sha256"],
                    "source_original_affine4_packed_sha256": origin["actual_original_AF4_packed_sha256"],
                }, parts)
                need(row["physical_bytes"] == COST[tier], "JE01_DIRECTION_ROLE_BYTES")
                rows.append(row)
            print(json.dumps({"event": "JE01_DIRECTION_ROLE_BUILT", "role_index": role_index,
                              "layer": layer, "expert": expert, "role": role,
                              "entries_complete": len(rows)}, sort_keys=True), flush=True)
        sink.flush()
        os.fsync(sink.fileno())
    validate_layout(rows, (output / BINARY).stat().st_size)
    need(len(rows) == 72 and (output / BINARY).stat().st_size == BANK_BYTES,
         "JE01_DIRECTION_REAL_BANK_SIZE")
    with (output / MANIFEST).open("x") as target:
        target.write(manifest_text(rows))
    record = {
        "schema": SCHEMA, "status": STATUS,
        "source_native_builder_git_blob_sha1": SOURCE_BLOB,
        "source_original_AF4_sha256": AF_SHA,
        "source_original_AF4_file_bytes": 9_814_343_680,
        "source_original_AF4_full_rehash": True,
        "source_cal240_ledger_receipt_sha256": original.LEDGER_SHA,
        "source_cal240_python_sha256": original.CAL_SRC_SHA,
        "native_codec_library_sha256": filehash(original.NATIVE / "result/je01_native_q4q5.dylib"),
        "builder_file_sha256": filehash(Path(__file__)),
        "physical_file_sha256": filehash(output / BINARY),
        "physical_file_size_bytes": BANK_BYTES,
        "role_manifest_sha256": filehash(output / MANIFEST),
        "role_count": 24, "catalog_entry_count": 72,
        "counts": {"AF4": 24, "Q4": 24, "Q5": 24},
        "catalog_is_same_budget_candidate": False,
        "entry_order": "frozen pairs; gate_proj/up_proj/down_proj; AF4/Q4/Q5",
        "per_entry": rows,
        "all_choices_built_without_loss_based_selection": True,
        "true_full_model_gold_NLL": "NOT_MEASURED",
        "empirical_fisher_or_parameter_gradients": "NOT_MEASURED",
        "heldout_validation_OOD_accessed": False,
        "original_model_weights_modified": False, "production_modified": False,
    }
    write_receipt(output / RECEIPT, record)
    verify_bank(output, record["receipt_sha256"])
    print(json.dumps({"status": STATUS, "receipt_sha256": record["receipt_sha256"],
                      "physical_file_sha256": record["physical_file_sha256"],
                      "bytes": BANK_BYTES}, sort_keys=True), flush=True)
    return record


def verify_bank(bank, expected_receipt_sha256=None):
    bank = Path(bank)
    need(bank.is_dir() and not bank.is_symlink(), "JE01_DIRECTION_BANK_MISSING")
    for name in (BINARY, MANIFEST, RECEIPT):
        path = bank / name
        need(path.is_file() and not path.is_symlink(), "JE01_DIRECTION_BANK_FILE_MISSING")
    record = json.loads((bank / RECEIPT).read_text())
    body = {k: v for k, v in record.items() if k != "receipt_sha256"}
    need(record.get("receipt_sha256") == canonical(body) and
         (expected_receipt_sha256 is None or record["receipt_sha256"] == expected_receipt_sha256),
         "JE01_DIRECTION_BANK_RECEIPT_HASH_MISMATCH")
    need(record["schema"] == SCHEMA and record["status"] == STATUS and
         record["source_original_AF4_sha256"] == AF_SHA and
         record["source_native_builder_git_blob_sha1"] == SOURCE_BLOB and
         record["source_cal240_ledger_receipt_sha256"] == LEDGER_RECEIPT and
         record["source_cal240_python_sha256"] == CAL_SOURCE_SHA and
         record["source_original_AF4_full_rehash"] is True and
         record["source_original_AF4_file_bytes"] == 9_814_343_680 and
         record["role_count"] == 24 and record["catalog_entry_count"] == 72 and
         record["counts"] == {"AF4": 24, "Q4": 24, "Q5": 24} and
         record["catalog_is_same_budget_candidate"] is False and
         record["heldout_validation_OOD_accessed"] is False and
         record["original_model_weights_modified"] is False and
         record["production_modified"] is False,
         "JE01_DIRECTION_BANK_SOURCE_SCOPE_MISMATCH")
    need((bank / BINARY).stat().st_size == record["physical_file_size_bytes"] == BANK_BYTES and
         filehash(bank / BINARY) == record["physical_file_sha256"] and
         filehash(bank / MANIFEST) == record["role_manifest_sha256"],
         "JE01_DIRECTION_BANK_FILE_HASH_MISMATCH")
    rows = record["per_entry"]
    source_roles = pinned_source_roles()
    need([(r["layer"], r["expert"], r["role"], r["tier"]) for r in rows] ==
         [(*key, tier) for key in ROLE_KEYS for tier in TIERS],
         "JE01_DIRECTION_BANK_ENTRY_ORDER_MISMATCH")
    for i, row in enumerate(rows):
        need(row["role_index"] == i // 3 and
             (row["output"], row["input"]) == shape_for(row["role"]) and
             row["physical_bytes"] == COST[row["tier"]],
             "JE01_DIRECTION_BANK_REAL_ROLE_SHAPE_MISMATCH")
        need(all(isinstance(row.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", row[k])
                 for k in ("source_original_bf16_tensor_sha256",
                           "source_original_affine4_packed_sha256")),
             "JE01_DIRECTION_BANK_SOURCE_TENSOR_HASH_MISSING")
        origin = source_roles[(row["layer"], row["expert"], row["role"])]
        need(row["source_original_bf16_tensor_sha256"] == origin["actual_original_BF16_tensor_sha256"] and
             row["source_original_affine4_packed_sha256"] == origin["actual_original_AF4_packed_sha256"] and
             (row["tier"] != "AF4" or row["part_sha256"]["packed"] ==
              origin["actual_original_AF4_packed_sha256"]),
             "JE01_DIRECTION_BANK_SOURCE_ROLE_RELABELLED")
    validate_layout(rows, BANK_BYTES)
    need((bank / MANIFEST).read_text() == manifest_text(rows),
         "JE01_DIRECTION_BANK_MANIFEST_LAYOUT_MISMATCH")
    # Hash every part as well as the full file. A self-consistent metadata
    # rewrite cannot silently relabel parts while preserving the binary.
    with (bank / BINARY).open("rb") as source:
        for row in rows:
            combined = hashlib.sha256()
            for part, (begin, end) in row["regions"].items():
                source.seek(begin)
                raw = source.read(end - begin)
                need(digest(raw) == row["part_sha256"][part],
                     "JE01_DIRECTION_BANK_PART_HASH_MISMATCH")
            for part in ("packed", "scale", "bias"):
                if part in row["regions"]:
                    begin, end = row["regions"][part]
                    source.seek(begin)
                    combined.update(source.read(end - begin))
            need(combined.hexdigest() == row["payload_sha256"],
                 "JE01_DIRECTION_BANK_PAYLOAD_HASH_MISMATCH")
    return record


def normalize_assignment(assignment, mode="fixed_12_6_6"):
    need(isinstance(assignment, dict), "JE01_DIRECTION_ASSIGNMENT_MUST_BE_MAPPING")
    result = {}
    for key, tier in assignment.items():
        if isinstance(key, str):
            parts = key.split("/")
            need(len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit(),
                 "JE01_DIRECTION_INVALID_ROLE_KEY")
            key = (int(parts[0]), int(parts[1]), parts[2])
        need(isinstance(key, tuple) and len(key) == 3 and
             type(key[0]) is int and type(key[1]) is int and
             key in ROLE_KEYS and key not in result and tier in TIERS,
             "JE01_DIRECTION_INVALID_OR_DUPLICATE_ASSIGNMENT")
        result[key] = tier
    need(set(result) == set(ROLE_KEYS), "JE01_DIRECTION_ASSIGNMENT_ROLE_COVERAGE")
    counts = {tier: list(result.values()).count(tier) for tier in TIERS}
    total = sum(COST[t] for t in result.values())
    if mode == "fixed_12_6_6":
        need(counts == {"AF4": 12, "Q4": 6, "Q5": 6} and total == BUDGET,
             "JE01_DIRECTION_REQUIRED_12_6_6_EXACT_BUDGET")
    elif mode == "equal_bytes":
        need(counts["Q4"] == counts["Q5"] and total == BUDGET,
             "JE01_DIRECTION_EXACT_BUDGET_REQUIRED")
    elif mode == "single_role_diagnostic":
        need(counts["AF4"] == 23 and counts["Q4"] + counts["Q5"] == 1 and
             total in (BUDGET - 180_224, BUDGET + 180_224),
             "JE01_DIRECTION_SINGLE_ROLE_DIAGNOSTIC_REQUIRED")
    else:
        raise ValueError("JE01_DIRECTION_UNKNOWN_MATERIALIZATION_MODE")
    return result, counts, total


def materialize(assignment, bank, output, *, mode="fixed_12_6_6",
                expected_bank_receipt_sha256=None):
    """Copy one already-built tier per role; never quantize or select by NLL.

The default is the preregistered 12/6/6 allocation. ``equal_bytes`` also
permits the all-AF4 control. ``single_role_diagnostic`` is deliberately
labelled unequal-budget and accepts only one Q4 or Q5 replacement.
    """
    assigned, counts, total = normalize_assignment(assignment, mode)
    output = fresh_output(output)
    record = verify_bank(bank, expected_bank_receipt_sha256)
    index = {(r["layer"], r["expert"], r["role"], r["tier"]): r
             for r in record["per_entry"]}
    output.mkdir(mode=0o700)
    binary, manifest = output / "JE01_ROLE_POLICY.bin", output / "JE01_ROLE_POLICY.tsv"
    rows = []
    with (Path(bank) / BINARY).open("rb") as source, binary.open("xb") as sink:
        for key in ROLE_KEYS:
            rows.append(copy_payload(source, sink, index[(*key, assigned[key])]))
        sink.flush()
        os.fsync(sink.fileno())
    validate_layout(rows, binary.stat().st_size)
    need(binary.stat().st_size == total, "JE01_DIRECTION_SELECTED_BUDGET_MISMATCH")
    with manifest.open("x") as target:
        target.write(manifest_text(rows))
    result = {
        "schema": "beglin-je01-native-role-direction-bank-materialization/1",
        "status": "JE01_SELECTED_ROLE_PAYLOADS_COPIED_AND_HASH_VERIFIED",
        "selection_mode": mode, "counts": counts,
        "same_budget_candidate": total == BUDGET,
        "source_bank_receipt_sha256": record["receipt_sha256"],
        "source_bank_file_sha256": record["physical_file_sha256"],
        "source_original_AF4_sha256": AF_SHA,
        "physical_file_size_bytes": total, "physical_file_sha256": filehash(binary),
        "role_manifest_sha256": filehash(manifest), "per_role": rows,
        "assignment": {"/".join(map(str, key)): assigned[key] for key in ROLE_KEYS},
        "selection_algorithm": "CALLER_SUPPLIED_ASSIGNMENT_NO_OPTIMIZATION",
        "full_model_NLL": "NOT_MEASURED", "heldout_validation_OOD_accessed": False,
        "original_model_weights_modified": False, "production_modified": False,
    }
    write_receipt(output / "JE01_ROLE_POLICY_RECEIPT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("build")
    create.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--bank", type=Path, required=True)
    check.add_argument("--expected-receipt", required=True)
    select = sub.add_parser("materialize")
    select.add_argument("--assignment", type=Path, required=True)
    select.add_argument("--bank", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--expected-bank-receipt", required=True)
    select.add_argument("--mode", choices=("fixed_12_6_6", "equal_bytes", "single_role_diagnostic"),
                        default="fixed_12_6_6")
    args = parser.parse_args()
    if args.command == "build":
        build(args.output)
    elif args.command == "verify":
        data = verify_bank(args.bank, args.expected_receipt)
        print(json.dumps({"status": "JE01_DIRECTION_BANK_VERIFIED", "receipt_sha256": data["receipt_sha256"]}))
    else:
        data = materialize(json.loads(args.assignment.read_text()), args.bank, args.output,
                           mode=args.mode, expected_bank_receipt_sha256=args.expected_bank_receipt)
        print(json.dumps({"status": data["status"], "receipt_sha256": data["receipt_sha256"],
                          "physical_file_size_bytes": data["physical_file_size_bytes"]}, sort_keys=True))
