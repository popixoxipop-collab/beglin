#!/usr/bin/env python3
"""Prepare frozen TRAIN rows 16..255 for native role-direction experiments.

All 256 original training prompt/gold identities are checked before writing;
only the 240 rows outside the previously inspected diagnostic pilot are
materialized. The native 512 receipt is read for provenance, but no validation
or OOD prompt binaries or gold labels are opened. This does not run a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

REPO = Path(__file__).resolve().parents[1]
WORK = Path("/Volumes/D50/TailnetCommander/work")
ROOT = WORK / "je01_role_directional_20261123_v1"
NATIVE_ROOT = WORK / "je01_real_original256_activation_20261122_v1/portable512/je01_inputs"
GOLD_FILE = ROOT / "train_gold_source/JE01_ORIGINAL_REAL_TRAIN256_FULL_GOLD_IDS.json"
BASELINE_FILE = REPO / "docs/research/JE01_ORIGINAL_AF_REAL256_FULL_TARGET_GOLD_NLL_20261118.json"
INPUT_SHA = "2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34"
GOLD_SHA = "355a169cfb7c9ab79901ae7848afa21a2f5a576d048a5476f527bca699136fb7"
BASE_SHA = "f71a4c4397cbadf938c7a9631ba8306547b31e33028dd9a54b556a1a758e3f41"
FROZEN_SHA = "a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"
BASELINE_NLL = 2.46035653719724
META_MAGIC = b"JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1\n"
SCHEMA = "beglin-je01-role-directional-train-input/1"
SOURCE_RECEIPT_NAME = "JE01_NATIVE_384_INPUTS_RECEIPT.json"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(data):
    return digest(json.dumps(data, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode())


def filehash(path):
    return digest(Path(path).read_bytes())


def sealed(path, expected):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "JE01_TRAIN240_SOURCE_FILE_MISSING")
    data = json.loads(path.read_text())
    need(data.get("receipt_sha256") == expected ==
         canonical({k: v for k, v in data.items() if k != "receipt_sha256"}),
         "JE01_TRAIN240_SOURCE_RECEIPT_HASH_MISMATCH:" + path.name)
    return data


def write_seal(path, data):
    data["receipt_sha256"] = canonical(data)
    with path.open("x") as target:
        target.write(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())
    return data


def source_documents(native_root, gold_file, baseline_file):
    """Load pinned source receipts; this reads no prompt binary or heldout gold."""
    native_root, gold_file, baseline_file = map(Path, (native_root, gold_file, baseline_file))
    need(native_root.is_dir() and not native_root.is_symlink() and
         (native_root / "calibration").is_dir() and
         not (native_root / "calibration").is_symlink(),
         "JE01_TRAIN240_NATIVE_ROOT_INVALID")
    native = sealed(native_root / SOURCE_RECEIPT_NAME, INPUT_SHA)
    gold = sealed(gold_file, GOLD_SHA)
    baseline = sealed(baseline_file, BASE_SHA)
    need(native["total_input_records"] == 512 and native["bos_id"] == 100000 and
         native["native_context_positions"] == 384 and
         native["source_manifest_sha256"] == FROZEN_SHA,
         "JE01_TRAIN240_FROZEN_NATIVE_SOURCE_DRIFT")
    need(gold["status"] == "TRAIN256_FULL_GOLD_SOURCE_READY" and
         gold["original_256_source_rows"] == 256 and
         gold["aligned_to_original_prompt_prefix_count"] == 256 and
         gold["same_source_validation_and_ood_labels_used"] is False,
         "JE01_TRAIN240_GOLD_SOURCE_SCOPE_DRIFT")
    need(baseline["status"] == "REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_CAL256_PASS" and
         baseline["source_gold_sha256"] == GOLD_SHA and
         baseline["original_512_input_sha256"] == INPUT_SHA and
         baseline["n_calibration_train_cases"] == 256 and
         baseline["gold_full_target_tokens_scored"] == 11024 and
         abs(baseline["mean_gold_token_NLL"] - BASELINE_NLL) < 1e-13 and
         baseline["training_only"] is True and baseline["heldout_gold_labels_used"] is False,
         "JE01_TRAIN240_ORIGINAL_BASELINE_SOURCE_DRIFT")
    prompts = native["cohorts"]["calibration"]["file_receipts"]
    cases, reference = gold["cases"], baseline["per_case_real_NLL"]
    need(len(prompts) == len(cases) == len(reference) == 256 and
         len({p["input_id"] for p in prompts}) == 256,
         "JE01_TRAIN240_SOURCE_ROW_COVERAGE")
    chunks = baseline["per_chunk_native_output_sha256"]
    need(len(chunks) == 16 and [c["chunk"] for c in chunks] == list(range(16)) and
         all(isinstance(c["result_sha256"], str) and
             re.fullmatch(r"[0-9a-f]{64}", c["result_sha256"]) for c in chunks),
         "JE01_TRAIN240_PRIOR_BASELINE_CHUNK_COVERAGE")
    return native, gold, baseline


def validate_case(native_root, prompt, gold, reference, index):
    """Check one training row against its three independently pinned sources."""
    input_id = prompt["input_id"]
    need(isinstance(input_id, str) and re.fullmatch(r"[0-9a-f]{64}", input_id),
         "JE01_TRAIN240_SOURCE_ID_FORMAT")
    path = Path(native_root) / "calibration" / (input_id + ".bin")
    need(path.is_file() and not path.is_symlink(), "JE01_TRAIN240_ORIGINAL_PROMPT_MISSING")
    raw = path.read_bytes()
    need(len(raw) >= 8 and len(raw) % 4 == 0 and
         len(raw) == prompt["native_int32_bytes"] and
         digest(raw) == prompt["native_int32_sha256"] == gold["native_BOS_prompt_sha256"] and
         input_id == gold["input_id"] == reference["source_id"] and
         gold["row"] == reference["row"] == index,
         "JE01_TRAIN240_ORIGINAL_CASE_IDENTITY_DRIFT:" + str(index))
    tokens = list(struct.unpack("<" + "i" * (len(raw) // 4), raw))
    targets = gold["gold_continuation_token_ids"]
    need(tokens[0] == 100000 and all(0 <= v < 102400 for v in tokens) and
         len(tokens) == gold["original_BOS_prompt_positions"] == prompt["token_count_with_bos"] and
         isinstance(targets, list) and len(targets) >= 1 and
         len(targets) == gold["gold_token_count"] == reference["tokens"] and
         all(type(v) is int and 0 <= v < 102400 for v in targets) and
         len(tokens) + len(targets) == gold["model_full_BOS_sequence_positions"] and
         len(tokens) + len(targets) <= 384 and
         all(isinstance(reference[k], (int, float)) and math.isfinite(reference[k]) and reference[k] >= 0
             for k in ("first_gold_nll", "nll_sum")),
         "JE01_TRAIN240_ORIGINAL_TOKEN_ALIGNMENT_DRIFT:" + str(index))
    return {"row": index, "input_id": input_id, "prompt_token_ids": tokens,
            "target_token_ids": targets, "original_prompt_path": str(path),
            "original_prompt_sha256": digest(raw), "original_prompt_positions": len(tokens),
            "gold_tokens": len(targets), "original_baseline_first_gold_nll": reference["first_gold_nll"],
            "original_baseline_nll_sum": reference["nll_sum"]}


def teacher_forced_bytes(row):
    tokens = row["prompt_token_ids"] + row["target_token_ids"][:-1]
    return struct.pack("<" + "i" * len(tokens), *tokens)


def metadata_bytes(rows):
    need(len(rows) == 16, "JE01_TRAIN240_NATIVE_CHUNK_REQUIRES16")
    raw = bytearray(META_MAGIC)
    for row in rows:
        targets = row["target_token_ids"]
        start = len(row["prompt_token_ids"])
        need(start >= 2 and 1 <= len(targets) and start + len(targets) <= 384,
             "JE01_TRAIN240_METADATA_BOUNDS")
        raw.extend(struct.pack("<II", start, len(targets)))
        raw.extend(struct.pack("<384i", *(targets + [-1] * (384 - len(targets)))))
    return bytes(raw)


def case_source_record(row):
    return {k: v for k, v in row.items() if k not in ("prompt_token_ids", "target_token_ids")}


def source_linkage(native_root, gold_file, baseline_file):
    return {"native_input_root": str(native_root),
            "native_input_receipt_path": str(Path(native_root) / SOURCE_RECEIPT_NAME),
            "native_input_receipt_sha256": INPUT_SHA,
            "gold_source_path": str(gold_file), "gold_source_receipt_sha256": GOLD_SHA,
            "baseline_source_path": str(baseline_file), "baseline_source_receipt_sha256": BASE_SHA,
            "frozen_manifest_sha256": FROZEN_SHA,
            "all_256_training_source_rows_validated": True,
            "excluded_diagnostic_case_ids": list(range(16))}


def validate_source_linkage(data):
    """Revalidate one 16-row dataset directly from the pinned original sources.

Called by the native runner before each TRAIN240 cell. This reads only the
16 requested original training prompt binaries, then independently derives
the complete teacher-forcing payload and metadata hashes from source gold.
    """
    need(data.get("schema") == SCHEMA and data.get("scope") == "TRAIN240" and
         data.get("independent_validation_ood_accessed") is False,
         "JE01_TRAIN240_DATASET_SCOPE_DRIFT")
    ids = data["case_ids"]
    need(len(ids) == 16 and all(type(i) is int for i in ids) and
         ids[0] in range(16, 256, 16) and ids == list(range(ids[0], ids[0] + 16)),
         "JE01_TRAIN240_DATASET_NOT_FROZEN_CHUNK")
    chunk = ids[0] // 16
    need(data["dataset_id"] == "train240_chunk" + str(chunk).zfill(2) and data["chunk"] == chunk,
         "JE01_TRAIN240_DATASET_ID_DRIFT")
    linkage = data["source_linkage"]
    native_root = Path(linkage["native_input_root"])
    gold_file, baseline_file = Path(linkage["gold_source_path"]), Path(linkage["baseline_source_path"])
    need(linkage == source_linkage(native_root, gold_file, baseline_file),
         "JE01_TRAIN240_SOURCE_LINKAGE_DRIFT")
    native, gold, baseline = source_documents(native_root, gold_file, baseline_file)
    rows = [validate_case(native_root, native["cohorts"]["calibration"]["file_receipts"][i],
                          gold["cases"][i], baseline["per_case_real_NLL"][i], i) for i in ids]
    need(data["target_token_ids"] == [r["target_token_ids"] for r in rows] and
         data["input_sha256"] == [digest(teacher_forced_bytes(r)) for r in rows] and
         data["gold_metadata_sha256"] == digest(metadata_bytes(rows)) and
         data["case_sources"] == [case_source_record(r) for r in rows],
         "JE01_TRAIN240_DERIVED_DATA_NOT_FROM_PINNED_SOURCE")
    old_chunk = baseline["per_chunk_native_output_sha256"][chunk]
    need(data["baseline_output_sha256"] == old_chunk["result_sha256"] and
         data["gold_tokens"] == sum(r["gold_tokens"] for r in rows) == old_chunk["gold_targets_scored"],
         "JE01_TRAIN240_CHUNK_BASELINE_DRIFT")
    manifest = Path(data["manifest"])
    need(manifest.is_file() and not manifest.is_symlink() and
         manifest.parent.name == "chunk" + str(chunk).zfill(2) and
         Path(data["gold_metadata"]) == manifest.parent / "gold_meta.bin",
         "JE01_TRAIN240_CHUNK_PATH_DRIFT")
    input_dir = manifest.parent.parent / "inputs"
    expected_manifest = "".join(str(input_dir / (r["input_id"] + ".bin")) + " 1\n" for r in rows)
    need(manifest.read_text() == expected_manifest and digest(expected_manifest.encode()) == data["manifest_sha256"],
         "JE01_TRAIN240_MANIFEST_NOT_ORIGINAL_SOURCE_ROWS")
    return rows


def prepare(native_root, gold_file, baseline_file, output):
    native_root, gold_file, baseline_file, output = map(Path, (native_root, gold_file, baseline_file, output))
    need(output.parent == ROOT and ROOT.is_dir() and not ROOT.is_symlink() and
         not output.exists() and not output.is_symlink() and
         not any(c.isspace() for c in str(output)) and len(str(output)) < 600,
         "JE01_TRAIN240_FRESH_ISOLATED_OUTPUT_REQUIRED")
    native, gold, baseline = source_documents(native_root, gold_file, baseline_file)
    rows = [validate_case(native_root, p, c, r, i) for i, (p, c, r) in enumerate(zip(
        native["cohorts"]["calibration"]["file_receipts"], gold["cases"], baseline["per_case_real_NLL"]))]
    need(sum(r["gold_tokens"] for r in rows) == 11024 and
         sum(r["gold_tokens"] for r in rows[:16]) == 656 and
         sum(r["gold_tokens"] for r in rows[16:]) == 10368,
         "JE01_TRAIN240_ORIGINAL_GOLD_COUNTS_DRIFT")
    output.mkdir(mode=0o700)
    inputs = output / "inputs"
    inputs.mkdir(mode=0o700)
    receipts = []
    linkage = source_linkage(native_root, gold_file, baseline_file)
    for chunk in range(1, 16):
        selected = rows[chunk * 16:(chunk + 1) * 16]
        folder = output / ("chunk" + str(chunk).zfill(2))
        folder.mkdir(mode=0o700)
        lines, hashes = [], []
        for row in selected:
            raw = teacher_forced_bytes(row)
            path = inputs / (row["input_id"] + ".bin")
            with path.open("xb") as target:
                need(target.write(raw) == len(raw), "JE01_TRAIN240_INPUT_SHORT_WRITE")
                target.flush()
                os.fsync(target.fileno())
            hashes.append(digest(raw))
            lines.append(str(path) + " 1\n")
        manifest, metadata = folder / "manifest.txt", folder / "gold_meta.bin"
        with manifest.open("x") as target:
            target.write("".join(lines))
        with metadata.open("xb") as target:
            target.write(metadata_bytes(selected))
        data = {"schema": SCHEMA, "scope": "TRAIN240", "chunk": chunk,
                "dataset_id": "train240_chunk" + str(chunk).zfill(2),
                "case_ids": [r["row"] for r in selected],
                "manifest": str(manifest), "gold_metadata": str(metadata),
                "manifest_sha256": filehash(manifest), "gold_metadata_sha256": filehash(metadata),
                "input_sha256": hashes,
                "target_token_ids": [r["target_token_ids"] for r in selected],
                "baseline_output_sha256": baseline["per_chunk_native_output_sha256"][chunk]["result_sha256"],
                "gold_tokens": sum(r["gold_tokens"] for r in selected),
                "case_sources": [case_source_record(r) for r in selected],
                "source_linkage": linkage,
                "baseline_output_is_historical_reference_not_new_execution": True,
                "independent_validation_ood_accessed": False,
                "native_model_forward": "NOT_RUN", "production_modified": False}
        validate_source_linkage(data)
        dataset = folder / "dataset.json"
        write_seal(dataset, data)
        receipts.append({"chunk": chunk, "dataset_id": data["dataset_id"],
                         "dataset_path": str(dataset), "dataset_receipt_sha256": data["receipt_sha256"],
                         "case_ids": data["case_ids"], "gold_tokens": data["gold_tokens"]})
    report = {"schema": "beglin-je01-native-role-directional-train240-source-preparation/1",
              "status": "TRAIN256_ALL_SOURCE_VERIFIED_TRAIN240_FIFTEEN_CHUNKS_PREPARED",
              "source_linkage": linkage, "validated_original_training_cases": 256,
              "prepared_training_cases": 240, "prepared_gold_tokens": 10368,
              "prepared_case_ids": list(range(16, 256)), "chunks": receipts,
              "independent_validation_ood_accessed": False,
              "native_model_forward": "NOT_RUN", "production_modified": False}
    write_seal(output / "JE01_TRAIN240_INPUT_PREPARATION_RECEIPT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-input-root", type=Path, default=NATIVE_ROOT)
    parser.add_argument("--gold-json", type=Path, default=GOLD_FILE)
    parser.add_argument("--baseline-json", type=Path, default=BASELINE_FILE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.native_input_root, args.gold_json, args.baseline_json, args.output)
    print(json.dumps({"status": result["status"], "receipt_sha256": result["receipt_sha256"],
                      "prepared_training_cases": 240, "prepared_gold_tokens": 10368}, sort_keys=True))
