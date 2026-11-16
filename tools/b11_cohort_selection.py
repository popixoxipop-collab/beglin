#!/usr/bin/env python3
"""B11 leakage-free deterministic 400-case cohort selection (pure offline).

Expects a pre-audited, outcome-free list of eligible ORIGINAL Qv21 tokenized
question metadata from pinned Arrow/tokenizer sources. Never opens labels,
model scores, Qv20 validation or Qv21 evaluation results, writes no files.
Caller must verify source hashes and exclude-list provenance separately.
"""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "BEGLIN-B11-pre-native-label-free-cohort-freeze/1"
SEED = "BEGLIN_B11_QV21_DISJOINT_20261117_V1"
SOURCE_SHA = {
    "arc_train": "2bb1e5e8875ff5944d3a6bf469ef6ea811217be8bc083abd01722ac62cdaa873",
    "ob_train": "076d0bfdd6c1059aa7a233f6b779b3712abe9205bf69737145cfe85b849854e8",
}
TOKENIZER_SHA = "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf"
QV20_CAL_SHA = "6bedf97854f140939b248001a491b6f7ae38a078b502a8af9a85d3a47cc99f28"
QV21_FREEZE_SHA = "eaf4633703f7ebc4139cbe6c970e36076a51693afa24141054d39d9fcaa93e40"
QV21_PRIVATE_SHA = {
    "cal_arc_train96": "982455c845d4c46ff66811e3ddd38fa70e9ade1cbcbba5880919943c8656fe2f",
    "eval_arc_train48": "c17fa191f40c4dc2facc4c8f2546cc68949593d29bdfbd82144c1cb35e07bb58",
    "eval_ob_train48": "5f8339b82edddceaae63f2f78a53cc0f64416ad4cd52b151a2bd644d800af78c",
}
FREEZE_SPLITS = (
    ("b11_fit_arc_train120", "arc_train", 60),
    ("b11_cal_arc_train80", "arc_train", 40),
    ("b11_eval_arc_train80", "arc_train", 40),
    ("b11_eval_ob_train120", "ob_train", 60),
)
EXCLUSION_SETS = (
    ("qv20_arc_train32", "arc_train", 32),
    ("qv21_cal_arc_train96", "arc_train", 96),
    ("qv21_eval_arc_train48", "arc_train", 48),
    ("qv21_eval_ob_train48", "ob_train", 48),
)
HEXSHA = re.compile(r"^[a-f0-9]{64}$")
ALLOWED_ROW_KEYS = {
    "source_id", "question_sha256", "full_prompt_sha256", "prefix_ids_sha256", "prefix_len",
}
EXCLUDED_ROW_KEYS = {"source_id", "question_sha256"}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def check_sha(value):
    if not isinstance(value, str) or not HEXSHA.fullmatch(value):
        raise ValueError("B11_SHA_INVALID")


def _check_id(value):
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError("B11_SOURCE_ID_INVALID")


def _candidate(row):
    if not isinstance(row, dict) or set(row) != ALLOWED_ROW_KEYS:
        raise ValueError("B11_CANDIDATE_FIELDS_LABEL_FREE_REQUIRED")
    _check_id(row["source_id"])
    for key in ("question_sha256", "full_prompt_sha256", "prefix_ids_sha256"):
        check_sha(row[key])
    n = row["prefix_len"]
    if type(n) is not int or not 18 <= n <= 208:
        raise ValueError("B11_PREFIX_LEN_INVALID")
    return row


def _exclusion(row):
    if not isinstance(row, dict) or set(row) != EXCLUDED_ROW_KEYS:
        raise ValueError("B11_EXCLUSION_FIELDS_LABEL_FREE_REQUIRED")
    _check_id(row["source_id"])
    check_sha(row["question_sha256"])


def select_cohorts(arc_metadata, ob_metadata, exclusion_sets):
    """Construct immutable *proposal* manifest if all splits are feasible.

    This function cannot certify actual source file hashes, original tokenizer
    output or absence of native scoring. Caller performs external SHA audit
    and create-only freeze before original-C measurement.
    """
    if not isinstance(exclusion_sets, dict) or set(exclusion_sets) != {
        x[0] for x in EXCLUSION_SETS
    }:
        raise ValueError("B11_EXCLUSION_SETS_MISSING")

    occupied_ids = {"arc_train": set(), "ob_train": set()}
    occupied_questions = set()
    exclusion_counts = {}
    for name, source, expected in EXCLUSION_SETS:
        rows = exclusion_sets[name]
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError("B11_EXCLUSION_COUNT_" + name)
        for row in rows:
            _exclusion(row)
            key = (source, row["source_id"])
            if key in occupied_ids[source] or row["question_sha256"] in occupied_questions:
                raise ValueError("B11_PRIOR_EXCLUSION_OVERLAP")
            occupied_ids[source].add(key)
            occupied_questions.add(row["question_sha256"])
        exclusion_counts[name] = len(rows)

    pools = {"arc_train": arc_metadata, "ob_train": ob_metadata}
    buckets = {}
    available_before = {}
    seen_questions = set()
    for source, rows in pools.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError("B11_MISSING_SOURCE_" + source)
        short, long = [], []
        source_ids = set()
        for entry in rows:
            row = _candidate(entry)
            key = (source, row["source_id"])
            if key in source_ids or row["question_sha256"] in seen_questions:
                raise ValueError("B11_SOURCE_DUPLICATE_OR_CROSS_SOURCE_QUESTION")
            source_ids.add(key)
            seen_questions.add(row["question_sha256"])
            if key in occupied_ids[source] or row["question_sha256"] in occupied_questions:
                continue
            stratum = "short" if row["prefix_len"] < 85 else "long"
            select_key = hashlib.sha256((
                SEED + "/" + source + "/" + row["source_id"] + "/" +
                row["question_sha256"]
            ).encode("utf-8")).hexdigest()
            item = {"corpus": source,
                    "sample_id": ("arc:" if source == "arc_train" else "ob:") +
                    row["question_sha256"][:32],
                    "source_id": row["source_id"],
                    "question_sha256": row["question_sha256"],
                    "full_prompt_sha256": row["full_prompt_sha256"],
                    "prefix_ids_sha256": row["prefix_ids_sha256"],
                    "prefix_len": row["prefix_len"],
                    "selection_key": select_key}
            (short if stratum == "short" else long).append(item)
        short.sort(key=lambda x: x["selection_key"])
        long.sort(key=lambda x: x["selection_key"])
        buckets[source] = {"short": short, "long": long}
        available_before[source] = {"short": len(short), "long": len(long)}

    cursor = {"arc_train": {"short": 0, "long": 0},
              "ob_train": {"short": 0, "long": 0}}
    output = {}
    for name, source, per_stratum in FREEZE_SPLITS:
        grouped = []
        for stratum in ("short", "long"):
            start = cursor[source][stratum]
            stop = start + per_stratum
            if stop > len(buckets[source][stratum]):
                raise ValueError("B11_INSUFFICIENT_FRESH_" + source + "_" + stratum)
            grouped.append(buckets[source][stratum][start:stop])
            cursor[source][stratum] = stop
        output[name] = [item for pair in zip(*grouped) for item in pair]

    combined = [x for rows in output.values() for x in rows]
    if len(combined) != 400 or len({x["sample_id"] for x in combined}) != 400 or (
        len({x["question_sha256"] for x in combined}) != 400
    ):
        raise ValueError("B11_SELECTION_FINAL_NOT_UNIQUE")
    public = {
        "schema": SCHEMA, "state": "PROPOSED_BEFORE_NEW_ORIGINAL_C_SCORING",
        "selection_seed": SEED,
        "source_arrow_sha256": SOURCE_SHA, "tokenizer_sha256": TOKENIZER_SHA,
        "qv20_calibration_manifest_sha256": QV20_CAL_SHA,
        "qv21_preexec_freeze_sha256": QV21_FREEZE_SHA,
        "qv21_private_split_sha256": QV21_PRIVATE_SHA,
        "exclusion_counts": exclusion_counts,
        "fresh_eligible_by_stratum": available_before,
        "split_counts": {key: len(value) for key, value in output.items()},
        "source_deduplication": True,
        "label_fields_accepted": False,
        "native_score_read_or_executed": False,
    }
    canonical = {"public": public, "selected": output}
    canonical["manifest_sha256"] = digest(canonical)
    return canonical
