#!/usr/bin/env python3
"""JE01 actual checkpoint tokenizer native 32-position feasibility (READ ONLY).

Replays frozen HF public row selection; rejects if exact 512-row SHA changes.
Reports length statistics only, not source text, not answers, not model quality.
Tokenizer is the original DeepSeek HF fast-tokenizers JSON; engine BOS and
runtime-tokenizer parity remain separate gates.
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

EXPECTED_MANIFEST = "a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"
EXPECTED_TOKENIZER = "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
SCHEMA = "beglin-je01-native-tokenizer-context-feasibility-v1"


def source_module(path):
    spec = importlib.util.spec_from_file_location("je01_data_freeze", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_for(df, cohort, row):
    if cohort in ("calibration", "same_source_holdout"):
        full = df.normal(row["text"])[:900]
        boundary = full.rfind(" ", 0,
                             min(len(full)-32, int(len(full)*.70)))
        if boundary < 64:
            raise ValueError("JE01_WIKITEXT_PROMPT_UNEXPECTED")
        return full[:boundary]
    if cohort == "ood_holdout":
        return df.normal(row["ctx"])
    raise ValueError("JE01_UNKNOWN_COHORT")


def run(source_path, tokenizer_path):
    import tokenizers
    df = source_module(source_path)
    source_sha = hashlib.sha256(pathlib.Path(source_path).read_bytes()).hexdigest()
    tok_sha = hashlib.sha256(pathlib.Path(tokenizer_path).read_bytes()).hexdigest()
    if tok_sha != EXPECTED_TOKENIZER:
        raise ValueError("JE01_ORIGINAL_TOKENIZER_SHA_DRIFT")
    plans = [(s, off) for s in df.SOURCES for off in df.planned_offsets(s)]
    grouped = {s[0]: [] for s in df.SOURCES}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(df.request_rows, s, off): s[0] for s, off in plans}
        for fut in as_completed(futures):
            grouped[futures[fut]].append(fut.result())
    sealed = df.freeze(grouped)
    if sealed["manifest_sha256"] != EXPECTED_MANIFEST:
        raise ValueError("JE01_FROZEN_512_ROWS_DRIFT")
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    result = {"schema": SCHEMA, "dataset_manifest_sha256": EXPECTED_MANIFEST,
              "data_freeze_source_sha256": source_sha,
              "tokenizer_sha256": tok_sha,
              "native_engine_max_positions": 32,
              "engine_bos_and_special_token_parity": "NOT_VERIFIED",
              "full_original_model_inference": "NOT_RUN",
              "gold_labels_used": False, "production_promotion_allowed": False,
              "cohorts": {}}
    for cohort in ("calibration", "same_source_holdout", "ood_holdout"):
        indexed = {}
        for pg in grouped[cohort]:
            for sample in pg["row_records"]:
                key = sample["row_idx"]
                if key in indexed:
                    if indexed[key]["row"] != sample["row"]:
                        raise ValueError("JE01_DUPLICATE_ROW_CONTENT_DRIFT")
                else:
                    indexed[key] = sample
        lengths, ids = [], []
        for rec in sealed["cohorts"][cohort]["records"]:
            idx = rec["row_idx"]
            orig = indexed[idx]["row"]
            raw = prompt_for(df, cohort, orig)
            if df.digest(raw) != rec["prompt_sha256"]:
                raise ValueError("JE01_SELECTED_PROMPT_CONTENT_DRIFT")
            token_ids = tokenizer.encode(raw, add_special_tokens=False).ids
            ids.append(token_ids)
            lengths.append(len(token_ids))
        if len(lengths) != sealed["cohorts"][cohort]["count"]:
            raise ValueError("JE01_COHORT_COUNT_DRIFT")
        ordered = sorted(lengths)
        one_target = [x for x in ordered if x+2 <= 32]
        result["cohorts"][cohort] = {
            "count": len(lengths), "min_raw_tokens": ordered[0],
            "median_raw_tokens": statistics.median(ordered),
            "p95_raw_tokens": ordered[int((len(ordered)-1)*.95)],
            "max_raw_tokens": ordered[-1],
            "raw_plus_one_bos_plus_one_gold_fits_32": len(one_target),
            "cannot_fit_one_gold_with_bos": len(lengths)-len(one_target),
            "token_ids_sha256": df.digest(ids),
            "token_lengths_sha256": df.digest(lengths),
        }
    result["receipt_sha256"] = df.digest(result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--tokenizer", required=True)
    a = p.parse_args()
    try:
        print(json.dumps(run(a.source, a.tokenizer), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "BLOCKED",
                          "error": str(exc)[:255], "native_model_forward": False}))
        raise SystemExit(2)
