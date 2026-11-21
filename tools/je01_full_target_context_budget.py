#!/usr/bin/env python3
"""JE01 pinned 512-source full-target context demand; no answers/weights/forward.

Read all original selected rows from immutable dataset revision, check original
frozen source row and prompt SHA and tokenizer SHA, estimate full gold-token
length for WikiText and ALL HellaSwag candidate endings, not label-chosen one.
Returns bounded aggregate lengths only. Native BOS parity still unverified.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import importlib.util

SCHEMA = "beglin-je01-512-full-target-context-v1"
EXPECTED_MANIFEST = "a9397bfabdd012f60d3dd3d33bcfab455ac1046eced30373d553217dd9409147"
EXPECTED_TOKENIZER = "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
CONTEXTS = (32, 160, 256, 320, 384, 512, 1024)


def _module(path):
    spec = importlib.util.spec_from_file_location("je01_freeze_context_budget", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _clean_target(df, cohort, item, expected_prompt_sha):
    source = item["row"]
    if cohort == "ood_holdout":
        prompt = df.normal(source["ctx"])
        endings = source.get("endings")
        if not isinstance(endings, list) or len(endings) != 4:
            raise ValueError("JE01_HELLASWAG_CHOICES_DRIFT")
        candidates = [prompt + " " + df.normal(end) for end in endings]
    else:
        full = df.normal(source["text"])[:900]
        boundary = full.rfind(" ", 0, min(len(full)-32, int(len(full)*.70)))
        if boundary < 64:
            raise ValueError("JE01_WIKITEXT_BOUNDARY_DRIFT")
        prompt = full[:boundary]
        candidates = [full]
    if df.digest(prompt) != expected_prompt_sha:
        raise ValueError("JE01_FROZEN_PROMPT_SHA_DRIFT")
    return prompt, candidates


def run(freeze_path, tokenizer_path):
    import tokenizers
    fp = Path(freeze_path)
    tp = Path(tokenizer_path)
    module = _module(fp)
    if hashlib.sha256(tp.read_bytes()).hexdigest() != EXPECTED_TOKENIZER:
        raise ValueError("JE01_TOKENIZER_SHA_DRIFT")
    tokenizer = tokenizers.Tokenizer.from_file(str(tp))
    work = [(src, i) for src in module.SOURCES
            for i in module.planned_offsets(src)]
    grouped = {src[0]: [] for src in module.SOURCES}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(module.request_rows, src, i): src[0] for src, i in work}
        for task in as_completed(futures):
            grouped[futures[task]].append(task.result())
    frozen = module.freeze(grouped)
    if frozen["manifest_sha256"] != EXPECTED_MANIFEST:
        raise ValueError("JE01_512_MANIFEST_DRIFT")
    summary = {
        "schema": SCHEMA, "input_row_manifest_sha256": EXPECTED_MANIFEST,
        "tokenizer_sha256": EXPECTED_TOKENIZER, "context_caps_tested": list(CONTEXTS),
        "native_full_model_forward": "NOT_RUN", "native_tokenizer_parity": "NOT_VERIFIED",
        "labels_read_for_selection": False, "promotion_allowed": False,
        "cohorts": {},
    }
    for cohort in ("calibration", "same_source_holdout", "ood_holdout"):
        lookup = {}
        for page in grouped[cohort]:
            for record in page["row_records"]:
                lookup.setdefault(record["row_idx"], record)
        lengths, prompt_lens, target_lens = [], [], []
        for record in frozen["cohorts"][cohort]["records"]:
            prompt, candidates = _clean_target(
                module, cohort, lookup[record["row_idx"]], record["prompt_sha256"]
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            per_candidate = [
                tokenizer.encode(full, add_special_tokens=False).ids
                for full in candidates
            ]
            # Full gold target or all four alternatives; no correct-label access.
            length = max(len(z) for z in per_candidate)
            lengths.append(length + 1)  # reserve 1 BOS per full target
            prompt_lens.append(len(prompt_ids) + 1)
            target_lens.append(length - len(prompt_ids))
        ordered = sorted(lengths)
        summary["cohorts"][cohort] = {
            "count": len(lengths),
            "max_bos_plus_complete_gold_or_all_4_endings": max(lengths),
            "min_bos_plus_full": min(lengths),
            "median_bos_plus_full": (ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2]) / 2,
            "p95_bos_plus_full": ordered[int(0.95*(len(ordered)-1))],
            "max_bos_plus_prompt_only": max(prompt_lens),
            "max_additional_candidate_tokens_approx": max(target_lens),
            "fits_context": {str(c): sum(n <= c for n in lengths) for c in CONTEXTS},
            "complete_sequence_lengths_sha256": module.digest(lengths),
            "prompt_lengths_sha256": module.digest(prompt_lens),
        }
    if [summary["cohorts"][x]["count"] for x in (
            "calibration", "same_source_holdout", "ood_holdout"
        )] != [256, 128, 128]:
        raise ValueError("JE01_COHORT_COUNT_DRIFT")
    summary["receipt_sha256"] = module.digest(summary)
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: script freeze_py_path tokenizer_json_path")
    try:
        print(json.dumps(run(sys.argv[1], sys.argv[2]), sort_keys=True, separators=(",",":")))
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "BLOCKED",
                          "error": str(exc)[:220], "model_forward": "NOT_RUN"}))
        raise SystemExit(2)
