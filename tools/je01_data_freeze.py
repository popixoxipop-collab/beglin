#!/usr/bin/env python3
"""JE01 real WikiText103/HellaSwag row-slice preregistration; no model inference.

Inputs: public Hugging Face dataset-viewer rows API only. Fails closed if the
server returns a different x-revision, a partial page or truncated cells.
Selections are from prompts/available choices only, NEVER correct labels.
Emits content hashes and source row numbers, not copyrighted raw text.
This is a row-slice identity freeze, NOT a full-source Parquet SHA attestation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import urllib.parse
import urllib.request

SCHEMA = "beglin-je01-preregistered-rows-v1"
SEED = "BEGLIN-JE01-2026-11-17-INDEPENDENT-SOURCE-ROWS-V1"
API = "https://datasets-server.huggingface.co/rows"
SOURCES = (
    ("calibration", "Salesforce/wikitext", "wikitext-103-raw-v1", "train",
     1801350, "b08601e04326c79dfdd32d625aee71d232d685c3", 20, 256),
    ("same_source_holdout", "Salesforce/wikitext", "wikitext-103-raw-v1", "validation",
     3760, "b08601e04326c79dfdd32d625aee71d232d685c3", 10, 128),
    ("ood_holdout", "Rowan/hellaswag", "default", "validation",
     10042, "218ec52e09a7e7462a5400043bb9a69a41d06b76", 4, 128),
)


def digest(value):
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normal(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def planned_offsets(source):
    name, dataset, config, split, n, sha, blocks, target = source
    offsets = []
    for b in range(blocks):
        lo = (b * (n - 100)) // blocks
        hi = ((b + 1) * (n - 100)) // blocks
        delta = int(digest([SEED, dataset, config, split, b]), 16)
        offsets.append(lo + delta % max(1, hi - lo))
    if len(set(offsets)) != len(offsets):
        raise ValueError("JE01_OFFSET_DUPLICATE")
    return offsets


def request_rows(source, offset, opener=urllib.request.urlopen):
    _, dataset, config, split, expected_n, sha, _, _ = source
    query = urllib.parse.urlencode({
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": 100,
    })
    url = API + "?" + query
    req = urllib.request.Request(url, headers={
        "User-Agent": "beglin-JE01-research-prereg-v1"})
    with opener(req, timeout=20) as response:
        revision = response.headers.get("x-revision")
        raw = response.read()
    if revision != sha:
        raise ValueError("JE01_DATASET_REVISION_DRIFT:" + repr(revision))
    if len(raw) > 2_500_000:
        raise ValueError("JE01_DATASET_PAGE_OVERSIZE")
    page = json.loads(raw)
    if (page.get("partial") is not False or
        page.get("num_rows_total") != expected_n or
        not isinstance(page.get("rows"), list) or
        len(page["rows"]) != 100):
        raise ValueError("JE01_ROW_PAGE_PARTIAL_OR_COUNT_DRIFT")
    ids = [x.get("row_idx") for x in page["rows"]]
    if ids != list(range(offset, offset + 100)):
        raise ValueError("JE01_ROW_INDEX_DRIFT")
    return {"offset": offset,
            "page_sha256": hashlib.sha256(raw).hexdigest(),
            "row_records": page["rows"]}


def candidates(source, pages):
    name, dataset, config, split, n, sha, blocks, target = source
    result = []
    for page in pages:
        for item in page["row_records"]:
            if item.get("truncated_cells"):
                continue
            row = item.get("row")
            if not isinstance(row, dict):
                continue
            idx = item["row_idx"]
            if dataset == "Salesforce/wikitext":
                full = normal(row.get("text", ""))[:900]
                if len(full) < 96 or full.startswith("="):
                    continue
                limit = min(len(full)-32, math.floor(len(full)*.70))
                boundary = full.rfind(" ", 0, limit)
                if boundary < 64:
                    continue
                prompt, target_text = full[:boundary], full[boundary:].strip()
                if len(target_text) < 24:
                    continue
                prompt_sha = digest(prompt)
                continuation_sha = digest(target_text)
                selection_basis = prompt_sha
                meta = {"target_sha256": continuation_sha}
            else:
                prompt = normal(row.get("ctx", ""))
                endings = row.get("endings")
                if len(prompt) < 32 or not isinstance(endings, list) or len(endings) != 4:
                    continue
                if any(len(normal(x)) < 3 for x in endings):
                    continue
                prompt_sha = digest(prompt)
                choices_sha = digest([normal(x) for x in endings])
                selection_basis = prompt_sha
                meta = {"choices_sha256": choices_sha}
                # Intentionally do not read row['label']: gold cannot choose a row.
            candidate_id = digest([dataset, sha, config, split, idx,
                                   selection_basis])
            result.append({
                "row_idx": idx, "source_row_id_sha256": candidate_id,
                "prompt_sha256": prompt_sha,
                "input_length_chars": len(prompt),
                **meta,
                "selection_rank": digest([SEED, dataset, split, idx, prompt_sha]),
            })
    return result


def freeze(pages_by_name):
    manifest = {
        "schema": SCHEMA, "seed_sha256": digest(SEED),
        "stage": "ROW_SLICE_PREREGISTRATION_ONLY",
        "full_dataset_file_sha_verified": False,
        "native_model_checkpoint_sha_verified": False,
        "native_model_forward_observed": False,
        "gold_labels_used_in_sampling": False,
        "cohorts": {}, "sources": {},
    }
    seen_prompt = set()
    for source in SOURCES:
        name, dataset, config, split, n, revision, blocks, target = source
        pages = pages_by_name[name]
        if sorted(x["offset"] for x in pages) != sorted(planned_offsets(source)):
            raise ValueError("JE01_PAGE_PLAN_TAMPER:" + name)
        all_candidates = candidates(source, pages)
        all_candidates.sort(key=lambda x: (x["selection_rank"], x["row_idx"]))
        selected = []
        for record in all_candidates:
            if record["prompt_sha256"] in seen_prompt:
                continue
            selected.append(record)
            seen_prompt.add(record["prompt_sha256"])
            if len(selected) == target:
                break
        if len(selected) != target:
            raise ValueError("JE01_INSUFFICIENT_QUALIFYING_PROMPTS:"
                             + name + ":" + str(len(selected)) + "/" + str(target))
        # Freeze rows in deterministic rank order, stripping rank helper.
        records = [{k: v for k, v in row.items() if k != "selection_rank"}
                   for row in selected]
        manifest["cohorts"][name] = {
            "count": len(records),
            "records": records,
            "cohort_sha256": digest(records),
        }
        manifest["sources"][name] = {
            "dataset": dataset, "config": config, "split": split,
            "revision": revision, "num_rows_total": n,
            "sample_windows": planned_offsets(source),
            "page_sha256_by_offset": {
                str(x["offset"]): x["page_sha256"] for x in sorted(
                    pages, key=lambda y: y["offset"]
                )},
            "eligible_candidates": len(all_candidates),
            "source_parquet_sha256": "NOT_MEASURED",
        }
    all_ids = [r["source_row_id_sha256"]
               for c in manifest["cohorts"].values() for r in c["records"]]
    if len(all_ids) != 512 or len(set(all_ids)) != len(all_ids):
        raise ValueError("JE01_FROZEN_SAMPLE_COUNT_OR_DUPLICATE")
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def execute(opener=urllib.request.urlopen):
    pairs = [(src, off) for src in SOURCES for off in planned_offsets(src)]
    pages_by_name = {src[0]: [] for src in SOURCES}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(request_rows, src, off, opener): src[0]
                   for src, off in pairs}
        for future in as_completed(futures):
            name = futures[future]
            pages_by_name[name].append(future.result())
    return freeze(pages_by_name)


if __name__ == "__main__":
    try:
        print(json.dumps(execute(), ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False))
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "stage": "BLOCKED",
                          "error": type(exc).__name__,
                          "reason": str(exc)[:256],
                          "native_inference_occurred": False}))
        raise SystemExit(2)
