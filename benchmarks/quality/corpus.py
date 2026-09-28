from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact import QualityArtifactError, sha256_json

CORPUS_SCHEMA = "beglin-quality-corpus/1"
TASK_TYPES = {"none", "exact_match", "contains_all", "numeric_tolerance"}


def _normalize_text(value: str, mode: str) -> str:
    if mode == "strip":
        return value.strip()
    if mode == "strip_casefold":
        return value.strip().casefold()
    if mode == "exact":
        return value
    raise QualityArtifactError(f"unsupported normalization: {mode}")


def render_prompt(entry: dict[str, Any]) -> str:
    prompt = entry.get("prompt")
    builder = entry.get("context_builder")
    if builder is None:
        if not isinstance(prompt, str):
            raise QualityArtifactError(f"{entry.get('id')}: prompt must be a string")
        return prompt
    if prompt is not None:
        raise QualityArtifactError(f"{entry.get('id')}: use prompt or context_builder, not both")
    if not isinstance(builder, dict):
        raise QualityArtifactError(f"{entry.get('id')}: context_builder must be an object")
    kind = builder.get("kind")
    unit = builder.get("unit")
    repeat = builder.get("repeat")
    suffix = builder.get("suffix")
    if not isinstance(unit, str) or not unit:
        raise QualityArtifactError(f"{entry.get('id')}: context_builder.unit invalid")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise QualityArtifactError(f"{entry.get('id')}: context_builder.repeat invalid")
    if not isinstance(suffix, str):
        raise QualityArtifactError(f"{entry.get('id')}: context_builder.suffix invalid")
    if kind == "repeat_prefix":
        prefix = ""
    elif kind == "sandwich_repeat":
        prefix = builder.get("prefix")
        if not isinstance(prefix, str):
            raise QualityArtifactError(f"{entry.get('id')}: context_builder.prefix invalid")
    else:
        raise QualityArtifactError(f"{entry.get('id')}: unsupported context_builder kind")
    return prefix + ((unit + " ") * repeat) + suffix


def validate_task(entry_id: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise QualityArtifactError(f"{entry_id}: task must be an object")
    task_type = raw.get("type")
    if task_type not in TASK_TYPES:
        raise QualityArtifactError(f"{entry_id}: unsupported task type {task_type!r}")
    normalization = raw.get("normalization", "strip")
    if normalization not in {"strip", "strip_casefold", "exact"}:
        raise QualityArtifactError(f"{entry_id}: invalid normalization")
    out = {"type": task_type, "normalization": normalization}
    if task_type == "exact_match":
        expected = raw.get("expected")
        if not isinstance(expected, str):
            raise QualityArtifactError(f"{entry_id}: exact_match expected must be string")
        out["expected"] = expected
    elif task_type == "contains_all":
        values = raw.get("expected_substrings")
        if not isinstance(values, list) or not values or not all(isinstance(x, str) and x for x in values):
            raise QualityArtifactError(f"{entry_id}: contains_all expected_substrings invalid")
        out["expected_substrings"] = values
    elif task_type == "numeric_tolerance":
        expected = raw.get("expected")
        tolerance = raw.get("tolerance")
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            raise QualityArtifactError(f"{entry_id}: numeric expected invalid")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise QualityArtifactError(f"{entry_id}: numeric tolerance invalid")
        out["expected"] = float(expected)
        out["tolerance"] = float(tolerance)
    return out


def validate_corpus(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != CORPUS_SCHEMA:
        raise QualityArtifactError(f"corpus.schema must be {CORPUS_SCHEMA}")
    corpus_id = raw.get("corpus_id")
    if not isinstance(corpus_id, str) or not corpus_id:
        raise QualityArtifactError("corpus_id must be non-empty")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise QualityArtifactError("corpus.entries must be a non-empty array")

    normalized = []
    seen = set()
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise QualityArtifactError(f"entries[{index}] must be an object")
        entry_id = item.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise QualityArtifactError(f"entries[{index}].id invalid")
        if entry_id in seen:
            raise QualityArtifactError(f"duplicate corpus id: {entry_id}")
        seen.add(entry_id)
        tags = item.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(x, str) and x for x in tags):
            raise QualityArtifactError(f"{entry_id}: tags invalid")
        min_context_tokens = item.get("min_context_tokens")
        if min_context_tokens is not None and (
            isinstance(min_context_tokens, bool)
            or not isinstance(min_context_tokens, int)
            or min_context_tokens < 1
        ):
            raise QualityArtifactError(f"{entry_id}: min_context_tokens invalid")
        use_for_perplexity = item.get("use_for_perplexity", False)
        if not isinstance(use_for_perplexity, bool):
            raise QualityArtifactError(f"{entry_id}: use_for_perplexity invalid")
        rendered = render_prompt(item)
        normalized.append({
            "id": entry_id,
            "prompt": rendered,
            "prompt_sha256": sha256_json({"prompt": rendered}),
            "tags": sorted(set(tags)),
            "min_context_tokens": min_context_tokens,
            "use_for_perplexity": use_for_perplexity,
            "task": validate_task(entry_id, item.get("task", {"type": "none"})),
        })

    canonical = {"schema": CORPUS_SCHEMA, "corpus_id": corpus_id, "entries": normalized}
    canonical["corpus_sha256"] = sha256_json(canonical)
    return canonical


def load_corpus(path: Path) -> dict[str, Any]:
    return validate_corpus(json.loads(path.read_text(encoding="utf-8")))


def score_task(task: dict[str, Any], output_text: str | None) -> float | None:
    task_type = task["type"]
    if task_type == "none":
        return None
    if output_text is None:
        return None
    mode = task.get("normalization", "strip")
    if task_type == "exact_match":
        return float(_normalize_text(output_text, mode) == _normalize_text(task["expected"], mode))
    if task_type == "contains_all":
        haystack = _normalize_text(output_text, mode)
        return float(all(_normalize_text(value, mode) in haystack for value in task["expected_substrings"]))
    if task_type == "numeric_tolerance":
        try:
            observed = float(output_text.strip())
        except ValueError:
            return 0.0
        return float(abs(observed - task["expected"]) <= task["tolerance"])
    raise QualityArtifactError(f"unsupported task type: {task_type}")
