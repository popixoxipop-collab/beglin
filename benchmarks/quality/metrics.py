from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from .artifact import QualityArtifactError, compare_identity
from .corpus import score_task


def _request_map(run: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (item["prompt_id"], item["repetition"]): item
        for item in run["requests"]
    }


def _canonical_output(item: dict[str, Any]) -> tuple[str, Any] | None:
    if item["output_token_ids"] is not None:
        return ("tokens", tuple(item["output_token_ids"]))
    if item["output_text"] is not None:
        return ("text", item["output_text"])
    return None


def _all_or_null(values: list[Any]) -> list[Any] | None:
    return values if values and all(value is not None for value in values) else None


def perplexity(run: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    required = {
        item["id"]
        for item in corpus["entries"]
        if item["use_for_perplexity"]
    }
    if not required:
        return {"value": None, "nll_sum": None, "token_count": None, "coverage": 0.0}

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in run["requests"]:
        by_prompt[request["prompt_id"]].append(request)

    total_nll = 0.0
    total_tokens = 0
    covered = set()
    for prompt_id in sorted(required):
        rows = by_prompt.get(prompt_id, [])
        if not rows:
            continue
        row = min(rows, key=lambda item: item["repetition"])
        if row["nll_sum"] is not None:
            total_nll += row["nll_sum"]
            total_tokens += row["nll_token_count"]
            covered.add(prompt_id)
        elif row["token_logprobs"] is not None and row["token_logprobs"]:
            total_nll += -sum(row["token_logprobs"])
            total_tokens += len(row["token_logprobs"])
            covered.add(prompt_id)

    coverage = len(covered) / len(required)
    if coverage != 1.0 or total_tokens == 0:
        return {
            "value": None,
            "nll_sum": total_nll if total_tokens else None,
            "token_count": total_tokens or None,
            "coverage": coverage,
        }
    mean_nll = total_nll / total_tokens
    if mean_nll > 700:
        raise QualityArtifactError("perplexity overflow; input NLL is not credible")
    return {
        "value": math.exp(mean_nll),
        "nll_sum": total_nll,
        "token_count": total_tokens,
        "coverage": coverage,
    }


def finite_logits(run: dict[str, Any]) -> dict[str, Any]:
    values = _all_or_null([item["finite_logits"] for item in run["requests"]])
    if values is None:
        known = sum(item["finite_logits"] is not None for item in run["requests"])
        return {
            "rate": None,
            "known_requests": known,
            "total_requests": len(run["requests"]),
        }
    return {
        "rate": sum(bool(value) for value in values) / len(values),
        "known_requests": len(values),
        "total_requests": len(values),
    }


def count_metric(run: dict[str, Any], field: str) -> int | None:
    values = _all_or_null([item[field] for item in run["requests"]])
    if values is None:
        return None
    return int(sum(values))


def quality_score(run: dict[str, Any]) -> float | None:
    values = [item["quality_score"] for item in run["requests"]]
    known = [value for value in values if value is not None]
    if not known or len(known) != len(values):
        return None
    return float(statistics.fmean(known))


def task_score(run: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    tasks = {
        item["id"]: item["task"]
        for item in corpus["entries"]
        if item["task"]["type"] != "none"
    }
    if not tasks:
        return {"value": None, "scored_prompts": 0, "expected_prompts": 0}

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in run["requests"]:
        by_prompt[request["prompt_id"]].append(request)

    values = []
    for prompt_id, task in sorted(tasks.items()):
        rows = by_prompt.get(prompt_id, [])
        if not rows:
            return {"value": None, "scored_prompts": len(values), "expected_prompts": len(tasks)}
        row = min(rows, key=lambda item: item["repetition"])
        score = score_task(task, row["output_text"])
        if score is None:
            return {"value": None, "scored_prompts": len(values), "expected_prompts": len(tasks)}
        values.append(score)
    return {
        "value": float(statistics.fmean(values)),
        "scored_prompts": len(values),
        "expected_prompts": len(tasks),
    }


def within_run_determinism(run: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in run["requests"]:
        groups[request["prompt_id"]].append(request)

    comparable = 0
    matches = 0
    incomplete = []
    for prompt_id, rows in sorted(groups.items()):
        outputs = [_canonical_output(row) for row in sorted(rows, key=lambda x: x["repetition"])]
        if len(outputs) < 2 or any(value is None for value in outputs):
            incomplete.append(prompt_id)
            continue
        comparable += 1
        if all(value == outputs[0] for value in outputs[1:]):
            matches += 1
    return {
        "rate": (matches / comparable) if comparable else None,
        "comparable_prompts": comparable,
        "matching_prompts": matches,
        "incomplete_prompts": incomplete,
    }


def reference_match(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = _request_map(reference)
    right = _request_map(candidate)
    keys = sorted(set(left) & set(right))
    if not keys:
        return {"rate": None, "comparable_requests": 0, "matching_requests": 0}
    comparable = 0
    matches = 0
    for key in keys:
        left_output = _canonical_output(left[key])
        right_output = _canonical_output(right[key])
        if left_output is None or right_output is None:
            continue
        comparable += 1
        if left_output == right_output:
            matches += 1
    return {
        "rate": (matches / comparable) if comparable else None,
        "comparable_requests": comparable,
        "matching_requests": matches,
    }


def _activation_aggregate(run: dict[str, Any]) -> dict[tuple[str, int, int | None], dict[str, float | None]]:
    buckets: dict[tuple[str, int, int | None], dict[str, list[float]]] = {}
    for request in run["requests"]:
        for item in request["activation_summaries"]:
            key = (item["role"], item["layer"], item["expert_id"])
            bucket = buckets.setdefault(key, {"mean_abs": [], "rms": []})
            if item["mean_abs"] is not None:
                bucket["mean_abs"].append(item["mean_abs"])
            if item["rms"] is not None:
                bucket["rms"].append(item["rms"])
    out = {}
    for key, values in buckets.items():
        out[key] = {
            "mean_abs": statistics.fmean(values["mean_abs"]) if values["mean_abs"] else None,
            "rms": statistics.fmean(values["rms"]) if values["rms"] else None,
        }
    return out


def activation_drift(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = _activation_aggregate(reference)
    right = _activation_aggregate(candidate)
    rows = []
    for key in sorted(set(left) & set(right)):
        l = left[key]
        r = right[key]
        mean_delta = None
        rms_delta = None
        if l["mean_abs"] is not None and r["mean_abs"] is not None:
            denom = max(abs(l["mean_abs"]), 1e-12)
            mean_delta = (r["mean_abs"] - l["mean_abs"]) / denom
        if l["rms"] is not None and r["rms"] is not None:
            denom = max(abs(l["rms"]), 1e-12)
            rms_delta = (r["rms"] - l["rms"]) / denom
        if mean_delta is None and rms_delta is None:
            continue
        rows.append({
            "role": key[0],
            "layer": key[1],
            "expert_id": key[2],
            "mean_abs_relative_delta": mean_delta,
            "rms_relative_delta": rms_delta,
        })
    mean_values = [
        abs(row["mean_abs_relative_delta"])
        for row in rows
        if row["mean_abs_relative_delta"] is not None
    ]
    rms_values = [
        abs(row["rms_relative_delta"])
        for row in rows
        if row["rms_relative_delta"] is not None
    ]
    return {
        "cells": rows,
        "cell_count": len(rows),
        "max_abs_mean_abs_relative_delta": max(mean_values) if mean_values else None,
        "max_abs_rms_relative_delta": max(rms_values) if rms_values else None,
    }


def router_near_tie(run: dict[str, Any]) -> dict[str, Any]:
    total_near = 0
    total_decisions = 0
    explicit_rates = []
    observed = 0
    for request in run["requests"]:
        router = request["router"]
        if router is None:
            continue
        observed += 1
        if router["near_tie_count"] is not None and router["decision_count"] is not None:
            total_near += router["near_tie_count"]
            total_decisions += router["decision_count"]
        elif router["near_tie_rate"] is not None:
            explicit_rates.append(router["near_tie_rate"])
    if total_decisions:
        rate = total_near / total_decisions
    elif explicit_rates and observed == len(run["requests"]):
        rate = statistics.fmean(explicit_rates)
    else:
        rate = None
    return {
        "rate": rate,
        "observed_requests": observed,
        "total_requests": len(run["requests"]),
        "near_tie_count": total_near if total_decisions else None,
        "decision_count": total_decisions or None,
    }


def long_context_stability(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    long_entries = {
        entry["id"]: entry
        for entry in corpus["entries"]
        if "long_context" in entry["tags"]
    }
    if not long_entries:
        return {
            "rate": None,
            "expected_prompts": 0,
            "covered_prompts": 0,
            "stable_prompts": 0,
        }

    left_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    right_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reference["requests"]:
        left_by_prompt[item["prompt_id"]].append(item)
    for item in candidate["requests"]:
        right_by_prompt[item["prompt_id"]].append(item)

    covered = 0
    stable = 0
    for prompt_id, entry in sorted(long_entries.items()):
        left_rows = left_by_prompt.get(prompt_id, [])
        right_rows = right_by_prompt.get(prompt_id, [])
        if not left_rows or not right_rows:
            continue
        left = min(left_rows, key=lambda item: item["repetition"])
        right = min(right_rows, key=lambda item: item["repetition"])
        min_tokens = entry["min_context_tokens"]
        if min_tokens is not None and right["context_tokens"] < min_tokens:
            continue
        left_output = _canonical_output(left)
        right_output = _canonical_output(right)
        if left_output is None or right_output is None or right["finite_logits"] is None:
            continue
        covered += 1
        if right["finite_logits"] and left_output == right_output:
            stable += 1

    return {
        "rate": (stable / len(long_entries)) if covered == len(long_entries) else None,
        "expected_prompts": len(long_entries),
        "covered_prompts": covered,
        "stable_prompts": stable,
    }


def evaluate_pair(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    identity = compare_identity(reference, candidate)
    if not identity["compatible"]:
        return {
            "status": "CONTROL_MISMATCH",
            "identity": identity,
            "reference_run_id": reference["run_id"],
            "candidate_run_id": candidate["run_id"],
            "candidate_n": candidate["variant"]["n"],
            "metrics": None,
        }

    reference_ppl = perplexity(reference, corpus)
    candidate_ppl = perplexity(candidate, corpus)
    ppl_ratio = None
    if reference_ppl["value"] is not None and candidate_ppl["value"] is not None:
        ppl_ratio = candidate_ppl["value"] / reference_ppl["value"]

    ref_task = task_score(reference, corpus)
    cand_task = task_score(candidate, corpus)
    task_delta = None
    if ref_task["value"] is not None and cand_task["value"] is not None:
        task_delta = cand_task["value"] - ref_task["value"]

    ref_quality = quality_score(reference)
    cand_quality = quality_score(candidate)
    quality_delta = None
    if ref_quality is not None and cand_quality is not None:
        quality_delta = cand_quality - ref_quality

    metrics = {
        "reference_perplexity": reference_ppl,
        "candidate_perplexity": candidate_ppl,
        "perplexity_ratio": ppl_ratio,
        "reference_quality_score": ref_quality,
        "candidate_quality_score": cand_quality,
        "quality_score_delta": quality_delta,
        "reference_task_score": ref_task,
        "candidate_task_score": cand_task,
        "task_score_delta": task_delta,
        "finite_logits": finite_logits(candidate),
        "reference_determinism": within_run_determinism(reference),
        "candidate_determinism": within_run_determinism(candidate),
        "reference_output_match": reference_match(reference, candidate),
        "correction_count": count_metric(candidate, "corrections"),
        "promotion_count": count_metric(candidate, "promotions"),
        "activation_drift": activation_drift(reference, candidate),
        "reference_router_near_tie": router_near_tie(reference),
        "candidate_router_near_tie": router_near_tie(candidate),
        "long_context_stability": long_context_stability(reference, candidate, corpus),
    }
    return {
        "status": "MEASURED",
        "identity": identity,
        "reference_run_id": reference["run_id"],
        "candidate_run_id": candidate["run_id"],
        "candidate_n": candidate["variant"]["n"],
        "metrics": metrics,
    }


def required_metric_paths() -> list[str]:
    return [
        "perplexity_ratio",
        "candidate_quality_score",
        "quality_score_delta",
        "candidate_task_score.value",
        "task_score_delta",
        "finite_logits.rate",
        "candidate_determinism.rate",
        "reference_output_match.rate",
        "correction_count",
        "promotion_count",
        "activation_drift.max_abs_rms_relative_delta",
        "candidate_router_near_tie.rate",
        "long_context_stability.rate",
    ]


def _read_path(root: dict[str, Any], path: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def metric_completeness(pair: dict[str, Any]) -> dict[str, Any]:
    if pair["status"] != "MEASURED":
        return {
            "complete": False,
            "missing_required_metrics": required_metric_paths(),
        }
    metrics = pair["metrics"]
    missing = [path for path in required_metric_paths() if _read_path(metrics, path) is None]
    return {
        "complete": not missing,
        "missing_required_metrics": missing,
    }
