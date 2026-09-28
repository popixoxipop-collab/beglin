from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "beglin-p9-independent-final-audit/1"
PHYSICAL_SCHEMA = "beglin-p9-physical-quality-run/1"
RUN_SCHEMA = "beglin-quality-run/1"
EVALUATION_SCHEMA = "beglin-quality-evaluation/1"
POLICY_SCHEMA = "beglin-quality-policy/1"
EXPECTED_VARIANTS = ("reference", "n4", "n5", "n6", "n7")
COMPARABLE_IDENTITY_FIELDS = (
    "source_commit",
    "source_tree",
    "binary_sha256",
    "checkpoint_sha256",
    "model_id",
    "hardware_id",
    "backend",
    "seed",
    "prompt_corpus_sha256",
    "runtime_config_hash",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be an object")
    return value


def hash_without(value: dict[str, Any], field: str) -> str:
    core = dict(value)
    core.pop(field, None)
    return sha256_json(core)


def read_metric(root: dict[str, Any], path: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def canonical_output(request: dict[str, Any]) -> tuple[str, Any] | None:
    tokens = request.get("output_token_ids")
    if tokens is not None:
        return ("tokens", tuple(tokens))
    text = request.get("output_text")
    if text is not None:
        return ("text", text)
    return None


def request_map(run: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (row["prompt_id"], row["repetition"]): row
        for row in run["requests"]
    }


def normalize_text(value: str, mode: str) -> str:
    if mode == "strip":
        return value.strip()
    if mode == "strip_casefold":
        return value.strip().casefold()
    if mode == "exact":
        return value
    raise ValueError(f"unsupported normalization: {mode}")


def score_task(task: dict[str, Any], output_text: str | None) -> float | None:
    task_type = task["type"]
    if task_type == "none":
        return None
    if output_text is None:
        return None
    mode = task.get("normalization", "strip")
    if task_type == "exact_match":
        return float(
            normalize_text(output_text, mode)
            == normalize_text(task["expected"], mode)
        )
    if task_type == "contains_all":
        haystack = normalize_text(output_text, mode)
        return float(
            all(
                normalize_text(value, mode) in haystack
                for value in task["expected_substrings"]
            )
        )
    if task_type == "numeric_tolerance":
        try:
            observed = float(output_text.strip())
        except ValueError:
            return 0.0
        return float(abs(observed - task["expected"]) <= task["tolerance"])
    raise ValueError(f"unsupported task type: {task_type}")


def normalized_corpus(raw: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for item in raw["entries"]:
        builder = item.get("context_builder")
        if builder is None:
            prompt = item["prompt"]
        else:
            prefix = builder.get("prefix", "")
            prompt = prefix + ((builder["unit"] + " ") * builder["repeat"]) + builder["suffix"]
        task = dict(item.get("task", {"type": "none"}))
        task.setdefault("normalization", "strip")
        if task["type"] == "numeric_tolerance":
            task["expected"] = float(task["expected"])
            task["tolerance"] = float(task["tolerance"])
        entries.append(
            {
                "id": item["id"],
                "prompt": prompt,
                "prompt_sha256": sha256_json({"prompt": prompt}),
                "tags": sorted(set(item.get("tags", []))),
                "min_context_tokens": item.get("min_context_tokens"),
                "use_for_perplexity": item.get("use_for_perplexity", False),
                "task": task,
            }
        )
    canonical = {
        "schema": raw["schema"],
        "corpus_id": raw["corpus_id"],
        "entries": entries,
    }
    canonical["corpus_sha256"] = sha256_json(canonical)
    return canonical


def perplexity(run: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    required = {
        row["id"] for row in corpus["entries"] if row["use_for_perplexity"]
    }
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
        elif row["token_logprobs"]:
            total_nll += -sum(row["token_logprobs"])
            total_tokens += len(row["token_logprobs"])
            covered.add(prompt_id)
    coverage = len(covered) / len(required)
    value = math.exp(total_nll / total_tokens) if coverage == 1.0 and total_tokens else None
    return {
        "value": value,
        "nll_sum": total_nll if total_tokens else None,
        "token_count": total_tokens or None,
        "coverage": coverage,
    }


def quality_score(run: dict[str, Any]) -> float | None:
    values = [row["quality_score"] for row in run["requests"]]
    if not values or any(value is None for value in values):
        return None
    return float(statistics.fmean(values))


def task_score(run: dict[str, Any], corpus: dict[str, Any]) -> dict[str, Any]:
    tasks = {
        row["id"]: row["task"]
        for row in corpus["entries"]
        if row["task"]["type"] != "none"
    }
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in run["requests"]:
        by_prompt[request["prompt_id"]].append(request)
    values = []
    for prompt_id, task in sorted(tasks.items()):
        rows = by_prompt.get(prompt_id, [])
        if not rows:
            return {
                "value": None,
                "scored_prompts": len(values),
                "expected_prompts": len(tasks),
            }
        row = min(rows, key=lambda item: item["repetition"])
        score = score_task(task, row["output_text"])
        if score is None:
            return {
                "value": None,
                "scored_prompts": len(values),
                "expected_prompts": len(tasks),
            }
        values.append(score)
    return {
        "value": float(statistics.fmean(values)),
        "scored_prompts": len(values),
        "expected_prompts": len(tasks),
    }


def finite_logits(run: dict[str, Any]) -> dict[str, Any]:
    values = [row["finite_logits"] for row in run["requests"]]
    known = [value for value in values if value is not None]
    rate = None if len(known) != len(values) else sum(bool(value) for value in known) / len(known)
    return {
        "rate": rate,
        "known_requests": len(known),
        "total_requests": len(values),
    }


def determinism(run: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in run["requests"]:
        groups[request["prompt_id"]].append(request)
    comparable = 0
    matches = 0
    incomplete = []
    for prompt_id, rows in sorted(groups.items()):
        outputs = [
            canonical_output(row)
            for row in sorted(rows, key=lambda item: item["repetition"])
        ]
        if len(outputs) < 2 or any(value is None for value in outputs):
            incomplete.append(prompt_id)
            continue
        comparable += 1
        if all(value == outputs[0] for value in outputs[1:]):
            matches += 1
    return {
        "rate": matches / comparable if comparable else None,
        "comparable_prompts": comparable,
        "matching_prompts": matches,
        "incomplete_prompts": incomplete,
    }


def reference_match(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = request_map(reference)
    right = request_map(candidate)
    comparable = 0
    matches = 0
    for key in sorted(set(left) & set(right)):
        left_output = canonical_output(left[key])
        right_output = canonical_output(right[key])
        if left_output is None or right_output is None:
            continue
        comparable += 1
        if left_output == right_output:
            matches += 1
    return {
        "rate": matches / comparable if comparable else None,
        "comparable_requests": comparable,
        "matching_requests": matches,
    }


def activation_drift(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    def aggregate(run: dict[str, Any]) -> dict[tuple[str, int, int | None], dict[str, float | None]]:
        buckets: dict[tuple[str, int, int | None], dict[str, list[float]]] = {}
        for request in run["requests"]:
            for item in request["activation_summaries"]:
                key = (item["role"], item["layer"], item["expert_id"])
                bucket = buckets.setdefault(key, {"mean_abs": [], "rms": []})
                if item["mean_abs"] is not None:
                    bucket["mean_abs"].append(item["mean_abs"])
                if item["rms"] is not None:
                    bucket["rms"].append(item["rms"])
        return {
            key: {
                "mean_abs": statistics.fmean(values["mean_abs"]) if values["mean_abs"] else None,
                "rms": statistics.fmean(values["rms"]) if values["rms"] else None,
            }
            for key, values in buckets.items()
        }

    left = aggregate(reference)
    right = aggregate(candidate)
    cells = []
    for key in sorted(set(left) & set(right)):
        left_cell = left[key]
        right_cell = right[key]
        mean_delta = None
        rms_delta = None
        if left_cell["mean_abs"] is not None and right_cell["mean_abs"] is not None:
            mean_delta = (right_cell["mean_abs"] - left_cell["mean_abs"]) / max(abs(left_cell["mean_abs"]), 1e-12)
        if left_cell["rms"] is not None and right_cell["rms"] is not None:
            rms_delta = (right_cell["rms"] - left_cell["rms"]) / max(abs(left_cell["rms"]), 1e-12)
        cells.append(
            {
                "role": key[0],
                "layer": key[1],
                "expert_id": key[2],
                "mean_abs_relative_delta": mean_delta,
                "rms_relative_delta": rms_delta,
            }
        )
    mean_values = [abs(row["mean_abs_relative_delta"]) for row in cells if row["mean_abs_relative_delta"] is not None]
    rms_values = [abs(row["rms_relative_delta"]) for row in cells if row["rms_relative_delta"] is not None]
    return {
        "cells": cells,
        "cell_count": len(cells),
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
    entries = {row["id"]: row for row in corpus["entries"] if "long_context" in row["tags"]}
    left: dict[str, list[dict[str, Any]]] = defaultdict(list)
    right: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference["requests"]:
        left[row["prompt_id"]].append(row)
    for row in candidate["requests"]:
        right[row["prompt_id"]].append(row)
    covered = 0
    stable = 0
    for prompt_id, entry in sorted(entries.items()):
        if not left[prompt_id] or not right[prompt_id]:
            continue
        left_row = min(left[prompt_id], key=lambda item: item["repetition"])
        right_row = min(right[prompt_id], key=lambda item: item["repetition"])
        if entry["min_context_tokens"] is not None and right_row["context_tokens"] < entry["min_context_tokens"]:
            continue
        left_output = canonical_output(left_row)
        right_output = canonical_output(right_row)
        if left_output is None or right_output is None or right_row["finite_logits"] is None:
            continue
        covered += 1
        if right_row["finite_logits"] and left_output == right_output:
            stable += 1
    return {
        "rate": stable / len(entries) if covered == len(entries) else None,
        "expected_prompts": len(entries),
        "covered_prompts": covered,
        "stable_prompts": stable,
    }


def recompute_metrics(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    reference_ppl = perplexity(reference, corpus)
    candidate_ppl = perplexity(candidate, corpus)
    reference_task = task_score(reference, corpus)
    candidate_task = task_score(candidate, corpus)
    reference_quality = quality_score(reference)
    candidate_quality = quality_score(candidate)
    reference_router = router_near_tie(reference)
    candidate_router = router_near_tie(candidate)
    return {
        "reference_perplexity": reference_ppl,
        "candidate_perplexity": candidate_ppl,
        "perplexity_ratio": candidate_ppl["value"] / reference_ppl["value"],
        "reference_quality_score": reference_quality,
        "candidate_quality_score": candidate_quality,
        "quality_score_delta": candidate_quality - reference_quality,
        "reference_task_score": reference_task,
        "candidate_task_score": candidate_task,
        "task_score_delta": candidate_task["value"] - reference_task["value"],
        "reference_finite_logits": finite_logits(reference),
        "candidate_finite_logits": finite_logits(candidate),
        "finite_logits": finite_logits(candidate),
        "reference_determinism": determinism(reference),
        "candidate_determinism": determinism(candidate),
        "reference_output_match": reference_match(reference, candidate),
        "correction_count": sum(row["corrections"] for row in candidate["requests"]),
        "promotion_count": sum(row["promotions"] for row in candidate["requests"]),
        "activation_drift": activation_drift(reference, candidate),
        "reference_router_near_tie": reference_router,
        "candidate_router_near_tie": candidate_router,
        "router_near_tie_delta": candidate_router["rate"] - reference_router["rate"],
        "long_context_stability": long_context_stability(reference, candidate, corpus),
    }


def rule_pass(observed: float, op: str, expected: float) -> bool:
    if op == "<=":
        return observed <= expected
    if op == ">=":
        return observed >= expected
    if op == "==":
        return observed == expected
    raise ValueError(f"unsupported policy operator: {op}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit a completed P9 physical quality run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    physical_path = args.run_dir / "physical_result.json"
    latest_path = args.run_dir / "p9_quality_latest.json"
    evaluation_path = args.run_dir / "evaluation.json"
    metadata_path = args.run_dir / "job_metadata.json"
    physical = load_json(physical_path)
    latest = load_json(latest_path)
    evaluation = load_json(evaluation_path)
    metadata = load_json(metadata_path)
    policy = load_json(args.policy)
    freeze = load_json(args.policy_freeze)
    registration = load_json(args.registration)
    corpus = normalized_corpus(load_json(args.corpus))

    checks: dict[str, bool] = {}
    failures: list[str] = []

    def require(name: str, condition: bool) -> None:
        checks[name] = bool(condition)
        if not condition:
            failures.append(name)

    require("physical_schema", physical.get("schema") == PHYSICAL_SCHEMA)
    require("physical_raw_status", physical.get("status") == "P9_PHYSICAL_RAW_PASS")
    require("physical_result_hash", physical.get("result_sha256") == hash_without(physical, "result_sha256"))
    require("latest_byte_identity", physical_path.read_bytes() == latest_path.read_bytes())
    require("production_write_denied", physical.get("production_write_allowed") is False)
    require("branch", physical.get("branch") == "prod/p9-integration")

    release = registration.get("release") or {}
    first_job = registration.get("first_physical_job") or {}
    require("registration_status", registration.get("status") == "P9_TAILNET_FIXTURE_REGISTERED")
    require("registered_argv", registration.get("argv") == ["python3", "beglin_p9_quality_fixture.py"])
    require("runner_sha_pinned", physical.get("runner_sha256") == release.get("runner_sha256"))
    require("job_id", metadata.get("job_id") == first_job.get("job_id"))
    require("job_request_hash", metadata.get("request_hash") == first_job.get("request_hash"))
    require("job_succeeded", metadata.get("state") == "SUCCEEDED" and metadata.get("exit_code") == 0)
    require("job_termination", metadata.get("termination_reason") == "exit_code")
    require("job_finished", metadata.get("ended_at") == physical.get("finished_at", "")[:19] + "Z")

    physical_identity = physical.get("identity") or {}
    require("source_commit", physical_identity.get("source_commit") == first_job.get("source_commit"))
    require("binary_sha", physical_identity.get("binary_sha256") == first_job.get("binary_sha256"))
    require("checkpoint_sha", physical_identity.get("checkpoint_sha256") == first_job.get("checkpoint_sha256"))
    require("metal_backend", physical_identity.get("backend") == "mlx_metal")
    require("model_identity", physical_identity.get("model_id") == "deepseek-v2-lite")

    policy_core = {
        "schema": policy.get("schema"),
        "policy_status": policy.get("policy_status"),
        "required_metrics": policy.get("required_metrics"),
        "rules": [
            {"metric": row["metric"], "op": row["op"], "value": float(row["value"])}
            for row in policy.get("rules", [])
        ],
    }
    policy_hash = sha256_json(policy_core)
    require("policy_schema", policy.get("schema") == POLICY_SCHEMA)
    require("policy_frozen", policy.get("policy_status") == "FROZEN")
    require("policy_file_hash", sha256_file(args.policy) == (freeze.get("policy") or {}).get("file_sha256"))
    require("policy_hash", policy_hash == (freeze.get("policy") or {}).get("policy_hash"))
    require("policy_observation_order_asserted", freeze.get("candidate_evidence_observed_before_freeze") is False)
    require("policy_baseline", policy.get("baseline_artifact_hash") == freeze.get("reference", {}).get("artifact_hash"))
    require("corpus_hash", corpus["corpus_sha256"] == physical_identity.get("prompt_corpus_sha256"))

    raw_runs = physical.get("runs", [])
    raw_by_variant = {row.get("variant"): row for row in raw_runs}
    require("physical_variant_set", set(raw_by_variant) == set(EXPECTED_VARIANTS) and len(raw_runs) == 5)
    raw_log_hashes: dict[str, str] = {}
    physical_rows = []
    for variant in EXPECTED_VARIANTS:
        row = raw_by_variant.get(variant) or {}
        raw_path = args.run_dir / "raw" / f"{variant}.log"
        actual_raw_sha = sha256_file(raw_path) if raw_path.is_file() else None
        raw_log_hashes[variant] = actual_raw_sha or "MISSING"
        row_ok = (
            row.get("status") == "PASS"
            and row.get("exit_code") == 0
            and row.get("request_count") == 16
            and row.get("parsed_request_count") == 16
            and row.get("metrics_complete") is True
            and row.get("finite_logits") is True
            and row.get("logits_checked", 0) > 0
            and row.get("log_sha256") == actual_raw_sha
        )
        require(f"physical_{variant}", row_ok)
        physical_rows.append(
            {
                "variant": variant,
                "status": row.get("status"),
                "request_count": row.get("request_count"),
                "log_sha256": actual_raw_sha,
            }
        )

    artifacts: dict[str, dict[str, Any]] = {}
    artifact_rows = []
    for variant in EXPECTED_VARIANTS:
        path = args.run_dir / f"{variant}.json"
        artifact = load_json(path)
        artifacts[variant] = artifact
        expected_kind = "reference" if variant == "reference" else "candidate"
        expected_n = None if variant == "reference" else int(variant[1:])
        identity = artifact.get("identity") or {}
        identity_ok = all(identity.get(field) == physical_identity.get(field) for field in COMPARABLE_IDENTITY_FIELDS)
        require(f"artifact_{variant}_schema", artifact.get("schema") == RUN_SCHEMA)
        require(f"artifact_{variant}_hash", artifact.get("artifact_hash") == hash_without(artifact, "artifact_hash"))
        require(f"artifact_{variant}_variant", artifact.get("variant", {}).get("kind") == expected_kind and artifact.get("variant", {}).get("n") == expected_n)
        require(f"artifact_{variant}_requests", len(artifact.get("requests", [])) == 16)
        require(f"artifact_{variant}_identity", identity_ok)
        artifact_rows.append(
            {
                "variant": variant,
                "artifact_hash": artifact.get("artifact_hash"),
                "file_sha256": sha256_file(path),
                "identity_compatible": identity_ok,
            }
        )

    reference = artifacts["reference"]
    require("freeze_reference_hash", freeze.get("reference", {}).get("artifact_hash") == reference.get("artifact_hash"))
    require("freeze_reference_file", freeze.get("reference", {}).get("file_sha256") == sha256_file(args.run_dir / "reference.json"))
    require("freeze_reference_raw", freeze.get("reference", {}).get("raw_log_sha256") == raw_log_hashes["reference"])

    require("evaluation_schema", evaluation.get("schema") == EVALUATION_SCHEMA)
    require("evaluation_hash", evaluation.get("evaluation_hash") == hash_without(evaluation, "evaluation_hash"))
    require("evaluation_policy", evaluation.get("policy_status") == "FROZEN" and evaluation.get("policy_hash") == policy_hash)
    require("evaluation_corpus", evaluation.get("corpus_sha256") == corpus["corpus_sha256"])
    require("evaluation_reference", evaluation.get("reference_artifact_hash") == reference.get("artifact_hash"))

    comparisons = evaluation.get("comparisons", [])
    comparison_by_n = {row.get("pair", {}).get("candidate_n"): row for row in comparisons}
    require("evaluation_candidate_set", set(comparison_by_n) == {4, 5, 6, 7} and len(comparisons) == 4)
    candidate_rows = []
    independently_derived_statuses: dict[int, str] = {}
    for n in (4, 5, 6, 7):
        comparison = comparison_by_n.get(n) or {}
        pair = comparison.get("pair") or {}
        candidate = artifacts[f"n{n}"]
        metrics = recompute_metrics(reference, candidate, corpus)
        require(f"n{n}_candidate_hash", comparison.get("candidate_artifact_hash") == candidate.get("artifact_hash"))
        require(f"n{n}_pair_measured", pair.get("status") == "MEASURED")
        require(f"n{n}_identity_compatible", pair.get("identity", {}).get("compatible") is True and pair.get("identity", {}).get("differing_fields") == [])
        require(f"n{n}_metrics_recomputed", pair.get("metrics") == metrics)

        missing = [path for path in policy_core["required_metrics"] if read_metric(metrics, path) is None]
        rule_results = []
        for rule in policy_core["rules"]:
            observed = read_metric(metrics, rule["metric"])
            passed = observed is not None and rule_pass(float(observed), rule["op"], rule["value"])
            rule_results.append({**rule, "observed": float(observed) if observed is not None else None, "passed": passed})
        if missing:
            derived_status = "INCOMPLETE_MISSING_METRICS"
        else:
            derived_status = "P9_QUALITY_PASS" if all(row["passed"] for row in rule_results) else "P9_QUALITY_FAIL"
        independently_derived_statuses[n] = derived_status
        gate = comparison.get("gate") or {}
        require(f"n{n}_missing_metrics", gate.get("missing_required_metrics") == missing)
        require(f"n{n}_rule_results", gate.get("rule_results") == rule_results)
        require(f"n{n}_gate_status", gate.get("status") == derived_status)
        candidate_rows.append(
            {
                "n": n,
                "status": derived_status,
                "perplexity_ratio": metrics["perplexity_ratio"],
                "quality_score": metrics["candidate_quality_score"],
                "reference_output_match": metrics["reference_output_match"]["rate"],
                "failed_rules": [row["metric"] for row in rule_results if not row["passed"]],
            }
        )

    derived_overall = (
        "P9_QUALITY_PASS"
        if all(status == "P9_QUALITY_PASS" for status in independently_derived_statuses.values())
        else "P9_QUALITY_FAIL"
    )
    require("overall_status", evaluation.get("status") == derived_overall)
    require("observed_matrix_outcome", independently_derived_statuses == {4: "P9_QUALITY_PASS", 5: "P9_QUALITY_FAIL", 6: "P9_QUALITY_FAIL", 7: "P9_QUALITY_FAIL"})

    evidence_status = "P9_EVIDENCE_INDEPENDENT_PASS" if not failures else "P9_EVIDENCE_INDEPENDENT_FAIL"
    eligible = [n for n, status in independently_derived_statuses.items() if status == "P9_QUALITY_PASS"]
    rejected = [n for n, status in independently_derived_statuses.items() if status != "P9_QUALITY_PASS"]
    audit_core = {
        "schema": AUDIT_SCHEMA,
        "observed_at_utc": args.observed_at,
        "status": evidence_status,
        "inputs": {
            "run_dir": str(args.run_dir),
            "physical_result_sha256": sha256_file(physical_path),
            "physical_result_claimed_hash": physical.get("result_sha256"),
            "evaluation_sha256": sha256_file(evaluation_path),
            "evaluation_claimed_hash": evaluation.get("evaluation_hash"),
            "policy_sha256": sha256_file(args.policy),
            "policy_hash": policy_hash,
            "policy_freeze_sha256": sha256_file(args.policy_freeze),
            "registration_sha256": sha256_file(args.registration),
            "job_metadata_sha256": sha256_file(metadata_path),
        },
        "tailnet_job": {
            "job_id": metadata.get("job_id"),
            "request_hash": metadata.get("request_hash"),
            "state": metadata.get("state"),
            "exit_code": metadata.get("exit_code"),
            "started_at": metadata.get("started_at"),
            "ended_at": metadata.get("ended_at"),
            "elapsed_ms": metadata.get("elapsed_ms"),
        },
        "control_identity": {field: physical_identity.get(field) for field in COMPARABLE_IDENTITY_FIELDS},
        "raw_runs": physical_rows,
        "normalized_artifacts": artifact_rows,
        "candidates": candidate_rows,
        "checks": checks,
        "failures": failures,
        "conclusion": {
            "physical_execution": physical.get("status"),
            "evidence_integrity": evidence_status,
            "p9_quality": derived_overall,
            "eligible_candidate_ns": eligible,
            "rejected_candidate_ns": rejected,
            "production_release_authorized": evidence_status == "P9_EVIDENCE_INDEPENDENT_PASS" and derived_overall == "P9_QUALITY_PASS",
            "production_mutation_performed": False,
            "policy_modified_after_freeze": False,
        },
    }
    audit = {**audit_core, "audit_sha256": sha256_json(audit_core)}
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
