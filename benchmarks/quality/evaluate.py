from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifact import QualityArtifactError, load_run_artifact, sha256_json
from .corpus import load_corpus
from .metrics import evaluate_pair, required_metric_paths

EVALUATION_SCHEMA = "beglin-quality-evaluation/1"
POLICY_SCHEMA = "beglin-quality-policy/1"


def _read_path(root: dict[str, Any], path: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def load_policy(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != POLICY_SCHEMA:
        raise QualityArtifactError(f"policy.schema must be {POLICY_SCHEMA}")
    state = raw.get("policy_status")
    if state not in {"UNFROZEN", "FROZEN"}:
        raise QualityArtifactError("policy_status must be UNFROZEN or FROZEN")
    required = raw.get("required_metrics")
    if not isinstance(required, list) or not required or not all(isinstance(x, str) and x for x in required):
        raise QualityArtifactError("required_metrics must be a non-empty string array")
    unknown = sorted(set(required) - set(required_metric_paths()))
    if unknown:
        raise QualityArtifactError(f"unknown required metrics: {unknown}")
    rules = raw.get("rules", [])
    if not isinstance(rules, list):
        raise QualityArtifactError("rules must be an array")
    parsed_rules = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise QualityArtifactError(f"rules[{index}] must be an object")
        metric = rule.get("metric")
        op = rule.get("op")
        value = rule.get("value")
        if metric not in required:
            raise QualityArtifactError(f"rules[{index}].metric must also be required")
        if op not in {"<=", ">=", "=="}:
            raise QualityArtifactError(f"rules[{index}].op unsupported")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QualityArtifactError(f"rules[{index}].value must be numeric")
        parsed_rules.append({"metric": metric, "op": op, "value": float(value)})
    if state == "UNFROZEN" and parsed_rules:
        raise QualityArtifactError("UNFROZEN policy must not enforce numeric rules")
    if state == "FROZEN" and not parsed_rules:
        raise QualityArtifactError("FROZEN policy must contain at least one rule")
    return {
        "schema": POLICY_SCHEMA,
        "policy_status": state,
        "required_metrics": required,
        "rules": parsed_rules,
        "policy_hash": sha256_json({
            "schema": POLICY_SCHEMA,
            "policy_status": state,
            "required_metrics": required,
            "rules": parsed_rules,
        }),
    }


def _rule_pass(observed: float, op: str, expected: float) -> bool:
    if op == "<=":
        return observed <= expected
    if op == ">=":
        return observed >= expected
    return observed == expected


def classify_pair(pair: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    if pair["status"] != "MEASURED":
        return {
            "status": pair["status"],
            "missing_required_metrics": list(policy["required_metrics"]),
            "rule_results": [],
        }
    metrics = pair["metrics"]
    missing = [
        path
        for path in policy["required_metrics"]
        if _read_path(metrics, path) is None
    ]
    if missing:
        return {
            "status": "INCOMPLETE_MISSING_METRICS",
            "missing_required_metrics": missing,
            "rule_results": [],
        }
    if policy["policy_status"] != "FROZEN":
        return {
            "status": "MEASURED_POLICY_NOT_FROZEN",
            "missing_required_metrics": [],
            "rule_results": [],
        }

    results = []
    for rule in policy["rules"]:
        observed = float(_read_path(metrics, rule["metric"]))
        passed = _rule_pass(observed, rule["op"], rule["value"])
        results.append({
            **rule,
            "observed": observed,
            "passed": passed,
        })
    return {
        "status": "P9_QUALITY_PASS" if all(item["passed"] for item in results) else "P9_QUALITY_FAIL",
        "missing_required_metrics": [],
        "rule_results": results,
    }


def evaluate(
    reference_path: Path,
    candidate_paths: list[Path],
    corpus_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    reference = load_run_artifact(reference_path)
    if reference["variant"]["kind"] != "reference":
        raise QualityArtifactError("reference artifact must have variant.kind=reference")
    corpus = load_corpus(corpus_path)
    policy = load_policy(policy_path)

    if reference["identity"]["prompt_corpus_sha256"] != corpus["corpus_sha256"]:
        raise QualityArtifactError("reference prompt_corpus_sha256 does not match corpus")

    rows = []
    for path in candidate_paths:
        candidate = load_run_artifact(path)
        if candidate["variant"]["kind"] != "candidate":
            raise QualityArtifactError(f"{path}: candidate artifact must have variant.kind=candidate")
        if candidate["identity"]["prompt_corpus_sha256"] != corpus["corpus_sha256"]:
            raise QualityArtifactError(f"{path}: prompt_corpus_sha256 does not match corpus")
        pair = evaluate_pair(reference, candidate, corpus)
        gate = classify_pair(pair, policy)
        rows.append({
            "candidate_path": str(path),
            "candidate_artifact_hash": candidate["artifact_hash"],
            "pair": pair,
            "gate": gate,
        })

    statuses = [row["gate"]["status"] for row in rows]
    if any(status == "CONTROL_MISMATCH" for status in statuses):
        overall = "FAIL_CONTROL_MISMATCH"
    elif any(status == "INCOMPLETE_MISSING_METRICS" for status in statuses):
        overall = "INCOMPLETE_MISSING_METRICS"
    elif policy["policy_status"] != "FROZEN":
        overall = "P9_QUALITY_HARNESS_READY"
    elif all(status == "P9_QUALITY_PASS" for status in statuses):
        overall = "P9_QUALITY_PASS"
    else:
        overall = "P9_QUALITY_FAIL"

    payload = {
        "schema": EVALUATION_SCHEMA,
        "status": overall,
        "reference_path": str(reference_path),
        "reference_artifact_hash": reference["artifact_hash"],
        "corpus_path": str(corpus_path),
        "corpus_sha256": corpus["corpus_sha256"],
        "policy_path": str(policy_path),
        "policy_hash": policy["policy_hash"],
        "policy_status": policy["policy_status"],
        "comparisons": rows,
    }
    payload["evaluation_hash"] = sha256_json(payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Beglin quality artifacts fail-closed.")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, action="append", required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = evaluate(args.reference, args.candidate, args.corpus, args.policy)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
