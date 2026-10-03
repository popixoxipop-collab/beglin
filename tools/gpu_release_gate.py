#!/usr/bin/env python3
"""Fail-closed G6 release qualification for MLX/Metal precision canaries.

This module does not deploy anything and never enables automatic promotion.
It only turns already-collected G4/G5/G6 evidence into a scope-limited release
qualification when a *predeclared* quality/performance budget is satisfied.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import gpu_runtime_control as grc
import precision_context as pc


BUDGET_SCHEMA = "gpu-release-budget-v1"
CALIBRATION_SCHEMA = "gpu-aa-calibration-v1"
RELEASE_SCHEMA = "gpu-release-qualification-v1"
BACKEND = "mlx_metal"

PERF_METRICS = (
    "ttft_p50_ms",
    "ttft_p95_ms",
    "token_latency_p50_ms",
    "token_latency_p95_ms",
    "throughput_tok_s",
    "cpu_rss_peak_bytes",
    "gpu_peak_memory_bytes",
)

RATIO_LIMITS = {
    "ttft_p50_ms": ("max_ttft_p50_ratio", "max"),
    "ttft_p95_ms": ("max_ttft_p95_ratio", "max"),
    "token_latency_p50_ms": ("max_token_latency_p50_ratio", "max"),
    "token_latency_p95_ms": ("max_token_latency_p95_ratio", "max"),
    "throughput_tok_s": ("min_throughput_ratio", "min"),
    "cpu_rss_peak_bytes": ("max_cpu_rss_peak_ratio", "max"),
    "gpu_peak_memory_bytes": ("max_gpu_peak_memory_ratio", "max"),
}

REQUIRED_BUDGET_NUMBERS = (
    "max_ttft_p50_ratio",
    "max_ttft_p95_ratio",
    "max_token_latency_p50_ratio",
    "max_token_latency_p95_ratio",
    "min_throughput_ratio",
    "max_cpu_rss_peak_ratio",
    "max_gpu_peak_memory_ratio",
    "max_drain_ms",
    "max_rollback_ms",
    "max_correction_required_rate",
    "min_holdout_delta",
    "max_aa_relative_spread",
)


class GpuReleaseGateError(RuntimeError):
    pass


def _finite_number(value: Any, field: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise GpuReleaseGateError(f"{field} must be numeric") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise GpuReleaseGateError(f"{field} must be finite")
    return value


def freeze_budget_spec(spec: Mapping[str, Any]) -> dict:
    """Validate and hash the budget before candidate measurements are admitted."""
    if not isinstance(spec, Mapping):
        raise GpuReleaseGateError("budget spec must be a mapping")
    scope = str(spec.get("scope", "")).strip()
    workload = str(spec.get("workload", "")).strip()
    if not scope or not workload:
        raise GpuReleaseGateError("budget spec requires non-empty scope and workload")

    normalized = {
        "schema": BUDGET_SCHEMA,
        "scope": scope,
        "workload": workload,
    }
    for key in REQUIRED_BUDGET_NUMBERS:
        value = _finite_number(spec.get(key), key)
        if key == "min_holdout_delta":
            # Quality deltas may legitimately be negative when the predeclared
            # budget permits a bounded loss.
            pass
        elif value < 0:
            raise GpuReleaseGateError(f"{key} must be non-negative")
        if key.endswith("_ratio") and value <= 0:
            raise GpuReleaseGateError(f"{key} must be positive")
        normalized[key] = value

    normalized["budget_spec_sha256"] = pc.sha256_json(normalized)
    return normalized


def require_frozen_budget(spec: Mapping[str, Any]) -> dict:
    if spec.get("schema") != BUDGET_SCHEMA:
        raise GpuReleaseGateError("budget spec is not frozen with gpu-release-budget-v1")
    expected = str(spec.get("budget_spec_sha256", ""))
    payload = {k: v for k, v in spec.items() if k != "budget_spec_sha256"}
    actual = pc.sha256_json(payload)
    if expected != actual:
        raise GpuReleaseGateError(
            f"budget spec hash mismatch: expected={expected} actual={actual}"
        )
    return dict(spec)


def validate_aa_calibration(
    calibration: Mapping[str, Any],
    *,
    budget_spec: Mapping[str, Any],
    context_hash: str,
    binary_sha256: str,
) -> dict:
    """Require A/A repeatability to be established against the frozen budget."""
    budget = require_frozen_budget(budget_spec)
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise GpuReleaseGateError("missing gpu-aa-calibration-v1 artifact")
    if calibration.get("budget_spec_sha256") != budget["budget_spec_sha256"]:
        raise GpuReleaseGateError("A/A calibration used a different budget spec")
    if calibration.get("context_hash") != context_hash:
        raise GpuReleaseGateError("A/A calibration context mismatch")
    if str(calibration.get("binary_sha256", "")).lower() != binary_sha256.lower():
        raise GpuReleaseGateError("A/A calibration binary mismatch")
    if int(calibration.get("repeat_count", 0)) < 2:
        raise GpuReleaseGateError("A/A calibration requires at least two repeats")

    spread = calibration.get("relative_spread")
    if not isinstance(spread, Mapping):
        raise GpuReleaseGateError("A/A calibration requires relative_spread")
    limit = float(budget["max_aa_relative_spread"])
    checked = {}
    for metric in PERF_METRICS:
        value = _finite_number(spread.get(metric), f"relative_spread.{metric}")
        if value < 0:
            raise GpuReleaseGateError(f"relative_spread.{metric} must be non-negative")
        checked[metric] = {
            "value": value,
            "limit": limit,
            "pass": value <= limit,
        }
    if not all(row["pass"] for row in checked.values()):
        raise GpuReleaseGateError("A/A reproducibility exceeds frozen spread budget")
    return {
        "status": "PASS",
        "checks": checked,
        "calibration_sha256": pc.sha256_json(calibration),
    }


def _metric_block(value: Mapping[str, Any], label: str) -> dict:
    if not isinstance(value, Mapping):
        raise GpuReleaseGateError(f"{label} metrics must be a mapping")
    out = {}
    for metric in PERF_METRICS:
        number = _finite_number(value.get(metric), f"{label}.{metric}")
        if number <= 0:
            raise GpuReleaseGateError(f"{label}.{metric} must be positive")
        out[metric] = number
    return out


def evaluate_budget(
    *,
    budget_spec: Mapping[str, Any],
    measurements: Mapping[str, Any],
) -> dict:
    budget = require_frozen_budget(budget_spec)
    baseline = _metric_block(measurements.get("baseline"), "baseline")
    candidate = _metric_block(measurements.get("candidate"), "candidate")

    checks = {}
    for metric, (limit_key, direction) in RATIO_LIMITS.items():
        ratio = candidate[metric] / baseline[metric]
        limit = float(budget[limit_key])
        passed = ratio <= limit if direction == "max" else ratio >= limit
        checks[f"{metric}_ratio"] = {
            "value": ratio,
            "limit": limit,
            "direction": direction,
            "pass": passed,
        }

    quality = measurements.get("quality")
    if not isinstance(quality, Mapping):
        raise GpuReleaseGateError("measurements require quality block")
    bool_checks = {
        "finite_logits": quality.get("finite_logits") is True,
        "target_replay_pass": quality.get("target_replay_pass") is True,
        "regression_panel_pass": quality.get("regression_panel_pass") is True,
    }
    for name, passed in bool_checks.items():
        checks[name] = {"value": bool(passed), "pass": bool(passed)}

    holdout_delta = _finite_number(quality.get("holdout_delta"), "quality.holdout_delta")
    checks["holdout_delta"] = {
        "value": holdout_delta,
        "limit": float(budget["min_holdout_delta"]),
        "direction": "min",
        "pass": holdout_delta >= float(budget["min_holdout_delta"]),
    }
    correction_rate = _finite_number(
        quality.get("correction_required_rate"),
        "quality.correction_required_rate",
    )
    if correction_rate < 0 or correction_rate > 1:
        raise GpuReleaseGateError("correction_required_rate must be in [0,1]")
    checks["correction_required_rate"] = {
        "value": correction_rate,
        "limit": float(budget["max_correction_required_rate"]),
        "direction": "max",
        "pass": correction_rate <= float(budget["max_correction_required_rate"]),
    }

    drain_ms = _finite_number(measurements.get("drain_ms"), "drain_ms")
    checks["drain_ms"] = {
        "value": drain_ms,
        "limit": float(budget["max_drain_ms"]),
        "direction": "max",
        "pass": 0 <= drain_ms <= float(budget["max_drain_ms"]),
    }
    passed = all(row["pass"] for row in checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
        "measurements_sha256": pc.sha256_json(measurements),
    }


def _require_capability(capability: Mapping[str, Any], context: pc.ExecutionContext) -> dict:
    try:
        worker = capability["worker"]
        gpu = capability["backends"]["mlx_metal"]
        runtime = gpu["runtime_control"]
    except (KeyError, TypeError) as exc:
        raise GpuReleaseGateError("malformed capability artifact") from exc
    if not gpu.get("compiled") or not runtime.get("compiled"):
        raise GpuReleaseGateError("release binary lacks verified MLX control plane")
    if str(worker.get("binary_sha256", "")).lower() != context.binary_sha256.lower():
        raise GpuReleaseGateError("capability binary does not match execution context")
    if worker.get("arch") not in {"arm64", "aarch64"}:
        raise GpuReleaseGateError("release capability is not Apple arm64")
    return {
        "worker_identity_sha256": capability.get("worker_identity_sha256"),
        "capability_sha256": pc.sha256_json(capability),
    }


def _require_cpu_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict:
    before_hash = pc.sha256_json(before)
    after_hash = pc.sha256_json(after)
    if before_hash != after_hash:
        raise GpuReleaseGateError(
            "CPU production identity/state changed during GPU qualification"
        )
    return {
        "cpu_identity_sha256": before_hash,
        "unchanged": True,
    }


def _require_rollback_drill(
    rollback_drill: Mapping[str, Any],
    *,
    budget_spec: Mapping[str, Any],
) -> dict:
    budget = require_frozen_budget(budget_spec)
    if rollback_drill.get("status") != "ROLLBACK_APPLIED":
        raise GpuReleaseGateError("rollback drill did not reach ROLLBACK_APPLIED")
    if rollback_drill.get("rollback_complete") is not True:
        raise GpuReleaseGateError("rollback drill is not complete")
    reconcile = rollback_drill.get("reconcile")
    if not isinstance(reconcile, Mapping) or reconcile.get("status") != "IN_SYNC":
        raise GpuReleaseGateError("rollback drill durable state is not IN_SYNC")
    rollback_ms = _finite_number(
        rollback_drill.get("rollback_latency_ms"), "rollback_latency_ms"
    )
    if rollback_ms < 0 or rollback_ms > float(budget["max_rollback_ms"]):
        raise GpuReleaseGateError(
            "rollback drill exceeded frozen rollback latency budget"
        )
    return {
        "status": "PASS",
        "rollback_latency_ms": rollback_ms,
        "rollback_drill_sha256": pc.sha256_json(rollback_drill),
    }


def no_approved_candidate(*, scope: str, workload: str, reason: str) -> dict:
    """Explicitly valid G6 outcome when no candidate qualified."""
    result = {
        "schema": RELEASE_SCHEMA,
        "status": "NO_APPROVED_CANDIDATE",
        "scope": str(scope),
        "workload": str(workload),
        "reason": str(reason),
        "gpu_auto_promotion_enabled": False,
    }
    result["release_manifest_sha256"] = pc.sha256_json(result)
    return result


def evaluate_release(
    *,
    context: pc.ExecutionContext,
    budget_spec: Mapping[str, Any],
    calibration: Mapping[str, Any],
    capability: Mapping[str, Any],
    preflight: Mapping[str, Any],
    canary_plan: Mapping[str, Any],
    canary_result: Mapping[str, Any],
    runtime_ack: Mapping[str, Any],
    candidate_policy,
    measurements: Mapping[str, Any],
    rollback_drill: Mapping[str, Any],
    cpu_identity_before: Mapping[str, Any],
    cpu_identity_after: Mapping[str, Any],
) -> dict:
    """Qualify one already-canary-tested GPU target for the declared scope."""
    if context.backend != BACKEND or context.correction_mode != "off":
        raise GpuReleaseGateError(
            "release context must be mlx_metal with correction_mode=off"
        )
    budget = require_frozen_budget(budget_spec)
    rows = pc.normalize_policy(candidate_policy)
    candidate_hash = pc.policy_hash(rows)

    cap = _require_capability(capability, context)
    calibration_result = validate_aa_calibration(
        calibration,
        budget_spec=budget,
        context_hash=context.context_hash,
        binary_sha256=context.binary_sha256,
    )

    if preflight.get("status") != "passed":
        raise GpuReleaseGateError("G4 isolated preflight did not pass")
    if preflight.get("backend") != BACKEND:
        raise GpuReleaseGateError("preflight backend mismatch")
    if preflight.get("correction_mode") != "off":
        raise GpuReleaseGateError("preflight correction mode is not OFF")
    if preflight.get("candidate_policy_hash") != candidate_hash:
        raise GpuReleaseGateError("preflight candidate policy mismatch")
    if str(preflight.get("binary_sha256", "")).lower() != context.binary_sha256.lower():
        raise GpuReleaseGateError("preflight binary mismatch")
    if not preflight.get("evidence_bundle"):
        raise GpuReleaseGateError("preflight evidence bundle is missing")

    if canary_plan.get("backend") != BACKEND:
        raise GpuReleaseGateError("canary plan backend mismatch")
    if canary_plan.get("context_hash") != context.context_hash:
        raise GpuReleaseGateError("canary plan context mismatch")
    if str(canary_plan.get("binary_sha256", "")).lower() != context.binary_sha256.lower():
        raise GpuReleaseGateError("canary plan binary mismatch")
    if canary_plan.get("candidate_policy_hash") != candidate_hash:
        raise GpuReleaseGateError("canary plan candidate policy mismatch")
    if canary_plan.get("limits", {}).get("auto_expand") is not False:
        raise GpuReleaseGateError("release canary must have auto_expand=false")

    if (
        canary_result.get("status") != "CANARY_PASS"
        or canary_result.get("decision") != "CANARY_PASS_NO_AUTO_EXPANSION"
        or canary_result.get("rollback_required") is not False
        or canary_result.get("auto_expand") is not False
    ):
        raise GpuReleaseGateError("G6 restart canary did not pass fail-closed criteria")

    ack = grc.normalize_runtime_ack(runtime_ack)
    if ack["active_policy_hash"] != candidate_hash:
        raise GpuReleaseGateError("device applied policy does not match candidate")
    if ack["correction_mode"] != "off":
        raise GpuReleaseGateError("device applied state has correction enabled")

    budget_result = evaluate_budget(
        budget_spec=budget,
        measurements=measurements,
    )
    rollback_result = _require_rollback_drill(
        rollback_drill,
        budget_spec=budget,
    )
    cpu_result = _require_cpu_unchanged(
        cpu_identity_before,
        cpu_identity_after,
    )

    evidence_hashes = {
        "context": pc.sha256_json(context.payload()),
        "budget": budget["budget_spec_sha256"],
        "calibration": calibration_result["calibration_sha256"],
        "capability": cap["capability_sha256"],
        "preflight": pc.sha256_json(preflight),
        "canary_plan": pc.sha256_json(canary_plan),
        "canary_result": pc.sha256_json(canary_result),
        "runtime_ack": ack["ack_sha256"],
        "measurements": budget_result["measurements_sha256"],
        "rollback_drill": rollback_result["rollback_drill_sha256"],
        "cpu_identity": cpu_result["cpu_identity_sha256"],
    }

    status = (
        "QUALIFIED_FOR_SCOPE"
        if budget_result["status"] == "PASS"
        else "REJECTED_BUDGET"
    )
    result = {
        "schema": RELEASE_SCHEMA,
        "status": status,
        "backend": BACKEND,
        "scope": budget["scope"],
        "workload": budget["workload"],
        "context_hash": context.context_hash,
        "binary_sha256": context.binary_sha256.lower(),
        "candidate_policy": rows,
        "candidate_policy_hash": candidate_hash,
        "applied_weight_epoch": ack["weight_epoch"],
        "budget_spec_sha256": budget["budget_spec_sha256"],
        "budget": budget_result,
        "aa_calibration": calibration_result,
        "rollback_drill": rollback_result,
        "cpu_state": cpu_result,
        "evidence_hashes": evidence_hashes,
        # Qualification is intentionally separate from enabling automation.
        "gpu_auto_promotion_enabled": False,
        "requires_explicit_opt_in_for_auto_promotion": True,
    }
    result["release_manifest_sha256"] = pc.sha256_json(result)
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_release_bundle(
    root: str | os.PathLike,
    *,
    release_manifest: Mapping[str, Any],
    budget_spec: Mapping[str, Any],
    measurements: Mapping[str, Any],
    rollback_drill: Mapping[str, Any],
    cpu_identity_before: Mapping[str, Any],
    cpu_identity_after: Mapping[str, Any],
) -> dict:
    """Persist the small G6 release evidence and a SHA256SUMS manifest."""
    root = Path(root)
    payloads = {
        "release_manifest.json": release_manifest,
        "budget_spec.json": budget_spec,
        "measurements.json": measurements,
        "rollback_drill.json": rollback_drill,
        "cpu_identity_before.json": cpu_identity_before,
        "cpu_identity_after.json": cpu_identity_after,
    }
    hashes = {}
    for name, value in payloads.items():
        path = root / name
        _atomic_json(path, value)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    sums_path = root / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    )
    with open(sums_path, "rb") as f:
        os.fsync(f.fileno())
    return {
        "root": str(root),
        "files": hashes,
        "sha256sums_sha256": hashlib.sha256(sums_path.read_bytes()).hexdigest(),
    }
