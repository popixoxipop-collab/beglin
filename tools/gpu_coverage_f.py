#!/usr/bin/env python3
"""Agent-F GPU coverage planning and evidence-matrix helper.

This module is intentionally independent of the live GPU control path. It
prepares bounded candidate plans, classifies observer decisions using the
existing autopilot_observer_v3 contract, and emits coverage rows that make the
evidence level explicit.

It does NOT start GPU workers, mutate production/shadow history, write
precision policy, or synthesize a real GPU PASS.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from autopilot_observer_v3 import ObservationEvidence, evaluate


SCHEMA = "gpu-coverage-f-v1"
DEFAULT_BACKEND = "mlx_metal"
DEFAULT_ARCH = "mla"

STATUS_REAL = {"PASS", "FAIL", "NO_BAD_CANDIDATE_WITHIN_BUDGET"}
EVIDENCE_LEVELS = {
    "historical_handoff",
    "static",
    "unit",
    "simulated",
    "decision_only",
    "real_gpu",
}


class CoverageError(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetSpec:
    role: str
    layer: int
    positive_n: int
    bad_candidates: tuple[int, ...]
    required_cases: tuple[str, ...]


HISTORICAL_TARGETS = (
    {
        "role": "shared_down_proj",
        "layer": 4,
        "positive_n": 6,
        "historical_g5_bad_n": 5,
        "historical_positive": "G4/G6 PASS",
        "historical_g5": "PASS",
        "source": "HANDOFF_GPU_TRACK_2026-09-25_PART2.md",
    },
    {
        "role": "kv_a_proj_with_mqa",
        "layer": 11,
        "positive_n": 7,
        "historical_g5_bad_n": None,
        "historical_positive": "custom Metal G4/G6 PASS",
        "historical_g5": "UNVERIFIED",
        "source": "HANDOFF_GPU_TRACK_2026-09-25_PART2.md",
    },
    {
        "role": "shared_up_proj",
        "layer": 3,
        "positive_n": 6,
        "historical_g5_bad_n": 7,
        "historical_positive": "G4/G6 PASS",
        "historical_g5": "PASS",
        "source": "HANDOFF_GPU_TRACK_2026-09-25_PART2.md",
    },
)

TARGET_SPECS = (
    TargetSpec(
        role="kv_a_proj_with_mqa",
        layer=11,
        positive_n=7,
        bad_candidates=(5, 6, 7),
        required_cases=("positive", "regression", "budget_exceeded", "inconclusive"),
    ),
    TargetSpec(
        role="shared_up_proj",
        layer=3,
        positive_n=6,
        bad_candidates=(7,),
        required_cases=("positive", "regression", "budget_exceeded", "inconclusive"),
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_json(path: str | os.PathLike[str], value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _require_int(row: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    if key not in row:
        raise CoverageError(f"missing observation field: {key}")
    try:
        value = int(row[key])
    except (TypeError, ValueError) as exc:
        raise CoverageError(f"observation field {key} must be integer") from exc
    if value < minimum:
        raise CoverageError(f"observation field {key} must be >= {minimum}")
    return value


def observation_from_json(row: Mapping[str, Any]) -> ObservationEvidence:
    for key in ("context_hash", "backend", "policy_hash"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise CoverageError(f"missing/invalid observation field: {key}")

    margin = row.get("median_margin")
    if margin is not None:
        try:
            margin = float(margin)
        except (TypeError, ValueError) as exc:
            raise CoverageError("median_margin must be numeric or null") from exc

    replay = row.get("target_replay_pass")
    if replay not in (True, False, None):
        raise CoverageError("target_replay_pass must be true/false/null")

    evidence = ObservationEvidence(
        context_hash=row["context_hash"],
        backend=row["backend"],
        policy_hash=row["policy_hash"],
        weight_epoch=_require_int(row, "weight_epoch"),
        requests_completed=_require_int(row, "requests_completed"),
        tokens_evaluated=_require_int(row, "tokens_evaluated"),
        near_tie_events=_require_int(row, "near_tie_events"),
        reference_checks_attempted=_require_int(row, "reference_checks_attempted"),
        effective_attribution_checks=_require_int(row, "effective_attribution_checks"),
        attribution_hits=_require_int(row, "attribution_hits"),
        run_errors=_require_int(row, "run_errors") if "run_errors" in row else 0,
        median_margin=margin,
        target_replay_pass=replay,
    )
    if evidence.attribution_hits > evidence.effective_attribution_checks:
        raise CoverageError("attribution_hits exceeds effective_attribution_checks")
    if evidence.effective_attribution_checks > evidence.reference_checks_attempted:
        raise CoverageError(
            "effective_attribution_checks exceeds reference_checks_attempted"
        )
    return evidence


def _decision_action(status: str, case_kind: str) -> str:
    if case_kind == "positive":
        return "CANARY_PASS" if status == "CANARY_PASS" else "ROLLBACK_REQUIRED"
    if case_kind == "regression":
        return "ROLLBACK_REQUIRED"
    if case_kind == "budget_exceeded":
        return "ROLLBACK_REQUIRED_BUDGET"
    if case_kind == "inconclusive":
        return "ROLLBACK_REQUIRED_INCONCLUSIVE"
    raise CoverageError(f"unsupported case kind: {case_kind}")


def evaluate_case(
    *,
    baseline: ObservationEvidence,
    post: ObservationEvidence,
    case_kind: str,
    min_requests: int = 50,
    min_effective_checks: int = 1,
    evidence_level: str = "decision_only",
    reused_hardware_run: bool = True,
) -> dict:
    if evidence_level not in EVIDENCE_LEVELS:
        raise CoverageError(f"invalid evidence_level: {evidence_level}")
    if case_kind not in {"positive", "regression", "budget_exceeded", "inconclusive"}:
        raise CoverageError(f"invalid case_kind: {case_kind}")

    result = evaluate(
        baseline,
        post,
        min_requests=min_requests,
        min_effective_checks=min_effective_checks,
    )
    action = _decision_action(result["status"], case_kind)
    return {
        "case_kind": case_kind,
        "observer_status": result["status"],
        "observer_reason": result["reason"],
        "expected_action": action,
        "evidence_level": evidence_level,
        "reused_hardware_run": bool(reused_hardware_run),
        "baseline_epoch": baseline.weight_epoch,
        "post_epoch": post.weight_epoch,
        "baseline_policy_hash": baseline.policy_hash,
        "post_policy_hash": post.policy_hash,
        "baseline_requests": baseline.requests_completed,
        "post_requests": post.requests_completed,
        "min_requests": int(min_requests),
        "min_effective_checks": int(min_effective_checks),
    }


def decision_quadrants(
    *,
    baseline: ObservationEvidence,
    passing_post: ObservationEvidence,
) -> list[dict]:
    """Return positive/budget/inconclusive decisions from one observation.

    These rows are decision-only. Regression requires a separate explicit
    failing observation and is not manufactured by mutating passing evidence.
    """
    return [
        evaluate_case(
            baseline=baseline,
            post=passing_post,
            case_kind="positive",
            min_requests=50,
            min_effective_checks=1,
            evidence_level="decision_only",
            reused_hardware_run=True,
        ),
        evaluate_case(
            baseline=baseline,
            post=passing_post,
            case_kind="budget_exceeded",
            min_requests=passing_post.requests_completed + 1,
            min_effective_checks=1,
            evidence_level="decision_only",
            reused_hardware_run=True,
        ),
        evaluate_case(
            baseline=baseline,
            post=passing_post,
            case_kind="inconclusive",
            min_requests=1,
            min_effective_checks=passing_post.effective_attribution_checks + 1,
            evidence_level="decision_only",
            reused_hardware_run=True,
        ),
    ]


def bounded_plan(
    *,
    base_sha: str | None,
    checkpoint_handoff: Mapping[str, Any] | None = None,
    max_candidates: int = 3,
    max_prompts: int = 3,
    max_requests_per_run: int = 100,
    max_restarts: int = 12,
) -> dict:
    if max_candidates < 1 or max_candidates > 3:
        raise CoverageError("max_candidates must be in [1,3]")
    if max_prompts < 1 or max_prompts > 3:
        raise CoverageError("max_prompts must be in [1,3]")
    if max_requests_per_run < 1 or max_requests_per_run > 100:
        raise CoverageError("max_requests_per_run must be in [1,100]")
    if max_restarts < 1 or max_restarts > 12:
        raise CoverageError("max_restarts must be in [1,12]")

    b_verified = False
    checkpoint_sha = None
    checkpoint_manifest = None
    b_reason = "B checkpoint handoff not supplied"
    if checkpoint_handoff:
        b_status = str(checkpoint_handoff.get("status", ""))
        checkpoint_sha = checkpoint_handoff.get("checkpoint_sha256")
        checkpoint_manifest = checkpoint_handoff.get("manifest_path")
        b_verified = (
            b_status == "VERIFIED"
            and isinstance(checkpoint_sha, str)
            and len(checkpoint_sha) == 64
            and all(c in "0123456789abcdefABCDEF" for c in checkpoint_sha)
        )
        if b_verified:
            b_reason = "B checkpoint handoff VERIFIED"
        else:
            b_reason = (
                "B handoff present but not VERIFIED with a 64-hex checkpoint identity"
            )

    cases = []
    for target in TARGET_SPECS:
        cases.append(
            {
                "target": {
                    "role": target.role,
                    "layer": target.layer,
                    "positive_n": target.positive_n,
                },
                "bad_candidate_budget": list(target.bad_candidates)[:max_candidates],
                "required_cases": list(target.required_cases),
                "max_prompts": max_prompts,
                "max_requests_per_run": max_requests_per_run,
                "max_restarts_total_plan": max_restarts,
            }
        )

    result = {
        "schema": SCHEMA,
        "generated_at": _now(),
        "agent_id": "F",
        "base_sha": base_sha,
        "backend": DEFAULT_BACKEND,
        "architecture": DEFAULT_ARCH,
        "real_gpu_ready": bool(b_verified),
        "real_gpu_block_reason": None if b_verified else b_reason,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_manifest": checkpoint_manifest,
        "resource_contract": {
            "requires_xox_gpu_lease": True,
            "requires_b_io_handoff": True,
            "production_mutation_allowed": False,
            "shadow_history_mutation_allowed": False,
        },
        "budgets": {
            "max_candidates_per_target": max_candidates,
            "max_prompts": max_prompts,
            "max_requests_per_run": max_requests_per_run,
            "max_restarts": max_restarts,
        },
        "historical_rows": [
            {
                **row,
                "evidence_level": "historical_handoff",
                "historical": True,
                "new_run": False,
            }
            for row in HISTORICAL_TARGETS
        ],
        "planned_targets": cases,
    }
    result["plan_sha256"] = sha256_json(result)
    return result


def coverage_row(
    *,
    role: str,
    layer: int,
    n: int | None,
    test_kind: str,
    status: str,
    evidence_level: str,
    historical: bool,
    new_run: bool,
    case_id: str,
    run_id: str | None,
    reused_hardware_run: bool,
    expected: Any = None,
    actual: Any = None,
    artifacts: Iterable[Mapping[str, Any]] = (),
    limitations: Iterable[str] = (),
) -> dict:
    if evidence_level not in EVIDENCE_LEVELS:
        raise CoverageError("invalid evidence_level")
    if evidence_level == "real_gpu" and not new_run:
        raise CoverageError("real_gpu evidence must be attached to a new run")
    if historical and new_run:
        raise CoverageError("row cannot be historical and new_run simultaneously")
    if status in STATUS_REAL and evidence_level not in {"real_gpu", "historical_handoff"}:
        raise CoverageError(
            f"status {status} requires real_gpu or historical_handoff evidence"
        )
    return {
        "role": str(role),
        "layer": int(layer),
        "n": None if n is None else int(n),
        "test_kind": str(test_kind),
        "status": str(status),
        "evidence_level": evidence_level,
        "historical": bool(historical),
        "new_run": bool(new_run),
        "case_id": str(case_id),
        "run_id": run_id,
        "reused_hardware_run": bool(reused_hardware_run),
        "expected": expected,
        "actual": actual,
        "artifacts": [dict(x) for x in artifacts],
        "limitations": [str(x) for x in limitations],
    }


def validate_matrix(rows: Iterable[Mapping[str, Any]]) -> dict:
    rows = [dict(r) for r in rows]
    ids = set()
    errors = []
    for idx, row in enumerate(rows):
        cid = row.get("case_id")
        if not cid:
            errors.append(f"row[{idx}] missing case_id")
        elif cid in ids:
            errors.append(f"duplicate case_id: {cid}")
        else:
            ids.add(cid)

        level = row.get("evidence_level")
        if level not in EVIDENCE_LEVELS:
            errors.append(f"{cid}: invalid evidence_level {level!r}")
        if level == "real_gpu" and not row.get("run_id"):
            errors.append(f"{cid}: real_gpu row missing run_id")
        if level == "real_gpu" and row.get("historical"):
            errors.append(f"{cid}: real_gpu row cannot be historical")
        if row.get("reused_hardware_run") and level == "real_gpu":
            errors.append(
                f"{cid}: reused_hardware_run decision row must not masquerade as real_gpu"
            )
        if row.get("status") in STATUS_REAL and level not in {
            "real_gpu",
            "historical_handoff",
        }:
            errors.append(
                f"{cid}: terminal real status {row.get('status')} lacks real evidence"
            )
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "row_count": len(rows),
        "errors": errors,
        "matrix_sha256": sha256_json(rows),
    }


def _load_json(path: str | os.PathLike[str]) -> Any:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--base-sha")
    plan.add_argument("--checkpoint-handoff")
    plan.add_argument("--output", required=True)

    decision = sub.add_parser("decision")
    decision.add_argument("--baseline", required=True)
    decision.add_argument("--post", required=True)
    decision.add_argument("--output", required=True)

    check = sub.add_parser("validate-matrix")
    check.add_argument("--matrix", required=True)

    args = ap.parse_args()
    try:
        if args.cmd == "plan":
            handoff = _load_json(args.checkpoint_handoff) if args.checkpoint_handoff else None
            value = bounded_plan(base_sha=args.base_sha, checkpoint_handoff=handoff)
            atomic_json(args.output, value)
        elif args.cmd == "decision":
            baseline = observation_from_json(_load_json(args.baseline))
            post = observation_from_json(_load_json(args.post))
            value = {
                "schema": SCHEMA,
                "generated_at": _now(),
                "rows": decision_quadrants(baseline=baseline, passing_post=post),
            }
            value["payload_sha256"] = sha256_json(value)
            atomic_json(args.output, value)
        else:
            obj = _load_json(args.matrix)
            rows = obj.get("rows") if isinstance(obj, dict) else obj
            if not isinstance(rows, list):
                raise CoverageError("matrix must be a list or object with rows")
            value = validate_matrix(rows)
            print(json.dumps(value, sort_keys=True))
            return 0 if value["ok"] else 2
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2

    print(json.dumps({"ok": True, "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
