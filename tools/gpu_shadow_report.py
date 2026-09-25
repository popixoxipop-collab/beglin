#!/usr/bin/env python3
"""Build a read-only observability report for GPU production-shadow state."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

SCHEMA = "gpu-shadow-report-v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
LAUNCH_FILE = "launcher_status.json"
CYCLE_FILE = "last_cycle.json"
HISTORY_FILE = "candidate_history.json"
DISCOVERY_FILE = "discovery.json"

PASS_WORDS = {
    "PASS",
    "PASSED",
    "ADMITTED",
    "CANARY_PASS",
    "CANARY_PASS_NO_AUTO_EXPANSION",
}
FAIL_MARKERS = ("FAIL", "REJECT", "ROLLBACK", "REGRESSION", "BLOCK", "ERROR")


class ShadowReportError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_json(root: Path, relative: str):
    path = (root / relative).resolve(strict=False)
    if not (path == root or _is_within(path, root)):
        raise ShadowReportError(f"state path escapes shadow root: {relative}")
    meta = {"path": relative, "exists": path.is_file()}
    if not path.is_file():
        return None, meta
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except Exception as exc:
        meta.update({
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "parse_error": type(exc).__name__,
        })
        return None, meta
    meta.update({
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    })
    if not isinstance(value, dict):
        meta["parse_error"] = "not_object"
        return None, meta
    return value, meta


def _safe_absolute_json(root: Path, value: str | None):
    if not value:
        return None, None
    path = Path(value).expanduser().resolve(strict=False)
    if not _is_within(path, root):
        return None, {
            "path": str(path),
            "exists": path.is_file(),
            "rejected": "outside_shadow_root",
        }
    return _safe_json(root, str(path.relative_to(root)))


def _parse_time(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _duration_seconds(started, finished):
    a, b = _parse_time(started), _parse_time(finished)
    if not a or not b or b < a:
        return None
    return round((b - a).total_seconds(), 6)


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _relation(launch: dict | None, cycle: dict | None) -> str:
    if not launch or not cycle:
        return "unknown"
    launch_id = launch.get("launch_id")
    cycle_launch_id = cycle.get("launch_id")
    if launch_id and cycle_launch_id:
        return "current" if launch_id == cycle_launch_id else "previous"
    if cycle_launch_id and not launch_id:
        return "previous"
    launch_started = _parse_time(launch.get("started_at"))
    cycle_finished = _parse_time(cycle.get("finished_at"))
    if launch_started and cycle_finished and cycle_finished < launch_started:
        return "previous"
    return "legacy_unlinked"


def _raw_status(value):
    if isinstance(value, str):
        return value.upper()
    if isinstance(value, dict):
        for key in ("final_status", "status", "decision", "action", "verdict"):
            if isinstance(value.get(key), str):
                return value[key].upper()
            nested = value.get(key)
            if isinstance(nested, dict):
                got = _raw_status(nested)
                if got:
                    return got
    return None


def _stage_status(payload: dict | None, stage: str):
    if not isinstance(payload, dict):
        return None
    if stage == "g4":
        direct = ("g4_status", "preflight_status", "isolated_preflight_status")
        nested = (
            "g4",
            "g4_result",
            "preflight",
            "preflight_result",
            "isolated_preflight",
            "isolated_preflight_result",
        )
    elif stage == "g6":
        direct = ("g6_status", "canary_status", "restart_canary_status")
        nested = (
            "g6",
            "g6_result",
            "canary",
            "canary_result",
            "restart_canary",
            "restart_canary_result",
        )
    else:
        direct = ("rollback_status",)
        nested = ("rollback", "rollback_result")

    for key in direct:
        if isinstance(payload.get(key), str):
            return payload[key].upper()
    for key in nested:
        if key in payload:
            got = _raw_status(payload[key])
            if got:
                return got
    return None


def _status_is_pass(value):
    if not value:
        return False
    value = str(value).upper()
    if value in PASS_WORDS:
        return True
    return "PASS" in value and not any(marker in value for marker in FAIL_MARKERS)


def _status_is_fail(value):
    if not value:
        return False
    value = str(value).upper()
    return any(marker in value for marker in FAIL_MARKERS)


def _pipeline_state(cycle: dict | None):
    if not cycle:
        return "unknown"
    status = str(cycle.get("status") or "").upper()
    if status in {
        "NO_READY_CANDIDATE",
        "NO_NEW_READY_CANDIDATE",
        "MANUAL_REVIEW_REQUIRED",
    }:
        return "not_run"
    if status == "SHADOW_CYCLE_COMPLETE":
        return "executed"
    if status in {"SHADOW_PIPELINE_ERROR", "SHADOW_CYCLE_FAILED"}:
        return "failed"
    return "unknown"


def _dedupe_reason(cycle: dict | None):
    if not cycle:
        return None
    if cycle.get("dedupe_reason"):
        return cycle.get("dedupe_reason")
    status = str(cycle.get("status") or "").upper()
    if status == "NO_READY_CANDIDATE":
        return "no_ready_candidate"
    if status == "NO_NEW_READY_CANDIDATE":
        return "reusable_terminal_same_candidate_and_runtime"
    if status == "MANUAL_REVIEW_REQUIRED":
        return "retry_budget_exhausted"
    return None


def _history_fingerprints(history: dict | None, candidate_ids):
    result = {}
    legacy = []
    if not isinstance(history, dict):
        return result, legacy
    candidates = history.get("candidates")
    if not isinstance(candidates, dict):
        return result, legacy
    for candidate_id in candidate_ids:
        row = candidates.get(str(candidate_id))
        if not isinstance(row, dict):
            continue
        result[str(candidate_id)] = {
            "candidate_fingerprint": row.get("fingerprint"),
            "validation_context_fingerprint": row.get("runtime_identity_sha256"),
            "shadow_status": row.get("shadow_status"),
            "reusable_terminal": row.get("reusable_terminal"),
            "attempt_count": row.get("attempt_count"),
            "launch_id": row.get("launch_id"),
            "cycle_id": row.get("cycle_id"),
            "shadow_run_id": row.get("shadow_run_id"),
        }
        if not row.get("runtime_identity_sha256"):
            legacy.append(str(candidate_id))
    return result, legacy


def build_report(shadow_root: str | Path, *, pid_alive_fn=_pid_alive) -> dict:
    root = Path(shadow_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ShadowReportError(f"shadow root is not a directory: {root}")

    launch, launch_meta = _safe_json(root, LAUNCH_FILE)
    cycle, cycle_meta = _safe_json(root, CYCLE_FILE)
    history, history_meta = _safe_json(root, HISTORY_FILE)
    discovery, discovery_meta = _safe_json(root, DISCOVERY_FILE)

    relation = _relation(launch, cycle)
    launcher_state = str((launch or {}).get("status") or "UNKNOWN").upper()
    pid = (launch or {}).get("pid") or (launch or {}).get("worker_pid")
    pid_alive = pid_alive_fn(pid) if pid is not None else None

    pipeline_state = _pipeline_state(cycle)
    cycle_status = (cycle or {}).get("status")
    shadow_run = (cycle or {}).get("shadow_run")
    if not isinstance(shadow_run, dict):
        shadow_run = None
    shadow_run_id = (
        (cycle or {}).get("shadow_run_id")
        or (shadow_run or {}).get("run_id")
        or (launch or {}).get("shadow_run_id")
    )

    run_result = None
    run_result_meta = None
    if shadow_run_id and SAFE_ID.fullmatch(str(shadow_run_id)):
        run_result, run_result_meta = _safe_json(
            root, f"executions/runs/{shadow_run_id}/shadow_result.json"
        )
    if run_result is None and shadow_run is not None:
        run_result = shadow_run

    child_payload = (run_result or {}).get("child_payload")
    if not isinstance(child_payload, dict):
        child_payload = None

    g4_status = _stage_status(child_payload, "g4")
    g6_status = _stage_status(child_payload, "g6")
    rollback_status = _stage_status(child_payload, "rollback")
    final_status = _raw_status(child_payload)
    shadow_status = (run_result or {}).get("shadow_status") or (shadow_run or {}).get(
        "shadow_status"
    )

    if pipeline_state == "not_run":
        gpu_validation_state = "not_run"
    elif pipeline_state == "failed":
        gpu_validation_state = "failed"
    elif _status_is_fail(g4_status) or _status_is_fail(g6_status):
        gpu_validation_state = "failed"
    elif _status_is_pass(g4_status) and _status_is_pass(g6_status):
        gpu_validation_state = "passed"
    else:
        gpu_validation_state = "unknown"

    if relation == "current" and gpu_validation_state in {"passed", "failed"}:
        evidence_level = "real_gpu"
    elif pipeline_state == "not_run":
        evidence_level = "not_run"
    elif relation == "legacy_unlinked":
        evidence_level = "legacy_unverified"
    elif run_result:
        evidence_level = "shadow_terminal"
    else:
        evidence_level = "unknown"

    if relation == "current":
        source_freshness = "current"
    elif relation == "previous":
        source_freshness = "stale_previous_cycle"
    elif relation == "legacy_unlinked":
        source_freshness = "legacy_unlinked"
    else:
        source_freshness = "unknown"

    selected = (cycle or {}).get("selected_candidate_id")
    candidate_ids = []
    if selected:
        candidate_ids.append(selected)
    for key in (
        "already_observed_candidate_ids",
        "manual_review_candidate_ids",
        "retrying_candidate_ids",
    ):
        for item in (cycle or {}).get(key) or []:
            if item not in candidate_ids:
                candidate_ids.append(item)
    fingerprints, legacy_candidates = _history_fingerprints(history, candidate_ids)

    spec, spec_meta = _safe_absolute_json(root, (cycle or {}).get("candidate_spec"))
    runtime_identity = (cycle or {}).get("runtime_identity")
    if not isinstance(runtime_identity, dict):
        runtime_identity = {}

    binary_sha256 = (spec or {}).get("binary_sha256") or runtime_identity.get(
        "binary_sha256"
    )
    checkpoint_sha256 = (spec or {}).get("checkpoint_sha256") or runtime_identity.get(
        "checkpoint_sha256"
    )
    checkpoint_identity = ((spec or {}).get("source") or {}).get(
        "checkpoint_identity"
    )
    checkpoint_manifest_ref = (
        "candidate_spec:source.checkpoint_identity"
        if isinstance(checkpoint_identity, dict)
        else None
    )

    artifacts = [
        meta
        for meta in (
            launch_meta,
            cycle_meta,
            history_meta,
            discovery_meta,
            spec_meta,
            run_result_meta,
        )
        if meta and meta.get("exists")
    ]

    warnings = []
    if relation == "previous":
        warnings.append("PREVIOUS_CYCLE_NOT_CURRENT_LAUNCH")
    if relation == "legacy_unlinked":
        warnings.append("LEGACY_CYCLE_WITHOUT_LAUNCH_ID")
    if launcher_state in {"PREPARING", "RUNNING"} and pid_alive is False:
        warnings.append("LAUNCHER_STATE_STALE_PROCESS_NOT_ALIVE")
    if legacy_candidates:
        warnings.append("LEGACY_CONTEXT_UNVERIFIED")
    if pipeline_state == "executed" and gpu_validation_state == "unknown":
        warnings.append("GPU_STAGE_STATUS_NOT_EXPLICIT")
    if cycle and cycle.get("production_write_allowed") is not False:
        warnings.append("PRODUCTION_WRITE_FLAG_NOT_FALSE")
    if run_result and run_result.get("production_write_allowed") is not False:
        warnings.append("SHADOW_RESULT_WRITE_FLAG_NOT_FALSE")

    report = {
        "schema": SCHEMA,
        "observed_at": _now(),
        "production_write_allowed": False,
        "source_freshness": source_freshness,
        "evidence_level": evidence_level,
        "launcher": {
            "state": launcher_state,
            "launch_id": (launch or {}).get("launch_id"),
            "pid": pid,
            "pid_alive": pid_alive,
            "returncode": (launch or {}).get("returncode"),
            "started_at": (launch or {}).get("started_at"),
            "finished_at": (launch or {}).get("finished_at"),
            "duration_seconds": _duration_seconds(
                (launch or {}).get("started_at"),
                (launch or {}).get("finished_at"),
            ),
        },
        "cycle": {
            "relation_to_launcher": relation,
            "status": cycle_status,
            "pipeline_state": pipeline_state,
            "launch_id": (cycle or {}).get("launch_id"),
            "cycle_id": (cycle or {}).get("cycle_id"),
            "shadow_run_id": shadow_run_id,
            "started_at": (cycle or {}).get("started_at"),
            "finished_at": (cycle or {}).get("finished_at"),
            "duration_seconds": _duration_seconds(
                (cycle or {}).get("started_at"),
                (cycle or {}).get("finished_at"),
            ),
            "candidate_count": (discovery or {}).get("candidate_count"),
            "ready_count": (cycle or {}).get("ready_count")
            if cycle
            else (discovery or {}).get("ready_count"),
            "selected_candidate": selected,
            "deferred_candidate_ids": (cycle or {}).get(
                "deferred_candidate_ids", []
            ),
            "dedupe_reason": _dedupe_reason(cycle),
            "manual_review_candidate_ids": (cycle or {}).get(
                "manual_review_candidate_ids", []
            ),
            "retrying_candidate_ids": (cycle or {}).get(
                "retrying_candidate_ids", []
            ),
        },
        "identity": {
            "validation_context_fingerprint": (cycle or {}).get(
                "runtime_identity_sha256"
            ),
            "binary_sha256": binary_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_manifest_ref": checkpoint_manifest_ref,
            "candidate_fingerprints": fingerprints,
        },
        "gpu_validation": {
            "state": gpu_validation_state,
            "shadow_status": shadow_status,
            "final_status": final_status,
            "g4_status": g4_status,
            "g6_status": g6_status,
            "rollback_status": rollback_status,
        },
        "artifacts": artifacts,
        "warnings": sorted(set(warnings)),
        "unknown_fields": [],
    }

    for dotted, value in (
        ("launcher.launch_id", report["launcher"]["launch_id"]),
        ("cycle.cycle_id", report["cycle"]["cycle_id"]),
        ("cycle.shadow_run_id", report["cycle"]["shadow_run_id"]),
        (
            "identity.validation_context_fingerprint",
            report["identity"]["validation_context_fingerprint"],
        ),
        ("identity.binary_sha256", report["identity"]["binary_sha256"]),
        ("identity.checkpoint_sha256", report["identity"]["checkpoint_sha256"]),
        ("gpu_validation.g4_status", report["gpu_validation"]["g4_status"]),
        ("gpu_validation.g6_status", report["gpu_validation"]["g6_status"]),
    ):
        if value is None:
            report["unknown_fields"].append(dotted)

    report["report_sha256"] = hashlib.sha256(
        _canonical_json(report).encode()
    ).hexdigest()
    return report


def render_markdown(report: dict) -> str:
    launch = report["launcher"]
    cycle = report["cycle"]
    gpu = report["gpu_validation"]
    identity = report["identity"]
    lines = [
        "# GPU Shadow Observability Report",
        "",
        "observed_at: " + str(report["observed_at"]),
        "source_freshness: " + str(report["source_freshness"]),
        "evidence_level: " + str(report["evidence_level"]),
        "production_write_allowed: " + str(report["production_write_allowed"]).lower(),
        "",
        "## Launcher",
        "state: " + str(launch["state"]),
        "launch_id: " + str(launch["launch_id"]),
        "pid_alive: " + str(launch["pid_alive"]),
        "returncode: " + str(launch["returncode"]),
        "",
        "## Cycle",
        "relation: " + str(cycle["relation_to_launcher"]),
        "status: " + str(cycle["status"]),
        "pipeline_state: " + str(cycle["pipeline_state"]),
        "cycle_id: " + str(cycle["cycle_id"]),
        "shadow_run_id: " + str(cycle["shadow_run_id"]),
        "ready_count: " + str(cycle["ready_count"]),
        "selected_candidate: " + str(cycle["selected_candidate"]),
        "dedupe_reason: " + str(cycle["dedupe_reason"]),
        "",
        "## Runtime identity",
        "validation_context_fingerprint: "
        + str(identity["validation_context_fingerprint"]),
        "binary_sha256: " + str(identity["binary_sha256"]),
        "checkpoint_sha256: " + str(identity["checkpoint_sha256"]),
        "",
        "## GPU validation",
        "state: " + str(gpu["state"]),
        "shadow_status: " + str(gpu["shadow_status"]),
        "final_status: " + str(gpu["final_status"]),
        "G4: " + str(gpu["g4_status"]),
        "G6: " + str(gpu["g6_status"]),
        "rollback: " + str(gpu["rollback_status"]),
    ]
    if report["warnings"]:
        lines += ["", "## Warnings"] + ["- " + x for x in report["warnings"]]
    if report["unknown_fields"]:
        lines += ["", "## Unknown fields"] + [
            "- " + x for x in report["unknown_fields"]
        ]
    lines += ["", "report_sha256: " + str(report["report_sha256"]), ""]
    return "\n".join(lines)


def _validate_output(path: str | None, shadow_root: Path) -> Path | None:
    if not path:
        return None
    root = Path(shadow_root).expanduser().resolve(strict=False)
    target = Path(path).expanduser().resolve(strict=False)
    if target == root or _is_within(target, root):
        raise ShadowReportError(
            "report output must be outside the observed shadow root"
        )
    return target


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shadow-root", required=True)
    ap.add_argument("--json-out")
    ap.add_argument("--markdown-out")
    args = ap.parse_args()
    try:
        root = Path(args.shadow_root).expanduser().resolve(strict=False)
        json_out = _validate_output(args.json_out, root)
        markdown_out = _validate_output(args.markdown_out, root)
        report = build_report(root)
        md = render_markdown(report)
        if json_out:
            _atomic_text(
                json_out, json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
        if markdown_out:
            _atomic_text(markdown_out, md)
    except Exception as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "status": "REPORT_ERROR",
            "production_write_allowed": False,
            "error_type": type(exc).__name__,
        }, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
