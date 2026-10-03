#!/usr/bin/env python3
"""Tamper-evident request lineage and aggregate metrics for precision closed-loop serving."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time

import precision_context as pc


SCHEMA = "beglin-precision-lineage-v1"
SUMMARY_SCHEMA = "beglin-precision-observability-summary-v1"


class PrecisionObservabilityError(RuntimeError):
    pass


def _sha(value: dict) -> str:
    return hashlib.sha256(pc.canonical_json(value).encode()).hexdigest()


def _evidence_trace(decision: dict) -> list[dict]:
    out = []
    selection = decision.get("selection") or {}
    for target in selection.get("targets", []):
        row = {
            "role": target.get("role"),
            "layer": target.get("layer"),
            "status": target.get("status"),
            "selected_n": target.get("selected_n"),
            "active_triggers": list(target.get("active_triggers") or []),
            "evidence": {},
        }
        evidence = target.get("evidence") or {}
        if isinstance(evidence, dict):
            for trigger, rows in sorted(evidence.items()):
                compact = []
                for item in rows or []:
                    if not isinstance(item, dict):
                        continue
                    compact.append({
                        "evidence_id": item.get("evidence_id"),
                        "evidence_sha256": item.get("evidence_sha256"),
                        "trigger_type": item.get("trigger_type"),
                        "from_n": item.get("from_n"),
                        "to_n": item.get("to_n"),
                        "requests": item.get("requests"),
                        "signal_bucket": item.get("signal_bucket"),
                    })
                row["evidence"][trigger] = compact
        out.append(row)
    return out


def _expected_cost(decision: dict) -> dict | None:
    policy = decision.get("policy_cost_optimizer")
    if isinstance(policy, dict):
        selected = policy.get("selected_cost")
        if isinstance(selected, dict):
            return {
                "scope": "policy",
                "expected_e2e_ms": selected.get("expected_e2e_ms"),
                "transition_p50_ms": selected.get("transition_p50_ms"),
                "steady_roundtrip_p50_ms": selected.get("steady_roundtrip_p50_ms"),
                "expected_inference_passes": selected.get("expected_inference_passes"),
                "resident_cache_bytes_after": selected.get("resident_cache_bytes_after"),
                "cache_state": selected.get("cache_state"),
                "active_weight_bytes": selected.get("active_weight_bytes"),
            }
    rows = []
    for target in (decision.get("selection") or {}).get("targets", []):
        cost = target.get("cost")
        if isinstance(cost, dict) and any(v is not None for v in cost.values()):
            rows.append({
                "role": target.get("role"),
                "layer": target.get("layer"),
                "selected_n": target.get("selected_n"),
                **cost,
            })
    if rows:
        return {"scope": "targets", "targets": rows}
    return None


def build_adaptive_decision(
    *,
    adaptive: dict,
    active_policy: list[dict],
    changes: list[dict],
    evidence_sha256: str,
) -> dict:
    """Normalize legacy adaptive L26 serving into the P6 lineage shape."""
    if not isinstance(adaptive, dict) or adaptive.get("enabled") is not True:
        raise PrecisionObservabilityError("adaptive precision metadata is invalid")
    action = str(adaptive.get("action", ""))
    triggered = action == "RECOVERY_N6"
    events = adaptive.get("base_events") or []
    margins = []
    for row in events:
        if isinstance(row, dict) and row.get("margin") is not None:
            try:
                margins.append(float(row["margin"]))
            except (TypeError, ValueError):
                pass
    margin = min(margins) if margins else None
    role = str(adaptive.get("role"))
    layer = int(adaptive.get("layer"))
    selected_n = int(adaptive.get("worker_left_at_n"))
    status = "TRIGGER_CONDITIONED_ALTERNATE" if triggered else "BASE_LOW_COST"
    evidence = {}
    if triggered:
        evidence = {
            "low_margin": [{
                "evidence_id": "adaptive-l26-certified",
                "evidence_sha256": evidence_sha256,
                "trigger_type": "low_margin",
                "from_n": int(adaptive.get("base_n")),
                "to_n": int(adaptive.get("recovery_n")),
                "requests": len(adaptive.get("trigger_request_indices") or []),
                "signal_bucket": {"margin_max": 0.02},
            }]
        }
    selection = {
        "targets": [{
            "role": role,
            "layer": layer,
            "status": status,
            "selected_n": selected_n,
            "active_triggers": ["low_margin"] if triggered else [],
            "evidence": evidence,
        }]
    }
    signal = {
        "active_triggers": ["low_margin"] if triggered else [],
        "margin": margin,
        "entropy": None,
        "routing_ambiguity_score": None,
        "raw": {"low_margin": triggered, "margin": margin},
    }
    return {
        "schema": "beglin-precision-adaptive-lineage-decision-v1",
        "status": "READY_FOR_OBSERVABILITY",
        "production_write_allowed": False,
        "signal": signal,
        "evidence_snapshot_sha256": evidence_sha256,
        "allocation_sha256": None,
        "selection_sha256": _sha(selection),
        "cost_evidence_sha256": None,
        "combined_policy_evidence": {"evidence_sha256": evidence_sha256},
        "selected_policy": pc.normalize_policy(active_policy),
        "changes": list(changes),
        "selection": selection,
        "policy_cost_optimizer": None,
    }


def build_record(
    *,
    admission_id: str,
    worker_pid: int | None,
    request_count: int,
    decision: dict,
    result: dict,
    prev_record_sha256: str | None,
) -> dict:
    epoch = result.get("precision_epoch") or {}
    transition = epoch.get("transition_cost") or {}
    signal = decision.get("signal") or {}
    responses = result.get("responses") or []
    outcome_payload = {
        "finite_logits": bool(result.get("finite_logits")),
        "responses": responses,
    }
    combined = decision.get("combined_policy_evidence")
    record = {
        "schema": SCHEMA,
        "ts_unix_ns": time.time_ns(),
        "admission_id": str(admission_id),
        "worker_pid": int(worker_pid) if worker_pid is not None else None,
        "request_count": int(request_count),
        "prev_record_sha256": prev_record_sha256,
        "signal": signal,
        "active_triggers": list(signal.get("active_triggers") or []),
        "evidence_snapshot_sha256": decision.get("evidence_snapshot_sha256"),
        "allocation_sha256": decision.get("allocation_sha256"),
        "selection_sha256": decision.get("selection_sha256"),
        "cost_evidence_sha256": decision.get("cost_evidence_sha256"),
        "combined_policy_evidence_sha256": (
            combined.get("evidence_sha256") if isinstance(combined, dict) else None
        ),
        "target_decisions": _evidence_trace(decision),
        "before_policy_hash": epoch.get("before_policy_hash"),
        "after_policy_hash": epoch.get("after_policy_hash"),
        "before_epoch": epoch.get("before_epoch"),
        "after_epoch": epoch.get("after_epoch"),
        "selected_policy": decision.get("selected_policy"),
        "changes": decision.get("changes") or [],
        "transitioned": bool(epoch.get("transitioned")),
        "expected_cost": _expected_cost(decision),
        "actual_cost": {
            "transition_wall_ms": float(transition.get("transition_wall_ms", 0.0)),
            "cache_hits": int(transition.get("cache_hits", 0)),
            "cache_misses": int(transition.get("cache_misses", 0)),
            "cache_bytes_added": int(transition.get("cache_bytes_added", 0)),
            "resident_cache_bytes": int(transition.get("resident_cache_bytes", 0)),
            "inference_passes": int(epoch.get("inference_passes", result.get("inference_passes", 1))),
            "engine_wall_ms": float(epoch.get("engine_wall_ms", result.get("engine_wall_ms", 0.0))),
            "roundtrip_ms": float(epoch.get("roundtrip_ms", result.get("roundtrip_ms", 0.0))),
        },
        "outcome": {
            "finite_logits": bool(result.get("finite_logits")),
            "response_count": len(responses),
            "responses_sha256": _sha(outcome_payload),
        },
    }
    record["record_sha256"] = _sha(record)
    return record


def verify_records(records: list[dict]) -> None:
    prev = None
    seen_admissions = set()
    for idx, raw in enumerate(records):
        if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
            raise PrecisionObservabilityError(f"invalid lineage record at index {idx}")
        admission = raw.get("admission_id")
        if admission in seen_admissions:
            raise PrecisionObservabilityError(f"duplicate admission_id: {admission}")
        seen_admissions.add(admission)
        if raw.get("prev_record_sha256") != prev:
            raise PrecisionObservabilityError(f"lineage chain break at index {idx}")
        expected = raw.get("record_sha256")
        body = dict(raw)
        body.pop("record_sha256", None)
        actual = _sha(body)
        if expected != actual:
            raise PrecisionObservabilityError(f"lineage record hash mismatch at index {idx}")
        prev = expected


def summarize(records: list[dict]) -> dict:
    verify_records(records)
    trigger_counts: dict[str, int] = {}
    target_transition_counts: dict[str, int] = {}
    policy_admissions: dict[str, int] = {}
    precision_residency: dict[str, int] = {}
    pass_hist: dict[str, int] = {}
    evidence_counts: dict[str, int] = {}
    cache_hits = cache_misses = cache_bytes_added = 0
    transitioned = triggered = extra_pass = finite = 0
    resident_current = resident_max = 0
    expected_e2e = []
    actual_e2e = []
    e2e_error = []
    total_requests = 0

    for row in records:
        total_requests += int(row.get("request_count", 0))
        active = row.get("active_triggers") or []
        if active:
            triggered += 1
        for trigger in active:
            trigger_counts[str(trigger)] = trigger_counts.get(str(trigger), 0) + 1
        if row.get("transitioned"):
            transitioned += 1
        for change in row.get("changes") or []:
            key = (
                f"{change.get('role')}/L{change.get('layer')}:"
                f"{change.get('from_n')}->{change.get('to_n')}"
            )
            target_transition_counts[key] = target_transition_counts.get(key, 0) + 1
        ph = str(row.get("after_policy_hash"))
        policy_admissions[ph] = policy_admissions.get(ph, 0) + 1
        for policy_row in row.get("selected_policy") or []:
            key = f"{policy_row.get('role')}/L{policy_row.get('layer')}/n{policy_row.get('n')}"
            precision_residency[key] = precision_residency.get(key, 0) + 1
        for target in row.get("target_decisions") or []:
            for rows in (target.get("evidence") or {}).values():
                for evidence in rows or []:
                    eid = evidence.get("evidence_id") or evidence.get("evidence_sha256")
                    if eid:
                        evidence_counts[str(eid)] = evidence_counts.get(str(eid), 0) + 1

        actual = row.get("actual_cost") or {}
        hits = int(actual.get("cache_hits", 0))
        misses = int(actual.get("cache_misses", 0))
        cache_hits += hits
        cache_misses += misses
        cache_bytes_added += int(actual.get("cache_bytes_added", 0))
        resident_current = int(actual.get("resident_cache_bytes", 0))
        resident_max = max(resident_max, resident_current)
        passes = int(actual.get("inference_passes", 1))
        pass_hist[str(passes)] = pass_hist.get(str(passes), 0) + 1
        if passes > 1:
            extra_pass += 1
        if (row.get("outcome") or {}).get("finite_logits") is True:
            finite += 1
        actual_ms = float(actual.get("roundtrip_ms", 0.0))
        actual_e2e.append(actual_ms)
        expected = row.get("expected_cost")
        if isinstance(expected, dict) and expected.get("scope") == "policy":
            value = expected.get("expected_e2e_ms")
            if value is not None:
                value = float(value)
                expected_e2e.append(value)
                e2e_error.append(actual_ms - value)

    admissions = len(records)
    denom_cache = cache_hits + cache_misses
    policy_share = {
        key: count / admissions if admissions else 0.0
        for key, count in sorted(policy_admissions.items())
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "PASS",
        "admissions": admissions,
        "requests": total_requests,
        "triggered_admissions": triggered,
        "trigger_rate": triggered / admissions if admissions else 0.0,
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "trigger_rates": {
            key: count / admissions if admissions else 0.0
            for key, count in sorted(trigger_counts.items())
        },
        "transitioned_admissions": transitioned,
        "transition_rate": transitioned / admissions if admissions else 0.0,
        "target_transition_counts": dict(sorted(target_transition_counts.items())),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / denom_cache if denom_cache else None,
        "cache_bytes_added": cache_bytes_added,
        "resident_cache_bytes_current": resident_current,
        "resident_cache_bytes_max": resident_max,
        "inference_pass_histogram": dict(sorted(pass_hist.items())),
        "extra_pass_admissions": extra_pass,
        "extra_pass_rate": extra_pass / admissions if admissions else 0.0,
        "finite_logits_admissions": finite,
        "finite_logits_rate": finite / admissions if admissions else 0.0,
        "policy_residency_admissions": dict(sorted(policy_admissions.items())),
        "policy_residency_share": policy_share,
        "precision_residency_admissions": dict(sorted(precision_residency.items())),
        "precision_residency_share": {
            key: count / admissions if admissions else 0.0
            for key, count in sorted(precision_residency.items())
        },
        "evidence_use_counts": dict(sorted(evidence_counts.items())),
        "expected_e2e_ms_mean": (
            sum(expected_e2e) / len(expected_e2e) if expected_e2e else None
        ),
        "actual_roundtrip_ms_mean": (
            sum(actual_e2e) / len(actual_e2e) if actual_e2e else None
        ),
        "e2e_error_ms_mean": (
            sum(e2e_error) / len(e2e_error) if e2e_error else None
        ),
        "lineage_head_sha256": records[-1]["record_sha256"] if records else None,
    }


class PrecisionObservability:
    def __init__(
        self,
        *,
        lineage_path: str | Path,
        snapshot_path: str | Path | None = None,
        strict: bool = False,
    ):
        self.lineage_path = Path(lineage_path)
        self.snapshot_path = (
            Path(snapshot_path) if snapshot_path is not None
            else self.lineage_path.with_suffix(self.lineage_path.suffix + ".summary.json")
        )
        self.strict = bool(strict)
        self.lock = threading.Lock()
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        records = self.read_records()
        verify_records(records)

    def read_records(self) -> list[dict]:
        if not self.lineage_path.is_file():
            return []
        rows = []
        for lineno, raw in enumerate(self.lineage_path.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise PrecisionObservabilityError(
                    f"invalid JSON lineage line {lineno}"
                ) from exc
        return rows

    def _write_snapshot(self, summary: dict) -> None:
        tmp = self.snapshot_path.with_name(
            self.snapshot_path.name + f".tmp.{os.getpid()}.{time.time_ns()}"
        )
        raw = json.dumps(summary, sort_keys=True, indent=2) + "\n"
        with tmp.open("w") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.snapshot_path)

    def record(
        self,
        *,
        admission_id: str,
        worker_pid: int | None,
        request_count: int,
        decision: dict,
        result: dict,
    ) -> dict:
        with self.lock:
            records = self.read_records()
            verify_records(records)
            if any(row.get("admission_id") == str(admission_id) for row in records):
                raise PrecisionObservabilityError(
                    f"duplicate admission_id: {admission_id}"
                )
            prev = records[-1]["record_sha256"] if records else None
            row = build_record(
                admission_id=admission_id,
                worker_pid=worker_pid,
                request_count=request_count,
                decision=decision,
                result=result,
                prev_record_sha256=prev,
            )
            raw = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            fd = os.open(
                self.lineage_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                payload = raw.encode()
                offset = 0
                while offset < len(payload):
                    written = os.write(fd, payload[offset:])
                    if written <= 0:
                        raise PrecisionObservabilityError(
                            "lineage append made no progress"
                        )
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)
            records.append(row)
            summary = summarize(records)
            self._write_snapshot(summary)
            return {
                "schema": "beglin-precision-observability-record-v1",
                "status": "RECORDED",
                "record_sha256": row["record_sha256"],
                "lineage_head_sha256": summary["lineage_head_sha256"],
                "admissions": summary["admissions"],
                "snapshot_path": str(self.snapshot_path),
            }

    def summary(self) -> dict:
        with self.lock:
            return summarize(self.read_records())
