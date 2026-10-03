#!/usr/bin/env python3
"""Tamper-evident request lineage and aggregate metrics for precision closed-loop serving."""
from __future__ import annotations

import hashlib
import copy
import fcntl
import math
import uuid
from collections import Counter
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


def build_explicit_policy_decision(*, target_policy: list[dict], changes: list[dict]) -> dict:
    policy = pc.normalize_policy(target_policy)
    selection = {
        "targets": [
            {
                "role": row["role"],
                "layer": int(row["layer"]),
                "status": "EXPLICIT_POLICY",
                "selected_n": int(row["n"]),
                "active_triggers": [],
                "evidence": {},
            }
            for row in policy
        ]
    }
    return {
        "schema": "beglin-precision-explicit-lineage-decision-v1",
        "status": "READY_FOR_OBSERVABILITY",
        "production_write_allowed": False,
        "signal": {
            "active_triggers": [],
            "margin": None,
            "entropy": None,
            "routing_ambiguity_score": None,
            "raw": {},
        },
        "evidence_snapshot_sha256": None,
        "allocation_sha256": None,
        "selection_sha256": _sha(selection),
        "cost_evidence_sha256": None,
        "combined_policy_evidence": None,
        "selected_policy": policy,
        "changes": list(changes),
        "selection": selection,
        "policy_cost_optimizer": None,
    }


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
    incoming_signal = decision.get("signal") or {}
    # Never persist arbitrary caller fields or raw prompt/token content.
    signal = {k: incoming_signal.get(k) for k in (
        "active_triggers", "margin", "entropy", "routing_ambiguity_score"
    )}
    signal["active_triggers"] = sorted(set(signal.get("active_triggers") or []))
    responses = result.get("responses") or []
    outcome_payload = {
        "finite_logits": bool(result.get("finite_logits")),
        "responses": responses,
    }
    combined = decision.get("combined_policy_evidence")
    record = {
        "schema": SCHEMA,
        "schema_revision": 2,
        "admission_path": result.get("admission_path", "unspecified"),
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
            **{k: transition.get(k) for k in (
                "transition_wall_ms", "cache_hits", "cache_misses",
                "cache_bytes_added", "resident_cache_bytes"
            )},
            **{k: epoch.get(k, result.get(k)) for k in (
                "inference_passes", "engine_wall_ms", "roundtrip_ms"
            )},
        },
        "outcome": {
            "status": result.get("lineage_outcome", "SUCCESS"),
            "error_type": result.get("lineage_error_type"),
            "finite_logits": result.get("finite_logits"),
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
        _validate_record_values(raw)
        if idx > 0:
            _validate_continuity(records[idx - 1], raw)
        prev = expected


def _validate_record_values(row):
    if not row.get("admission_id"):
        raise PrecisionObservabilityError("admission_id must be nonempty")
    if type(row.get("request_count")) is not int or row["request_count"] < 0:
        raise PrecisionObservabilityError("request_count must be a nonnegative integer")
    for name, value in (row.get("actual_cost") or {}).items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PrecisionObservabilityError(f"invalid actual cost: {name}")
        if not math.isfinite(value) or value < 0:
            raise PrecisionObservabilityError(f"invalid actual cost: {name}")
    for name in ("inference_passes", "cache_hits", "cache_misses", "cache_bytes_added", "resident_cache_bytes"):
        value = (row.get("actual_cost") or {}).get(name)
        if value is not None and type(value) is not int:
            raise PrecisionObservabilityError(f"integer metric required: {name}")
    # allow_nan=False also validates signal, expected-cost and evidence fields.
    try:
        json.dumps(row, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise PrecisionObservabilityError("record is not finite JSON") from exc


def _validate_continuity(prior, row):
    if prior is None:
        return
    prior_id = prior.get("worker_instance_id")
    current_id = row.get("worker_instance_id")
    same_worker = (prior_id == current_id if prior_id and current_id else
                   prior.get("worker_pid") is not None and
                   prior.get("worker_pid") == row.get("worker_pid"))
    if same_worker:
        for old, new in (("after_policy_hash", "before_policy_hash"),
                         ("after_epoch", "before_epoch")):
            if prior.get(old) is not None and row.get(new) is not None:
                if prior[old] != row[new]:
                    raise PrecisionObservabilityError("unobserved policy/epoch transition")


class _Metrics:
    """Incremental accumulator. No request/response payloads retained in memory."""
    def __init__(self):
        self.counts = Counter()
        self.triggers = Counter()
        self.transitions = Counter()
        self.policies = Counter()
        self.precisions = Counter()
        self.passes = Counter()
        self.evidence = Counter()
        self.outcomes = Counter()
        self.missing = Counter()
        self.paths = Counter()
        self.resident_current = None
        self.resident_max = None
        self.head = None

    def add(self, row):
        c = self.counts
        c['admissions'] += 1
        c['requests'] += row.get('request_count', 0)
        status = (row.get('outcome') or {}).get('status', 'SUCCESS')
        self.outcomes[status] += 1
        self.paths[row.get('admission_path', 'unspecified')] += 1
        active = set(row.get('active_triggers') or [])
        c['triggered'] += bool(active)
        self.triggers.update(active)
        c['transitioned'] += bool(row.get('transitioned'))
        for r in row.get('changes') or []:
            self.transitions[f"{r.get('role')}/L{r.get('layer')}:{r.get('from_n')}->{r.get('to_n')}"] += 1
        if status == 'SUCCESS':
            ph = row.get('after_policy_hash')
            if ph is not None:
                self.policies[ph] += 1
            for r in row.get('selected_policy') or []:
                self.precisions[f"{r.get('role')}/L{r.get('layer')}/n{r.get('n')}"] += 1
        for target in row.get('target_decisions') or []:
            for records in (target.get('evidence') or {}).values():
                for ev in records or []:
                    key = ev.get('evidence_id') or ev.get('evidence_sha256')
                    if key:
                        self.evidence[key] += 1
        cost = row.get('actual_cost') or {}
        for key in ('cache_hits', 'cache_misses', 'cache_bytes_added'):
            v = cost.get(key)
            if v is not None:
                c[key] += v
            else:
                self.missing[key] += 1
        resident = cost.get('resident_cache_bytes')
        if resident is not None:
            self.resident_current = resident
            self.resident_max = max(self.resident_max or 0, resident)
        passes = cost.get('inference_passes')
        if passes is not None:
            c['pass_measured'] += passes > 0
            self.passes[str(passes)] += 1
            c['extra_pass'] += passes > 1
        else:
            self.missing['inference_passes'] += 1
        finite = (row.get('outcome') or {}).get('finite_logits')
        if finite is not None:
            c['finite_measured'] += 1
            c['finite'] += finite is True
        actual = cost.get('roundtrip_ms')
        if actual is not None:
            c['actual_n'] += 1
            c['actual_sum'] += actual
        expected = row.get('expected_cost')
        if isinstance(expected, dict) and expected.get('scope') == 'policy':
            ex = expected.get('expected_e2e_ms')
            if ex is not None:
                c['expected_n'] += 1
                c['expected_sum'] += ex
                if actual is not None:
                    c['error_n'] += 1
                    c['error_sum'] += actual - ex
        self.head = row['record_sha256']

    def summary(self):
        c = self.counts
        n = c['admissions']
        success = self.outcomes['SUCCESS']
        ratio = lambda num, den: num / den if den else None
        mean = lambda key: ratio(c[key + '_sum'], c[key + '_n'])
        ordered = lambda x: dict(sorted(x.items()))
        return {
            'schema': SUMMARY_SCHEMA, 'status': 'PASS',
            'admissions': n, 'requests': c['requests'],
            'outcome_counts': ordered(self.outcomes),
            'admission_path_counts': ordered(self.paths),
            'successful_admissions': success,
            'rejected_admissions': self.outcomes['REJECTED'],
            'error_admissions': self.outcomes['ERROR'],
            'triggered_admissions': c['triggered'],
            'trigger_rate': ratio(c['triggered'], n),
            'trigger_counts': ordered(self.triggers),
            'trigger_rates': {k: ratio(v, n) for k,v in sorted(self.triggers.items())},
            'transitioned_admissions': c['transitioned'],
            'transition_rate': ratio(c['transitioned'], n),
            'target_transition_counts': ordered(self.transitions),
            'cache_hits': c['cache_hits'], 'cache_misses': c['cache_misses'],
            'cache_hit_rate': ratio(c['cache_hits'], c['cache_hits'] + c['cache_misses']),
            'cache_bytes_added': c['cache_bytes_added'],
            'resident_cache_bytes_current': self.resident_current,
            'resident_cache_bytes_max': self.resident_max,
            'inference_pass_histogram': ordered(self.passes),
            'extra_pass_admissions': c['extra_pass'],
            'extra_pass_rate': ratio(c['extra_pass'], c['pass_measured']),
            'finite_logits_admissions': c['finite'],
            'finite_logits_rate': ratio(c['finite'], c['finite_measured']),
            'policy_residency_admissions': ordered(self.policies),
            'policy_residency_share': {k: ratio(v, success) for k,v in sorted(self.policies.items())},
            'precision_residency_admissions': ordered(self.precisions),
            'precision_residency_share': {k: ratio(v, success) for k,v in sorted(self.precisions.items())},
            'evidence_use_counts': ordered(self.evidence),
            'expected_e2e_ms_mean': mean('expected'),
            'actual_roundtrip_ms_mean': mean('actual'),
            'e2e_error_ms_mean': mean('error'),
            'missing_measurement_counts': ordered(self.missing),
            'denominators': {'triggers': n, 'residency': success,
                             'extra_pass': c['pass_measured'], 'finite_logits': c['finite_measured'],
                             'cache_hit': c['cache_hits'] + c['cache_misses']},
            'residency_unit': 'successful_admission_end_state_not_wall_time',
            'lineage_head_sha256': self.head,
        }


def summarize(records: list[dict]) -> dict:
    verify_records(records)
    metrics = _Metrics()
    for row in records:
        _validate_record_values(row)
        metrics.add(row)
    return metrics.summary()


class PrecisionObservability:
    def __init__(self, *, lineage_path, snapshot_path=None, strict=False, identity=None, max_bytes=0, max_rotated_files=0):
        self.lineage_path = Path(lineage_path)
        self.snapshot_path = (Path(snapshot_path) if snapshot_path is not None else
                              self.lineage_path.with_suffix(self.lineage_path.suffix + '.summary.json'))
        if self.lineage_path.resolve() == self.snapshot_path.resolve():
            raise PrecisionObservabilityError('snapshot path must differ from lineage path')
        self.strict = bool(strict)
        self.identity = dict(identity or {})
        self.max_bytes = max(0, int(max_bytes))
        self.max_rotated_files = max(0, int(max_rotated_files))
        self.worker_instance_id = self.identity.get('worker_instance_id') or uuid.uuid4().hex
        self.lock = threading.RLock()
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.lineage_path.with_suffix(self.lineage_path.suffix + '.lock')
        self._fingerprint = None
        self._seen = set()
        self._last = None
        self._metrics = _Metrics()
        self.last_error_type = None
        self.sink_error_count = 0
        with self.lock, self._file_lock():
            self._reload()

    def _file_lock(self):
        from contextlib import contextmanager
        @contextmanager
        def locked():
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        return locked()

    def _stat(self):
        try:
            st = self.lineage_path.stat()
            return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
        except FileNotFoundError:
            return None

    def _reload(self):
        records = self.read_records()
        verify_records(records)
        metrics = _Metrics()
        for row in records:
            _validate_record_values(row)
            metrics.add(row)
        self._seen = {r['admission_id'] for r in records}
        self._last = records[-1] if records else None
        self._metrics = metrics
        self._fingerprint = self._stat()

    def read_records(self):
        if not self.lineage_path.is_file():
            return []
        rows = []
        with self.lineage_path.open('rb') as handle:
            for lineno, raw in enumerate(handle, 1):
                if not raw.endswith(b'\n'):
                    raise PrecisionObservabilityError(f'incomplete lineage tail at line {lineno}')
                if not raw.strip():
                    continue
                try:
                    rows.append(json.loads(raw))
                except (ValueError, UnicodeError) as exc:
                    raise PrecisionObservabilityError(f'invalid lineage JSON at line {lineno}') from exc
        return rows

    def _write_snapshot(self, summary):
        tmp = self.snapshot_path.with_name(self.snapshot_path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as handle:
                handle.write(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.snapshot_path)
        finally:
            tmp.unlink(missing_ok=True)

    def _rotate_if_needed(self, incoming_bytes):
        if self.max_bytes <= 0:
            return None
        current = self.lineage_path.stat().st_size if self.lineage_path.exists() else 0
        if current == 0 or current + int(incoming_bytes) <= self.max_bytes:
            return None
        stamp = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        rotated = self.lineage_path.with_name(
            self.lineage_path.name + f".{stamp}.rotated"
        )
        os.replace(self.lineage_path, rotated)
        self._seen = set()
        self._last = None
        self._metrics = _Metrics()
        self._fingerprint = None
        files = sorted(
            self.lineage_path.parent.glob(self.lineage_path.name + ".*.rotated"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        if self.max_rotated_files >= 0:
            for old in files[self.max_rotated_files:]:
                old.unlink(missing_ok=True)
        return rotated

    def record(self, *, admission_id, worker_pid, request_count, decision, result):
        started = time.monotonic()
        with self.lock, self._file_lock():
            if self._stat() != self._fingerprint:
                self._reload()
            if str(admission_id) in self._seen:
                raise PrecisionObservabilityError(f'duplicate admission_id: {admission_id}')
            row = build_record(admission_id=admission_id, worker_pid=worker_pid,
                               request_count=request_count, decision=decision, result=result,
                               prev_record_sha256=self._last['record_sha256'] if self._last else None)
            row['worker_instance_id'] = self.worker_instance_id
            row['runtime_identity'] = self.identity
            row.pop('record_sha256')
            row['record_sha256'] = _sha(row)
            _validate_record_values(row)
            _validate_continuity(self._last, row)
            # Validate aggregation BEFORE durable append: invalid input must not poison the chain.
            updated = copy.deepcopy(self._metrics)
            updated.add(row)
            summary = updated.summary()
            raw = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
            rotated = self._rotate_if_needed(len(raw))
            if rotated is not None:
                row['prev_record_sha256'] = None
                row.pop('record_sha256', None)
                row['record_sha256'] = _sha(row)
                updated = _Metrics()
                updated.add(row)
                summary = updated.summary()
                raw = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
            fd = os.open(self.lineage_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(fd, raw[offset:])
                    if written <= 0:
                        raise PrecisionObservabilityError('lineage append made no progress')
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)
            self._seen.add(str(admission_id))
            self._last = row
            self._metrics = updated
            self._fingerprint = self._stat()
            snapshot_status = 'CURRENT'
            try:
                self._write_snapshot(summary)
            except Exception as exc:
                self.last_error_type = type(exc).__name__
                self.sink_error_count += 1
                snapshot_status = 'ERROR'
                if self.strict:
                    raise
            return {'schema': 'beglin-precision-observability-record-v1',
                    'status': 'RECORDED', 'record_sha256': row['record_sha256'],
                    'lineage_head_sha256': summary['lineage_head_sha256'],
                    'admissions': summary['admissions'], 'snapshot_path': str(self.snapshot_path),
                    'snapshot_status': snapshot_status,
                    'recording_wall_ms': (time.monotonic() - started) * 1000.0}

    def summary(self, *, verify=True):
        with self.lock, self._file_lock():
            if verify or self._stat() != self._fingerprint:
                self._reload()
            return self._metrics.summary()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify and summarize a precision lineage journal (read-only)")
    parser.add_argument("--lineage", required=True)
    args = parser.parse_args()
    path = Path(args.lineage)
    records = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise PrecisionObservabilityError("incomplete lineage tail")
            if line.strip():
                records.append(json.loads(line))
    print(json.dumps(summarize(records), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
