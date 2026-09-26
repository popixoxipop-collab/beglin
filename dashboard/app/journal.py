from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evidence import build_canary, build_production, evidence_root

DB_PATH = Path(os.environ.get("BEGLIN_DASHBOARD_DB", str(Path(__file__).resolve().parents[1] / ".data" / "dashboard.db"))).expanduser().resolve()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    with _connect() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS event_journal (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                experiment_id TEXT,
                session_id TEXT,
                source_commit TEXT NOT NULL,
                binary_sha256 TEXT NOT NULL,
                checkpoint_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_journal_type_seq "
            "ON event_journal(event_type, seq)"
        )


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def append_event(
    *,
    event_id: str,
    event_type: str,
    occurred_at: str,
    experiment_id: str | None,
    source_commit: str,
    binary_sha256: str,
    checkpoint_sha256: str,
    payload: dict[str, Any],
    session_id: str | None = None,
) -> int:
    init_db()
    payload_json = _stable_json(payload)
    payload_sha256 = hashlib.sha256(payload_json.encode()).hexdigest()
    ingested_at = datetime.now(timezone.utc).isoformat()
    with _connect() as db:
        db.execute(
            """
            INSERT OR IGNORE INTO event_journal (
                event_id, event_type, occurred_at, ingested_at,
                experiment_id, session_id, source_commit,
                binary_sha256, checkpoint_sha256, payload_json, payload_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                occurred_at,
                ingested_at,
                experiment_id,
                session_id,
                source_commit,
                binary_sha256,
                checkpoint_sha256,
                payload_json,
                payload_sha256,
            ),
        )
        row = db.execute(
            "SELECT seq FROM event_journal WHERE event_id = ?", (event_id,)
        ).fetchone()
    if row is None:
        raise RuntimeError("event insert/readback failed")
    return int(row["seq"])


def list_events(after_seq: int = 0, limit: int = 100) -> list[dict[str, Any]]:
    init_db()
    bounded = max(1, min(int(limit), 500))
    with _connect() as db:
        rows = db.execute(
            """
            SELECT * FROM event_journal
            WHERE seq > ?
            ORDER BY seq ASC
            LIMIT ?
            """,
            (int(after_seq), bounded),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        out.append(item)
    return out


def ensure_a0_seeded() -> None:
    canary = build_canary()
    production = build_production()
    source_commit = production.evidence.content.get("source_commit") if hasattr(production.evidence, "content") else None

    # Source identity comes from the exact A0 evidence-bound views.
    from .evidence import build_summary

    summary = build_summary()
    append_event(
        event_id=f"a0:{canary.run_id}:canary.passed",
        event_type="canary.passed",
        occurred_at=canary.finished_at,
        experiment_id=canary.run_id,
        source_commit=summary.source_commit,
        binary_sha256=summary.binary_sha256,
        checkpoint_sha256=summary.checkpoint_sha256,
        payload={
            "target": canary.target.model_dump(),
            "before_n": canary.before_n,
            "pre_requests": canary.pre_requests,
            "post_requests": canary.post_requests,
            "attribution_before": canary.attribution_before,
            "attribution_after": canary.attribution_after,
            "target_replay_pass": canary.target_replay_pass,
            "rollback_required": canary.rollback_required,
            "auto_expand": canary.auto_expand,
            "decision": canary.decision,
            "evidence_id": canary.evidence.evidence_id,
            "evidence_sha256": canary.evidence.sha256,
        },
    )
    append_event(
        event_id=f"a0:{production.run_id}:production.policy_verified",
        event_type="production.policy_verified",
        occurred_at=production.applied_at,
        experiment_id=production.run_id,
        source_commit=summary.source_commit,
        binary_sha256=summary.binary_sha256,
        checkpoint_sha256=summary.checkpoint_sha256,
        payload={
            "target": production.active_policy[0].model_dump(),
            "active_policy": [row.model_dump() for row in production.active_policy],
            "restart_verification": [
                row.model_dump() for row in production.restart_verification
            ],
            "decision": production.decision,
            "rollback_ready": production.rollback.ready,
            "rollback_mode": production.rollback.mode,
            "resident_worker": production.resident_worker,
            "evidence_id": production.evidence.evidence_id,
            "evidence_sha256": production.evidence.sha256,
        },
    )
