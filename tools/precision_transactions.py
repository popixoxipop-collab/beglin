#!/usr/bin/env python3
"""Durable precision transaction state machine for G3 drain/apply/rollback."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import precision_context as pc
from backend_adapters import BackendAdapter, BackendError, StaleCommand


TERMINAL = {
    "APPLIED",
    "ROLLBACK_APPLIED",
    "FAILED_ISOLATED",
    "STALE_COMMAND",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class TransactionJournal:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, txn_id):
        safe = "".join(c for c in str(txn_id) if c.isalnum() or c in "-_.")
        if not safe or safe != str(txn_id):
            raise ValueError("invalid transaction id")
        return self.root / f"{safe}.json"

    def load(self, txn_id):
        path = self.path(txn_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def save(self, state):
        state = dict(state)
        state["updated_at"] = _now()
        _atomic_json(self.path(state["txn_id"]), state)
        return state


def _transition(journal, state, status, **updates):
    state = dict(state)
    state.update(updates)
    state["status"] = status
    history = list(state.get("history", []))
    history.append({"status": status, "at": _now()})
    state["history"] = history
    return journal.save(state)


def _new_state(
    txn_id,
    backend,
    context_hash,
    expected_epoch,
    expected_policy_hash,
    requested_policy,
):
    normalized = pc.normalize_policy(requested_policy)
    return {
        "version": 1,
        "txn_id": txn_id,
        "backend": backend,
        "context_hash": context_hash,
        "status": "REQUESTED",
        "created_at": _now(),
        "expected_epoch": int(expected_epoch),
        "expected_policy_hash": expected_policy_hash,
        "requested_policy": normalized,
        "requested_policy_hash": pc.policy_hash(normalized),
        "history": [{"status": "REQUESTED", "at": _now()}],
    }


def execute_policy_transaction(
    adapter: BackendAdapter,
    journal: TransactionJournal,
    *,
    txn_id: str,
    expected_context_hash: str,
    expected_epoch: int,
    expected_policy_hash: str,
    requested_policy,
):
    existing = journal.load(txn_id)
    if existing:
        if existing.get("status") in TERMINAL:
            return existing
        state = existing
    else:
        state = journal.save(_new_state(
            txn_id,
            adapter.name,
            expected_context_hash,
            expected_epoch,
            expected_policy_hash,
            requested_policy,
        ))

    context = adapter.collect_context()
    if context.context_hash != state["context_hash"] or context.backend != adapter.name:
        return _transition(
            journal,
            state,
            "STALE_COMMAND",
            error="execution context changed before transaction",
        )

    current = adapter.query_applied_state()
    if (
        current.epoch != int(state["expected_epoch"])
        or current.policy_hash != state["expected_policy_hash"]
    ):
        return _transition(
            journal,
            state,
            "STALE_COMMAND",
            observed_epoch=current.epoch,
            observed_policy_hash=current.policy_hash,
            error="epoch or policy preimage no longer matches",
        )

    snapshot = None
    try:
        adapter.pause_admission()
        state = _transition(journal, state, "ADMISSION_PAUSED")

        adapter.drain()
        state = _transition(journal, state, "DRAINED")

        adapter.synchronize()
        state = _transition(journal, state, "GPU_SYNCED")

        snapshot = adapter.snapshot()
        state = _transition(
            journal,
            state,
            "SNAPSHOT_DURABLE",
            rollback_snapshot=snapshot,
        )

        applied = adapter.apply_policy(state["requested_policy"])
        state = _transition(
            journal,
            state,
            "VERIFYING",
            observed_epoch=applied.epoch,
            observed_policy_hash=applied.policy_hash,
        )

        verified = adapter.verify_policy(state["requested_policy_hash"])
        if verified.active_requests != 0:
            raise BackendError("new epoch became active before durable ACK")

        # APPLIED is the durable ACK. Admission resumes only after this write succeeds.
        state = _transition(
            journal,
            state,
            "APPLIED",
            applied_epoch=verified.epoch,
            applied_policy_hash=verified.policy_hash,
            applied_at=_now(),
        )
        adapter.resume_admission()
        state = journal.save({
            **state,
            "admission_resumed_at": _now(),
        })
        return state

    except Exception as exc:
        state = _transition(
            journal,
            state,
            "ROLLBACK_REQUESTED",
            error=f"{type(exc).__name__}: {exc}",
        )
        if snapshot is None:
            # No weight mutation happened yet. Preserve paused isolation until the caller
            # explicitly reconciles the worker; never claim rollback was applied.
            return _transition(
                journal,
                state,
                "FAILED_ISOLATED",
                rollback_error="no durable snapshot available",
            )
        try:
            restored = adapter.restore(snapshot)
            expected_restore_hash = pc.policy_hash(snapshot["policy"])
            adapter.verify_policy(expected_restore_hash)
            state = _transition(
                journal,
                state,
                "ROLLBACK_APPLIED",
                rollback_epoch=restored.epoch,
                rollback_policy_hash=expected_restore_hash,
                rollback_at=_now(),
            )
            adapter.resume_admission()
            return journal.save({
                **state,
                "admission_resumed_at": _now(),
            })
        except Exception as rollback_exc:
            return _transition(
                journal,
                state,
                "FAILED_ISOLATED",
                rollback_error=(
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                ),
            )


def reconcile(adapter: BackendAdapter, journal: TransactionJournal, txn_id: str):
    state = journal.load(txn_id)
    if state is None:
        raise KeyError(txn_id)
    actual = adapter.query_applied_state()
    return {
        "txn_id": txn_id,
        "journal_status": state["status"],
        "journal_policy_hash": (
            state.get("applied_policy_hash")
            or state.get("rollback_policy_hash")
            or state.get("expected_policy_hash")
        ),
        "actual_policy_hash": actual.policy_hash,
        "actual_epoch": actual.epoch,
        "admission_paused": actual.admission_paused,
        "consistent": (
            (
                state["status"] == "APPLIED"
                and actual.policy_hash == state.get("applied_policy_hash")
            )
            or (
                state["status"] == "ROLLBACK_APPLIED"
                and actual.policy_hash == state.get("rollback_policy_hash")
            )
            or state["status"] in {"STALE_COMMAND", "FAILED_ISOLATED"}
        ),
    }
