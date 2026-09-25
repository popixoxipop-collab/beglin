#!/usr/bin/env python3
"""Dry-run-only manual canary controller and kill-switch state machine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from manual_canary_contract import normalize_proposal, validate_approval


class ManualCanaryControllerError(RuntimeError):
    pass


STATES = {
    "PROPOSED",
    "SHADOW_EVIDENCE_VERIFIED",
    "AWAITING_MANUAL_APPROVAL",
    "APPROVAL_VALIDATED",
    "PREPARING_CANARY",
    "CANARY_RUNNING",
    "CANARY_PASS_REVIEW_REQUIRED",
    "ROLLBACK_REQUIRED",
    "ROLLBACK_PENDING",
    "ROLLBACK_VERIFIED",
    "ISOLATED",
}

TERMINAL = {"CANARY_PASS_REVIEW_REQUIRED", "ROLLBACK_VERIFIED", "ISOLATED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _policy_map(rows: list[dict]) -> dict[tuple[str, int], int]:
    return {
        (str(row["role"]), int(row["layer"])): int(row["n"])
        for row in rows
    }


def _require_single_target_preimage(
    *,
    baseline_policy: list[dict],
    candidate_policy: list[dict],
    target: Mapping[str, Any],
) -> None:
    before = _policy_map(baseline_policy)
    after = _policy_map(candidate_policy)
    key = (str(target["role"]), int(target["layer"]))
    before_n = int(target["before_n"])
    after_n = int(target["after_n"])
    if before.get(key) != before_n:
        raise ManualCanaryControllerError(
            f"baseline preimage mismatch for {key}: expected n={before_n} actual={before.get(key)}"
        )
    if after.get(key) != after_n:
        raise ManualCanaryControllerError(
            f"candidate target mismatch for {key}: expected n={after_n} actual={after.get(key)}"
        )
    differing = {
        k for k in set(before) | set(after)
        if before.get(k) != after.get(k)
    }
    if differing != {key}:
        raise ManualCanaryControllerError(
            "manual canary candidate must change exactly one role/layer"
        )


@dataclass
class DryRunAdapter:
    """In-memory adapter used only to validate the manual-canary state machine."""

    policy: list[dict]
    epoch: int
    policy_hash: str
    fail_apply: bool = False
    fail_restore: bool = False
    fail_sync: bool = False

    def query(self) -> dict:
        return {
            "policy": json.loads(json.dumps(self.policy)),
            "epoch": int(self.epoch),
            "policy_hash": str(self.policy_hash),
        }

    def apply_candidate(self, *, policy: list[dict], policy_hash: str) -> dict:
        if self.fail_apply:
            raise ManualCanaryControllerError("injected dry-run apply failure")
        self.policy = json.loads(json.dumps(policy))
        self.policy_hash = str(policy_hash)
        self.epoch += 1
        return self.query()

    def synchronize(self) -> None:
        if self.fail_sync:
            raise ManualCanaryControllerError("injected dry-run synchronize failure")

    def restore_baseline(self, *, policy: list[dict], policy_hash: str, txn_id: str) -> dict:
        if self.fail_restore:
            raise ManualCanaryControllerError("injected dry-run restore failure")
        previous_epoch = int(self.epoch)
        self.policy = json.loads(json.dumps(policy))
        self.policy_hash = str(policy_hash)
        self.epoch += 1
        return {
            **self.query(),
            "txn_id": str(txn_id),
            "previous_epoch": previous_epoch,
        }


class ManualCanaryStore:
    def __init__(self, root: str | Path, run_id: str):
        self.root = Path(root) / str(run_id)
        self.state_path = self.root / "state.json"
        self.journal_path = self.root / "events.jsonl"
        self.nonce_path = self.root.parent / "consumed_nonces.json"
        self.kill_path = self.root / "kill_switch.json"

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with open(path) as f:
            return json.load(f)

    def state(self) -> dict | None:
        return self._read_json(self.state_path, None)

    def consumed_nonces(self) -> set[str]:
        obj = self._read_json(self.nonce_path, {"nonces": []})
        return {str(x) for x in obj.get("nonces", [])}

    def consume_nonce(self, nonce: str) -> None:
        values = self.consumed_nonces()
        if nonce in values:
            raise ManualCanaryControllerError("approval nonce already consumed")
        values.add(nonce)
        _atomic_json(self.nonce_path, {
            "schema": "manual-canary-consumed-nonces-v1",
            "nonces": sorted(values),
            "updated_at": _now(),
        })

    def transition(self, state: str, **payload: Any) -> dict:
        if state not in STATES:
            raise ManualCanaryControllerError(f"unsupported state: {state}")
        value = {
            "schema": "manual-canary-state-v1",
            "state": state,
            "production_write_allowed": False,
            "updated_at": _now(),
            **payload,
        }
        _atomic_json(self.state_path, value)
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(value, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return value

    def request_kill(self, *, reason: str, requested_by: str) -> dict:
        if not reason or not requested_by:
            raise ManualCanaryControllerError("kill switch requires reason and requested_by")
        value = {
            "schema": "manual-canary-kill-v1",
            "requested": True,
            "reason": str(reason),
            "requested_by": str(requested_by),
            "requested_at": _now(),
            "production_write_allowed": False,
        }
        _atomic_json(self.kill_path, value)
        return value

    def kill_requested(self) -> bool:
        return bool(self._read_json(self.kill_path, {}).get("requested"))


class ManualCanaryController:
    """Dry-run controller with strict state ordering and no production adapter."""

    def __init__(
        self,
        *,
        store: ManualCanaryStore,
        adapter: DryRunAdapter,
        proposal: Mapping[str, Any],
        baseline_policy: list[dict],
        candidate_policy: list[dict],
    ):
        self.store = store
        self.adapter = adapter
        self.proposal = normalize_proposal(proposal)
        self.baseline_policy = json.loads(json.dumps(baseline_policy))
        self.candidate_policy = json.loads(json.dumps(candidate_policy))
        _require_single_target_preimage(
            baseline_policy=self.baseline_policy,
            candidate_policy=self.candidate_policy,
            target=self.proposal["single_target"],
        )
        if _sha256_json(self.baseline_policy) != self.proposal["baseline_policy_hash"]:
            raise ManualCanaryControllerError("baseline_policy_hash does not match baseline policy")
        if _sha256_json(self.candidate_policy) != self.proposal["candidate_policy_hash"]:
            raise ManualCanaryControllerError("candidate_policy_hash does not match candidate policy")

    def _require_state(self, *allowed: str) -> dict:
        state = self.store.state()
        actual = None if state is None else state.get("state")
        if actual not in allowed:
            raise ManualCanaryControllerError(
                f"invalid state transition: current={actual!r} required={allowed}"
            )
        return state

    def initialize(self) -> dict:
        if self.store.state() is not None:
            raise ManualCanaryControllerError("controller run is already initialized")
        return self.store.transition(
            "PROPOSED",
            proposal_id=self.proposal["proposal_id"],
            proposal_digest=_sha256_json(self.proposal),
            target=self.proposal["single_target"],
            baseline_policy_hash=self.proposal["baseline_policy_hash"],
            candidate_policy_hash=self.proposal["candidate_policy_hash"],
        )

    def verify_shadow_evidence(self, evidence: Mapping[str, Any]) -> dict:
        self._require_state("PROPOSED")
        refs = {(r["kind"], r["run_id"], r["sha256"]) for r in self.proposal["evidence_refs"]}
        rows = evidence.get("refs", [])
        supplied = {
            (str(r["kind"]), str(r["run_id"]), str(r["sha256"]).lower())
            for r in rows
        }
        if refs != supplied or len(rows) != len(refs):
            raise ManualCanaryControllerError("shadow evidence references do not match proposal")
        if evidence.get("production_write_allowed") is not False:
            raise ManualCanaryControllerError("shadow evidence must be production-write disabled")
        if evidence.get("g4_status") != "PASS" or evidence.get("g6_status") != "PASS":
            raise ManualCanaryControllerError("G4 and G6 must both be PASS before approval")
        return self.store.transition(
            "SHADOW_EVIDENCE_VERIFIED",
            proposal_id=self.proposal["proposal_id"],
            evidence_ref_count=len(refs),
        )

    def await_approval(self) -> dict:
        self._require_state("SHADOW_EVIDENCE_VERIFIED")
        return self.store.transition(
            "AWAITING_MANUAL_APPROVAL",
            proposal_id=self.proposal["proposal_id"],
        )

    def validate_manual_approval(self, approval: Mapping[str, Any], *, now: datetime) -> dict:
        self._require_state("AWAITING_MANUAL_APPROVAL")
        _, normalized = validate_approval(
            proposal=self.proposal,
            approval=approval,
            now=now,
            consumed_nonces=self.store.consumed_nonces(),
        )
        self.store.consume_nonce(normalized["nonce"])
        return self.store.transition(
            "APPROVAL_VALIDATED",
            approval_id=normalized["approval_id"],
            approval_issuer=normalized["issuer"],
            approval_nonce=normalized["nonce"],
            signature_status=normalized["signature_status"],
        )

    def prepare_canary(self) -> dict:
        self._require_state("APPROVAL_VALIDATED")
        if self.store.kill_requested():
            return self.store.transition("ISOLATED", reason="kill switch requested before canary")
        actual = self.adapter.query()
        if actual["epoch"] != self.proposal["expected_epoch"]:
            raise ManualCanaryControllerError(
                f"baseline epoch drift: expected={self.proposal['expected_epoch']} actual={actual['epoch']}"
            )
        if actual["policy_hash"] != self.proposal["baseline_policy_hash"]:
            raise ManualCanaryControllerError("runtime baseline policy drift")
        return self.store.transition(
            "PREPARING_CANARY",
            expected_epoch=self.proposal["expected_epoch"],
            runtime_policy_hash=actual["policy_hash"],
        )

    def start_canary(self) -> dict:
        self._require_state("PREPARING_CANARY")
        if self.store.kill_requested():
            return self.store.transition("ISOLATED", reason="kill switch requested before candidate apply")
        try:
            applied = self.adapter.apply_candidate(
                policy=self.candidate_policy,
                policy_hash=self.proposal["candidate_policy_hash"],
            )
        except Exception as exc:
            return self.store.transition("ISOLATED", reason=f"candidate apply failed: {exc}")
        if applied["policy_hash"] != self.proposal["candidate_policy_hash"]:
            return self.store.transition("ISOLATED", reason="candidate self-report policy mismatch")
        if int(applied["epoch"]) <= int(self.proposal["expected_epoch"]):
            return self.store.transition("ISOLATED", reason="candidate epoch did not advance")
        return self.store.transition(
            "CANARY_RUNNING",
            candidate_epoch=applied["epoch"],
            candidate_policy_hash=applied["policy_hash"],
        )

    def observe(
        self,
        *,
        requests: int,
        tokens: int,
        duration_ms: int,
        memory_bytes: int,
        regression: bool,
        inconclusive: bool = False,
    ) -> dict:
        self._require_state("CANARY_RUNNING")
        metrics = {
            "requests": int(requests),
            "tokens": int(tokens),
            "duration_ms": int(duration_ms),
            "memory_bytes": int(memory_bytes),
        }
        if any(v < 0 for v in metrics.values()):
            raise ManualCanaryControllerError("observation metrics must be non-negative")
        budget = self.proposal["budget"]
        exceeded = [
            name
            for name, key in (
                ("requests", "max_requests"),
                ("tokens", "max_tokens"),
                ("duration_ms", "max_duration_ms"),
                ("memory_bytes", "max_memory_bytes"),
            )
            if metrics[name] > budget[key]
        ]
        if self.store.kill_requested():
            return self.store.transition(
                "ROLLBACK_REQUIRED",
                reason="kill switch requested during canary",
                metrics=metrics,
            )
        if exceeded:
            return self.store.transition(
                "ROLLBACK_REQUIRED",
                reason="budget exceeded: " + ",".join(exceeded),
                metrics=metrics,
            )
        if regression:
            return self.store.transition("ROLLBACK_REQUIRED", reason="regression detected", metrics=metrics)
        if inconclusive:
            return self.store.transition("ROLLBACK_REQUIRED", reason="inconclusive canary", metrics=metrics)
        return self.store.transition(
            "CANARY_PASS_REVIEW_REQUIRED",
            reason="dry-run canary met bounded observation criteria",
            metrics=metrics,
        )

    def rollback(self, *, txn_id: str) -> dict:
        self._require_state("ROLLBACK_REQUIRED")
        if not txn_id:
            raise ManualCanaryControllerError("rollback requires txn_id")
        before = self.adapter.query()
        if before["policy_hash"] != self.proposal["candidate_policy_hash"]:
            return self.store.transition(
                "ISOLATED",
                reason="rollback precondition is not the approved candidate policy",
                txn_id=str(txn_id),
            )
        self.store.transition(
            "ROLLBACK_PENDING",
            txn_id=str(txn_id),
            candidate_epoch=before["epoch"],
            expected_baseline_policy_hash=self.proposal["baseline_policy_hash"],
        )
        try:
            self.adapter.synchronize()
            restored = self.adapter.restore_baseline(
                policy=self.baseline_policy,
                policy_hash=self.proposal["baseline_policy_hash"],
                txn_id=str(txn_id),
            )
        except Exception as exc:
            return self.store.transition("ISOLATED", reason=f"rollback failed: {exc}", txn_id=str(txn_id))

        if restored.get("txn_id") != str(txn_id):
            return self.store.transition("ISOLATED", reason="rollback ACK txn mismatch", txn_id=str(txn_id))
        if int(restored.get("previous_epoch", -1)) != int(before["epoch"]):
            return self.store.transition("ISOLATED", reason="rollback ACK previous_epoch mismatch", txn_id=str(txn_id))
        if int(restored.get("epoch", -1)) <= int(before["epoch"]):
            return self.store.transition("ISOLATED", reason="rollback ACK epoch did not advance", txn_id=str(txn_id))
        if restored.get("policy_hash") != self.proposal["baseline_policy_hash"]:
            return self.store.transition("ISOLATED", reason="rollback ACK policy mismatch", txn_id=str(txn_id))
        return self.store.transition(
            "ROLLBACK_VERIFIED",
            txn_id=str(txn_id),
            restored_epoch=restored["epoch"],
            restored_policy_hash=restored["policy_hash"],
        )

    def reconcile_after_restart(self) -> dict:
        state = self.store.state()
        if state is None:
            raise ManualCanaryControllerError("no durable controller state")
        if state["state"] in TERMINAL:
            return state
        if state["state"] in {
            "PROPOSED",
            "SHADOW_EVIDENCE_VERIFIED",
            "AWAITING_MANUAL_APPROVAL",
            "APPROVAL_VALIDATED",
        }:
            return state

        actual = self.adapter.query()
        if state["state"] == "ROLLBACK_PENDING" and actual["policy_hash"] == self.proposal["baseline_policy_hash"]:
            return self.store.transition(
                "ROLLBACK_VERIFIED",
                reason="restart reconciliation found exact baseline after pending rollback",
                restored_epoch=actual["epoch"],
                restored_policy_hash=actual["policy_hash"],
            )
        if actual["policy_hash"] == self.proposal["candidate_policy_hash"]:
            return self.store.transition(
                "ROLLBACK_REQUIRED",
                reason="restart reconciliation found candidate still applied",
                candidate_epoch=actual["epoch"],
            )
        if actual["policy_hash"] == self.proposal["baseline_policy_hash"]:
            return self.store.transition(
                "ISOLATED",
                reason="baseline observed without a pending rollback ACK",
                runtime_policy_hash=actual["policy_hash"],
            )
        return self.store.transition(
            "ISOLATED",
            reason="restart reconciliation found unknown runtime policy",
            runtime_policy_hash=actual["policy_hash"],
        )
