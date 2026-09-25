#!/usr/bin/env python3
"""Generate auditable Agent-E dry-run journals; no production adapter is used."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import manual_canary_contract as mc
import manual_canary_controller as ctl


BASELINE = [{"role": "shared_down_proj", "layer": 4, "n": 5}]
CANDIDATE = [{"role": "shared_down_proj", "layer": 4, "n": 6}]
NOW = datetime(2026, 9, 26, 0, 1, tzinfo=timezone.utc)


def h(ch: str) -> str:
    return ch * 64


def proposal() -> dict:
    return {
        "mode": "dry_run",
        "production_write_allowed": False,
        "proposal_id": "agent-e-fixture",
        "proposer": "planner-agent",
        "environment_id": "fixture-env",
        "model_revision": "fixture-model",
        "backend": "mlx_metal",
        "architecture": "deepseek-v2-lite",
        "source_commit": "fixture-only",
        "binary_sha256": h("a"),
        "checkpoint_sha256": h("b"),
        "baseline_policy_hash": mc.sha256_json(BASELINE),
        "candidate_policy_hash": mc.sha256_json(CANDIDATE),
        "single_target": {
            "role": "shared_down_proj",
            "layer": 4,
            "before_n": 5,
            "after_n": 6,
        },
        "evidence_refs": [
            {"kind": "G4_A_B_R", "run_id": "g4-fixture", "sha256": h("c")},
            {"kind": "G6_RESTART_CANARY", "run_id": "g6-fixture", "sha256": h("d")},
        ],
        "budget": {
            "max_requests": 50,
            "max_tokens": 1000,
            "max_duration_ms": 10000,
            "max_memory_bytes": 1024 * 1024,
        },
        "expected_epoch": 7,
        "restart_instance_id": "worker-pre",
        "kill_switch_scope": "single-target",
        "rollback_plan": "restore exact baseline preimage",
    }


def approval(p: dict, nonce: str) -> dict:
    return {
        "approval_id": "approval-" + nonce,
        "proposal_digest": mc.proposal_digest(p),
        "issuer": "human-reviewer-fixture",
        "issued_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(minutes=10)).isoformat(),
        "nonce": nonce,
        "mode": "dry_run",
        "signature_status": "DRY_RUN_TEST_ONLY",
        "production_write_allowed": False,
    }


def evidence(p: dict) -> dict:
    row = mc.normalize_proposal(p)
    return {
        "production_write_allowed": False,
        "g4_status": "PASS",
        "g6_status": "PASS",
        "refs": row["evidence_refs"],
    }


def make_controller(root: Path, run_id: str):
    p = proposal()
    adapter = ctl.DryRunAdapter(
        policy=BASELINE.copy(),
        epoch=7,
        policy_hash=mc.sha256_json(BASELINE),
    )
    store = ctl.ManualCanaryStore(root, run_id)
    controller = ctl.ManualCanaryController(
        store=store,
        adapter=adapter,
        proposal=p,
        baseline_policy=BASELINE,
        candidate_policy=CANDIDATE,
    )
    return p, adapter, store, controller


def prepare(p, controller, nonce):
    controller.initialize()
    controller.verify_shadow_evidence(evidence(p))
    controller.await_approval()
    controller.validate_manual_approval(approval(p, nonce), now=NOW)
    controller.prepare_canary()
    controller.start_canary()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)

    happy_p, _, happy_store, happy = make_controller(root, "happy")
    prepare(happy_p, happy, "nonce-happy")
    happy_final = happy.observe(
        requests=20, tokens=500, duration_ms=5000, memory_bytes=1000,
        regression=False,
    )

    rollback_p, _, rollback_store, rollback = make_controller(root, "rollback")
    prepare(rollback_p, rollback, "nonce-rollback")
    rollback.observe(
        requests=2, tokens=20, duration_ms=50, memory_bytes=1000,
        regression=True,
    )
    rollback_final = rollback.rollback(txn_id="txn-fixture-rollback")

    summary = {
        "schema": "manual-canary-agent-e-evidence-v1",
        "production_write_allowed": False,
        "evidence_level": "simulated",
        "happy_final_state": happy_final["state"],
        "rollback_final_state": rollback_final["state"],
        "happy_journal_sha256": sha256(happy_store.journal_path),
        "rollback_journal_sha256": sha256(rollback_store.journal_path),
        "happy_state_sha256": sha256(happy_store.state_path),
        "rollback_state_sha256": sha256(rollback_store.state_path),
        "note": "fixture evidence only; not a real GPU or production canary",
    }
    out = root / "summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
