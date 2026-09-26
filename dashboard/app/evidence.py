from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    CanaryView,
    ControlledVariablesSnapshot,
    DashboardSummaryView,
    EvidenceRef,
    EvidenceView,
    ExperimentCard,
    ExperimentalVariablesSnapshot,
    PolicyView,
    ProductionView,
    RestartVerification,
    RollbackView,
    TargetRef,
    VariablesView,
)

FINAL = "A0_FINAL_CERTIFICATION_STATUS_20260926.json"
CANARY = "A0_SIGNED_PRODUCTION_CANARY_EVIDENCE_20260926.json"
PRODUCTION = "A0_PERSISTENT_PRODUCTION_EVIDENCE_20260926.json"
ROLLBACK = "A0_PERSISTENT_ROLLBACK_READINESS_20260926.json"

EVIDENCE_META = {
    "a0-final": (FINAL, "CERTIFICATION_MANIFEST"),
    "signed-canary": (CANARY, "SIGNED_CANARY"),
    "persistent-production": (PRODUCTION, "PERSISTENT_APPLY"),
    "rollback-readiness": (ROLLBACK, "ROLLBACK_READINESS"),
}


def evidence_root() -> Path:
    configured = os.environ.get("BEGLIN_EVIDENCE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _path(name: str) -> Path:
    path = evidence_root() / name
    if path.is_symlink():
        raise RuntimeError(f"evidence file may not be a symlink: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"missing evidence file: {path}")
    return path


def _bytes(name: str) -> bytes:
    return _path(name).read_bytes()


def _sha(name: str) -> str:
    return hashlib.sha256(_bytes(name)).hexdigest()


def _json(name: str) -> dict[str, Any]:
    value = json.loads(_bytes(name))
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence root must be a JSON object: {name}")
    return value


def _ref(evidence_id: str, run_id: str | None = None) -> EvidenceRef:
    name, kind = EVIDENCE_META[evidence_id]
    return EvidenceRef(
        evidence_id=evidence_id,
        kind=kind,
        path=str(_path(name)),
        sha256=_sha(name),
        run_id=run_id,
    )


def _target(value: dict[str, Any], n_key: str = "n") -> TargetRef:
    return TargetRef(
        role=str(value["role"]),
        layer=int(value["layer"]),
        n=int(value[n_key]),
    )


def build_summary() -> DashboardSummaryView:
    final = _json(FINAL)
    persistent = dict(final["persistent_production_bridge"])
    policy = [_target(row) for row in persistent.get("live_policy", [])]
    return DashboardSummaryView(
        generated_at=datetime.now(timezone.utc).isoformat(),
        certification_status=str(final["status"]),
        source_commit=str(final["source_commit"]),
        binary_sha256=str(final["binary_sha256"]),
        checkpoint_sha256=str(final["checkpoint_sha256"]),
        canary_status=str(final["production_canary_bridge"]["status"]),
        production_state=str(final.get("production_state") or "UNKNOWN"),
        resident_worker=bool(persistent.get("resident_worker", False)),
        active_policy=policy,
        known_runs=2,
        production_write_allowed=bool(final.get("production_write_allowed", False)),
    )


def build_variables() -> VariablesView:
    final = _json(FINAL)
    canary = _json(CANARY)
    production = _json(PRODUCTION)
    target = dict(canary["target"])
    unsupported = [
        "model_id",
        "dataset_id",
        "prompt_set_id",
        "hardware_id",
        "backend",
        "architecture",
        "seed",
    ]
    controlled = ControlledVariablesSnapshot(
        source_commit=str(final["source_commit"]),
        source_tree=str(final["source_tree"]),
        binary_sha256=str(final["binary_sha256"]),
        checkpoint_sha256=str(final["checkpoint_sha256"]),
    )
    experimental = ExperimentalVariablesSnapshot(
        target_role=str(target["role"]),
        target_layer=int(target["layer"]),
        before_n=int(target["before_n"]),
        after_n=int(target["after_n"]),
        canary_mode="restart",
        rollback_policy=str(production["durable_rollback"]["mode"]),
        auto_expand=bool(canary["auto_expand"]),
        request_budget=int(canary["post"]["requests"]),
        token_budget=int(canary["post"]["tokens"]),
    )
    return VariablesView(
        controlled=controlled,
        experimental=experimental,
        unsupported_fields=unsupported,
    )


def build_experiments() -> list[ExperimentCard]:
    canary = _json(CANARY)
    production = _json(PRODUCTION)
    ctarget = dict(canary["target"])
    ptarget = dict(production["target"])
    return [
        ExperimentCard(
            experiment_id=str(canary["run_id"]),
            ordinal=1,
            title="Signed restart canary",
            status=str(canary["status"]),
            phase="CANARY",
            target=TargetRef(
                role=str(ctarget["role"]),
                layer=int(ctarget["layer"]),
                n=int(ctarget["after_n"]),
            ),
            signed=canary["signature"]["status"] == "VERIFIED",
            rollback_required=bool(canary["rollback_required"]),
            persistent=False,
            resident_worker=False,
            evidence_id="signed-canary",
        ),
        ExperimentCard(
            experiment_id=str(production["run_id"]),
            ordinal=2,
            title="Persistent cold-start policy",
            status=str(production["status"]),
            phase="PERSISTENT",
            target=TargetRef(
                role=str(ptarget["role"]),
                layer=int(ptarget["layer"]),
                n=int(ptarget["after_n"]),
            ),
            signed=production["signature"]["status"] == "VERIFIED",
            rollback_required=False,
            persistent=bool(production["persistent_mutation"]),
            resident_worker=bool(production["resident_worker"]),
            evidence_id="persistent-production",
        ),
    ]


def build_policy() -> PolicyView:
    final = _json(FINAL)
    persistent = dict(final["persistent_production_bridge"])
    return PolicyView(
        observed_at=str(final["observed_at"]),
        policy_path=str(persistent["production_path"]),
        policy_hash=str(persistent["live_sha256"]),
        active_targets=[_target(row) for row in persistent["live_policy"]],
        source_commit=str(final["source_commit"]),
        binary_sha256=str(final["binary_sha256"]),
        checkpoint_sha256=str(final["checkpoint_sha256"]),
        persistent=bool(persistent["persistent_mutation"]),
        resident_worker=bool(persistent["resident_worker"]),
    )


def build_canary() -> CanaryView:
    data = _json(CANARY)
    target = dict(data["target"])
    return CanaryView(
        run_id=str(data["run_id"]),
        signed=data["signature"]["status"] == "VERIFIED",
        signature_status=str(data["signature"]["status"]),
        target=TargetRef(
            role=str(target["role"]),
            layer=int(target["layer"]),
            n=int(target["after_n"]),
        ),
        before_n=int(target["before_n"]),
        candidate_policy_hash=str(data["candidate_policy_hash"]),
        pre_worker_id=str(data["runtime_preimage"]["worker_instance_id"]),
        post_worker_id=f"pid-{int(data['post']['pid'])}",
        pre_requests=int(data["pre"]["requests"]),
        post_requests=int(data["post"]["requests"]),
        attribution_before=float(data["observer"]["baseline_attribution_rate"]),
        attribution_after=float(data["observer"]["post_attribution_rate"]),
        target_replay_pass=bool(data["observer"]["target_replay_pass"]),
        rollback_required=bool(data["rollback_required"]),
        auto_expand=bool(data["auto_expand"]),
        decision=str(data["decision"]),
        finished_at=str(data["finished_at"]),
        evidence=_ref("signed-canary", str(data["run_id"])),
    )


def build_production() -> ProductionView:
    data = _json(PRODUCTION)
    rollback_data = _json(ROLLBACK)
    target = dict(data["target"])
    restart = [RestartVerification(**row) for row in data["restart_verification"]]
    rollback = RollbackView(
        ready=bool(rollback_data["rollback_cas_precondition"]),
        status=str(rollback_data["status"]),
        mode=str(rollback_data["rollback_mode"]),
        preimage_sha256=str(rollback_data["preimage_sha256"]),
        postimage_sha256=str(rollback_data["postimage_sha256"]),
        live_sha256=str(rollback_data["live_sha256"]),
        exercised=bool(rollback_data["rollback_exercised"]),
    )
    return ProductionView(
        run_id=str(data["run_id"]),
        state=str(data["status"]),
        production_path=str(data["production_path"]),
        preimage_sha256=str(data["preimage_sha256"]),
        postimage_sha256=str(data["postimage_sha256"]),
        live_sha256=str(rollback_data["live_sha256"]),
        active_policy=[_target(row) for row in data["active_policy"]],
        restart_verification=restart,
        production_write_allowed=bool(data["production_write_allowed"]),
        resident_worker=bool(data["resident_worker"]),
        service_state=str(data["service_state"]),
        decision=str(data["decision"]),
        rollback=rollback,
        applied_at=str(data["applied_at"]),
        evidence=_ref("persistent-production", str(data["run_id"])),
    )


def build_evidence(evidence_id: str) -> EvidenceView:
    if evidence_id not in EVIDENCE_META:
        raise KeyError(evidence_id)
    name, _kind = EVIDENCE_META[evidence_id]
    content = _json(name)
    run_id = content.get("run_id")
    return EvidenceView(
        evidence=_ref(evidence_id, str(run_id) if run_id else None),
        schema_name=str(content["schema"]) if content.get("schema") else None,
        status=str(content["status"]) if content.get("status") else None,
        content=content,
    )


def evidence_root_readable() -> bool:
    try:
        for name, _kind in EVIDENCE_META.values():
            _path(name)
        return True
    except (OSError, RuntimeError):
        return False
