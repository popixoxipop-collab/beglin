from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from .journal import ensure_a0_seeded, list_events
from .models import TargetRef
from .projectors import coverage, heatmap, metrics
from .replay import ensure_replay_seeded
from .research_models import (
    ControlConsistency,
    ControlIdentity,
    ResearchComparisonEntry,
    ResearchComparisonView,
    ResearchExportView,
)

router = APIRouter(prefix="/api/v1/research", tags=["research"])


def _all_events() -> list[dict]:
    ensure_a0_seeded()
    ensure_replay_seeded()
    return list_events(after_seq=0, limit=500)


def _target(payload: dict) -> TargetRef | None:
    raw = payload.get("target")
    if not isinstance(raw, dict):
        return None
    try:
        return TargetRef(**raw)
    except Exception:
        return None


def _entry_from_event(row: dict) -> ResearchComparisonEntry | None:
    event_type = str(row["event_type"])
    payload = dict(row["payload"])
    control = ControlIdentity(
        source_commit=str(row["source_commit"]),
        binary_sha256=str(row["binary_sha256"]),
        checkpoint_sha256=str(row["checkpoint_sha256"]),
    )

    if event_type == "canary.passed":
        target = _target(payload)
        return ResearchComparisonEntry(
            event_id=str(row["event_id"]),
            experiment_id=row.get("experiment_id"),
            source_kind="A0_CANARY",
            phase="CANARY",
            status=str(payload.get("decision") or "CANARY_PASS"),
            target=target,
            target_text=f"{target.role}/L{target.layer}" if target else None,
            sample_count=int(payload.get("post_requests") or 0),
            real_gpu=True,
            decision_only=False,
            rollback_required=bool(payload.get("rollback_required", False)),
            evidence_id=payload.get("evidence_id"),
            notes="Human-signed restart canary.",
            control=control,
        )

    if event_type == "production.policy_verified":
        target = _target(payload)
        samples = sum(
            int(item.get("requests") or 0)
            for item in (payload.get("restart_verification") or [])
        )
        return ResearchComparisonEntry(
            event_id=str(row["event_id"]),
            experiment_id=row.get("experiment_id"),
            source_kind="A0_PERSISTENT",
            phase="PERSISTENT",
            status=str(payload.get("decision") or "PERSISTENT_APPLY_VERIFIED"),
            target=target,
            target_text=f"{target.role}/L{target.layer}" if target else None,
            sample_count=samples,
            real_gpu=True,
            decision_only=False,
            rollback_required=False,
            evidence_id=payload.get("evidence_id"),
            notes="Persistent cold-start policy verified by independent restart workers.",
            control=control,
        )

    if event_type == "experiment.completed":
        target = _target(payload)
        decision_only = bool(payload.get("decision_only", False))
        return ResearchComparisonEntry(
            event_id=str(row["event_id"]),
            experiment_id=row.get("experiment_id"),
            source_kind="DECISION_ONLY" if decision_only else str(payload.get("evidence_mode") or "REPLAY"),
            phase=str(payload.get("phase") or "REAL_GPU"),
            status=str(payload.get("status") or "OBSERVED"),
            target=target,
            target_text=(
                f"{target.role}/L{target.layer}" if target
                else payload.get("target_text")
            ),
            sample_count=(
                int(payload["sample_count"])
                if payload.get("sample_count") is not None
                else None
            ),
            real_gpu=bool(payload.get("real_gpu", False)),
            decision_only=decision_only,
            rollback_required=bool(payload.get("rollback_required", False)),
            evidence_id=payload.get("evidence_id"),
            evidence_mode=payload.get("evidence_mode"),
            notes=payload.get("notes"),
            control=control,
            raw_matrix_sha256=payload.get("raw_matrix_sha256"),
        )

    if event_type == "experiment.decision_evaluated":
        return ResearchComparisonEntry(
            event_id=str(row["event_id"]),
            experiment_id=row.get("experiment_id"),
            source_kind="DECISION_ONLY" if payload.get("decision_only") else str(payload.get("evidence_mode") or "DECISION"),
            phase="POST_MORTEM",
            status=str(payload.get("status") or "OBSERVED"),
            target=None,
            target_text=payload.get("target_text"),
            sample_count=None,
            real_gpu=bool(payload.get("real_gpu", False)),
            decision_only=bool(payload.get("decision_only", False)),
            rollback_required=bool(payload.get("rollback_required", False)),
            evidence_id=payload.get("evidence_id"),
            evidence_mode=payload.get("evidence_mode"),
            notes=payload.get("notes"),
            control=control,
            raw_matrix_sha256=payload.get("raw_matrix_sha256"),
        )

    return None


def comparison_view() -> ResearchComparisonView:
    entries = [
        entry for row in _all_events()
        if (entry := _entry_from_event(row)) is not None
    ]
    commits = sorted({entry.control.source_commit for entry in entries})
    binaries = sorted({entry.control.binary_sha256 for entry in entries})
    checkpoints = sorted({entry.control.checkpoint_sha256 for entry in entries})
    return ResearchComparisonView(
        generated_at=datetime.now(timezone.utc).isoformat(),
        entries=entries,
        control_consistency=ControlConsistency(
            same_source_commit=len(commits) <= 1,
            same_binary_sha256=len(binaries) <= 1,
            same_checkpoint_sha256=len(checkpoints) <= 1,
            distinct_source_commits=commits,
            distinct_binary_sha256=binaries,
            distinct_checkpoint_sha256=checkpoints,
        ),
    )


def export_view() -> ResearchExportView:
    comparison = comparison_view()
    hm = heatmap(window="CUMULATIVE")
    cov = coverage()
    met = metrics()
    columns = [
        "event_id",
        "experiment_id",
        "source_kind",
        "phase",
        "status",
        "role",
        "layer",
        "n",
        "sample_count",
        "real_gpu",
        "decision_only",
        "rollback_required",
        "source_commit",
        "binary_sha256",
        "checkpoint_sha256",
        "evidence_id",
    ]
    csv_rows: list[dict] = []
    for entry in comparison.entries:
        csv_rows.append(
            {
                "event_id": entry.event_id,
                "experiment_id": entry.experiment_id,
                "source_kind": entry.source_kind,
                "phase": entry.phase,
                "status": entry.status,
                "role": entry.target.role if entry.target else None,
                "layer": entry.target.layer if entry.target else None,
                "n": entry.target.n if entry.target else None,
                "sample_count": entry.sample_count,
                "real_gpu": entry.real_gpu,
                "decision_only": entry.decision_only,
                "rollback_required": entry.rollback_required,
                "source_commit": entry.control.source_commit,
                "binary_sha256": entry.control.binary_sha256,
                "checkpoint_sha256": entry.control.checkpoint_sha256,
                "evidence_id": entry.evidence_id,
            }
        )
    svg_ready = [
        {
            "layer": cell.layer,
            "role_group": cell.role_group,
            "sample_count": cell.sample_count,
            "promotion_hits": cell.promotion_hits,
            "correction_hits": cell.correction_hits,
            "precision_n": cell.current_precision_n,
            "live_state": cell.live_state,
        }
        for cell in hm.cells
    ]
    return ResearchExportView(
        generated_at=datetime.now(timezone.utc).isoformat(),
        comparison=comparison,
        heatmap=hm,
        coverage=cov,
        metrics=met,
        csv_columns=columns,
        csv_rows=csv_rows,
        svg_ready_heatmap=svg_ready,
        notes=[
            "CSV rows are flattened from canonical comparison entries.",
            "SVG-ready heatmap data contains no fabricated activation magnitude.",
            "NO_BAD_CANDIDATE_WITHIN_BUDGET is preserved as source status and is never relabeled PASS.",
        ],
    )


@router.get(
    "/comparison",
    response_model=ResearchComparisonView,
    operation_id="getResearchComparison",
)
def get_research_comparison() -> ResearchComparisonView:
    return comparison_view()


@router.get(
    "/export",
    response_model=ResearchExportView,
    operation_id="getResearchExport",
)
def get_research_export() -> ResearchExportView:
    return export_view()
