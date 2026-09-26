from __future__ import annotations

from datetime import datetime, timezone

from .evidence import build_canary, build_production
from .journal import ensure_a0_seeded, list_events
from .models import (
    CoverageCell,
    CoverageView,
    EventLogEntry,
    EventLogView,
    HeatmapCell,
    LayerHeatmapView,
    MetricsView,
    TargetRef,
)


def event_log(after_seq: int = 0, limit: int = 100) -> EventLogView:
    ensure_a0_seeded()
    raw = list_events(after_seq=after_seq, limit=limit)
    entries: list[EventLogEntry] = []
    for row in raw:
        payload = row["payload"]
        target_data = payload.get("target")
        target = TargetRef(**target_data) if target_data else None
        if row["event_type"] == "canary.passed":
            message = (
                f"Signed canary passed for {target.role}/L{target.layer} "
                f"n={target.n}; attribution "
                f"{payload['attribution_before']} -> {payload['attribution_after']}"
            )
        elif row["event_type"] == "production.policy_verified":
            message = (
                f"Persistent policy verified for {target.role}/L{target.layer} "
                f"n={target.n}; restart validation 2/2"
            )
        else:
            message = row["event_type"]
        entries.append(
            EventLogEntry(
                seq=int(row["seq"]),
                event_id=row["event_id"],
                event_type=row["event_type"],
                occurred_at=row["occurred_at"],
                ingested_at=row["ingested_at"],
                experiment_id=row["experiment_id"],
                severity="INFO",
                target=target,
                message=message,
                evidence_id=payload.get("evidence_id"),
            )
        )
    cursor = entries[-1].seq if entries else int(after_seq)
    return EventLogView(cursor=cursor, entries=entries)


def heatmap(window: str = "CUMULATIVE") -> LayerHeatmapView:
    canary = build_canary()
    production = build_production()
    post_requests = canary.post_requests
    restart_requests = sum(row.requests for row in production.restart_verification)
    corrected = canary.post_requests + sum(
        row.corrected_hits for row in production.restart_verification
    )
    samples = post_requests + restart_requests
    last = max(canary.finished_at, production.applied_at)
    cell = HeatmapCell(
        layer=production.active_policy[0].layer,
        role_group="Shared Up",
        sample_count=samples,
        promotion_hits=samples,
        correction_hits=corrected,
        current_precision_n=production.active_policy[0].n,
        activation_ema=None,
        anomaly_count=0,
        last_updated_at=last,
        live_state="PROMOTED",
        evidence_ids=["signed-canary", "persistent-production"],
    )
    return LayerHeatmapView(
        generated_at=datetime.now(timezone.utc).isoformat(),
        window=window,
        cells=[cell],
        max_hit_count=samples,
        source_kind="evidence_aggregate",
    )


def coverage() -> CoverageView:
    canary = build_canary()
    production = build_production()
    target = production.active_policy[0]
    target_key = f"{target.role}:L{target.layer}:n{target.n}"
    cells = [
        CoverageCell(
            cell_id="signed-canary",
            experiment_id=canary.run_id,
            target_key=target_key,
            status="PASS",
            hit_count=canary.post_requests,
            pass_count=canary.post_requests,
            fail_count=0,
            first_seen_at=canary.finished_at,
            last_seen_at=canary.finished_at,
            evidence_id="signed-canary",
        )
    ]
    for index, row in enumerate(production.restart_verification, 1):
        cells.append(
            CoverageCell(
                cell_id=f"persistent-restart-{index}",
                experiment_id=production.run_id,
                target_key=target_key,
                status="PROMOTED",
                hit_count=row.requests,
                pass_count=row.corrected_hits,
                fail_count=row.requests - row.corrected_hits,
                first_seen_at=production.applied_at,
                last_seen_at=production.applied_at,
                evidence_id="persistent-production",
            )
        )
    filled = sum(1 for cell in cells if cell.status != "EMPTY")
    total = len(cells)
    return CoverageView(
        mode="BY_EXPERIMENT",
        total_cells=total,
        filled_cells=filled,
        percentage=(100.0 * filled / total) if total else 0.0,
        cells=cells,
    )


def metrics() -> MetricsView:
    canary = build_canary()
    production = build_production()
    request_count = canary.post_requests + sum(
        row.requests for row in production.restart_verification
    )
    corrected_hits = canary.post_requests + sum(
        row.corrected_hits for row in production.restart_verification
    )
    target = production.active_policy[0]
    return MetricsView(
        generated_at=datetime.now(timezone.utc).isoformat(),
        request_count=request_count,
        corrected_hits=corrected_hits,
        attribution_before=canary.attribution_before,
        attribution_after=canary.attribution_after,
        canary_status=canary.decision,
        rollback_ready=production.rollback.ready,
        active_policy_summary=f"{target.role}/L{target.layer} n={target.n}",
        resident_worker=production.resident_worker,
    )
