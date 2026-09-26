from fastapi import APIRouter, HTTPException, Query

from .evidence import (
    build_canary,
    build_evidence,
    build_experiments,
    build_policy,
    build_production,
    build_summary,
    build_variables,
)
from .models import (
    CanaryView,
    CoverageView,
    DashboardSummaryView,
    EvidenceView,
    EventLogView,
    ExperimentCard,
    LayerHeatmapView,
    MetricsView,
    PolicyView,
    ProductionView,
    VariablesView,
)
from .projectors import coverage, event_log, heatmap, metrics

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


@router.get(
    "/dashboard/summary",
    response_model=DashboardSummaryView,
    operation_id="getDashboardSummary",
)
def get_dashboard_summary() -> DashboardSummaryView:
    return build_summary()


@router.get(
    "/dashboard/variables",
    response_model=VariablesView,
    operation_id="getDashboardVariables",
)
def get_dashboard_variables() -> VariablesView:
    return build_variables()


@router.get(
    "/experiments",
    response_model=list[ExperimentCard],
    operation_id="listExperiments",
)
def list_experiments() -> list[ExperimentCard]:
    return build_experiments()


@router.get(
    "/policy",
    response_model=PolicyView,
    operation_id="getPolicy",
)
def get_policy() -> PolicyView:
    return build_policy()


@router.get(
    "/canary",
    response_model=CanaryView,
    operation_id="getCanary",
)
def get_canary() -> CanaryView:
    return build_canary()


@router.get(
    "/production",
    response_model=ProductionView,
    operation_id="getProduction",
)
def get_production() -> ProductionView:
    return build_production()


@router.get(
    "/events",
    response_model=EventLogView,
    operation_id="getEvents",
)
def get_events(
    after_seq: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> EventLogView:
    return event_log(after_seq=after_seq, limit=limit)


@router.get(
    "/heatmap",
    response_model=LayerHeatmapView,
    operation_id="getHeatmap",
)
def get_heatmap(
    window: str = Query(default="CUMULATIVE"),
) -> LayerHeatmapView:
    allowed = {"REALTIME", "1M", "5M", "1H", "SESSION", "CUMULATIVE"}
    if window not in allowed:
        raise HTTPException(status_code=422, detail="unsupported heatmap window")
    return heatmap(window=window)


@router.get(
    "/coverage",
    response_model=CoverageView,
    operation_id="getCoverage",
)
def get_coverage() -> CoverageView:
    return coverage()


@router.get(
    "/metrics",
    response_model=MetricsView,
    operation_id="getMetrics",
)
def get_metrics() -> MetricsView:
    return metrics()


@router.get(
    "/evidence/{evidence_id}",
    response_model=EvidenceView,
    operation_id="getEvidence",
)
def get_evidence(evidence_id: str) -> EvidenceView:
    try:
        return build_evidence(evidence_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown evidence id") from exc
