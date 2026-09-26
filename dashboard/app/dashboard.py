from fastapi import APIRouter, HTTPException

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
    DashboardSummaryView,
    EvidenceView,
    ExperimentCard,
    PolicyView,
    ProductionView,
    VariablesView,
)

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
    "/evidence/{evidence_id}",
    response_model=EvidenceView,
    operation_id="getEvidence",
)
def get_evidence(evidence_id: str) -> EvidenceView:
    try:
        return build_evidence(evidence_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown evidence id") from exc
