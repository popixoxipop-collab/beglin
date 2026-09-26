from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .models import CoverageView, LayerHeatmapView, MetricsView, TargetRef


class ControlIdentity(BaseModel):
    source_commit: str
    binary_sha256: str
    checkpoint_sha256: str


class ResearchComparisonEntry(BaseModel):
    event_id: str
    experiment_id: str | None = None
    source_kind: str
    phase: str
    status: str
    target: TargetRef | None = None
    target_text: str | None = None
    sample_count: int | None = Field(default=None, ge=0)
    real_gpu: bool
    decision_only: bool
    rollback_required: bool
    evidence_id: str | None = None
    evidence_mode: str | None = None
    notes: str | None = None
    control: ControlIdentity
    raw_matrix_sha256: str | None = None


class ControlConsistency(BaseModel):
    same_source_commit: bool
    same_binary_sha256: bool
    same_checkpoint_sha256: bool
    distinct_source_commits: list[str]
    distinct_binary_sha256: list[str]
    distinct_checkpoint_sha256: list[str]


class ResearchComparisonView(BaseModel):
    generated_at: str
    entries: list[ResearchComparisonEntry]
    control_consistency: ControlConsistency


class ResearchExportView(BaseModel):
    schema_version: str = "beglin-research-export/1"
    generated_at: str
    comparison: ResearchComparisonView
    heatmap: LayerHeatmapView
    coverage: CoverageView
    metrics: MetricsView
    csv_columns: list[str]
    csv_rows: list[dict[str, Any]]
    svg_ready_heatmap: list[dict[str, Any]]
    notes: list[str]
