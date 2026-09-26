from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TargetRef(BaseModel):
    role: str
    layer: int = Field(ge=0)
    n: int = Field(ge=2, le=15)


class EvidenceRef(BaseModel):
    evidence_id: str
    kind: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_host: str = "eoe"
    run_id: str | None = None
    immutable: bool = True


class ControlledVariablesSnapshot(BaseModel):
    source_commit: str
    source_tree: str
    binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str | None = None
    dataset_id: str | None = None
    prompt_set_id: str | None = None
    hardware_id: str | None = None
    backend: str | None = None
    architecture: str | None = None
    seed: int | None = None
    baseline_precision_n: int = 4
    quant_format: str = "qNg64"
    group_size: int = 64


class ExperimentalVariablesSnapshot(BaseModel):
    target_role: str
    target_layer: int = Field(ge=0)
    before_n: int = Field(ge=2, le=15)
    after_n: int = Field(ge=2, le=15)
    canary_mode: str
    rollback_policy: str
    auto_expand: bool
    request_budget: int | None = None
    token_budget: int | None = None


class VariablesView(BaseModel):
    controlled: ControlledVariablesSnapshot
    experimental: ExperimentalVariablesSnapshot
    unsupported_fields: list[str]


class ExperimentCard(BaseModel):
    experiment_id: str
    ordinal: int = Field(ge=1)
    title: str
    status: str
    phase: str
    target: TargetRef
    signed: bool | None = None
    rollback_required: bool | None = None
    persistent: bool | None = None
    resident_worker: bool | None = None
    evidence_id: str


class DashboardSummaryView(BaseModel):
    generated_at: str
    certification_status: str
    source_commit: str
    binary_sha256: str
    checkpoint_sha256: str
    canary_status: str
    production_state: str
    resident_worker: bool
    active_policy: list[TargetRef]
    known_runs: int
    production_write_allowed: bool


class PolicyView(BaseModel):
    observed_at: str
    policy_path: str
    policy_hash: str
    active_targets: list[TargetRef]
    source_commit: str
    binary_sha256: str
    checkpoint_sha256: str
    persistent: bool
    resident_worker: bool


class CanaryView(BaseModel):
    run_id: str
    signed: bool
    signature_status: str
    target: TargetRef
    before_n: int
    candidate_policy_hash: str
    pre_worker_id: str
    post_worker_id: str
    pre_requests: int
    post_requests: int
    attribution_before: float
    attribution_after: float
    target_replay_pass: bool
    rollback_required: bool
    auto_expand: bool
    decision: str
    finished_at: str
    evidence: EvidenceRef


class RestartVerification(BaseModel):
    pid: int
    requests: int
    tokens: int
    corrected_hits: int
    elapsed_ms: int
    peak_rss_bytes: int
    ack_sha256: str


class RollbackView(BaseModel):
    ready: bool
    status: str
    mode: str
    preimage_sha256: str
    postimage_sha256: str
    live_sha256: str
    exercised: bool


class ProductionView(BaseModel):
    run_id: str
    state: str
    production_path: str
    preimage_sha256: str
    postimage_sha256: str
    live_sha256: str
    active_policy: list[TargetRef]
    restart_verification: list[RestartVerification]
    production_write_allowed: bool
    resident_worker: bool
    service_state: str
    decision: str
    rollback: RollbackView
    applied_at: str
    evidence: EvidenceRef


class EvidenceView(BaseModel):
    evidence: EvidenceRef
    schema_name: str | None = None
    status: str | None = None
    content: dict[str, Any]


class HealthView(BaseModel):
    status: Literal["ok"] = "ok"
    evidence_root_readable: bool
