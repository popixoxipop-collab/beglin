-- Precision Evidence Contract v3 (G1).
-- Additive only: v2 tables remain untouched and are not promoted into v3 evidence.
-- Apply in staging first. RLS/grants are intentionally not changed here; inspect the
-- deployment role and existing policies before adding least-privilege access.

create table if not exists moe_execution_contexts_v3 (
  context_id text primary key,
  schema_version text not null default 'precision-context-v3',
  model_id text not null,
  architecture text not null,
  checkpoint_sha256 text not null,
  tokenizer_sha256 text not null,
  base_artifact_sha256 text not null,
  backend text not null check (backend in ('cpu', 'mlx_metal')),
  device_fingerprint text not null,
  binary_sha256 text not null,
  build_manifest_sha256 text not null,
  kernel_version text not null,
  precision_mode text not null,
  quant_format text not null,
  group_size integer not null check (group_size > 0),
  encoder_version text not null,
  execution_mode text not null,
  runtime_config jsonb not null default '{}'::jsonb,
  canonical_context jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists moe_execution_contexts_v3_backend_idx
  on moe_execution_contexts_v3 (backend, model_id, created_at desc);

create table if not exists moe_validation_runs_v3 (
  run_id text primary key,
  context_id text not null references moe_execution_contexts_v3(context_id),
  run_kind text not null check (
    run_kind in ('baseline', 'candidate', 'oracle', 'negative_control', 'rollback_drill')
  ),
  status text not null,
  correction_enabled boolean not null,
  preimage_policy_sha256 text not null,
  candidate_policy_sha256 text,
  manifest_sha256 text,
  prompt_token_sha256 text,
  generated_prefix_sha256 text,
  batch_schedule_sha256 text,
  seed bigint,
  evidence_path text,
  evidence_sha256 text,
  exit_code integer,
  started_at timestamptz not null default now(),
  finished_at timestamptz
);

create index if not exists moe_validation_runs_v3_context_idx
  on moe_validation_runs_v3 (context_id, run_kind, started_at desc);

create table if not exists moe_attribution_provenance_v3 (
  id bigserial primary key,
  run_id text not null references moe_validation_runs_v3(run_id),
  context_id text not null references moe_execution_contexts_v3(context_id),
  req integer not null,
  pos integer not null,
  role text not null,
  layer integer not null,
  orig_argmax integer,
  corrected_argmax integer,
  margin double precision,
  effective_attribution_check boolean not null default false,
  attribution_hit boolean not null default false,
  source_record_id text,
  observed_at timestamptz not null default now(),
  unique (run_id, req, pos, role, layer, orig_argmax, corrected_argmax)
);

create index if not exists moe_attribution_provenance_v3_target_idx
  on moe_attribution_provenance_v3
  (context_id, role, layer, observed_at desc);

create table if not exists moe_live_preflight_results_v3 (
  id bigserial primary key,
  run_id text not null references moe_validation_runs_v3(run_id),
  context_id text not null references moe_execution_contexts_v3(context_id),
  backend text not null check (backend in ('cpu', 'mlx_metal')),
  role text not null,
  layer integer not null,
  n integer not null,
  req integer,
  pos integer,
  orig_token integer,
  corrected_token integer,
  emitted_token integer,
  correction_enabled boolean not null,
  correction_required boolean,
  pass boolean not null,
  status text not null,
  reason text,
  preimage_policy_sha256 text not null,
  candidate_policy_sha256 text not null,
  evidence_path text,
  evidence_sha256 text,
  tested_at timestamptz not null default now(),
  unique (
    run_id, context_id, role, layer, n,
    preimage_policy_sha256, candidate_policy_sha256
  )
);

create index if not exists moe_live_preflight_results_v3_lookup_idx
  on moe_live_preflight_results_v3
  (context_id, role, layer, n, preimage_policy_sha256, tested_at desc);

create table if not exists moe_precision_transactions_v3 (
  txn_id text primary key,
  context_id text not null references moe_execution_contexts_v3(context_id),
  backend text not null check (backend in ('cpu', 'mlx_metal')),
  action text not null check (action in ('promote', 'demote', 'restore')),
  status text not null,
  expected_epoch bigint not null,
  resulting_epoch bigint,
  expected_policy_sha256 text not null,
  desired_policy_sha256 text not null,
  applied_policy_sha256 text,
  rollback_policy_sha256 text,
  requested_at timestamptz not null default now(),
  admission_paused_at timestamptz,
  drain_completed_at timestamptz,
  applied_ack_at timestamptz,
  rollback_ack_at timestamptz,
  error_reason text,
  journal_path text
);

create index if not exists moe_precision_transactions_v3_context_idx
  on moe_precision_transactions_v3
  (context_id, expected_epoch, requested_at desc);

comment on table moe_execution_contexts_v3 is
'Immutable backend/device/binary/model/numeric context. Cross-context PASS reuse is forbidden.';

comment on table moe_validation_runs_v3 is
'Backend-scoped baseline/candidate/oracle runs. Candidate correction_enabled=false is required for independent promotion validation.';

comment on table moe_precision_transactions_v3 is
'Precision control transaction journal with epoch, desired/applied hashes, and durable apply/rollback acknowledgements.';
