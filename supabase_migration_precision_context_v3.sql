-- Precision Evidence Contract v3.
-- Additive-only schema; v2 tables remain untouched and legacy/unscoped.

create table if not exists moe_execution_contexts_v3 (
  context_hash text primary key,
  created_at timestamptz not null default now(),
  schema_version text not null default 'precision-context-v3',
  model_id text not null,
  architecture text not null,
  checkpoint_sha256 text not null,
  tokenizer_sha256 text not null,
  base_artifact_sha256 text not null,
  backend text not null check (backend in ('cpu','mlx_metal')),
  device_fingerprint text not null,
  binary_sha256 text not null,
  build_manifest_sha256 text not null,
  kernel_revision text not null,
  execution_mode text not null,
  runtime_config_sha256 text not null,
  quant_format text not null,
  group_size integer not null check (group_size > 0),
  correction_mode text not null,
  metadata jsonb not null default '{}'::jsonb
);

create table if not exists moe_validation_runs_v3 (
  run_id text primary key,
  created_at timestamptz not null default now(),
  finished_at timestamptz,
  context_hash text not null references moe_execution_contexts_v3(context_hash),
  backend text not null check (backend in ('cpu','mlx_metal')),
  run_kind text not null check (run_kind in ('baseline','candidate','reference','observer')),
  status text not null,
  model_id text not null,
  role text,
  layer integer,
  n integer,
  policy_preimage_sha256 text,
  policy_postimage_sha256 text,
  weight_epoch bigint,
  manifest_sha256 text,
  prompt_tokens_sha256 text,
  generated_prefix_sha256 text,
  req integer,
  pos integer,
  reference_run_id text references moe_validation_runs_v3(run_id),
  pass boolean,
  reason text,
  evidence_path text,
  evidence_sha256 text,
  metrics jsonb not null default '{}'::jsonb
);

create index if not exists moe_validation_runs_v3_target_idx
  on moe_validation_runs_v3
  (context_hash, role, layer, n, created_at desc);

create table if not exists moe_attribution_provenance_v3 (
  id bigserial primary key,
  created_at timestamptz not null default now(),
  context_hash text not null references moe_execution_contexts_v3(context_hash),
  run_id text not null references moe_validation_runs_v3(run_id),
  model_id text not null,
  role text not null,
  layer integer not null,
  manifest_sha256 text not null,
  prompt_tokens_sha256 text not null,
  req integer not null,
  pos integer not null,
  orig_argmax integer not null,
  corrected_argmax integer not null,
  threshold double precision,
  margin double precision,
  unique (context_hash, run_id, role, layer, req, pos, orig_argmax, corrected_argmax)
);

create index if not exists moe_attribution_provenance_v3_target_idx
  on moe_attribution_provenance_v3
  (context_hash, role, layer, created_at desc);

create table if not exists moe_live_preflight_results_v3 (
  id bigserial primary key,
  tested_at timestamptz not null default now(),
  context_hash text not null references moe_execution_contexts_v3(context_hash),
  baseline_run_id text references moe_validation_runs_v3(run_id),
  candidate_run_id text references moe_validation_runs_v3(run_id),
  model_id text not null,
  role text not null,
  layer integer not null,
  n integer not null,
  baseline_policy_hash text not null,
  requested_policy_hash text not null,
  applied_policy_hash text,
  expected_epoch bigint,
  observed_epoch bigint,
  pass boolean not null,
  status text not null,
  reason text,
  correction_required boolean,
  emitted_token integer,
  reference_token integer,
  evidence_sha256 text
);

create index if not exists moe_live_preflight_results_v3_lookup_idx
  on moe_live_preflight_results_v3
  (context_hash, role, layer, n, tested_at desc);

create table if not exists moe_precision_transactions_v3 (
  txn_id text primary key,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  context_hash text not null references moe_execution_contexts_v3(context_hash),
  backend text not null check (backend in ('cpu','mlx_metal')),
  status text not null,
  role text,
  layer integer,
  n integer,
  expected_epoch bigint not null,
  resulting_epoch bigint,
  expected_policy_hash text not null,
  requested_policy_hash text not null,
  applied_policy_hash text,
  rollback_policy_hash text,
  requested_at timestamptz not null default now(),
  applied_at timestamptz,
  rollback_at timestamptz,
  ack_sha256 text,
  error text,
  journal jsonb not null default '{}'::jsonb
);

create index if not exists moe_precision_transactions_v3_context_idx
  on moe_precision_transactions_v3
  (context_hash, updated_at desc);

comment on table moe_execution_contexts_v3 is
'Immutable backend/device/binary/runtime identity. CPU and MLX/Metal evidence never share a context hash.';
comment on table moe_precision_transactions_v3 is
'Desired/applied precision policy transactions with durable epoch and rollback state.';
