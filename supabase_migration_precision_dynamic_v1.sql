-- Trigger-conditioned precision evidence and selector decisions.
-- Internal evidence only. No direct production mutation path.

create table if not exists public.moe_precision_trigger_evidence_v1 (
  evidence_id text primary key,
  created_at timestamptz not null default now(),
  model_id text not null,
  context_hash text,
  role text not null,
  layer integer not null,
  from_n integer not null,
  to_n integer not null,
  trigger_type text not null check (
    trigger_type in ('near_tie','low_margin','high_entropy','routing_ambiguity')
  ),
  signal_bucket jsonb not null default '{}'::jsonb,
  requests integer not null check (requests > 0),
  pass boolean not null,
  status text not null,
  metrics jsonb not null default '{}'::jsonb,
  source_run_id text,
  evidence_sha256 text
);

create index if not exists moe_precision_trigger_evidence_v1_lookup_idx
  on public.moe_precision_trigger_evidence_v1
  (model_id, role, layer, from_n, to_n, trigger_type, created_at desc);

create table if not exists public.moe_precision_dynamic_decisions_v1 (
  decision_id text primary key,
  created_at timestamptz not null default now(),
  model_id text not null,
  schema_version text not null default 'beglin-dynamic-precision-selector-v1',
  status text not null,
  signal jsonb not null default '{}'::jsonb,
  selected_policy jsonb not null default '[]'::jsonb,
  target_decisions jsonb not null default '[]'::jsonb,
  production_write_allowed boolean not null default false
);

create index if not exists moe_precision_dynamic_decisions_v1_model_created_idx
  on public.moe_precision_dynamic_decisions_v1 (model_id, created_at desc);

alter table public.moe_precision_trigger_evidence_v1 enable row level security;
alter table public.moe_precision_dynamic_decisions_v1 enable row level security;

revoke all privileges on table
  public.moe_precision_trigger_evidence_v1,
  public.moe_precision_dynamic_decisions_v1
from anon, authenticated;

grant all privileges on table
  public.moe_precision_trigger_evidence_v1,
  public.moe_precision_dynamic_decisions_v1
to service_role;

comment on table public.moe_precision_trigger_evidence_v1 is
'Trigger-conditioned precision calibration evidence. Higher n is never presumed safer.';
comment on table public.moe_precision_dynamic_decisions_v1 is
'Fail-closed dynamic precision decisions. Decision-only; production_write_allowed=false.';
