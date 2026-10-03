-- Beglin Precision Allocator v1 decision evidence.
-- Internal research/control-plane evidence only; never exposed to anon/authenticated.

create table if not exists public.moe_precision_allocator_decisions_v1 (
  decision_id text primary key,
  created_at timestamptz not null default now(),
  model_id text not null,
  schema_version text not null default 'beglin-precision-allocator-v1',
  status text not null,
  current_policy jsonb not null default '[]'::jsonb,
  proposed_policy jsonb not null default '[]'::jsonb,
  objective jsonb not null default '{}'::jsonb,
  constraints jsonb not null default '{}'::jsonb,
  target_decisions jsonb not null default '[]'::jsonb,
  source_contexts jsonb not null default '[]'::jsonb,
  production_write_allowed boolean not null default false
);

create index if not exists moe_precision_allocator_decisions_v1_model_created_idx
  on public.moe_precision_allocator_decisions_v1 (model_id, created_at desc);

alter table public.moe_precision_allocator_decisions_v1 enable row level security;

revoke all privileges on table public.moe_precision_allocator_decisions_v1
from anon, authenticated;

grant all privileges on table public.moe_precision_allocator_decisions_v1
to service_role;

comment on table public.moe_precision_allocator_decisions_v1 is
'Evidence-gated nonmonotonic precision allocator proposals. Proposal-only; production_write_allowed is false.';
