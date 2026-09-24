-- Durable telemetry ingest idempotency for JSONL outbox replay.
alter table moe_neartie_events
  add column if not exists ingest_id text;

create unique index if not exists moe_neartie_events_ingest_id_uq
  on moe_neartie_events (ingest_id)
  where ingest_id is not null;

create table if not exists moe_attribution_ingest_keys (
  ingest_id text primary key,
  created_at timestamptz not null default now(),
  model text not null,
  corpus text not null,
  role text not null,
  layer integer not null
);

create or replace function increment_role_precision_idempotent(
  p_ingest_id text,
  p_model text,
  p_corpus text,
  p_role text,
  p_layer integer,
  p_margin double precision default null
)
returns boolean as $$
declare
  claimed integer;
begin
  insert into moe_attribution_ingest_keys
    (ingest_id, model, corpus, role, layer)
  values
    (p_ingest_id, p_model, p_corpus, p_role, p_layer)
  on conflict (ingest_id) do nothing;

  get diagnostics claimed = row_count;
  if claimed = 0 then
    return false;
  end if;

  update moe_role_precision_state
  set event_count = event_count + 1,
      min_margin_observed = case
        when p_margin is null then min_margin_observed
        when min_margin_observed is null then p_margin
        else least(min_margin_observed, p_margin)
      end,
      last_updated = now()
  where model = p_model
    and corpus = p_corpus
    and role = p_role
    and layer = p_layer;

  return true;
end;
$$ language plpgsql;

comment on table moe_attribution_ingest_keys is
'Idempotency claims for attribution aggregate increments. Replaying the same JSONL record never increments event_count twice.';
