-- D-l4-11: durable attribution provenance for direct replay.
create table if not exists moe_attribution_provenance (
  id bigserial primary key,
  first_seen_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  source text not null default 'jsonl_push',
  model text not null,
  corpus text not null,
  role text not null,
  layer integer not null,
  manifest text not null,
  req integer not null,
  pos integer not null,
  orig_argmax integer not null,
  corrected_argmax integer not null,
  threshold double precision,
  margin double precision,
  batch_size integer,
  replay_margin_b1 double precision,
  attribution_ts_unix bigint,
  event_ts_unix bigint,
  source_jsonl text,
  unique (model, corpus, role, layer, manifest, req, pos, orig_argmax, corrected_argmax)
);

create index if not exists moe_attribution_provenance_target_idx
  on moe_attribution_provenance
  (model, role, layer, last_seen_at desc);

create index if not exists moe_attribution_provenance_event_idx
  on moe_attribution_provenance
  (model, corpus, manifest, req, pos);
