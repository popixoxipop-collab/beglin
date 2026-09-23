-- P5 Evidence Contract v2: durable actual-serving-path preflight evidence.
create table if not exists moe_live_preflight_results (
  id bigserial primary key,
  tested_at timestamptz not null default now(),
  source text not null default 'qng64_live_preflight',
  model text not null,
  role text not null,
  layer integer not null,
  n integer not null,
  corpus text,
  req integer,
  pos integer,
  promotion_preimage_sha256 text not null,
  promotion_postimage_sha256 text not null,
  orig_token integer,
  corrected_token integer,
  emitted_token integer,
  correction_required boolean,
  pass boolean not null,
  status text not null,
  reason text,
  manifest text,
  baseline_rc integer,
  candidate_rc integer,
  evidence_path text,
  engine_commit text
);

create index if not exists moe_live_preflight_results_lookup_idx
  on moe_live_preflight_results
  (model, role, layer, n, promotion_preimage_sha256, tested_at desc);

create index if not exists moe_live_preflight_results_status_idx
  on moe_live_preflight_results (status, tested_at desc);

comment on table moe_live_preflight_results is
'P5 Evidence Contract v2: actual QWEN_MOE_PROMOTION_FILE_NQ serving-path preflight evidence, keyed to the exact promotion preimage.';
