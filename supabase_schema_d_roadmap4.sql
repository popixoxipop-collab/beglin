-- D-roadmap-4: live role×layer near-tie attribution + closed-loop precision promotion
-- Project: beglin (btdjbfgqzglucifcnuoc)
-- Run once via Supabase MCP (once its project-scope approval lands) or the SQL editor.

create table if not exists moe_neartie_events (
  id bigserial primary key,
  ts timestamptz not null default now(),
  req integer not null,
  pos integer not null,
  predicted_token integer not null,
  competing_token integer not null,
  margin double precision not null,
  model text not null,
  corpus text not null,
  attributed_role text,
  attributed_layer integer
);

-- D-quant-supabase-1 note (2026-09-02): this table's PK was originally just
-- (role, layer) -- widened to (model, role, layer) by the migration block
-- near the end of this file. The create-table below is kept as the
-- original historical shape (a fresh DB run of this whole file still ends
-- up correct, since the migration block runs right after); do not add
-- `model` here directly, the migration block documents WHY that column
-- exists.
create table if not exists moe_role_precision_state (
  role text not null,
  layer integer not null,
  current_bits integer not null default 8,
  event_count integer not null default 0,
  min_margin_observed double precision,
  promoted boolean not null default false,
  promoted_at timestamptz,
  last_updated timestamptz not null default now(),
  primary key (role, layer)
);

create index if not exists moe_neartie_events_ts_idx
  on moe_neartie_events (ts);

-- Real implementation found attribution is one-to-MANY per flagged event (a single real flip
-- routinely gets reproduced by more than one single-role promotion -- pilot data: 13/108 combos
-- for one event), which the original single attributed_role/attributed_layer columns above can't
-- represent. Those columns are kept (harmless, unused) for forward compat with a future per-hit
-- join table; the real accumulation happens here instead, directly on the map -- each attribution
-- hit increments the matching (role,layer) row's event_count. Plain PostgREST UPDATE can't do a
-- read-modify-write increment atomically, hence this RPC.
create or replace function increment_role_precision(p_role text, p_layer integer, p_margin double precision default null)
returns void as $$
  update moe_role_precision_state
  set event_count = event_count + 1,
      min_margin_observed = case
        when p_margin is null then min_margin_observed
        when min_margin_observed is null then p_margin
        else least(min_margin_observed, p_margin)
      end,
      last_updated = now()
  where role = p_role and layer = p_layer;
$$ language sql;

-- Seed: real MOE_ST_ATTN_ROLES_MLA (qwen_infer.c:12311-12316) x NL=27
-- + MOE_ST_DENSE_ROLES (qwen_infer.c:12381-12385) x layer 0 only (FIRST_DENSE_LAYERS=1)
-- + MOE_ST_SHARED_ROLES (qwen_infer.c:12386-12391) x layers 1..26 (MOE_N_SHARED=2>0)
-- + embed_tokens/lm_head wildcards (layer=-1, default bits=32 per moe_role_bits() call sites)
-- 191 rows total (108 attn + 3 dense + 78 shared + 2 wildcard) -- computed, not guessed.
insert into moe_role_precision_state (role, layer, current_bits) values
('q_proj', 0, 8),
('kv_a_proj_with_mqa', 0, 8),
('kv_b_proj', 0, 8),
('o_proj', 0, 8),
('q_proj', 1, 8),
('kv_a_proj_with_mqa', 1, 8),
('kv_b_proj', 1, 8),
('o_proj', 1, 8),
('q_proj', 2, 8),
('kv_a_proj_with_mqa', 2, 8),
('kv_b_proj', 2, 8),
('o_proj', 2, 8),
('q_proj', 3, 8),
('kv_a_proj_with_mqa', 3, 8),
('kv_b_proj', 3, 8),
('o_proj', 3, 8),
('q_proj', 4, 8),
('kv_a_proj_with_mqa', 4, 8),
('kv_b_proj', 4, 8),
('o_proj', 4, 8),
('q_proj', 5, 8),
('kv_a_proj_with_mqa', 5, 8),
('kv_b_proj', 5, 8),
('o_proj', 5, 8),
('q_proj', 6, 8),
('kv_a_proj_with_mqa', 6, 8),
('kv_b_proj', 6, 8),
('o_proj', 6, 8),
('q_proj', 7, 8),
('kv_a_proj_with_mqa', 7, 8),
('kv_b_proj', 7, 8),
('o_proj', 7, 8),
('q_proj', 8, 8),
('kv_a_proj_with_mqa', 8, 8),
('kv_b_proj', 8, 8),
('o_proj', 8, 8),
('q_proj', 9, 8),
('kv_a_proj_with_mqa', 9, 8),
('kv_b_proj', 9, 8),
('o_proj', 9, 8),
('q_proj', 10, 8),
('kv_a_proj_with_mqa', 10, 8),
('kv_b_proj', 10, 8),
('o_proj', 10, 8),
('q_proj', 11, 8),
('kv_a_proj_with_mqa', 11, 8),
('kv_b_proj', 11, 8),
('o_proj', 11, 8),
('q_proj', 12, 8),
('kv_a_proj_with_mqa', 12, 8),
('kv_b_proj', 12, 8),
('o_proj', 12, 8),
('q_proj', 13, 8),
('kv_a_proj_with_mqa', 13, 8),
('kv_b_proj', 13, 8),
('o_proj', 13, 8),
('q_proj', 14, 8),
('kv_a_proj_with_mqa', 14, 8),
('kv_b_proj', 14, 8),
('o_proj', 14, 8),
('q_proj', 15, 8),
('kv_a_proj_with_mqa', 15, 8),
('kv_b_proj', 15, 8),
('o_proj', 15, 8),
('q_proj', 16, 8),
('kv_a_proj_with_mqa', 16, 8),
('kv_b_proj', 16, 8),
('o_proj', 16, 8),
('q_proj', 17, 8),
('kv_a_proj_with_mqa', 17, 8),
('kv_b_proj', 17, 8),
('o_proj', 17, 8),
('q_proj', 18, 8),
('kv_a_proj_with_mqa', 18, 8),
('kv_b_proj', 18, 8),
('o_proj', 18, 8),
('q_proj', 19, 8),
('kv_a_proj_with_mqa', 19, 8),
('kv_b_proj', 19, 8),
('o_proj', 19, 8),
('q_proj', 20, 8),
('kv_a_proj_with_mqa', 20, 8),
('kv_b_proj', 20, 8),
('o_proj', 20, 8),
('q_proj', 21, 8),
('kv_a_proj_with_mqa', 21, 8),
('kv_b_proj', 21, 8),
('o_proj', 21, 8),
('q_proj', 22, 8),
('kv_a_proj_with_mqa', 22, 8),
('kv_b_proj', 22, 8),
('o_proj', 22, 8),
('q_proj', 23, 8),
('kv_a_proj_with_mqa', 23, 8),
('kv_b_proj', 23, 8),
('o_proj', 23, 8),
('q_proj', 24, 8),
('kv_a_proj_with_mqa', 24, 8),
('kv_b_proj', 24, 8),
('o_proj', 24, 8),
('q_proj', 25, 8),
('kv_a_proj_with_mqa', 25, 8),
('kv_b_proj', 25, 8),
('o_proj', 25, 8),
('q_proj', 26, 8),
('kv_a_proj_with_mqa', 26, 8),
('kv_b_proj', 26, 8),
('o_proj', 26, 8),
('dense_gate_proj', 0, 8),
('dense_up_proj', 0, 8),
('dense_down_proj', 0, 8),
('shared_gate_proj', 1, 8),
('shared_up_proj', 1, 8),
('shared_down_proj', 1, 8),
('shared_gate_proj', 2, 8),
('shared_up_proj', 2, 8),
('shared_down_proj', 2, 8),
('shared_gate_proj', 3, 8),
('shared_up_proj', 3, 8),
('shared_down_proj', 3, 8),
('shared_gate_proj', 4, 8),
('shared_up_proj', 4, 8),
('shared_down_proj', 4, 8),
('shared_gate_proj', 5, 8),
('shared_up_proj', 5, 8),
('shared_down_proj', 5, 8),
('shared_gate_proj', 6, 8),
('shared_up_proj', 6, 8),
('shared_down_proj', 6, 8),
('shared_gate_proj', 7, 8),
('shared_up_proj', 7, 8),
('shared_down_proj', 7, 8),
('shared_gate_proj', 8, 8),
('shared_up_proj', 8, 8),
('shared_down_proj', 8, 8),
('shared_gate_proj', 9, 8),
('shared_up_proj', 9, 8),
('shared_down_proj', 9, 8),
('shared_gate_proj', 10, 8),
('shared_up_proj', 10, 8),
('shared_down_proj', 10, 8),
('shared_gate_proj', 11, 8),
('shared_up_proj', 11, 8),
('shared_down_proj', 11, 8),
('shared_gate_proj', 12, 8),
('shared_up_proj', 12, 8),
('shared_down_proj', 12, 8),
('shared_gate_proj', 13, 8),
('shared_up_proj', 13, 8),
('shared_down_proj', 13, 8),
('shared_gate_proj', 14, 8),
('shared_up_proj', 14, 8),
('shared_down_proj', 14, 8),
('shared_gate_proj', 15, 8),
('shared_up_proj', 15, 8),
('shared_down_proj', 15, 8),
('shared_gate_proj', 16, 8),
('shared_up_proj', 16, 8),
('shared_down_proj', 16, 8),
('shared_gate_proj', 17, 8),
('shared_up_proj', 17, 8),
('shared_down_proj', 17, 8),
('shared_gate_proj', 18, 8),
('shared_up_proj', 18, 8),
('shared_down_proj', 18, 8),
('shared_gate_proj', 19, 8),
('shared_up_proj', 19, 8),
('shared_down_proj', 19, 8),
('shared_gate_proj', 20, 8),
('shared_up_proj', 20, 8),
('shared_down_proj', 20, 8),
('shared_gate_proj', 21, 8),
('shared_up_proj', 21, 8),
('shared_down_proj', 21, 8),
('shared_gate_proj', 22, 8),
('shared_up_proj', 22, 8),
('shared_down_proj', 22, 8),
('shared_gate_proj', 23, 8),
('shared_up_proj', 23, 8),
('shared_down_proj', 23, 8),
('shared_gate_proj', 24, 8),
('shared_up_proj', 24, 8),
('shared_down_proj', 24, 8),
('shared_gate_proj', 25, 8),
('shared_up_proj', 25, 8),
('shared_down_proj', 25, 8),
('shared_gate_proj', 26, 8),
('shared_up_proj', 26, 8),
('shared_down_proj', 26, 8),
('embed_tokens', -1, 32),
('lm_head', -1, 32)
on conflict (role, layer) do nothing;

-- =============================================================================
-- D-quant-supabase-1 (2026-09-02, executed live against the real DB, verified
-- via independent SELECT + a live RPC-call-and-revert round-trip -- not just
-- trusted from the ALTER/UPDATE statements' own row counts):
--   WHY: schema was single-model (DeepSeek-V2-Lite only, seeded above). A
--   concurrent session (macstudio, user eoe) runs a DIFFERENT architecture
--   (Qwen3-30B-A3B, GQA) against this SAME Supabase project/URL/key. Role
--   names overlap by coincidence (q_proj/o_proj exist in both role tables)
--   at the same layer index, so a cross-model push would silently conflate
--   two unrelated models' accumulated event_count/min_margin under one
--   (role,layer) row. moe_neartie_events already tracked `model` per-event;
--   the aggregate map didn't -- this closes that gap.
--   COST: existing 191 rows backfilled with the one real model string
--   already in use ('deepseek-v2-lite', confirmed from the actually-pushed
--   JSONL, not guessed); increment_role_precision()'s old 3-arg signature
--   was dropped (not kept as dead code -- a stale caller using it would
--   otherwise silently match nothing, or the wrong row, once the PK
--   widened). d4_supabase_push.py updated to pass model through.
--   EXIT: to widen further (e.g. per-corpus too), same pattern -- add
--   column, backfill, drop+recreate PK, update the RPC signature and the
--   push script together.
-- =============================================================================

alter table moe_role_precision_state add column if not exists model text;

update moe_role_precision_state set model = 'deepseek-v2-lite' where model is null;

alter table moe_role_precision_state alter column model set not null;

alter table moe_role_precision_state drop constraint if exists moe_role_precision_state_pkey;
alter table moe_role_precision_state add primary key (model, role, layer);

drop function if exists increment_role_precision(text, integer, double precision);

create or replace function increment_role_precision(p_model text, p_role text, p_layer integer, p_margin double precision default null)
returns void as $$
  update moe_role_precision_state
  set event_count = event_count + 1,
      min_margin_observed = case
        when p_margin is null then min_margin_observed
        when min_margin_observed is null then p_margin
        else least(min_margin_observed, p_margin)
      end,
      last_updated = now()
  where model = p_model and role = p_role and layer = p_layer;
$$ language sql;

-- =============================================================================
-- D-quant-supabase-2 (2026-09-02, executed live, verified via independent
-- SELECT + PK-definition + RPC-call-and-revert -- same discipline as
-- D-quant-supabase-1 above):
--   WHY: moe_neartie_events already tags every row with `corpus`;
--   moe_role_precision_state didn't. WikiText-103 (Phase 7/8) is about to
--   push into the same table -- without this, its hits would silently
--   blend into the same (model,role,layer) rows as WikiText-2's, making
--   "is this finding dataset-specific" unanswerable. Same fix as
--   D-quant-supabase-1, one dimension further.
--   COST: PK widens to (model, corpus, role, layer); existing 191 rows
--   backfilled with the one real corpus string already in use
--   ('wikitext-2-raw-v1-validation-short-fullext', confirmed by querying
--   moe_neartie_events directly); increment_role_precision() becomes
--   5-arg (old 4-arg dropped, not kept as dead code). Seeding a NEW
--   (model,corpus) pair is no longer a hardcoded block in this file --
--   use tools/seed_precision_map.py <model> <corpus> | psql "$CONN"
--   instead (same 191-row shape, parameterized).
--   EXIT: same widen-the-PK pattern again if another dimension is needed.
-- =============================================================================

alter table moe_role_precision_state add column if not exists corpus text;

update moe_role_precision_state set corpus = 'wikitext-2-raw-v1-validation-short-fullext' where corpus is null;

alter table moe_role_precision_state alter column corpus set not null;

alter table moe_role_precision_state drop constraint if exists moe_role_precision_state_pkey;
alter table moe_role_precision_state add primary key (model, corpus, role, layer);

drop function if exists increment_role_precision(text, integer, double precision);

create or replace function increment_role_precision(p_model text, p_corpus text, p_role text, p_layer integer, p_margin double precision default null)
returns void as $$
  update moe_role_precision_state
  set event_count = event_count + 1,
      min_margin_observed = case
        when p_margin is null then min_margin_observed
        when min_margin_observed is null then p_margin
        else least(min_margin_observed, p_margin)
      end,
      last_updated = now()
  where model = p_model and corpus = p_corpus and role = p_role and layer = p_layer;
$$ language sql;

-- =============================================================================
-- D-quant-mono-1 (2026-09-02, executed live): durable, corpus-tagged home for
-- ROI-G's arbitrary-n quantization sweep data (tools/quant_sim_n.py /
-- quant_sim_analyze.py / quant_search_n.py).
--   WHY: the first 90 real sweep data points (6 targets x n=2..16, all
--   WikiText-2) only existed as local TSV files -- exactly the kind of
--   thing that should be in the shared map. Without a durable, queryable,
--   corpus-tagged store, "is this target's monotonicity violation
--   WikiText-2-specific" has nowhere to be answered once WikiText-103
--   data exists to compare against.
--   COST: one new table, raw per-n-value rows (source of truth).
--   Classification (knee, tier-or-not, monotonic-or-not) stays a derived
--   query/script output, not a second table that can drift from the raw
--   rows.
--   EXIT: promote to a materialized view over this table if classification
--   needs to be queried frequently instead of recomputed.
-- =============================================================================

create table if not exists moe_quant_sweep_results (
  id bigserial primary key,
  model text not null,
  corpus text not null,
  role text not null,
  layer integer not null,
  req integer not null,
  pos integer not null,
  n integer not null,
  eff_bpw double precision not null,
  pass boolean not null,
  rel_l2 double precision,
  tested_at timestamptz not null default now()
);

create index if not exists moe_quant_sweep_results_target_idx
  on moe_quant_sweep_results (model, corpus, role, layer);

-- =============================================================================
-- D-quant-supabase-3 (2026-09-02, executed live, verified via independent
-- SELECT): moe_role_precision_state gets a 5th role family, ROUTED EXPERTS.
--   WHY: the user reviewed the SLAM MAP artifact (4 families: ATTENTION/
--   DENSE FFN/SHARED FFN/GLOBAL) and asked directly whether it showed every
--   independently-precision-controllable element of the engine. It didn't --
--   qwen_infer.c registers a 5th family, MOE_ST_EXPERT_ROLES (expert_gate_
--   proj/expert_up_proj/expert_down_proj, qwen_infer.c:13376), via
--   st_register_moe_role() at every l >= MOE_FIRST_DENSE_LAYERS (:13749-13754,
--   same layer range as MOE_ST_SHARED_ROLES), and it is also a first-class
--   member of the attribution role vocabulary (MOE_ATTRIB_EXPERT_GATE/UP/
--   DOWN, :5583, D-d5-9). tools/seed_precision_map.py simply never seeded it
--   -- an oversight in the seed script, not a deliberate scope decision, so
--   the dashboard was silently incomplete since it was first built.
--   COST: 78 new rows per (model,corpus) pair (3 roles x layers 1..26; layer
--   0 has no routed experts, same architecture fact as DENSE FFN being
--   layer-0-only in reverse). Backfilled live for the one existing pair
--   (deepseek-v2-lite, wikitext-2-raw-v1-validation-short-fullext):
--   191 -> 269 rows, verified via SELECT count(*) filter (role like
--   'expert%') = 78. Every new row starts at event_count=0 -- checked live
--   before seeding (moe_role_precision_state had zero expert_* rows,
--   moe_neartie_events has no per-combo role column to check, and every
--   WikiText-2 raw JSONL still on bob was grepped for expert_gate_proj/
--   expert_up_proj/expert_down_proj with zero matches) -- no real
--   attribution run has ever swept these roles yet, so 0 hits is an honest
--   "trackable, unmeasured" state, not a finding.
--   EXIT: per-expert-index granularity (vs. today's per-layer-promotes-all-
--   64-experts) would need a new `expert_idx` PK column -- this change is
--   additive only, no migration of the 191 already-seeded rows required.
-- =============================================================================
