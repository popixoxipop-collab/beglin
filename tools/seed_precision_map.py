#!/usr/bin/env python3
# D-quant-supabase-2 follow-up: seeding moe_role_precision_state used to be a
# one-time hardcoded 191-row INSERT block in supabase_schema_d_roadmap4.sql --
# fine when there was exactly one (model,corpus) pair. Now that the PK is
# (model, corpus, role, layer), a second corpus (WikiText-103) needs its own
# seed set, so this became a reusable script instead of a second hardcoded
# block. Mirrors qwen_infer.c's real role tables exactly:
# MOE_ST_ATTN_ROLES_MLA (q_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj, layers
# 0-26), MOE_ST_DENSE_ROLES (layer 0 only), MOE_ST_SHARED_ROLES (layers
# 1-26), plus the embed_tokens/lm_head wildcards (layer -1) -- 108+3+78+2=191,
# same total the original hardcoded seed produced.
#
# D-quant-supabase-3 (2026-09-02, user-requested after reviewing the SLAM MAP
# artifact): routed experts added as a 5th role family.
#   WHY: qwen_infer.c actually tracks precision for a 5th family the map never
#   seeded -- MOE_ST_EXPERT_ROLES (expert_gate_proj/expert_up_proj/
#   expert_down_proj, qwen_infer.c:13376), registered via st_register_moe_role()
#   at :13749-13754 for every l >= MOE_FIRST_DENSE_LAYERS (same range as
#   MOE_ST_SHARED_ROLES), and also present in the attribution role vocabulary
#   (MOE_ATTRIB_EXPERT_GATE/UP/DOWN, :5583, valid range confirmed identical at
#   moe_attrib_role_valid_at(), :5610-5615). The user directly asked why the
#   dashboard didn't show it -- it was a real gap in this seed script, not a
#   deliberate scope choice.
#   COST: 3 roles x 26 layers (1..26) = 78 more rows per (model,corpus) pair
#   (191 -> 269). Each row represents ALL E=64 routed experts of that layer's
#   projection promoted together (one AF tensor per projection, not
#   per-expert-index -- matches how the engine itself stores/registers them,
#   confirmed at :13752-13753). No real attribution run has ever swept these
#   roles yet (checked live: 0 rows in moe_neartie_events/moe_role_precision_
#   state with an expert_* role, and grepped every WikiText-2 raw JSONL on
#   bob for expert_gate_proj/expert_up_proj/expert_down_proj -- zero matches)
#   -- so every seeded row starts at hits=0, honestly reflecting "trackable,
#   not yet measured" rather than implying evidence that doesn't exist.
#   EXIT: if per-expert-index granularity is ever needed instead of
#   per-layer-promotes-all-E, that's a new PK dimension (add an `expert_idx`
#   column) -- this seed only needs re-running with the wider row set, no
#   migration of already-seeded rows required (additive, same PK shape).
#
# Usage: python3 tools/seed_precision_map.py <model> <corpus> | psql "$CONN"
# Prints SQL to stdout (does not connect to the DB itself -- no db driver
# installed locally, and this keeps the script usable from any machine with
# just python3 + a psql pipe).

import sys

NL = 27
ATTN_ROLES = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
DENSE_ROLES = ["dense_gate_proj", "dense_up_proj", "dense_down_proj"]
SHARED_ROLES = ["shared_gate_proj", "shared_up_proj", "shared_down_proj"]
ROUTED_ROLES = ["expert_gate_proj", "expert_up_proj", "expert_down_proj"]  # D-quant-supabase-3
FIRST_DENSE_LAYERS = 1  # DeepSeek-V2-Lite: only layer 0 is dense, N_SHARED=2>0 for layers 1..26


def sql_escape(s):
    return s.replace("'", "''")


def main():
    if len(sys.argv) != 3:
        print("usage: seed_precision_map.py <model> <corpus>", file=sys.stderr)
        sys.exit(1)
    model, corpus = sys.argv[1], sys.argv[2]
    m, c = sql_escape(model), sql_escape(corpus)

    rows = []
    for role in ATTN_ROLES:
        for l in range(NL):
            rows.append((role, l, 8))
    for role in DENSE_ROLES:
        for l in range(FIRST_DENSE_LAYERS):
            rows.append((role, l, 8))
    for role in SHARED_ROLES:
        for l in range(FIRST_DENSE_LAYERS, NL):
            rows.append((role, l, 8))
    for role in ROUTED_ROLES:  # D-quant-supabase-3
        for l in range(FIRST_DENSE_LAYERS, NL):
            rows.append((role, l, 8))
    rows.append(("embed_tokens", -1, 32))
    rows.append(("lm_head", -1, 32))

    assert len(rows) == 269, f"expected 269 rows, got {len(rows)}"

    print(f"-- seed_precision_map.py: model='{model}' corpus='{corpus}' ({len(rows)} rows)")
    print("insert into moe_role_precision_state (model, corpus, role, layer, current_bits) values")
    lines = [f"('{m}', '{c}', '{sql_escape(role)}', {layer}, {bits})" for role, layer, bits in rows]
    print(",\n".join(lines))
    print("on conflict (model, corpus, role, layer) do nothing;")


if __name__ == "__main__":
    main()
