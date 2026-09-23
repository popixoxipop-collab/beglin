#!/usr/bin/env python3
"""D-roadmap-4 Phase 4: batched push of the local JSONL event/attribution log to Supabase.
Reads QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY from env (never hardcoded). Best-effort, non-fatal
on any single request failure (D3: telemetry must never be allowed to break the pipeline) --
prints a warning and continues. Tracks a byte offset in a sidecar file so re-running is safe
(only pushes newly-appended lines).

Usage: python3 d4_supabase_push.py <jsonl_path> [--batch-size N]
"""
import json, os, sys, time
import urllib.request, urllib.error

BATCH = 20

def post(url, key, path, rows):
    if not rows:
        return True
    req = urllib.request.Request(
        url + path,
        data=json.dumps(rows).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except urllib.error.HTTPError as e:
        print(f"[d4 push] WARN HTTP {e.code} pushing {len(rows)} rows to {path}: {e.read()[:300]}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[d4 push] WARN {type(e).__name__} pushing {len(rows)} rows to {path}: {e}", file=sys.stderr)
        return False



def upsert_provenance(url, key, rows):
    if not rows:
        return True
    conflict = 'model,corpus,role,layer,manifest,req,pos,orig_argmax,corrected_argmax'
    req = urllib.request.Request(
        url + '/rest/v1/moe_attribution_provenance?on_conflict=' + conflict,
        data=json.dumps(rows).encode(),
        headers={
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'Prefer': 'resolution=merge-duplicates,return=minimal',
        },
        method='POST',
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f'[d4 push] WARN provenance upsert failed: {e}', file=sys.stderr)
        return False

def rpc_increment(url, key, model, corpus, role, layer, margin=None):
    # D-quant-supabase-2: increment_role_precision() now takes p_corpus too --
    # moe_role_precision_state's PK widened to (model,corpus,role,layer) so a
    # different corpus's data (e.g. WikiText-103, once Phase 7/8 pushes it)
    # can't silently conflate its counts with WikiText-2's under the same
    # (model,role,layer) row. See migrate_corpus_pk.sql / RESULTS.md.
    req = urllib.request.Request(
        url + "/rest/v1/rpc/increment_role_precision",
        data=json.dumps({"p_model": model, "p_corpus": corpus, "p_role": role, "p_layer": layer, "p_margin": margin}).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"[d4 push] WARN increment_role_precision({model},{corpus},{role},{layer},margin={margin}) failed: {e}", file=sys.stderr)
        return False

def main():
    if len(sys.argv) < 2:
        print("usage: d4_supabase_push.py <jsonl_path>", file=sys.stderr); sys.exit(1)
    path = sys.argv[1]
    url = os.environ.get("QWEN_SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("QWEN_SUPABASE_KEY", "")
    if not url or not key:
        print("FATAL: QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY not set", file=sys.stderr); sys.exit(1)

    offset_path = path + ".pushed_offset"
    offset = 0
    if os.path.exists(offset_path):
        offset = int(open(offset_path).read().strip() or "0")

    # D-push-margin-1 (this session): moe_role_precision_state.min_margin_observed
    # was ALWAYS null in the live DB (269/269 rows, verified via SELECT) --
    # root cause: attribs tuples never carried a margin value in the first
    # place, so every rpc_increment() call used the p_margin=None default.
    #   WHY this fix: attribution rows (kind="attribution") never carried
    #   their own margin field (see qwen_infer.c's attribution fprintf --
    #   only corrected_argmax/orig_argmax/threshold). The margin instead
    #   lives on the "event" row for the same (req,pos).
    #   COST/caveat: (req,pos) is only unique within one manifest -- per
    #   D-qNg64-9's own comment, req numbering restarts at 0 per manifest
    #   file. If a single JSONL file accumulates runs from MULTIPLE
    #   manifests for the SAME (model,corpus) pair, a stale/wrong-run
    #   margin could be joined by coincidence. Event rows don't carry a
    #   manifest field today (only attribution rows do) so this can't be
    #   fully guarded here -- documented, not silently assumed safe.
    #   EXIT: if this becomes a real problem, add "manifest" to the event
    #   row's JSON too and widen the join key to (model,corpus,manifest,req,pos).
    #
    # D-push-margin-2 (this session, SAME-DAY correction of D-push-margin-1's
    # own bug): the first version above built margin_by_key incrementally
    # while streaming forward and assumed the event row always precedes its
    # attribution rows in the file -- verified FALSE by actually looking at a
    # real JSONL: moe_neartie_maybe_correct() (which writes attribution rows)
    # runs BEFORE moe_neartie_maybe_log() (which writes the event row) in
    # both emit loops (qwen_infer.c decode/prefill columns) -- same ordering
    # fact promotion_controller.py's own docstring already documented, which
    # this fix somehow missed the first time despite reading that exact file
    # this session. Caught by re-checking a real fresh JSONL's actual line
    # order after the "fix" landed instead of trusting it worked -- min_margin
    # was STILL null after D-push-margin-1's own supposed fix, which is what
    # forced this second look.
    #   FIX: two passes over the new lines instead of one streaming pass --
    #   pass 1 builds the COMPLETE margin_by_key index from every event row
    #   in this batch, pass 2 resolves every attribution row against it,
    #   independent of which kind physically comes first in the file.
    lines = []
    with open(path) as f:
        f.seek(offset)
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
        new_offset = f.tell()

    event_by_key = {}
    ambiguous_event_keys = set()
    for row in lines:
        if row.get("kind") == "event":
            key = (row["model"], row["corpus"], row["req"], row["pos"])
            if key in event_by_key:
                ambiguous_event_keys.add(key)
            else:
                event_by_key[key] = row

    events, attribs, provenance = [], [], []
    n_events = n_attribs = 0
    for row in lines:
        if row.get("kind") == "event":
            events.append({
                "req": row["req"], "pos": row["pos"],
                "predicted_token": row["predicted_token"], "competing_token": row["competing_token"],
                "margin": row["margin"], "model": row["model"], "corpus": row["corpus"],
                "batch_size": row.get("batch_size"),
                # D-neartie-batch-2: paired B=1 single-stream replay margin for the same
                # (req,pos) -- absent (None) on any JSONL line from before this instrumentation.
                "replay_margin_b1": row.get("replay_margin_b1"),
            })
            n_events += 1
        elif row.get("kind") == "attribution":
            ev_key = (row["model"], row["corpus"], row["req"], row["pos"])
            ev = None if ev_key in ambiguous_event_keys else event_by_key.get(ev_key)
            margin = None if ev is None else ev.get("margin")
            attribs.append((row["model"], row["corpus"], row["role"], row["layer"], margin))
            manifest = row.get("manifest")
            if manifest and row.get("orig_argmax") is not None and row.get("corrected_argmax") is not None:
                provenance.append({
                    "model": row["model"], "corpus": row["corpus"],
                    "role": row["role"], "layer": row["layer"],
                    "manifest": manifest, "req": row["req"], "pos": row["pos"],
                    "orig_argmax": row["orig_argmax"],
                    "corrected_argmax": row["corrected_argmax"],
                    "threshold": row.get("threshold"),
                    "margin": margin,
                    "batch_size": None if ev is None else ev.get("batch_size"),
                    "replay_margin_b1": None if ev is None else ev.get("replay_margin_b1"),
                    "attribution_ts_unix": row.get("ts_unix"),
                    "event_ts_unix": None if ev is None else ev.get("ts_unix"),
                    "source_jsonl": os.path.abspath(path),
                    "last_seen_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                })
            n_attribs += 1
        if len(events) >= BATCH:
            post(url, key, "/rest/v1/moe_neartie_events", events); events = []

    post(url, key, "/rest/v1/moe_neartie_events", events)
    for model, corpus, role, layer, margin in attribs:
        rpc_increment(url, key, model, corpus, role, layer, margin)
    for i in range(0, len(provenance), BATCH):
        upsert_provenance(url, key, provenance[i:i + BATCH])

    with open(offset_path, "w") as f:
        f.write(str(new_offset))
    print(f"[d4 push] pushed {n_events} events, {n_attribs} attribution increments, {len(provenance)} replayable provenance row(s) (offset {offset}->{new_offset})")

if __name__ == "__main__":
    main()
