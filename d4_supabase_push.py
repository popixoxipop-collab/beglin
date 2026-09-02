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

def rpc_increment(url, key, role, layer, margin=None):
    req = urllib.request.Request(
        url + "/rest/v1/rpc/increment_role_precision",
        data=json.dumps({"p_role": role, "p_layer": layer, "p_margin": margin}).encode(),
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
        print(f"[d4 push] WARN increment_role_precision({role},{layer}) failed: {e}", file=sys.stderr)
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

    events, attribs = [], []
    n_events = n_attribs = 0
    with open(path) as f:
        f.seek(offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") == "event":
                events.append({
                    "req": row["req"], "pos": row["pos"],
                    "predicted_token": row["predicted_token"], "competing_token": row["competing_token"],
                    "margin": row["margin"], "model": row["model"], "corpus": row["corpus"],
                })
                n_events += 1
            elif row.get("kind") == "attribution":
                attribs.append((row["role"], row["layer"]))
                n_attribs += 1
            if len(events) >= BATCH:
                post(url, key, "/rest/v1/moe_neartie_events", events); events = []
        new_offset = f.tell()

    post(url, key, "/rest/v1/moe_neartie_events", events)
    for role, layer in attribs:
        rpc_increment(url, key, role, layer)

    with open(offset_path, "w") as f:
        f.write(str(new_offset))
    print(f"[d4 push] pushed {n_events} events, {n_attribs} attribution increments (offset {offset}->{new_offset})")

if __name__ == "__main__":
    main()
