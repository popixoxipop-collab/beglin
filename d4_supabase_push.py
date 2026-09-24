#!/usr/bin/env python3
"""D-roadmap-4 Phase 4: batched push of the local JSONL event/attribution log to Supabase.
Reads QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY from env (never hardcoded). Best-effort, non-fatal
on any single request failure (D3: telemetry must never be allowed to break the pipeline) --
prints a warning and continues. Tracks a byte offset in a sidecar file so re-running is safe
(only pushes newly-appended lines).

Usage: python3 d4_supabase_push.py <jsonl_path> [--batch-size N]
"""
import hashlib, json, os, sys, time
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



def upsert_events(url, key, rows):
    if not rows:
        return True
    req = urllib.request.Request(
        url + "/rest/v1/moe_neartie_events?on_conflict=ingest_id",
        data=json.dumps(rows).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=ignore-duplicates,return=minimal",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(
            f"[d4 push] WARN event upsert failed: {e}",
            file=sys.stderr,
        )
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

def rpc_increment(url, key, ingest_id, model, corpus, role, layer, margin=None):
    # D-quant-supabase-2: increment_role_precision() now takes p_corpus too --
    # moe_role_precision_state's PK widened to (model,corpus,role,layer) so a
    # different corpus's data (e.g. WikiText-103, once Phase 7/8 pushes it)
    # can't silently conflate its counts with WikiText-2's under the same
    # (model,role,layer) row. See migrate_corpus_pk.sql / RESULTS.md.
    req = urllib.request.Request(
        url + "/rest/v1/rpc/increment_role_precision_idempotent",
        data=json.dumps({
            "p_ingest_id": ingest_id,
            "p_model": model,
            "p_corpus": corpus,
            "p_role": role,
            "p_layer": layer,
            "p_margin": margin,
        }).encode(),
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
        print(f"[d4 push] WARN increment_role_precision_idempotent({ingest_id},{model},{corpus},{role},{layer},margin={margin}) failed: {e}", file=sys.stderr)
        return False

def _atomic_write_json(path, value):
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(value, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_text(path, value):
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_complete_records(path, offset):
    size = os.path.getsize(path)
    if size < offset:
        raise RuntimeError(
            f"source log shrank below checkpoint: size={size} offset={offset}"
        )
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    if not data:
        return [], offset, b""
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        return [], offset, b""
    complete = data[:last_nl + 1]
    rows = []
    cursor = int(offset)
    for i, raw_with_nl in enumerate(complete.splitlines(keepends=True), 1):
        raw = raw_with_nl.rstrip(b"\r\n")
        if raw.strip():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid complete JSONL record at relative line {i}: {exc}"
                ) from exc
            seed = (
                os.path.abspath(path).encode()
                + b":"
                + str(cursor).encode()
                + b":"
                + raw
            )
            row["_ingest_id"] = hashlib.sha256(seed).hexdigest()
            rows.append(row)
        cursor += len(raw_with_nl)
    return rows, offset + last_nl + 1, complete


def _build_operations(lines, source_path):
    event_by_key = {}
    ambiguous_event_keys = set()
    for row in lines:
        if row.get("kind") == "event":
            event_key = (row["model"], row["corpus"], row["req"], row["pos"])
            if event_key in event_by_key:
                ambiguous_event_keys.add(event_key)
            else:
                event_by_key[event_key] = row

    events, attribs, provenance = [], [], []
    n_events = n_attribs = 0
    for row in lines:
        if row.get("kind") == "event":
            events.append({
                "req": row["req"], "pos": row["pos"],
                "predicted_token": row["predicted_token"],
                "competing_token": row["competing_token"],
                "margin": row["margin"], "model": row["model"],
                "corpus": row["corpus"],
                "batch_size": row.get("batch_size"),
                "replay_margin_b1": row.get("replay_margin_b1"),
                "ingest_id": row["_ingest_id"],
            })
            n_events += 1
        elif row.get("kind") == "attribution":
            event_key = (row["model"], row["corpus"], row["req"], row["pos"])
            event = (
                None if event_key in ambiguous_event_keys
                else event_by_key.get(event_key)
            )
            margin = None if event is None else event.get("margin")
            attribs.append({
                "ingest_id": row["_ingest_id"],
                "model": row["model"], "corpus": row["corpus"],
                "role": row["role"], "layer": row["layer"],
                "margin": margin,
            })
            manifest = row.get("manifest")
            if (
                manifest
                and row.get("orig_argmax") is not None
                and row.get("corrected_argmax") is not None
            ):
                provenance.append({
                    "model": row["model"], "corpus": row["corpus"],
                    "role": row["role"], "layer": row["layer"],
                    "manifest": manifest, "req": row["req"], "pos": row["pos"],
                    "orig_argmax": row["orig_argmax"],
                    "corrected_argmax": row["corrected_argmax"],
                    "threshold": row.get("threshold"),
                    "margin": margin,
                    "batch_size": None if event is None else event.get("batch_size"),
                    "replay_margin_b1": (
                        None if event is None else event.get("replay_margin_b1")
                    ),
                    "attribution_ts_unix": row.get("ts_unix"),
                    "event_ts_unix": None if event is None else event.get("ts_unix"),
                    "source_jsonl": os.path.abspath(source_path),
                    "last_seen_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                })
            n_attribs += 1

    ops = []
    for i in range(0, len(events), BATCH):
        ops.append({"kind": "events", "rows": events[i:i + BATCH], "done": False})
    for item in attribs:
        ops.append({"kind": "increment", "args": item, "done": False})
    for i in range(0, len(provenance), BATCH):
        ops.append({
            "kind": "provenance",
            "rows": provenance[i:i + BATCH],
            "done": False,
        })
    return ops, n_events, n_attribs, len(provenance)


def _execute_operation(url, api_key, op):
    if op["kind"] == "events":
        return upsert_events(url, api_key, op["rows"])
    if op["kind"] == "increment":
        a = op["args"]
        return rpc_increment(
            url, api_key, a["ingest_id"], a["model"], a["corpus"],
            a["role"], a["layer"], a.get("margin"),
        )
    if op["kind"] == "provenance":
        return upsert_provenance(url, api_key, op["rows"])
    raise RuntimeError(f"unknown outbox operation kind: {op['kind']}")


def run_once(path, url, api_key):
    offset_path = path + ".pushed_offset"
    outbox_path = path + ".push_outbox.json"
    offset = 0
    if os.path.exists(offset_path):
        offset = int(open(offset_path).read().strip() or "0")

    if os.path.exists(outbox_path):
        with open(outbox_path) as f:
            outbox = json.load(f)
        if outbox.get("source_path") != os.path.abspath(path):
            raise RuntimeError("outbox source path does not match current source")
        if int(outbox.get("start_offset", -1)) != offset:
            raise RuntimeError(
                "outbox start offset does not match committed checkpoint"
            )
        end = int(outbox["end_offset"])
        with open(path, "rb") as f:
            f.seek(offset)
            complete = f.read(end - offset)
        digest = hashlib.sha256(complete).hexdigest()
        if digest != outbox.get("source_sha256"):
            raise RuntimeError("source bytes changed underneath durable outbox")
    else:
        lines, end, complete = _read_complete_records(path, offset)
        if not lines:
            print(
                f"[d4 push] no complete new JSONL records "
                f"(checkpoint remains {offset})"
            )
            return True
        ops, n_events, n_attribs, n_provenance = _build_operations(lines, path)
        outbox = {
            "version": 1,
            "source_path": os.path.abspath(path),
            "start_offset": offset,
            "end_offset": end,
            "source_sha256": hashlib.sha256(complete).hexdigest(),
            "counts": {
                "events": n_events,
                "attributions": n_attribs,
                "provenance": n_provenance,
            },
            "operations": ops,
        }
        _atomic_write_json(outbox_path, outbox)

    for i, op in enumerate(outbox["operations"]):
        if op.get("done"):
            continue
        if not _execute_operation(url, api_key, op):
            print(
                f"[d4 push] ERROR operation {i}/{len(outbox['operations'])} "
                f"({op['kind']}) failed; checkpoint stays at {offset}",
                file=sys.stderr,
            )
            return False
        op["done"] = True
        _atomic_write_json(outbox_path, outbox)

    end = int(outbox["end_offset"])
    _atomic_write_text(offset_path, str(end))
    os.unlink(outbox_path)
    c = outbox["counts"]
    print(
        f"[d4 push] committed {c['events']} events, "
        f"{c['attributions']} attribution increments, "
        f"{c['provenance']} provenance row(s) "
        f"(offset {offset}->{end})"
    )
    return True


def main():
    if len(sys.argv) < 2:
        print("usage: d4_supabase_push.py <jsonl_path>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    url = os.environ.get("QWEN_SUPABASE_URL", "").rstrip("/")
    api_key = os.environ.get("QWEN_SUPABASE_KEY", "")
    if not url or not api_key:
        print(
            "FATAL: QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY not set",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        ok = run_once(path, url, api_key)
    except Exception as exc:
        print(f"[d4 push] FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
