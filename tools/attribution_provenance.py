#!/usr/bin/env python3
"""Durable attribution provenance lookup for replayable P5 evidence."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

TABLE = "moe_attribution_provenance"


class ProvenanceStoreUnavailable(RuntimeError):
    pass


def _credentials():
    url = os.environ.get("QWEN_SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("QWEN_SUPABASE_KEY", "")
    if not url or not key:
        raise ProvenanceStoreUnavailable(
            "QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY are required"
        )
    return url, key


def fetch_latest(model, role, layer, limit=20):
    """Return newest replayable provenance rows for a target."""
    url, key = _credentials()
    params = {
        "model": f"eq.{model}",
        "role": f"eq.{role}",
        "layer": f"eq.{int(layer)}",
        "select": (
            "id,first_seen_at,last_seen_at,source,model,corpus,role,layer,"
            "manifest,req,pos,orig_argmax,corrected_argmax,threshold,margin,"
            "batch_size,replay_margin_b1,attribution_ts_unix,event_ts_unix,"
            "source_jsonl"
        ),
        "order": "last_seen_at.desc,id.desc",
        "limit": str(int(limit)),
    }
    qs = urllib.parse.urlencode(params, safe=".,")
    req = urllib.request.Request(
        f"{url}/rest/v1/{TABLE}?{qs}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        if exc.code in (400, 404) and TABLE in body:
            raise ProvenanceStoreUnavailable(
                f"attribution provenance table unavailable: HTTP {exc.code}: "
                f"{body[:300]}"
            ) from exc
        raise RuntimeError(
            f"provenance lookup failed: HTTP {exc.code}: {body[:500]}"
        ) from exc
    return [
        r for r in rows
        if r.get("manifest")
        and r.get("orig_argmax") is not None
        and r.get("corrected_argmax") is not None
    ]


def fetch_best(model, role, layer):
    rows = fetch_latest(model, role, layer, limit=1)
    return rows[0] if rows else None
