#!/usr/bin/env python3
"""REST persistence helpers for Precision Evidence Contract v3."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


class EvidenceV3Unavailable(RuntimeError):
    pass


def _credentials():
    url = os.environ.get("QWEN_SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("QWEN_SUPABASE_KEY", "")
    if not url or not key:
        raise EvidenceV3Unavailable(
            "QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY are required"
        )
    return url, key


def _headers(key, *, write=False):
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    if write:
        h["Content-Type"] = "application/json"
    return h


def _write(table, payload, *, conflict=None):
    url, key = _credentials()
    suffix = f"?on_conflict={conflict}" if conflict else ""
    headers = {
        **_headers(key, write=True),
        "Prefer": (
            "resolution=merge-duplicates,return=representation"
            if conflict else "return=representation"
        ),
    }
    req = urllib.request.Request(
        f"{url}/rest/v1/{table}{suffix}",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        if exc.code in (400, 404) and table in body:
            raise EvidenceV3Unavailable(
                f"v3 table {table} unavailable: HTTP {exc.code}: {body[:300]}"
            ) from exc
        raise RuntimeError(
            f"v3 write failed for {table}: HTTP {exc.code}: {body[:500]}"
        ) from exc
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"unverified v3 write for {table}: {rows!r}")
    return rows


def upsert_execution_context(context):
    payload = context.payload()
    payload["schema_version"] = payload.pop("schema")
    payload["context_hash"] = context.context_hash
    return _write(
        "moe_execution_contexts_v3",
        payload,
        conflict="context_hash",
    )[0]


def insert_validation_run(row):
    return _write("moe_validation_runs_v3", row)[0]


def insert_transaction(row):
    return _write(
        "moe_precision_transactions_v3",
        row,
        conflict="txn_id",
    )[0]


def fetch_latest_preflight(context_hash, role, layer, n):
    url, key = _credentials()
    params = {
        "context_hash": f"eq.{context_hash}",
        "role": f"eq.{role}",
        "layer": f"eq.{int(layer)}",
        "n": f"eq.{int(n)}",
        "select": "*",
        "order": "tested_at.desc",
        "limit": "1",
    }
    qs = urllib.parse.urlencode(params, safe=".,")
    req = urllib.request.Request(
        f"{url}/rest/v1/moe_live_preflight_results_v3?{qs}",
        headers=_headers(key),
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        if exc.code in (400, 404) and "moe_live_preflight_results_v3" in body:
            raise EvidenceV3Unavailable(
                f"v3 preflight table unavailable: HTTP {exc.code}: {body[:300]}"
            ) from exc
        raise
    return rows[0] if rows else None
