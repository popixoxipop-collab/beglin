#!/usr/bin/env python3
"""Read-only production evidence discovery for GPU precision shadow mode.

This module intentionally performs no POST/PATCH/DELETE and writes no live
promotion/control state. It reads the existing Supabase evidence tables,
computes the already-reviewed qNg64 safe-n decision, joins replayable
attribution provenance, and emits a local shadow candidate queue.

The queue is not production approval. A READY row only means there is enough
read-only evidence to hand the candidate to the separate GPU shadow runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import urllib.parse
import urllib.request

import attribution_provenance as provenance
import promotion_writeback as pwb


SCHEMA = "gpu-shadow-discovery-v1"
BACKEND = "mlx_metal"


class ShadowDiscoveryError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _credentials(env=None):
    env = os.environ if env is None else env
    url = str(env.get("QWEN_SUPABASE_URL", "")).rstrip("/")
    key = str(env.get("QWEN_SUPABASE_KEY", ""))
    if not url or not key:
        raise ShadowDiscoveryError(
            "QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY are required for read-only discovery"
        )
    return url, key


def fetch_candidates(model: str, limit: int = 100, *, opener=None, env=None):
    """Read undecided production candidates using one GET request."""
    url, key = _credentials(env)
    opener = urllib.request.urlopen if opener is None else opener
    params = {
        "model": f"eq.{model}",
        "event_count": "gt.0",
        "promoted": "eq.false",
        "select": "model,role,layer,event_count,current_bits,min_margin_observed",
        "order": "event_count.desc,role.asc,layer.asc",
        "limit": str(int(limit)),
    }
    qs = urllib.parse.urlencode(params, safe=".,")
    req = urllib.request.Request(
        f"{url}/rest/v1/moe_role_precision_state?{qs}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        method="GET",
    )
    with opener(req, timeout=30) as resp:
        rows = json.loads(resp.read())
    if not isinstance(rows, list):
        raise ShadowDiscoveryError("candidate query did not return a JSON array")
    return rows


def _candidate_id(model: str, role: str, layer: int, n: int, provenance_id) -> str:
    raw = {
        "model": model,
        "role": role,
        "layer": int(layer),
        "n": int(n),
        "provenance_id": provenance_id,
    }
    return sha256_json(raw)[:24]


def _ready_row(model: str, candidate: dict, n: int, detail: dict, prov: dict) -> dict:
    role = str(candidate["role"])
    layer = int(candidate["layer"])
    orig = int(prov["orig_argmax"])
    corrected = int(prov["corrected_argmax"])
    if orig == corrected:
        raise ShadowDiscoveryError(
            f"provenance has no actual token flip for {role}/L{layer}"
        )
    return {
        "candidate_id": _candidate_id(model, role, layer, n, prov.get("id")),
        "status": "READY",
        "backend": BACKEND,
        "model": model,
        "role": role,
        "layer": layer,
        "n": int(n),
        "event_count": int(candidate.get("event_count") or 0),
        "current_bits": candidate.get("current_bits"),
        "min_margin_observed": candidate.get("min_margin_observed"),
        "event": {
            "orig_token": orig,
            "corrected_token": corrected,
            "pos": int(prov["pos"]),
            "req": int(prov["req"]),
        },
        "reference": {"emitted_token": corrected},
        "provenance": {
            "id": prov.get("id"),
            "corpus": prov.get("corpus"),
            "manifest": prov.get("manifest"),
            "source_jsonl": prov.get("source_jsonl"),
            "margin": prov.get("margin"),
            "batch_size": prov.get("batch_size"),
            "replay_margin_b1": prov.get("replay_margin_b1"),
            "last_seen_at": prov.get("last_seen_at"),
        },
        "safe_n_evidence": detail,
    }


def discover(
    model: str,
    limit: int = 100,
    *,
    fetch_candidates_fn=fetch_candidates,
    target_safe_n_fn=pwb.target_safe_n,
    fetch_best_provenance_fn=provenance.fetch_best,
) -> dict:
    rows = fetch_candidates_fn(model, limit)
    decisions = []
    ready = []

    for candidate in rows:
        role = str(candidate["role"])
        layer = int(candidate["layer"])
        n, detail = target_safe_n_fn(model, role, layer)
        base = {
            "model": model,
            "role": role,
            "layer": layer,
            "event_count": int(candidate.get("event_count") or 0),
            "current_bits": candidate.get("current_bits"),
            "min_margin_observed": candidate.get("min_margin_observed"),
        }
        if n is None:
            decisions.append({
                **base,
                "status": "SKIP_NO_SAFE_N",
                "reason": (detail or {}).get("reason", "no deployment-safe qNg64 n"),
                "safe_n_evidence": detail,
            })
            continue

        try:
            prov = fetch_best_provenance_fn(model, role, layer)
        except provenance.ProvenanceStoreUnavailable as exc:
            decisions.append({
                **base,
                "status": "PROVENANCE_STORE_UNAVAILABLE",
                "reason": str(exc),
                "safe_n_evidence": detail,
            })
            continue
        if prov is None:
            decisions.append({
                **base,
                "status": "NEEDS_ATTRIBUTION_PROVENANCE",
                "reason": "safe n exists but no replayable attribution provenance row exists",
                "safe_n_evidence": detail,
            })
            continue

        item = _ready_row(model, candidate, n, detail, prov)
        ready.append(item)
        decisions.append(item)

    result = {
        "schema": SCHEMA,
        "mode": "read_only",
        "production_write_allowed": False,
        "backend": BACKEND,
        "model": model,
        "generated_at": _now(),
        "candidate_count": len(rows),
        "ready_count": len(ready),
        "ready": ready,
        "decisions": decisions,
    }
    result["payload_sha256"] = sha256_json(result)
    return result


def _atomic_json(path: str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-v2-lite")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        result = discover(args.model, args.limit)
        _atomic_json(args.output, result)
    except Exception as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "mode": "read_only",
            "production_write_allowed": False,
            "status": "DISCOVERY_ERROR",
            "error": str(exc),
        }, sort_keys=True))
        return 2

    print(json.dumps({
        "schema": SCHEMA,
        "mode": "read_only",
        "production_write_allowed": False,
        "status": "DISCOVERY_COMPLETE",
        "candidate_count": result["candidate_count"],
        "ready_count": result["ready_count"],
        "payload_sha256": result["payload_sha256"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
