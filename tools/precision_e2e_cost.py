#!/usr/bin/env python3
"""Measured end-to-end cost enrichment for precision allocator candidates.

Correctness/evidence remains a hard gate elsewhere. This module only attaches
cost dimensions to candidates that are already being considered by the
allocator. Runtime cache state comes from the durable native ACK.
"""
from __future__ import annotations

import copy
import json

import precision_context as pc


class PrecisionCostError(RuntimeError):
    pass


def _key(role, layer, n):
    return str(role), int(layer), int(n)


def _profile_index(evidence: dict) -> dict:
    if not isinstance(evidence, dict):
        raise PrecisionCostError("cost evidence must be an object")
    if evidence.get("schema") != "beglin-precision-e2e-cost-v1":
        raise PrecisionCostError("unexpected cost evidence schema")
    rows = evidence.get("target_profiles")
    if not isinstance(rows, list) or not rows:
        raise PrecisionCostError("cost evidence target_profiles must be non-empty")
    out = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PrecisionCostError(f"target_profiles[{idx}] must be an object")
        try:
            key = _key(row["role"], row["layer"], row["n"])
            steady_engine = float(row["steady_engine_p50_ms"])
            steady_roundtrip = float(row["steady_roundtrip_p50_ms"])
            rss = int(row["rss_bytes"])
            passes = int(row.get("expected_inference_passes", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise PrecisionCostError(f"target_profiles[{idx}] is invalid") from exc
        if min(steady_engine, steady_roundtrip, rss, passes) < 0 or passes < 1:
            raise PrecisionCostError(f"target_profiles[{idx}] has invalid costs")
        if key in out:
            raise PrecisionCostError(f"duplicate target cost profile: {key}")
        out[key] = copy.deepcopy(row)
    return out


def _cache_index(runtime_state: dict) -> dict[tuple[str, int, int], int]:
    rows = runtime_state.get("qng64_cache", []) if isinstance(runtime_state, dict) else []
    out = {}
    for row in rows:
        key = _key(row["role"], row["layer"], row["n"])
        out[key] = int(row["bytes"])
    return out
def _transition(profile: dict, from_n: int, cache_state: str) -> dict | None:
    for row in profile.get("transitions", []):
        if int(row.get("from_n", -1)) == int(from_n) and row.get("cache_state") == cache_state:
            try:
                out = {
                    "transition_p50_ms": float(row["p50_transition_ms"]),
                    "cache_hits": int(row.get("cache_hits", 0)),
                    "cache_misses": int(row.get("cache_misses", 0)),
                    "cache_bytes_added": int(row.get("cache_bytes_added", 0)),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise PrecisionCostError("invalid transition cost row") from exc
            if min(out.values()) < 0:
                raise PrecisionCostError("transition cost row contains negative value")
            return out
    return None


def enrich_candidates(
    candidates: list[dict],
    *,
    current_policy: list[dict],
    runtime_state: dict,
    cost_evidence: dict,
) -> list[dict]:
    profiles = _profile_index(cost_evidence)
    current = {
        (str(row["role"]), int(row["layer"])): int(row["n"])
        for row in pc.normalize_policy(current_policy)
    }
    cache = _cache_index(runtime_state)
    resident_before = int(runtime_state.get("resident_qng64_cache_bytes", 0) or 0)
    if resident_before < 0:
        raise PrecisionCostError("runtime resident cache bytes are invalid")

    out = []
    for raw in candidates:
        row = copy.deepcopy(raw)
        key = _key(row["role"], row["layer"], row["n"])
        profile = profiles.get(key)
        if profile is None:
            row["cost_evidence_complete"] = False
            out.append(row)
            continue
        role_layer = (key[0], key[1])
        from_n = current.get(role_layer)
        if from_n is None:
            raise PrecisionCostError(
                f"current policy lacks cost target {key[0]}/L{key[1]}"
            )
        target_cached = key in cache
        if from_n == key[2]:
            transition = {
                "transition_p50_ms": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_bytes_added": 0,
            }
            cache_state = "active"
        else:
            cache_state = "warm" if target_cached else "cold"
            transition = _transition(profile, from_n, cache_state)
            if transition is None:
                row["cost_evidence_complete"] = False
                row["cost_missing_transition"] = {
                    "from_n": from_n,
                    "to_n": key[2],
                    "cache_state": cache_state,
                }
                out.append(row)
                continue
        steady_engine = float(profile["steady_engine_p50_ms"])
        steady_roundtrip = float(profile["steady_roundtrip_p50_ms"])
        passes = int(profile.get("expected_inference_passes", 1))
        resident_after = resident_before + int(transition["cache_bytes_added"])
        row.update({
            "persistent_p50_ms": steady_engine,
            "persistent_rss_bytes": int(profile["rss_bytes"]),
            "steady_roundtrip_p50_ms": steady_roundtrip,
            "expected_inference_passes": passes,
            "transition_cache_state": cache_state,
            "transition_p50_ms": float(transition["transition_p50_ms"]),
            "transition_cache_hits": int(transition["cache_hits"]),
            "transition_cache_misses": int(transition["cache_misses"]),
            "transition_cache_bytes_added": int(transition["cache_bytes_added"]),
            "resident_cache_bytes_after": resident_after,
            "expected_compute_ms": float(transition["transition_p50_ms"]) + steady_engine * passes,
            "expected_e2e_ms": float(transition["transition_p50_ms"]) + steady_roundtrip * passes,
            "cost_evidence_complete": True,
        })
        out.append(row)
    return out
def snapshot_sha256(evidence: dict) -> str:
    import hashlib
    return hashlib.sha256(pc.canonical_json(evidence).encode()).hexdigest()


def main() -> int:
    import argparse
    from pathlib import Path
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates",required=True)
    ap.add_argument("--current-policy",required=True)
    ap.add_argument("--runtime-state",required=True)
    ap.add_argument("--cost-evidence",required=True)
    args=ap.parse_args()
    got=enrich_candidates(
        json.loads(Path(args.candidates).read_text()),
        current_policy=json.loads(Path(args.current_policy).read_text()),
        runtime_state=json.loads(Path(args.runtime_state).read_text()),
        cost_evidence=json.loads(Path(args.cost_evidence).read_text()),
    )
    print(json.dumps(got,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
