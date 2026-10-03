#!/usr/bin/env python3
"""Evidence-gated precision allocator for non-monotonic qNg64 policies.

This module never writes a production promotion file. It treats each
(role, layer, n) as an independent discrete candidate and admits a candidate
only when all of the following are true:
  - real qNg64 sweep evidence exists and has no recorded FAIL event,
  - G4 isolated preflight passed,
  - G6 observer/canary passed with zero remaining attribution hits.

Resource cost is estimated from the real checkpoint tensor size and qNg64
effective bits/weight. Optional persistent-worker benchmark evidence adds
latency/RSS dimensions. Cross-target composition is coordinate-wise for v1;
the result explicitly requires pairwise/interaction evidence before production.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import struct
import urllib.parse
import urllib.request

import backend_capabilities as bc


GROUP_SIZE = 64
SUPPORTED_QNG64 = tuple(sorted(set(bc.GPU_NATIVE_QNG64) | set(bc.GPU_CUSTOM_QNG64)))
ATTN = {
    "q_proj": "q_proj",
    "kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
    "kv_b_proj": "kv_b_proj",
    "o_proj": "o_proj",
}
DENSE = {
    "dense_gate_proj": "gate_proj",
    "dense_up_proj": "up_proj",
    "dense_down_proj": "down_proj",
}
SHARED = {
    "shared_gate_proj": "gate_proj",
    "shared_up_proj": "up_proj",
    "shared_down_proj": "down_proj",
}
EXPERT = {
    "expert_gate_proj": "gate_proj",
    "expert_up_proj": "up_proj",
    "expert_down_proj": "down_proj",
}


class AllocatorError(RuntimeError):
    pass


def effective_bpw(n: int) -> float:
    n = int(n)
    if n not in SUPPORTED_QNG64:
        raise AllocatorError(f"unsupported qNg64 width: {n}")
    return n + 32.0 / (GROUP_SIZE * n)


def target_tensor_names(weight_map: dict, role: str, layer: int) -> list[str]:
    layer = int(layer)
    if role in ATTN:
        name = f"model.layers.{layer}.self_attn.{ATTN[role]}.weight"
        return [name] if name in weight_map else []
    if role in DENSE:
        name = f"model.layers.{layer}.mlp.{DENSE[role]}.weight"
        return [name] if name in weight_map else []
    if role in SHARED:
        name = f"model.layers.{layer}.mlp.shared_experts.{SHARED[role]}.weight"
        return [name] if name in weight_map else []
    if role in EXPERT:
        suffix = EXPERT[role]
        pat = re.compile(
            rf"^model\.layers\.{layer}\.mlp\.experts\.\d+\.{suffix}\.weight$"
        )
        return sorted(name for name in weight_map if pat.match(name))
    raise AllocatorError(f"unsupported precision role: {role}")


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise AllocatorError(f"invalid safetensors file: {path}")
        (hlen,) = struct.unpack("<Q", raw)
        return json.loads(handle.read(hlen))


def target_numel(checkpoint_index: str | Path, role: str, layer: int) -> dict:
    index_path = Path(checkpoint_index)
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise AllocatorError("checkpoint index lacks weight_map")
    names = target_tensor_names(weight_map, role, layer)
    if not names:
        raise AllocatorError(f"no checkpoint tensor matched {role}/L{layer}")

    headers: dict[str, dict] = {}
    total = 0
    shapes = {}
    for name in names:
        shard_name = weight_map[name]
        if shard_name not in headers:
            headers[shard_name] = _read_safetensors_header(index_path.parent / shard_name)
        info = headers[shard_name].get(name)
        if not isinstance(info, dict) or "shape" not in info:
            raise AllocatorError(f"tensor metadata missing for {name}")
        shape = [int(v) for v in info["shape"]]
        numel = math.prod(shape)
        total += numel
        shapes[name] = shape
    return {
        "role": role,
        "layer": int(layer),
        "tensor_count": len(names),
        "numel": total,
        "tensors": names,
        "shapes": shapes,
    }


def _credentials() -> tuple[str, str]:
    url = os.environ.get("QWEN_SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("QWEN_SUPABASE_KEY", "")
    if not url or not key:
        raise AllocatorError("QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY are required")
    return url, key


def _get(table: str, params: dict) -> list[dict]:
    url, key = _credentials()
    qs = urllib.parse.urlencode(params, safe=".,*()")
    req = urllib.request.Request(
        f"{url}/rest/v1/{table}?{qs}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        rows = json.loads(resp.read())
    if not isinstance(rows, list):
        raise AllocatorError(f"{table} did not return a JSON array")
    return rows


def fetch_evidence(model: str) -> dict:
    sweeps = _get("moe_quant_sweep_results", {
        "model": f"eq.{model}",
        "source": "eq.qng64_real",
        "select": "corpus,role,layer,req,pos,n,pass,eff_bpw,tested_at",
        "order": "tested_at.asc",
    })
    preflight = _get("moe_live_preflight_results_v3", {
        "model_id": f"eq.{model}",
        "select": "context_hash,role,layer,n,pass,status,tested_at,candidate_run_id",
        "order": "tested_at.asc",
    })
    validation = _get("moe_validation_runs_v3", {
        "model_id": f"eq.{model}",
        "run_kind": "eq.observer",
        "select": "context_hash,role,layer,n,status,pass,created_at,metrics",
        "order": "created_at.asc",
    })
    return {"sweeps": sweeps, "preflight": preflight, "validation": validation}


def _truthy(value) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def build_candidates(evidence: dict, target_sizes: dict, benchmark: dict | None = None) -> list[dict]:
    benchmark = benchmark or {}
    sweep = {}
    for row in evidence.get("sweeps", []):
        key = (str(row["role"]), int(row["layer"]), int(row["n"]))
        event = (str(row["corpus"]), int(row["req"]), int(row["pos"]))
        slot = sweep.setdefault(key, {})
        slot.setdefault(event, set()).add(bool(row["pass"]))

    g4 = {}
    for row in evidence.get("preflight", []):
        key = (str(row["role"]), int(row["layer"]), int(row["n"]))
        if row.get("status") == "PASS" and row.get("pass") is True:
            g4[key] = row

    g6 = {}
    for row in evidence.get("validation", []):
        if row.get("n") is None:
            continue
        key = (str(row["role"]), int(row["layer"]), int(row["n"]))
        metrics = row.get("metrics") or {}
        passed = (
            row.get("status") == "CANARY_PASS"
            and row.get("pass") is True
            and _truthy(metrics.get("target_replay_pass"))
            and int(metrics.get("post_attribution_hits", 1)) == 0
            and not _truthy(metrics.get("rollback_required"))
        )
        if passed:
            g6[key] = row

    out = []
    keys = sorted(set(sweep) | set(g4) | set(g6))
    for role, layer, n in keys:
        if n not in SUPPORTED_QNG64:
            continue
        events = sweep.get((role, layer, n), {})
        pass_events = sum(flags == {True} for flags in events.values())
        fail_events = sum(False in flags for flags in events.values())
        conflicting_events = sum(flags == {False, True} for flags in events.values())
        size = target_sizes.get((role, layer))
        if not size:
            continue
        numel = int(size["numel"])
        bpw = effective_bpw(n)
        estimated_bytes = numel * bpw / 8.0
        bench = benchmark.get(f"{role}:{layer}:{n}", {})
        feasible = (
            pass_events > 0
            and fail_events == 0
            and conflicting_events == 0
            and (role, layer, n) in g4
            and (role, layer, n) in g6
        )
        reasons = []
        if pass_events == 0:
            reasons.append("NO_REAL_PASS_EVENT")
        if fail_events:
            reasons.append("REAL_FAIL_EVENT")
        if conflicting_events:
            reasons.append("CONTRADICTORY_REAL_EVENT")
        if (role, layer, n) not in g4:
            reasons.append("NO_G4_PASS")
        if (role, layer, n) not in g6:
            reasons.append("NO_G6_PASS")
        out.append({
            "role": role,
            "layer": layer,
            "n": n,
            "feasible": feasible,
            "blocked_reasons": reasons,
            "real_pass_events": pass_events,
            "real_fail_events": fail_events,
            "real_event_count": len(events),
            "g4_context_hash": (g4.get((role, layer, n)) or {}).get("context_hash"),
            "g6_context_hash": (g6.get((role, layer, n)) or {}).get("context_hash"),
            "tensor_count": int(size["tensor_count"]),
            "numel": numel,
            "effective_bpw": bpw,
            "estimated_bytes": estimated_bytes,
            "persistent_p50_ms": bench.get("p50_engine_ms"),
            "persistent_p95_ms": bench.get("p95_engine_ms"),
            "persistent_rss_bytes": bench.get("rss_bytes"),
            "benchmark_pid": bench.get("pid"),
        })
    return out


def _dominates(a: dict, b: dict, dimensions: list[str]) -> bool:
    av = [a.get(k) for k in dimensions]
    bv = [b.get(k) for k in dimensions]
    if any(v is None for v in av + bv):
        return False
    return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))


def pareto_frontier(rows: list[dict]) -> list[dict]:
    feasible = [r for r in rows if r["feasible"]]
    dims = ["estimated_bytes"]
    if feasible and all(r.get("persistent_p50_ms") is not None for r in feasible):
        dims.append("persistent_p50_ms")
    if feasible and all(r.get("persistent_rss_bytes") is not None for r in feasible):
        dims.append("persistent_rss_bytes")
    frontier = [
        row for row in feasible
        if not any(other is not row and _dominates(other, row, dims) for other in feasible)
    ]
    return sorted(frontier, key=lambda r: (r["estimated_bytes"], r["n"]))


def choose_target(rows: list[dict], *, memory_weight=1.0, latency_weight=0.0, rss_weight=0.0) -> dict | None:
    feasible = [r for r in rows if r["feasible"]]
    if not feasible:
        return None
    mem0 = min(r["estimated_bytes"] for r in feasible)
    lat_values = [r["persistent_p50_ms"] for r in feasible if r.get("persistent_p50_ms") is not None]
    rss_values = [r["persistent_rss_bytes"] for r in feasible if r.get("persistent_rss_bytes") is not None]
    lat0 = min(lat_values) if lat_values else None
    rss0 = min(rss_values) if rss_values else None

    scored = []
    for row in feasible:
        score = memory_weight * (row["estimated_bytes"] / mem0)
        used = ["memory"]
        if latency_weight and lat0 is not None and row.get("persistent_p50_ms") is not None:
            score += latency_weight * (row["persistent_p50_ms"] / lat0)
            used.append("latency")
        if rss_weight and rss0 is not None and row.get("persistent_rss_bytes") is not None:
            score += rss_weight * (row["persistent_rss_bytes"] / rss0)
            used.append("rss")
        scored.append((score, -row["real_pass_events"], row["n"], row, used))
    _, _, _, chosen, used = min(scored, key=lambda x: (x[0], x[1], x[2]))
    return {**chosen, "objective_score": min(x[0] for x in scored), "objective_dimensions": used}


def optimize(candidates: list[dict], current_policy: list[dict] | None = None, **weights) -> dict:
    current = {
        (str(row["role"]), int(row["layer"])): int(row["n"])
        for row in (current_policy or [])
    }
    grouped = {}
    for row in candidates:
        grouped.setdefault((row["role"], row["layer"]), []).append(row)

    proposals = []
    proposed_policy = dict(current)
    for target, rows in sorted(grouped.items()):
        chosen = choose_target(rows, **weights)
        frontier = pareto_frontier(rows)
        if chosen is None:
            proposals.append({
                "role": target[0], "layer": target[1],
                "status": "NO_FEASIBLE_CANDIDATE",
                "current_n": current.get(target),
                "pareto_frontier": frontier,
            })
            continue
        proposed_policy[target] = int(chosen["n"])
        alternates = [
            {"n": r["n"], "real_pass_events": r["real_pass_events"],
             "persistent_p50_ms": r.get("persistent_p50_ms")}
            for r in rows if r["feasible"] and r["n"] != chosen["n"]
        ]
        proposals.append({
            "role": target[0],
            "layer": target[1],
            "status": "PROPOSED",
            "current_n": current.get(target),
            "selected_n": chosen["n"],
            "selected": chosen,
            "pareto_frontier": frontier,
            "dynamic_escalation": {
                "status": "EVIDENCE_REQUIRED",
                "reason": (
                    "v1 does not assume higher n is more accurate; an alternate is enabled "
                    "only after contextual trigger-conditioned evidence exists"
                ),
                "candidate_alternates": alternates,
                "future_triggers": ["near_tie", "low_margin", "high_entropy", "routing_ambiguity"],
            },
        })

    normalized = [
        {"role": role, "layer": layer, "n": n}
        for (role, layer), n in sorted(proposed_policy.items())
    ]
    return {
        "schema": "beglin-precision-allocator-v1",
        "status": "PROPOSAL_ONLY",
        "production_write_allowed": False,
        "interaction_model": "coordinate_independent_v1",
        "pairwise_evidence_required_before_combined_production": True,
        "proposed_policy": normalized,
        "targets": proposals,
    }


def _load_benchmark(path: str | None) -> dict:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text())
    rows = obj.get("rows", obj)
    out = {}
    if isinstance(rows, list):
        for row in rows:
            out[f"{row['role']}:{int(row['layer'])}:{int(row['n'])}"] = row
    elif isinstance(rows, dict):
        out = rows
    return out


def _parse_policy(text: str | None) -> list[dict]:
    if not text:
        return []
    rows = []
    for item in text.split(","):
        role, layer, n = item.split(":")
        rows.append({"role": role, "layer": int(layer), "n": int(n)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-v2-lite")
    ap.add_argument("--checkpoint-index", required=True)
    ap.add_argument("--current-policy", help="role:layer:n,role:layer:n")
    ap.add_argument("--benchmark-json")
    ap.add_argument("--memory-weight", type=float, default=1.0)
    ap.add_argument("--latency-weight", type=float, default=0.0)
    ap.add_argument("--rss-weight", type=float, default=0.0)
    ap.add_argument("--output")
    args = ap.parse_args()

    evidence = fetch_evidence(args.model)
    targets = {
        (str(row["role"]), int(row["layer"]))
        for source in (evidence["sweeps"], evidence["preflight"], evidence["validation"])
        for row in source
        if row.get("role") is not None and row.get("layer") is not None
    }
    sizes = {}
    for target in sorted(targets):
        try:
            sizes[target] = target_numel(args.checkpoint_index, *target)
        except AllocatorError:
            continue
    candidates = build_candidates(evidence, sizes, _load_benchmark(args.benchmark_json))
    result = optimize(
        candidates,
        current_policy=_parse_policy(args.current_policy),
        memory_weight=args.memory_weight,
        latency_weight=args.latency_weight,
        rss_weight=args.rss_weight,
    )
    result["model"] = args.model
    result["candidate_count"] = len(candidates)
    result["feasible_candidate_count"] = sum(r["feasible"] for r in candidates)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
