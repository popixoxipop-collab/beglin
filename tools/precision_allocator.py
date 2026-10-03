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
import hashlib
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
    trigger = _get("moe_precision_trigger_evidence_v1", {
        "model_id": f"eq.{model}",
        "select": "*",
        "order": "created_at.asc",
    })
    return {
        "sweeps": sweeps,
        "preflight": preflight,
        "validation": validation,
        "trigger": trigger,
    }


def _truthy(value) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _trigger_proves_benefit(row: dict) -> bool:
    if row.get("status") != "PASS" or row.get("pass") is not True:
        return False
    if int(row.get("requests", 0)) <= 0:
        return False
    metrics = row.get("metrics") or {}
    try:
        if "base_failures" in metrics and "target_failures" in metrics:
            return int(metrics["base_failures"]) > 0 and int(metrics["target_failures"]) == 0
    except (TypeError, ValueError):
        return False
    return metrics.get("base_pass") is False and metrics.get("target_pass") is True


def _trigger_source_event(row: dict):
    event = (row.get("metrics") or {}).get("source_event") or {}
    try:
        return (
            str(event["corpus"]),
            int(event["req"]),
            int(event["pos"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


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
        pass_event_keys = [
            {"corpus": ev[0], "req": ev[1], "pos": ev[2]}
            for ev, flags in sorted(events.items())
            if flags == {True}
        ]
        fail_event_keys = [
            {"corpus": ev[0], "req": ev[1], "pos": ev[2]}
            for ev, flags in sorted(events.items())
            if False in flags
        ]
        out.append({
            "role": role,
            "layer": layer,
            "n": n,
            "feasible": feasible,
            "conditionally_feasible": False,
            "conditional_recoveries": [],
            "blocked_reasons": reasons,
            "real_pass_events": pass_events,
            "real_fail_events": fail_events,
            "real_event_count": len(events),
            "real_pass_event_keys": pass_event_keys,
            "real_fail_event_keys": fail_event_keys,
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
    by_key = {
        (row["role"], int(row["layer"]), int(row["n"])): row
        for row in out
    }
    triggers = evidence.get("trigger", [])
    for base in out:
        # A conditional base is intentionally allowed to fail only on events
        # that have exact trigger-conditioned recovery evidence. It must still
        # have its own G4/G6 pass on at least one non-trigger event.
        if (
            base["real_pass_events"] <= 0
            or base["real_fail_events"] <= 0
            or "CONTRADICTORY_REAL_EVENT" in base["blocked_reasons"]
            or "NO_G4_PASS" in base["blocked_reasons"]
            or "NO_G6_PASS" in base["blocked_reasons"]
        ):
            continue
        fail_events = {
            (row["corpus"], int(row["req"]), int(row["pos"]))
            for row in base["real_fail_event_keys"]
        }
        covered = {event: [] for event in fail_events}
        for trigger in triggers:
            if (
                str(trigger.get("role")) != base["role"]
                or int(trigger.get("layer", -1)) != int(base["layer"])
                or int(trigger.get("from_n", -1)) != int(base["n"])
                or not _trigger_proves_benefit(trigger)
            ):
                continue
            event = _trigger_source_event(trigger)
            if event not in fail_events:
                continue
            try:
                to_n = int(trigger["to_n"])
            except (KeyError, TypeError, ValueError):
                continue
            target = by_key.get((base["role"], int(base["layer"]), to_n))
            if target is None or target.get("feasible") is not True:
                continue
            covered[event].append({
                "evidence_id": trigger.get("evidence_id"),
                "trigger_type": trigger.get("trigger_type"),
                "signal_bucket": trigger.get("signal_bucket") or {},
                "from_n": int(base["n"]),
                "to_n": to_n,
                "source_event": {
                    "corpus": event[0],
                    "req": event[1],
                    "pos": event[2],
                },
                "requests": int(trigger.get("requests", 0)),
                "evidence_sha256": trigger.get("evidence_sha256"),
                "target_g4_context_hash": target.get("g4_context_hash"),
                "target_g6_context_hash": target.get("g6_context_hash"),
            })
        if fail_events and all(covered[event] for event in fail_events):
            base["conditionally_feasible"] = True
            base["conditional_recoveries"] = [
                row
                for event in sorted(covered)
                for row in covered[event]
            ]

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


def _objective_dimensions(weights: dict) -> list[tuple[str, str, float]]:
    specs = [
        ("memory", "estimated_bytes", float(weights.get("memory_weight", 1.0))),
        ("latency", "persistent_p50_ms", float(weights.get("latency_weight", 0.0))),
        ("rss", "persistent_rss_bytes", float(weights.get("rss_weight", 0.0))),
        ("transition", "transition_p50_ms", float(weights.get("transition_weight", 0.0))),
        ("cache", "resident_cache_bytes_after", float(weights.get("cache_weight", 0.0))),
        ("inference_passes", "expected_inference_passes", float(weights.get("inference_pass_weight", 0.0))),
        ("e2e", "expected_e2e_ms", float(weights.get("e2e_weight", 0.0))),
    ]
    for name, _, weight in specs:
        if weight < 0:
            raise AllocatorError(f"objective weight {name} must be non-negative")
    return [row for row in specs if row[2] != 0.0]


def _score_candidate(row: dict, universe: list[dict], dimensions: list[tuple[str, str, float]]) -> tuple[float, list[str]]:
    score = 0.0
    used = []
    for name, field, weight in dimensions:
        values = [candidate.get(field) for candidate in universe]
        if any(value is None for value in values):
            raise AllocatorError(
                f"objective dimension {name} requires complete measured field {field}"
            )
        vals = [float(value) for value in values]
        if any(value < 0 for value in vals):
            raise AllocatorError(f"objective dimension {name} contains negative values")
        value = float(row[field])
        baseline = min(vals)
        if baseline > 0:
            normalized = value / baseline
        else:
            scale = max(max(vals), 1.0)
            normalized = value / scale
        score += weight * normalized
        used.append(name)
    return score, used


def _choose_scored(candidates: list[dict], universe: list[dict], **weights) -> dict | None:
    if not candidates:
        return None
    dimensions = _objective_dimensions(weights)
    if not dimensions:
        raise AllocatorError("at least one objective weight must be non-zero")
    scored = []
    for row in candidates:
        score, used = _score_candidate(row, universe, dimensions)
        scored.append((score, -int(row.get("real_pass_events", 0)), int(row["n"]), row, used))
    score, _, _, chosen, used = min(scored, key=lambda x: (x[0], x[1], x[2]))
    return {**chosen, "objective_score": score, "objective_dimensions": used}


def choose_target(rows: list[dict], **weights) -> dict | None:
    feasible = [r for r in rows if r["feasible"]]
    universe = [r for r in rows if r["feasible"] or r.get("conditionally_feasible")]
    return _choose_scored(feasible, universe or feasible, **weights)


def choose_conditional_base(rows: list[dict], static_safe: dict | None, **weights) -> dict | None:
    candidates = [r for r in rows if r.get("conditionally_feasible")]
    universe = [r for r in rows if r["feasible"] or r.get("conditionally_feasible")]
    chosen = _choose_scored(candidates, universe or candidates, **weights)
    if chosen is None:
        return None
    if static_safe is not None and chosen["objective_score"] >= static_safe["objective_score"]:
        return None
    return chosen


def _alternate_summary(row: dict) -> dict:
    out = {
        "n": row["n"],
        "real_pass_events": row["real_pass_events"],
        "persistent_p50_ms": row.get("persistent_p50_ms"),
    }
    for key in (
        "transition_p50_ms",
        "resident_cache_bytes_after",
        "expected_inference_passes",
        "expected_e2e_ms",
        "transition_cache_state",
    ):
        if row.get(key) is not None:
            out[key] = row.get(key)
    return out


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
        static_safe = choose_target(rows, **weights)
        conditional_base = choose_conditional_base(rows, static_safe, **weights)
        frontier = pareto_frontier(rows)

        if conditional_base is not None:
            proposed_policy[target] = int(conditional_base["n"])
            recovery_ns = sorted({
                int(row["to_n"])
                for row in conditional_base["conditional_recoveries"]
            })
            alternates = []
            for n in recovery_ns:
                target_row = next(
                    r for r in rows
                    if int(r["n"]) == n and r.get("feasible") is True
                )
                alternates.append(_alternate_summary(target_row))
            proposals.append({
                "role": target[0],
                "layer": target[1],
                "status": "PROPOSED",
                "policy_mode": "CONDITIONAL",
                "current_n": current.get(target),
                "selected_n": conditional_base["n"],
                "static_safe_n": static_safe["n"] if static_safe else None,
                "selected": conditional_base,
                "pareto_frontier": frontier,
                "dynamic_escalation": {
                    "status": "EVIDENCE_READY",
                    "reason": (
                        "low-cost base is admitted only outside exact failure events; "
                        "all observed base failures have trigger-conditioned recovery "
                        "evidence to a backend-admitted alternate"
                    ),
                    "candidate_alternates": alternates,
                    "recoveries": conditional_base["conditional_recoveries"],
                    "future_triggers": ["near_tie", "low_margin", "high_entropy", "routing_ambiguity"],
                },
            })
            continue

        chosen = static_safe
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
            _alternate_summary(r)
            for r in rows if r["feasible"] and r["n"] != chosen["n"]
        ]
        proposals.append({
            "role": target[0],
            "layer": target[1],
            "status": "PROPOSED",
            "policy_mode": "STATIC",
            "current_n": current.get(target),
            "selected_n": chosen["n"],
            "selected": chosen,
            "pareto_frontier": frontier,
            "dynamic_escalation": {
                "status": "EVIDENCE_REQUIRED",
                "reason": (
                    "v1 does not assume higher n is more accurate; an alternate is enabled "
                    "only after contextual trigger-conditioned evidence proves the base "
                    "actually fails and the alternate removes that failure"
                ),
                "candidate_alternates": alternates,
                "future_triggers": ["near_tie", "low_margin", "high_entropy", "routing_ambiguity"],
            },
        })

    normalized = [
        {"role": role, "layer": layer, "n": n}
        for (role, layer), n in sorted(proposed_policy.items())
    ]
    current_normalized = [
        {"role": role, "layer": layer, "n": n}
        for (role, layer), n in sorted(current.items())
    ]
    return {
        "schema": "beglin-precision-allocator-v1",
        "status": "PROPOSAL_ONLY",
        "production_write_allowed": False,
        "interaction_model": "coordinate_independent_v1",
        "pairwise_evidence_required_before_combined_production": True,
        "current_policy": current_normalized,
        "proposed_policy": normalized,
        "objective_weights": {
            "memory": float(weights.get("memory_weight", 1.0)),
            "latency": float(weights.get("latency_weight", 0.0)),
            "rss": float(weights.get("rss_weight", 0.0)),
            "transition": float(weights.get("transition_weight", 0.0)),
            "cache": float(weights.get("cache_weight", 0.0)),
            "inference_passes": float(weights.get("inference_pass_weight", 0.0)),
            "e2e": float(weights.get("e2e_weight", 0.0)),
        },
        "targets": proposals,
    }


def persist_decision(result: dict, model: str) -> dict:
    url, key = _credentials()
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    decision_id = hashlib.sha256(canonical.encode()).hexdigest()
    contexts = sorted({
        ctx
        for target in result.get("targets", [])
        for row in (
            list(target.get("pareto_frontier", []))
            + ([target.get("selected")] if target.get("selected") else [])
        )
        for ctx in (row.get("g4_context_hash"), row.get("g6_context_hash"))
        if ctx
    })
    payload = {
        "decision_id": decision_id,
        "model_id": model,
        "schema_version": result["schema"],
        "status": result["status"],
        "current_policy": result.get("current_policy", []),
        "proposed_policy": result.get("proposed_policy", []),
        "objective": {
            "weights": result.get("objective_weights", {}),
            "interaction_model": result.get("interaction_model"),
        },
        "constraints": {
            "quality_is_hard_constraint": True,
            "pairwise_evidence_required_before_combined_production": result.get(
                "pairwise_evidence_required_before_combined_production", True
            ),
        },
        "target_decisions": result.get("targets", []),
        "source_contexts": contexts,
        "production_write_allowed": False,
    }
    req = urllib.request.Request(
        f"{url}/rest/v1/moe_precision_allocator_decisions_v1?on_conflict=decision_id",
        data=json.dumps(payload).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=representation",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        rows = json.loads(resp.read())
    if not isinstance(rows, list) or len(rows) != 1:
        raise AllocatorError("allocator decision persistence was not verified")
    return rows[0]


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
    ap.add_argument("--persist", action="store_true")
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
    if args.persist:
        persisted = persist_decision(result, args.model)
        result["persisted_decision_id"] = persisted["decision_id"]
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
