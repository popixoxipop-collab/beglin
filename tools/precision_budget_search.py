#!/usr/bin/env python3
"""Exact bounded per-tensor qNg64 precision-budget search, CPU or MLX/Metal.

This is an OFFLINE planner, not a runtime promotion or quality certificate.
Candidates must carry measured/simulated per-tensor rel_l2 from an identified
checkpoint. The objective is a WEIGHTED RECONSTRUCTION-ERROR PROXY, not actual
next-token loss, accuracy, latency, or GPU peak-memory. Non-monotone candidates
are intentionally searched exhaustively by Pareto-frontier dynamic programming.

Usage: python3 tools/precision_budget_search.py --input profile.json [--output plan.json]
Never reads model weights or touches serving state.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from qng64_storage import (
    QuantizationConfigurationError, check_backend_n, packed_weight_bytes,
    effective_bits_per_weight,
)

SCHEMA = "beglin-qng64-per-tensor-budget-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PlanError(ValueError):
    pass


class InfeasibleBudget(PlanError):
    pass


class StateLimitExceeded(PlanError):
    pass


def _integer(label, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise PlanError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(label, value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PlanError(f"{label} must be finite")
    value = float(value)
    if positive and value <= 0:
        raise PlanError(f"{label} must be strictly positive")
    if not positive and value < 0:
        raise PlanError(f"{label} cannot be negative")
    return value


def _parse(profile):
    if not isinstance(profile, dict) or profile.get("schema") != SCHEMA:
        raise PlanError(f"profile schema must be {SCHEMA}")
    backend = profile.get("backend")
    if backend not in ("cpu", "mlx_metal"):
        raise PlanError("backend must be cpu or mlx_metal")
    model_id = profile.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise PlanError("model_id is required")
    checkpoint = profile.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or not HEX64.fullmatch(checkpoint):
        raise PlanError("checkpoint_sha256 must be a pinned 64-character lowercase digest")
    budget = _integer("budget_bytes", profile.get("budget_bytes"), 1)
    rows = profile.get("tensors")
    if not isinstance(rows, list) or not rows:
        raise PlanError("at least one tensor is required")
    seen = set()
    names = set()
    tensors = []
    for t in rows:
        if not isinstance(t, dict):
            raise PlanError("tensor rows must be objects")
        name = t.get("name")
        role = t.get("role")
        layer = _integer("layer", t.get("layer"))
        if not isinstance(name, str) or not name.strip() or not isinstance(role, str) or not role.strip():
            raise PlanError("tensor name and role must be nonempty")
        expert = t.get("expert_id")
        if expert is not None:
            expert = _integer("expert_id", expert)
        identity = (role, layer, expert)
        if identity in seen:
            raise PlanError(f"duplicate role/layer/expert_id: {identity}")
        if name in names:
            raise PlanError(f"duplicate tensor name: {name}")
        seen.add(identity)
        names.add(name)
        elements = _integer("elements", t.get("elements"), 1)
        weight = _finite("importance", t.get("importance", 1), positive=True)
        bound = t.get("max_rel_l2")
        if bound is not None:
            bound = _finite("max_rel_l2", bound)
        options = t.get("candidates")
        if not isinstance(options, list) or not options:
            raise PlanError(f"no candidates for {name}")
        candidates = []
        bit_ids = set()
        for c in options:
            if not isinstance(c, dict):
                raise PlanError("candidates must be objects")
            n = c.get("n")
            try:
                check_backend_n(backend, n)
                nbytes = packed_weight_bytes(elements, n)
            except QuantizationConfigurationError as exc:
                raise PlanError(f"{name}: {exc}") from exc
            if n in bit_ids:
                raise PlanError(f"duplicate candidate bit width for {name}: {n}")
            bit_ids.add(n)
            rel = _finite("rel_l2", c.get("rel_l2"))
            if bound is not None and rel > bound:
                continue
            # An explicitly weighted proxy; it is NOT a model-level quality score.
            score = weight * rel * rel
            if not math.isfinite(score):
                raise PlanError("weighted reconstruction score overflow")
            candidates.append({"n": n, "bytes": nbytes, "rel_l2": rel, "cost": score})
        if not candidates:
            raise PlanError(f"no valid candidates after per-tensor quality bound: {name}")
        candidates.sort(key=lambda c: (c["bytes"], c["cost"], c["n"]))
        tensors.append({"name": name, "role": role, "layer": layer,
                        "expert_id": expert, "elements": elements,
                        "importance": weight, "candidates": candidates})
    return backend, model_id, checkpoint, budget, tensors


def solve(profile, max_states=200000):
    max_states = _integer("max_states", max_states, 1)
    backend, model_id, checkpoint, budget, tensors = _parse(profile)
    minimum = sum(min(c["bytes"] for c in t["candidates"]) for t in tensors)
    if minimum > budget:
        raise InfeasibleBudget(f"minimum qNg64 serialized bytes {minimum} > budget {budget}")

    # Each state is cost(bytes) -> (weighted proxy loss, n-choice tuple).
    # Pareto pruning is exact for nonnegative additive costs and loss.
    frontier = {0: (0.0, ())}
    peaks = [1]
    for t in tensors:
        proposed = {}
        for old_bytes, (old_score, choices) in frontier.items():
            for c in t["candidates"]:
                total = old_bytes + c["bytes"]
                if total > budget:
                    continue
                score = old_score + c["cost"]
                choice = choices + (c["n"],)
                prior = proposed.get(total)
                if prior is None or (score, choice) < prior:
                    proposed[total] = (score, choice)
            if len(proposed) > max_states:
                raise StateLimitExceeded("exact frontier exceeds max_states; no approximate plan issued")
        if not proposed:
            raise InfeasibleBudget("no assignment meets budget")
        pruned = {}
        best_error = float("inf")
        for nbytes, option in sorted(proposed.items()):
            if option[0] < best_error:
                pruned[nbytes] = option
                best_error = option[0]
        frontier = pruned
        peaks.append(len(frontier))
    size, (score, choices) = min(frontier.items(), key=lambda pair: (pair[1][0], pair[0], pair[1][1]))
    selections = []
    for t, n in zip(tensors, choices):
        c = next(x for x in t["candidates"] if x["n"] == n)
        selections.append({"name": t["name"], "role": t["role"], "layer": t["layer"],
                           "expert_id": t["expert_id"], "n": n, "packed_bytes": c["bytes"],
                           "modeled_bpw": effective_bits_per_weight(n),
                           "rel_l2": c["rel_l2"], "importance": t["importance"]})

    total_elements = sum(t["elements"] for t in tensors)
    bf16_bytes = 2 * total_elements
    plan = {
        "schema": "beglin-qng64-budget-plan-v1",
        "mode": "OFFLINE_RESEARCH_ONLY_NOT_QUALIFIED_FOR_LIVE",
        "model_id": model_id,
        "checkpoint_sha256": checkpoint,
        "backend": backend,
        "quant_format": "qng64_g64_ef_v1",
        "budget_bytes": budget,
        "serialized_qng64_weight_bytes": size,
        "theoretical_bf16_weight_bytes": bf16_bytes,
        "theoretical_bytes_saved_vs_bf16": bf16_bytes - size,
        "weighted_relative_l2_squared_proxy": score,
        "objective": "minimize sum(importance * rel_l2**2) under serialized qNg64 weight-byte budget",
        "frontier_states_peak": max(peaks),
        "quality_validated": False,
        "latency_measured": False,
        "gpu_resident_memory_measured": False,
        "automatic_live_promotion": False,
        "notes": "rel_l2 is an offline tensor reconstruction proxy, NOT model perplexity; serialized weights exclude CPU/GPU repacks, cache and runtime overhead",
        "selections": selections,
    }
    # Digest the selected logical policy, not the enclosing JSON's file path/time.
    identity = {"model_id": model_id, "checkpoint_sha256": checkpoint, "backend": backend,
                "format": plan["quant_format"],
                "targets": [{"role": x["role"], "layer": x["layer"],
                             "expert_id": x["expert_id"], "n": x["n"]} for x in selections]}
    plan["policy_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON with measured/simulated per-tensor candidates")
    parser.add_argument("--output", type=Path, help="optional result JSON; no weight or serving-state writes")
    parser.add_argument("--max-states", type=int, default=200000, help="fail if exact search exceeds this bound")
    args = parser.parse_args(argv)
    result = solve(json.loads(args.input.read_text(encoding="utf-8")), max_states=args.max_states)
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
