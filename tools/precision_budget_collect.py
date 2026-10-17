#!/usr/bin/env python3
"""Hash-pinned quant_sim_n summary collection -> per-tensor qNg64 plan.

Offline research only: a summary's SHA binds its metric bytes but does not
prove the checkpoint, engine identity, model task quality or runtime performance.
Legacy summaries containing the old wrong bpw formula are rejected explicitly.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from precision_budget_search import SCHEMA, PlanError, solve
from qng64_storage import effective_bits_per_weight

COLLECT_SCHEMA = "beglin-qng64-budget-collect-v1"


def _summary_path(root, relative):
    if not isinstance(relative, str) or not relative.strip():
        raise PlanError("summary_path must be a nonempty relative file path")
    fragment = Path(relative)
    if fragment.is_absolute() or ".." in fragment.parts:
        raise PlanError("summary_path must remain inside manifest directory")
    target = (root / fragment).resolve(strict=True)
    if not target.is_relative_to(root.resolve(strict=True)) or not target.is_file():
        raise PlanError("summary_path escapes manifest directory or is not a file")
    return target


def collect(manifest_path):
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(m, dict) or m.get("schema") != COLLECT_SCHEMA:
        raise PlanError(f"manifest schema must be {COLLECT_SCHEMA}")
    tensors = m.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise PlanError("manifest tensors must be a nonempty list")
    plan_input = {"schema": SCHEMA, "model_id": m.get("model_id"),
                  "backend": m.get("backend"),
                  "checkpoint_sha256": m.get("checkpoint_sha256"),
                  "budget_bytes": m.get("budget_bytes"), "tensors": []}
    source_summaries = []
    for t in tensors:
        if not isinstance(t, dict):
            raise PlanError("tensor manifest entries must be objects")
        file = _summary_path(manifest_path.parent, t.get("summary_path"))
        raw = file.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != t.get("summary_sha256"):
            raise PlanError(f"summary content SHA mismatch: {t.get('summary_path')}")
        summary = json.loads(raw)
        if not isinstance(summary, dict) or summary.get("tensor") != t.get("name"):
            raise PlanError("summary tensor identity mismatch")
        shape = summary.get("shape")
        if (not isinstance(shape, list) or len(shape) != 2 or
                any(type(x) is not int or x <= 0 for x in shape)):
            raise PlanError("summary shape must be [out, in] with positive integers")
        if shape[1] % 64:
            raise PlanError("summary tensor input width must be group-64 aligned")
        candidates = []
        for c in summary.get("results", []):
            if not isinstance(c, dict):
                raise PlanError("summary candidates must be objects")
            n = c.get("n")
            # This throws on invalid widths; do not silently reinterpret.
            try:
                expected = effective_bits_per_weight(n)
            except ValueError as exc:
                raise PlanError("unsupported qNg64 bit-width in summary") from exc
            reported = c.get("effective_bpw")
            if type(reported) not in (int, float) or not math.isfinite(reported) or not math.isclose(float(reported), expected, rel_tol=0, abs_tol=1e-9):
                raise PlanError("incorrect/legacy effective_bpw: rerun quant_sim_n.py with fixed n+0.5 accounting")
            candidates.append({"n": n, "rel_l2": c.get("rel_l2")})
        row = {"name": t["name"], "role": t.get("role"), "layer": t.get("layer"),
               "elements": shape[0] * shape[1], "importance": t.get("importance", 1),
               "candidates": candidates}
        if "expert_id" in t:
            row["expert_id"] = t["expert_id"]
        if "max_rel_l2" in t:
            row["max_rel_l2"] = t["max_rel_l2"]
        plan_input["tensors"].append(row)
        source_summaries.append({"tensor": t["name"],
                                 "relative_path": t["summary_path"],
                                 "sha256": observed})
    plan = solve(plan_input)
    plan["source_metric_provenance"] = {
        "type": "HASH_VERIFIED_SIMULATOR_SUMMARIES_NOT_MODEL_AUTHORITY",
        "summaries": source_summaries,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = collect(args.manifest)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
