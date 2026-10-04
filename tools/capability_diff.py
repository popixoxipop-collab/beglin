#!/usr/bin/env python3
"""P12 model/backend capability diff.

Compares two immutable ModelCapabilityBundles compiled from model artifacts.
The diff is read-only and deterministic: it never mutates model files or runtime
state and never upgrades unverified capability to VERIFIED.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import model_capability as mc
import model_evidence_registry as mer


class CapabilityDiffError(RuntimeError):
    pass


def _compile(path: str, *, backend: str | None, evidence_paths: list[str]) -> dict:
    evidence = {}
    if evidence_paths:
        source = mc.inspect_model_source(path)
        descriptor = mc.build_architecture_descriptor(source)
        registry = mer.VerificationEvidenceRegistry.from_paths(evidence_paths)
        evidence = registry.resolve_model_set(
            architecture_id=descriptor["architecture_id"],
            checkpoint_identity_sha256=source["checkpoint_identity_sha256"],
        )
    return mc.compile_model_capabilities(path, backend=backend, **evidence)


def _target_rows(
    bundle: Mapping[str, Any], *, semantic_only: bool = False
) -> dict[tuple, dict]:
    backend = {
        (str(row["target_key"]), str(row["backend"])): row
        for row in bundle["backend_capability_matrix"].get("rows", [])
    }
    mutation = {
        (str(row["target_key"]), str(row["backend"])): row
        for row in bundle["runtime_mutation_matrix"].get("rows", [])
    }
    quant = {
        (str(row["target_key"]), str(row["backend"])): row
        for row in bundle["quant_capability_matrix"].get("rows", [])
    }
    out = {}
    for source_key, row in backend.items():
        q = quant.get(source_key, {})
        m = mutation.get(source_key, {})
        semantic = (
            str(row.get("role")),
            -1 if row.get("layer") is None else int(row.get("layer")),
            -1 if row.get("expert_id") is None else int(row.get("expert_id")),
        )
        key = semantic if semantic_only else (str(row["backend"]), *semantic)
        if key in out:
            raise CapabilityDiffError(
                f"duplicate capability semantic key: {key!r}"
            )
        out[key] = {
            "target_key": str(row["target_key"]),
            "backend": str(row["backend"]),
            "role": row.get("role"),
            "layer": row.get("layer"),
            "expert_id": row.get("expert_id"),
            "inference_status": row.get("inference_status"),
            "qng64_status": row.get("qng64_status"),
            "supported_n": sorted(int(n) for n in row.get("supported_n", [])),
            "mutation_mode": m.get("mutation_mode", row.get("mutation_mode")),
            "allowed_target_precisions": sorted(
                int(n) for n in m.get("allowed_target_precisions", [])
            ),
            "source_precision": q.get("source_precision"),
        }
    return out


def _architecture_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict]:
    keys = (
        "architecture_id", "architecture_family", "architecture_variant",
        "dense_or_moe", "attention_kind", "rope_kind", "norm_kind",
        "qk_norm_kind", "hidden_size", "intermediate_size", "num_layers",
        "num_attention_heads", "num_kv_heads", "head_dim", "vocab_size",
        "context_length", "expert_count", "experts_per_token",
        "shared_expert_count", "sliding_window", "attention_sink",
    )
    out = []
    for key in keys:
        a = left.get(key)
        b = right.get(key)
        if a != b:
            out.append({"field": key, "left": a, "right": b})
    return out


def build_diff(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_backend: str | None = None,
    right_backend: str | None = None,
) -> dict:
    for label, bundle in (("left", left), ("right", right)):
        if bundle.get("schema") != "beglin-model-capability-bundle-v1":
            raise CapabilityDiffError(f"{label} is not a model capability bundle")
        if bundle.get("bundle_sha256") != mc.stable_identity_sha256(bundle):
            raise CapabilityDiffError(f"{label} capability bundle hash mismatch")

    semantic_only = left_backend is not None and right_backend is not None
    lrows = _target_rows(left, semantic_only=semantic_only)
    rrows = _target_rows(right, semantic_only=semantic_only)
    keys = sorted(set(lrows) | set(rrows))
    target_changes = []
    counts = {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}
    for key in keys:
        lrow = lrows.get(key)
        rrow = rrows.get(key)
        if lrow is None:
            kind = "added"
        elif rrow is None:
            kind = "removed"
        elif lrow == rrow:
            kind = "unchanged"
        else:
            kind = "changed"
        counts[kind] += 1
        if kind != "unchanged":
            target_changes.append({
                "change": kind.upper(),
                "semantic_key": list(key),
                "target_key": (
                    (lrow or rrow).get("target_key")
                ),
                "left_backend": lrow.get("backend") if lrow else None,
                "right_backend": rrow.get("backend") if rrow else None,
                "left": lrow,
                "right": rrow,
            })

    ldesc = left["architecture_descriptor"]
    rdesc = right["architecture_descriptor"]
    out = {
        "schema": "beglin-capability-diff-v1",
        "left": {
            "model_id": left["model_id"],
            "checkpoint_identity_sha256": left["checkpoint_identity_sha256"],
            "architecture_id": ldesc["architecture_id"],
            "backend_filter": left_backend,
            "bundle_sha256": left["bundle_sha256"],
            "eligibility": left["p8_p11_eligibility"]["status"],
            "tokenizer_status": left["tokenizer_contract"]["status"],
            "loader_status": left["loader_contract"]["status"],
        },
        "right": {
            "model_id": right["model_id"],
            "checkpoint_identity_sha256": right["checkpoint_identity_sha256"],
            "architecture_id": rdesc["architecture_id"],
            "backend_filter": right_backend,
            "bundle_sha256": right["bundle_sha256"],
            "eligibility": right["p8_p11_eligibility"]["status"],
            "tokenizer_status": right["tokenizer_contract"]["status"],
            "loader_status": right["loader_contract"]["status"],
        },
        "same_checkpoint": (
            left["checkpoint_identity_sha256"]
            == right["checkpoint_identity_sha256"]
        ),
        "same_architecture": ldesc["architecture_id"] == rdesc["architecture_id"],
        "architecture_changes": _architecture_diff(ldesc, rdesc),
        "target_counts": counts,
        "target_changes": target_changes,
        "automatic_live_promotion": False,
        "production_write_allowed": False,
    }
    out["diff_sha256"] = mc.stable_identity_sha256(out)
    return out


def summary(value: Mapping[str, Any]) -> dict:
    return {
        "left": value["left"],
        "right": value["right"],
        "same_checkpoint": value["same_checkpoint"],
        "same_architecture": value["same_architecture"],
        "architecture_change_count": len(value["architecture_changes"]),
        "target_counts": value["target_counts"],
        "diff_sha256": value["diff_sha256"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare Beglin P12 model/backend capability bundles."
    )
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--left-backend", choices=["cpu", "mlx_metal"])
    ap.add_argument("--right-backend", choices=["cpu", "mlx_metal"])
    ap.add_argument("--left-evidence", action="append", default=[])
    ap.add_argument("--right-evidence", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        left = _compile(
            args.left, backend=args.left_backend, evidence_paths=args.left_evidence
        )
        right = _compile(
            args.right, backend=args.right_backend, evidence_paths=args.right_evidence
        )
        value = build_diff(
            left, right,
            left_backend=args.left_backend,
            right_backend=args.right_backend,
        )
    except (
        mc.ModelCapabilityError,
        mer.EvidenceRegistryError,
        CapabilityDiffError,
    ) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(value if args.json else summary(value), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
