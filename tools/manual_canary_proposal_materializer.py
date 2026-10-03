#!/usr/bin/env python3
"""Materialize a fully-bound Agent E proposal from a launch-time runtime preimage.

This module does not launch workers, mutate runtime state, or enable production.
It only converts already-observed runtime/preflight data into the normalized
manual-canary proposal contract.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import manual_canary_contract as mc


class ProposalMaterializerError(RuntimeError):
    pass


def _positive_int(name: str, value: Any) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ProposalMaterializerError(f"{name} must be an integer") from exc
    if out <= 0:
        raise ProposalMaterializerError(f"{name} must be positive")
    return out


def derive_budget(
    observed: Mapping[str, Any],
    *,
    physical_memory_bytes: int | None = None,
) -> dict:
    requests = _positive_int("observed.requests", observed.get("requests"))
    tokens = _positive_int("observed.tokens", observed.get("tokens"))
    duration_ms = _positive_int("observed.duration_ms", observed.get("duration_ms"))
    memory_bytes = _positive_int("observed.memory_bytes", observed.get("memory_bytes"))

    budget = {
        "max_requests": max(16, math.ceil(requests * 1.5)),
        "max_tokens": max(256, math.ceil(tokens * 1.5)),
        "max_duration_ms": max(30_000, math.ceil(duration_ms * 2.0)),
        "max_memory_bytes": math.ceil(memory_bytes * 1.25),
    }

    if physical_memory_bytes is not None:
        physical = _positive_int("physical_memory_bytes", physical_memory_bytes)
        if memory_bytes >= physical:
            raise ProposalMaterializerError(
                "observed memory is not below physical memory; refuse to derive a budget"
            )
        headroom_cap = math.floor(physical * 0.90)
        if budget["max_memory_bytes"] > headroom_cap:
            budget["max_memory_bytes"] = headroom_cap
        if budget["max_memory_bytes"] <= memory_bytes:
            raise ProposalMaterializerError(
                "physical-memory cap leaves no positive memory headroom"
            )

    return budget


def _policy_map(rows: list[dict]) -> dict[tuple[str, int], int]:
    out: dict[tuple[str, int], int] = {}
    for row in rows:
        try:
            key = (str(row["role"]), int(row["layer"]))
            n = int(row["n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProposalMaterializerError("invalid policy row") from exc
        if not key[0] or key[1] < 0 or n <= 0:
            raise ProposalMaterializerError("invalid policy target")
        if key in out:
            raise ProposalMaterializerError("duplicate role/layer in policy")
        out[key] = n
    return out


def materialize_proposal(
    *,
    template: Mapping[str, Any],
    runtime_preimage: Mapping[str, Any],
    candidate_policy: list[dict],
    observed_metrics: Mapping[str, Any],
    physical_memory_bytes: int | None = None,
) -> dict:
    value = copy.deepcopy(dict(template))
    try:
        active_policy = list(runtime_preimage["active_policy"])
        active_policy_hash = str(runtime_preimage["active_policy_hash"])
        weight_epoch = int(runtime_preimage["weight_epoch"])
        ack_sha256 = str(runtime_preimage["ack_sha256"])
        worker_instance_id = str(runtime_preimage["worker_instance_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProposalMaterializerError("runtime preimage is incomplete") from exc

    if not ack_sha256 or not worker_instance_id:
        raise ProposalMaterializerError("runtime ACK and worker instance must be non-empty")
    canonical_baseline_hash = mc.sha256_json(active_policy)
    if canonical_baseline_hash != active_policy_hash:
        raise ProposalMaterializerError("runtime active_policy hash mismatch")

    target = dict(value.get("single_target") or {})
    try:
        key = (str(target["role"]), int(target["layer"]))
        after_n = int(target["after_n"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProposalMaterializerError("template single_target is invalid") from exc
    if not key[0] or key[1] < 0 or after_n <= 0:
        raise ProposalMaterializerError("template single_target is invalid")

    before = _policy_map(active_policy)
    after = _policy_map(candidate_policy)
    if after.get(key) != after_n:
        raise ProposalMaterializerError("candidate policy target does not match template")
    differing = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if differing != {key}:
        raise ProposalMaterializerError("candidate policy must change exactly one target")

    before_n = before.get(key)
    budget = derive_budget(observed_metrics, physical_memory_bytes=physical_memory_bytes)

    value["schema"] = "manual-canary-proposal-v1"
    value["status"] = "READY_FOR_HUMAN_SIGNATURE"
    value["baseline_policy_hash"] = canonical_baseline_hash
    value["candidate_policy_hash"] = mc.sha256_json(candidate_policy)
    value["single_target"] = {
        "role": key[0],
        "layer": key[1],
        "before_n": before_n,
        "after_n": after_n,
    }
    value["budget"] = budget
    value["expected_epoch"] = weight_epoch
    value["restart_instance_id"] = worker_instance_id
    value["missing_required_runtime_fields"] = []
    value["runtime_preimage_binding"] = {
        "ack_sha256": ack_sha256,
        "active_policy_hash": active_policy_hash,
        "weight_epoch": weight_epoch,
        "worker_instance_id": worker_instance_id,
    }
    value["budget_observed_metrics"] = {
        "requests": int(observed_metrics["requests"]),
        "tokens": int(observed_metrics["tokens"]),
        "duration_ms": int(observed_metrics["duration_ms"]),
        "memory_bytes": int(observed_metrics["memory_bytes"]),
    }
    if physical_memory_bytes is not None:
        value["budget_physical_memory_bytes"] = int(physical_memory_bytes)

    normalized = mc.normalize_proposal(value)
    return {
        "schema": "manual-canary-proposal-materialization-v1",
        "status": "READY_FOR_HUMAN_SIGNATURE",
        "production_write_allowed": False,
        "proposal": normalized,
        "proposal_digest": mc.sha256_json(normalized),
        "runtime_preimage_binding": value["runtime_preimage_binding"],
        "budget_observed_metrics": value["budget_observed_metrics"],
        "physical_memory_bytes": value.get("budget_physical_memory_bytes"),
    }


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", required=True)
    ap.add_argument("--runtime-preimage", required=True)
    ap.add_argument("--candidate-policy", required=True)
    ap.add_argument("--observed-metrics", required=True)
    ap.add_argument("--physical-memory-bytes", type=int)
    ap.add_argument("--output")
    args = ap.parse_args()

    result = materialize_proposal(
        template=_load(args.template),
        runtime_preimage=_load(args.runtime_preimage),
        candidate_policy=_load(args.candidate_policy),
        observed_metrics=_load(args.observed_metrics),
        physical_memory_bytes=args.physical_memory_bytes,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
