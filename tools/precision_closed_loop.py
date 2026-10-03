#!/usr/bin/env python3
"""Evidence-gated allocator -> selector -> precision-epoch bridge.

The module does not fetch remote evidence on the serving hot path. Candidate
rows, trigger evidence and exact combined-policy acceptance records are loaded
or refreshed outside admission, then pinned in one engine instance.

A request-time admission performs:
  current runtime policy
    -> evidence-gated Precision Allocator
    -> trigger-conditioned Dynamic Selector
    -> exact combined-policy acceptance gate
    -> Precision Epoch Scheduler

No component may directly write production routing state. The only runtime
mutation is the scheduler's already-CAS-protected resident qNg64 REBIND_SET.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

import precision_allocator as pa
import precision_context as pc
import precision_dynamic_selector as ds


class PrecisionClosedLoopError(RuntimeError):
    pass


def _policy_map(policy: list[dict]) -> dict[tuple[str, int], int]:
    normalized = pc.normalize_policy(policy)
    return {
        (str(row["role"]), int(row["layer"])): int(row["n"])
        for row in normalized
    }


def _canonical_sha(value) -> str:
    return hashlib.sha256(pc.canonical_json(value).encode()).hexdigest()


def _combined_evidence_index(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for idx, row in enumerate(rows or []):
        if not isinstance(row, dict):
            raise PrecisionClosedLoopError(
                f"combined_policy_evidence[{idx}] must be an object"
            )
        policy_hash = str(row.get("policy_hash", "")).lower()
        if len(policy_hash) != 64 or any(
            c not in "0123456789abcdef" for c in policy_hash
        ):
            raise PrecisionClosedLoopError(
                f"combined_policy_evidence[{idx}] has invalid policy_hash"
            )
        if row.get("status") != "PASS" or row.get("pass") is not True:
            continue
        evidence_sha = str(row.get("evidence_sha256", "")).lower()
        if len(evidence_sha) != 64 or any(
            c not in "0123456789abcdef" for c in evidence_sha
        ):
            raise PrecisionClosedLoopError(
                f"combined_policy_evidence[{idx}] has invalid evidence_sha256"
            )
        if row.get("production_touched") is not False:
            raise PrecisionClosedLoopError(
                "combined-policy acceptance must explicitly prove "
                "production_touched=false"
            )
        old = out.get(policy_hash)
        if old is not None and old != row:
            raise PrecisionClosedLoopError(
                f"conflicting combined-policy evidence for {policy_hash}"
            )
        out[policy_hash] = dict(row)
    return out


def _allocator_target_index(allocation: dict) -> dict[tuple[str, int], dict]:
    out = {}
    for row in allocation.get("targets", []):
        try:
            key = (str(row["role"]), int(row["layer"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PrecisionClosedLoopError("allocator target is malformed") from exc
        if key in out:
            raise PrecisionClosedLoopError(
                f"allocator emitted duplicate target {key[0]}/L{key[1]}"
            )
        out[key] = row
    return out


def materialize_selected_policy(
    *,
    current_policy: list[dict],
    allocation: dict,
    selection: dict,
    combined_policy_evidence: list[dict],
) -> dict:
    """Turn proposal-only allocator/selector output into a scheduler-safe policy."""
    current = pc.normalize_policy(current_policy)
    current_map = _policy_map(current)
    current_hash = pc.policy_hash(current)

    if allocation.get("schema") != "beglin-precision-allocator-v1":
        raise PrecisionClosedLoopError("unexpected allocator schema")
    if allocation.get("status") != "PROPOSAL_ONLY":
        raise PrecisionClosedLoopError("allocator result is not proposal-only")
    if allocation.get("production_write_allowed") is not False:
        raise PrecisionClosedLoopError("allocator unexpectedly allows production writes")
    if pc.normalize_policy(allocation.get("current_policy", [])) != current:
        raise PrecisionClosedLoopError(
            "allocator current_policy does not match runtime preimage"
        )

    proposed = pc.normalize_policy(allocation.get("proposed_policy", []))
    if set(_policy_map(proposed)) != set(current_map):
        raise PrecisionClosedLoopError(
            "allocator proposed policy changes resident policy shape"
        )

    if selection.get("schema") != "beglin-dynamic-precision-selector-v1":
        raise PrecisionClosedLoopError("unexpected selector schema")
    if selection.get("status") != "DECISION_ONLY":
        raise PrecisionClosedLoopError("selector result is not decision-only")
    if selection.get("production_write_allowed") is not False:
        raise PrecisionClosedLoopError("selector unexpectedly allows production writes")
    if selection.get("nonmonotonic_precision") is not True:
        raise PrecisionClosedLoopError(
            "selector did not preserve non-monotonic precision contract"
        )

    target_map = _policy_map(proposed)
    allocator_targets = _allocator_target_index(allocation)

    seen = set()
    selector_rows = selection.get("targets", [])
    if not isinstance(selector_rows, list):
        raise PrecisionClosedLoopError("selector targets must be a list")
    for row in selector_rows:
        try:
            key = (str(row["role"]), int(row["layer"]))
            selected_n = int(row["selected_n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PrecisionClosedLoopError("selector target is malformed") from exc
        if key in seen:
            raise PrecisionClosedLoopError(
                f"selector emitted duplicate target {key[0]}/L{key[1]}"
            )
        seen.add(key)
        target = allocator_targets.get(key)
        if target is None or target.get("status") != "PROPOSED":
            raise PrecisionClosedLoopError(
                f"selector target {key[0]}/L{key[1]} lacks allocator proposal"
            )
        base_n = int(target["selected_n"])
        allowed = {base_n}
        for alt in (
            target.get("dynamic_escalation", {})
            .get("candidate_alternates", [])
        ):
            allowed.add(int(alt["n"]))
        if selected_n not in allowed:
            raise PrecisionClosedLoopError(
                f"selector chose unadmitted n={selected_n} for {key[0]}/L{key[1]}"
            )
        status = str(row.get("status"))
        if status == "TRIGGER_CONDITIONED_ALTERNATE":
            if selected_n == base_n:
                raise PrecisionClosedLoopError(
                    "trigger-conditioned alternate did not leave base precision"
                )
            evidence = row.get("evidence")
            if not isinstance(evidence, dict) or not evidence:
                raise PrecisionClosedLoopError(
                    "trigger-conditioned alternate lacks contextual evidence"
                )
        elif selected_n != base_n:
            raise PrecisionClosedLoopError(
                f"selector status {status!r} may not leave allocator base"
            )
        if key not in target_map:
            raise PrecisionClosedLoopError(
                f"selector target {key[0]}/L{key[1]} is not resident"
            )
        target_map[key] = selected_n

    selected_policy = pc.normalize_policy([
        {"role": role, "layer": layer, "n": n}
        for (role, layer), n in target_map.items()
    ])
    if set(_policy_map(selected_policy)) != set(current_map):
        raise PrecisionClosedLoopError(
            "closed-loop selected policy changes resident policy shape"
        )
    target_hash = pc.policy_hash(selected_policy)

    changes = [
        {
            "role": role,
            "layer": layer,
            "from_n": current_map[(role, layer)],
            "to_n": _policy_map(selected_policy)[(role, layer)],
        }
        for role, layer in sorted(current_map)
        if current_map[(role, layer)]
        != _policy_map(selected_policy)[(role, layer)]
    ]

    combined = None
    if (
        changes
        and allocation.get(
            "pairwise_evidence_required_before_combined_production"
        ) is True
        and len(selected_policy) > 1
    ):
        combined = _combined_evidence_index(
            combined_policy_evidence
        ).get(target_hash)
        if combined is None:
            raise PrecisionClosedLoopError(
                "exact combined-policy acceptance is required before "
                f"scheduling policy {target_hash}"
            )

    return {
        "schema": "beglin-precision-closed-loop-decision-v1",
        "status": "READY_FOR_SCHEDULER",
        "production_write_allowed": False,
        "current_policy": current,
        "current_policy_hash": current_hash,
        "selected_policy": selected_policy,
        "selected_policy_hash": target_hash,
        "changes": changes,
        "allocation_sha256": _canonical_sha(allocation),
        "selection_sha256": _canonical_sha(selection),
        "combined_policy_evidence": combined,
        "signal": selection.get("signal"),
        "allocation": allocation,
        "selection": selection,
    }


class PrecisionClosedLoopEngine:
    """Pure request-time allocator/selector decision engine.

    Candidate and evidence snapshots are intentionally constructor inputs so a
    serving process never performs a network fetch while holding admission.
    """

    def __init__(
        self,
        *,
        candidates: list[dict],
        trigger_evidence: list[dict],
        combined_policy_evidence: list[dict],
        memory_weight: float = 1.0,
        latency_weight: float = 0.0,
        rss_weight: float = 0.0,
    ):
        if not isinstance(candidates, list) or not candidates:
            raise PrecisionClosedLoopError("candidates must be a non-empty list")
        self.candidates = json.loads(json.dumps(candidates))
        self.trigger_evidence = json.loads(json.dumps(trigger_evidence or []))
        self.combined_policy_evidence = json.loads(
            json.dumps(combined_policy_evidence or [])
        )
        self.weights = {
            "memory_weight": float(memory_weight),
            "latency_weight": float(latency_weight),
            "rss_weight": float(rss_weight),
        }
        # Validate combined evidence once at configuration time.
        _combined_evidence_index(self.combined_policy_evidence)
        self.snapshot_sha256 = _canonical_sha({
            "candidates": self.candidates,
            "trigger_evidence": self.trigger_evidence,
            "combined_policy_evidence": self.combined_policy_evidence,
            "weights": self.weights,
        })

    def decide(self, *, current_policy: list[dict], signal: dict) -> dict:
        current = pc.normalize_policy(current_policy)
        allocation = pa.optimize(
            self.candidates,
            current_policy=current,
            **self.weights,
        )
        selection = ds.select(
            allocation,
            signal,
            self.trigger_evidence,
        )
        result = materialize_selected_policy(
            current_policy=current,
            allocation=allocation,
            selection=selection,
            combined_policy_evidence=self.combined_policy_evidence,
        )
        result["evidence_snapshot_sha256"] = self.snapshot_sha256
        result["objective_weights"] = dict(self.weights)
        return result
