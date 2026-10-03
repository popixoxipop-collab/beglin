#!/usr/bin/env python3
"""Admission-scoped precision policy scheduler for the persistent MLX worker.

v1 deliberately allows only n->n changes for role/layer targets already present
in the resident worker's startup policy.  Adding or removing a target still
requires a worker restart so no request can silently materialize a new policy
shape outside the certified startup preimage.
"""
from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Callable

import gpu_runtime_control as grc
import precision_context as pc


SAFE_ADMISSION = re.compile(r"^[A-Za-z0-9_.-]+$")


class PrecisionEpochSchedulerError(RuntimeError):
    pass


def _policy_map(policy: list[dict]) -> dict[tuple[str, int], int]:
    normalized = pc.normalize_policy(policy)
    return {(row["role"], int(row["layer"])): int(row["n"]) for row in normalized}


def plan_transition(current_policy: list[dict], target_policy: list[dict]) -> list[dict]:
    current = _policy_map(current_policy)
    target = _policy_map(target_policy)
    if set(current) != set(target):
        added = sorted(set(target) - set(current))
        removed = sorted(set(current) - set(target))
        raise PrecisionEpochSchedulerError(
            f"policy shape change requires worker restart: added={added} removed={removed}"
        )
    changes = []
    for role, layer in sorted(current):
        before = current[(role, layer)]
        after = target[(role, layer)]
        if before == after:
            continue
        if before not in grc.SUPPORTED_QNG64 or after not in grc.SUPPORTED_QNG64:
            raise PrecisionEpochSchedulerError(
                f"unsupported qNg64 transition {role}/L{layer}: {before}->{after}"
            )
        changes.append({
            "role": role,
            "layer": layer,
            "expected_n": before,
            "target_n": after,
        })
    if len(changes) > grc.MAX_REBIND_SET_TARGETS:
        raise PrecisionEpochSchedulerError(
            f"transition has {len(changes)} targets; max={grc.MAX_REBIND_SET_TARGETS}"
        )
    return changes


class PrecisionEpochScheduler:
    """Serialize admission, transition a whole policy in one epoch, then infer."""

    def __init__(
        self,
        *,
        ack_path: str | Path,
        txn_path: str | Path,
        submit_fn: Callable[[list[tuple[list[int], int]]], dict],
        admission_lock=None,
    ):
        self.ack_path = Path(ack_path)
        self.txn_path = Path(txn_path)
        self.submit_fn = submit_fn
        # When embedded in PersistentRouteWorker this is the exact same RLock
        # used by ordinary submit(), so direct and policy-aware admissions
        # cannot interleave between preimage read, transaction publication,
        # inference and terminal ACK verification.
        self.lock = admission_lock if admission_lock is not None else threading.Lock()

    def run(
        self,
        parsed: list[tuple[list[int], int]],
        *,
        target_policy: list[dict],
        admission_id: str | None = None,
    ) -> dict:
        if not parsed:
            raise PrecisionEpochSchedulerError("admission batch cannot be empty")
        admission_id = admission_id or f"admit-{uuid.uuid4().hex}"
        if not SAFE_ADMISSION.fullmatch(admission_id):
            raise PrecisionEpochSchedulerError("admission_id contains unsupported characters")

        with self.lock:
            before = grc.read_runtime_ack(self.ack_path)
            normalized_target = pc.normalize_policy(target_policy)
            changes = plan_transition(before["active_policy"], normalized_target)
            target_hash = pc.policy_hash(normalized_target)
            txn_id = None
            if changes:
                txn_id = f"epoch-{uuid.uuid4().hex}"
                requested = grc.prepare_rebind_set(
                    ack_path=self.ack_path,
                    txn_path=self.txn_path,
                    txn_id=txn_id,
                    expected_epoch=int(before["weight_epoch"]),
                    expected_policy_hash=str(before["active_policy_hash"]),
                    changes=changes,
                )
                if requested["target_policy_hash"] != target_hash:
                    raise PrecisionEpochSchedulerError(
                        "controller target policy hash disagrees with scheduler"
                    )

            try:
                result = self.submit_fn(parsed)
            except Exception as exc:
                raise PrecisionEpochSchedulerError(
                    f"admission {admission_id} failed while precision epoch was active"
                ) from exc

            if changes:
                after = grc.verify_terminal_ack(
                    ack_path=self.ack_path,
                    txn_id=txn_id,
                    allowed_statuses={"REBIND_SET_APPLIED"},
                )
                if int(after.get("changed_targets", -1)) != len(changes):
                    raise PrecisionEpochSchedulerError(
                        "runtime ACK changed_targets does not match transition plan"
                    )
                if int(after["weight_epoch"]) != int(before["weight_epoch"]) + 1:
                    raise PrecisionEpochSchedulerError(
                        "multi-target transition did not advance exactly one weight epoch"
                    )
            else:
                after = grc.read_runtime_ack(self.ack_path)
                if int(after["weight_epoch"]) != int(before["weight_epoch"]):
                    raise PrecisionEpochSchedulerError(
                        "weight epoch changed during no-op admission"
                    )

            if after["active_policy_hash"] != target_hash:
                raise PrecisionEpochSchedulerError(
                    "runtime policy after admission differs from requested precision policy"
                )
            if after["active_policy"] != normalized_target:
                raise PrecisionEpochSchedulerError(
                    "runtime active policy rows differ from requested precision policy"
                )

            if changes:
                transition_cost = {
                    "transition_wall_ms": float(after.get("transition_wall_ms", 0.0)),
                    "cache_hits": int(after.get("transition_cache_hits", 0)),
                    "cache_misses": int(after.get("transition_cache_misses", 0)),
                    "cache_bytes_added": int(after.get("transition_cache_bytes_added", 0)),
                    "resident_cache_bytes": int(after.get("resident_qng64_cache_bytes", 0)),
                }
            else:
                transition_cost = {
                    "transition_wall_ms": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_bytes_added": 0,
                    "resident_cache_bytes": int(after.get("resident_qng64_cache_bytes", 0)),
                }
            inference_passes = int(result.get("inference_passes", 1))
            if inference_passes < 1:
                raise PrecisionEpochSchedulerError("inference_passes must be >= 1")
            out = dict(result)
            out["precision_epoch"] = {
                "schema": "beglin-precision-epoch-admission-v1",
                "admission_id": admission_id,
                "txn_id": txn_id,
                "transitioned": bool(changes),
                "changed_targets": changes,
                "before_epoch": int(before["weight_epoch"]),
                "after_epoch": int(after["weight_epoch"]),
                "before_policy_hash": str(before["active_policy_hash"]),
                "after_policy_hash": str(after["active_policy_hash"]),
                "target_policy_hash": target_hash,
                "transition_cost": transition_cost,
                "inference_passes": inference_passes,
                "engine_wall_ms": float(result.get("engine_wall_ms", 0.0)),
                "roundtrip_ms": float(result.get("roundtrip_ms", 0.0)),
            }
            return out
