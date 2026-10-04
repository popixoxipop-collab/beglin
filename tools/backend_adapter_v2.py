#!/usr/bin/env python3
"""P12 backend-symmetric transition planning contract.

No implementation here mutates production.  CPU and MLX planners return the
same schema; backend differences are represented as capability data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import model_capability as mc


class BackendPlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendState:
    backend: str
    epoch: int
    policy: list[dict]
    policy_hash: str


def normalize_policy(policy: Iterable[Mapping]) -> list[dict]:
    rows = [
        {"role": str(r["role"]), "layer": int(r["layer"]), "n": int(r["n"])}
        for r in policy
    ]
    keys = [(r["role"], r["layer"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise BackendPlanError("duplicate role/layer in policy")
    return sorted(rows, key=lambda r: (r["role"], r["layer"]))


def policy_hash(policy: Iterable[Mapping]) -> str:
    return mc.sha256_json(normalize_policy(policy))


def transition_changes(current: Iterable[Mapping], target: Iterable[Mapping]) -> list[dict]:
    before = {(r["role"], r["layer"]): r["n"] for r in normalize_policy(current)}
    after = {(r["role"], r["layer"]): r["n"] for r in normalize_policy(target)}
    if set(before) != set(after):
        return [{
            "kind": "POLICY_SHAPE_CHANGE",
            "added": sorted([list(k) for k in set(after) - set(before)]),
            "removed": sorted([list(k) for k in set(before) - set(after)]),
        }]
    return [
        {
            "kind": "PRECISION_CHANGE",
            "role": role,
            "layer": layer,
            "expected_n": before[(role, layer)],
            "target_n": after[(role, layer)],
        }
        for role, layer in sorted(before)
        if before[(role, layer)] != after[(role, layer)]
    ]


class BackendAdapterV2:
    name = "abstract"

    def plan_transition(self, *, state: BackendState, target_policy: Iterable[Mapping]) -> dict:
        raise NotImplementedError


class CpuBackendAdapterV2(BackendAdapterV2):
    name = "cpu"

    def __init__(self, *, verified_targets: Iterable[str] = ()):
        self.verified_targets = set(verified_targets)

    def plan_transition(self, *, state: BackendState, target_policy: Iterable[Mapping]) -> dict:
        if state.backend != self.name:
            raise BackendPlanError("CPU adapter received non-CPU state")
        target = normalize_policy(target_policy)
        changes = transition_changes(state.policy, target)
        shape = any(c["kind"] == "POLICY_SHAPE_CHANGE" for c in changes)
        return {
            "schema": "beglin-backend-transition-plan-v2",
            "backend": self.name,
            "expected_epoch": int(state.epoch),
            "expected_policy_hash": state.policy_hash,
            "target_policy": target,
            "target_policy_hash": policy_hash(target),
            "changes": changes,
            "action": "RESTART_REQUIRED" if changes else "NOOP",
            "mutation_mode": "RESTART_REQUIRED" if changes else "IMMUTABLE",
            "policy_shape_change": shape,
            "production_write_allowed": False,
        }


class MlxMetalBackendAdapterV2(BackendAdapterV2):
    name = "mlx_metal"

    def __init__(self, *, verified_hot_targets: Iterable[str], max_atomic_targets: int = 8):
        self.verified_hot_targets = set(verified_hot_targets)
        self.max_atomic_targets = int(max_atomic_targets)
        if self.max_atomic_targets <= 0:
            raise ValueError("max_atomic_targets must be positive")

    def _target_key(self, role: str, layer: int) -> str:
        return f"L{int(layer)}/{str(role)}"

    def plan_transition(self, *, state: BackendState, target_policy: Iterable[Mapping]) -> dict:
        if state.backend != self.name:
            raise BackendPlanError("MLX adapter received non-MLX state")
        target = normalize_policy(target_policy)
        changes = transition_changes(state.policy, target)
        if not changes:
            action, mode, shape = "NOOP", "IMMUTABLE", False
        elif any(c["kind"] == "POLICY_SHAPE_CHANGE" for c in changes):
            action, mode, shape = "RESTART_REQUIRED", "RESTART_REQUIRED", True
        else:
            unknown = [
                self._target_key(c["role"], c["layer"])
                for c in changes
                if self._target_key(c["role"], c["layer"]) not in self.verified_hot_targets
            ]
            if unknown:
                action, mode, shape = "RESTART_REQUIRED", "RESTART_REQUIRED", False
            elif len(changes) == 1:
                action, mode, shape = "HOT_REBIND_SINGLE", "HOT_REBIND_SINGLE", False
            elif len(changes) <= self.max_atomic_targets:
                action, mode, shape = "HOT_REBIND_MULTI", "HOT_REBIND_MULTI", False
            else:
                action, mode, shape = "RESTART_REQUIRED", "RESTART_REQUIRED", False
        return {
            "schema": "beglin-backend-transition-plan-v2",
            "backend": self.name,
            "expected_epoch": int(state.epoch),
            "expected_policy_hash": state.policy_hash,
            "target_policy": target,
            "target_policy_hash": policy_hash(target),
            "changes": changes,
            "action": action,
            "mutation_mode": mode,
            "policy_shape_change": shape,
            "production_write_allowed": False,
        }
