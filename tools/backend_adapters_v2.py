#!/usr/bin/env python3
"""P12 symmetric backend planning contract.

This module does not mutate a runtime. It normalizes CPU and MLX planning
semantics into one schema so upper layers do not branch on backend details.
Execution adapters can later consume these plans.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import model_capability as mc
import precision_context as pc


class BackendV2Error(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendStateV2:
    backend: str
    epoch: int
    policy: list[dict]
    policy_hash: str
    model_capability_bundle_sha256: str

    @classmethod
    def build(
        cls,
        *,
        backend: str,
        epoch: int,
        policy: list[dict],
        model_capability_bundle_sha256: str,
    ) -> "BackendStateV2":
        if backend not in pc.SUPPORTED_BACKENDS:
            raise BackendV2Error(f"unsupported backend={backend}")
        normalized = pc.normalize_policy(policy)
        return cls(
            backend=backend,
            epoch=int(epoch),
            policy=normalized,
            policy_hash=pc.policy_hash(normalized),
            model_capability_bundle_sha256=str(model_capability_bundle_sha256),
        )


class BackendAdapterV2:
    name = "abstract"

    def __init__(self, capability_bundle: Mapping[str, Any]):
        self.bundle = copy.deepcopy(dict(capability_bundle))
        if self.bundle.get("schema") != "beglin-model-capability-bundle-v1":
            raise BackendV2Error("invalid model capability bundle")
        self.bundle_sha = str(self.bundle.get("bundle_sha256") or "")
        if not self.bundle_sha:
            raise BackendV2Error("capability bundle SHA missing")
        actual = mc.stable_identity_sha256(self.bundle)
        if actual != self.bundle_sha:
            raise BackendV2Error(
                f"capability bundle hash mismatch: expected={self.bundle_sha} actual={actual}"
            )

    def probe_capabilities(self) -> dict:
        rows = [
            dict(r)
            for r in self.bundle["backend_capability_matrix"]["rows"]
            if r["backend"] == self.name
        ]
        return {
            "schema": "beglin-backend-v2-probe-v1",
            "backend": self.name,
            "model_capability_bundle_sha256": self.bundle_sha,
            "target_count": len(rows),
            "rows": rows,
        }

    def _mutation_row(self, target_key: str) -> dict:
        for row in self.bundle["runtime_mutation_matrix"]["rows"]:
            if row["backend"] == self.name and row["target_key"] == target_key:
                return dict(row)
        raise BackendV2Error(
            f"no runtime mutation capability for backend={self.name} target={target_key}"
        )

    @staticmethod
    def _policy_map(policy: list[dict]) -> dict[tuple[str, int], int]:
        return {
            (str(row["role"]), int(row["layer"])): int(row["n"])
            for row in pc.normalize_policy(policy)
        }

    def plan_transition(
        self,
        *,
        state: BackendStateV2,
        target_policy: list[dict],
        target_keys: Mapping[tuple[str, int], str],
    ) -> dict:
        if state.backend != self.name:
            raise BackendV2Error(
                f"state backend mismatch: state={state.backend} adapter={self.name}"
            )
        if state.model_capability_bundle_sha256 != self.bundle_sha:
            raise BackendV2Error("stale model capability bundle")
        before = self._policy_map(state.policy)
        normalized_target = pc.normalize_policy(target_policy)
        after = self._policy_map(normalized_target)
        if set(before) != set(after):
            action = "RESTART_REQUIRED"
            changes = []
            added = sorted(set(after) - set(before))
            removed = sorted(set(before) - set(after))
            reason = {
                "code": "POLICY_SHAPE_CHANGE",
                "added": [list(x) for x in added],
                "removed": [list(x) for x in removed],
            }
        else:
            changes = []
            modes = []
            for key in sorted(before):
                if before[key] == after[key]:
                    continue
                if key not in target_keys:
                    raise BackendV2Error(f"missing target-key binding for {key}")
                target_key = str(target_keys[key])
                cap = self._mutation_row(target_key)
                if after[key] not in cap.get("allowed_target_precisions", []):
                    raise BackendV2Error(
                        f"precision n={after[key]} unsupported for {target_key}"
                    )
                modes.append(str(cap["mutation_mode"]))
                changes.append({
                    "target_key": target_key,
                    "role": key[0],
                    "layer": key[1],
                    "expected_n": before[key],
                    "target_n": after[key],
                    "capability_mode": cap["mutation_mode"],
                })
            action, reason = self._resolve_action(modes, len(changes))
        plan = {
            "schema": "beglin-backend-transition-plan-v1",
            "backend": self.name,
            "model_capability_bundle_sha256": self.bundle_sha,
            "expected_epoch": state.epoch,
            "expected_policy_hash": state.policy_hash,
            "target_policy": normalized_target,
            "target_policy_hash": pc.policy_hash(normalized_target),
            "action": action,
            "changes": changes,
            "reason": reason,
            "production_write_allowed": False,
            "automatic_live_promotion": False,
        }
        plan["transition_plan_sha256"] = mc.stable_identity_sha256(plan)
        return plan

    def _resolve_action(self, modes: list[str], count: int) -> tuple[str, dict | None]:
        raise NotImplementedError


class CpuBackendAdapterV2(BackendAdapterV2):
    name = "cpu"

    def _resolve_action(self, modes: list[str], count: int):
        if count == 0:
            return "NOOP", None
        # P12 v1 wraps the proven CPU controller conservatively. Runtime hot
        # mutation is not inferred from GPU semantics.
        return "RESTART_REQUIRED", {"code": "CPU_HOT_MUTATION_NOT_CERTIFIED"}


class MlxMetalBackendAdapterV2(BackendAdapterV2):
    name = "mlx_metal"

    def _resolve_action(self, modes: list[str], count: int):
        if count == 0:
            return "NOOP", None
        if any(mode in {"RESTART_REQUIRED", "IMMUTABLE"} for mode in modes):
            return "RESTART_REQUIRED", {"code": "TARGET_REQUIRES_RESTART"}
        if any(mode == "IMPLEMENTED_UNVERIFIED" for mode in modes):
            return "VALIDATION_REQUIRED", {"code": "HOT_REBIND_NOT_VERIFIED"}
        if count == 1 and all(mode == "HOT_REBIND_SINGLE" for mode in modes):
            return "HOT_REBIND_SINGLE", None
        if count <= 8 and all(
            mode in {"HOT_REBIND_SINGLE", "HOT_REBIND_MULTI"} for mode in modes
        ):
            return "HOT_REBIND_MULTI", None
        return "RESTART_REQUIRED", {"code": "ATOMIC_TARGET_LIMIT_OR_MODE"}


def adapter_v2(name: str, capability_bundle: Mapping[str, Any]) -> BackendAdapterV2:
    if name == "cpu":
        return CpuBackendAdapterV2(capability_bundle)
    if name == "mlx_metal":
        return MlxMetalBackendAdapterV2(capability_bundle)
    raise BackendV2Error(f"unsupported backend={name}")
