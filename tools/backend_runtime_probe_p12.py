#!/usr/bin/env python3
"""P12 backend-symmetric read-only runtime surface.

This module bridges the existing immutable binary capability report and runtime
ACK/state into one CPU/MLX API.  It deliberately has no mutation implementation.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Callable, Mapping

import backend_adapter_v2 as bav2
import model_capability as mc


class BackendRuntimeProbeError(RuntimeError):
    pass


def normalize_binary_capability(report: Mapping[str, Any]) -> dict:
    """Normalize legacy precision-capability-v1 into deterministic P12 form.

    Observation time, host name, binary path and link-report text are excluded
    from the stable probe identity. Binary hash, architecture and compiled
    control symbols remain identity-bearing.
    """
    if report.get("schema") != "precision-capability-v1":
        raise BackendRuntimeProbeError("unsupported backend capability schema")
    worker = dict(report.get("worker") or {})
    binary_sha = mc.require_sha("binary sha256", worker.get("binary_sha256"))
    arch = str(worker.get("arch") or "")
    if not arch:
        raise BackendRuntimeProbeError("worker architecture is missing")

    backends = report.get("backends") or {}
    cpu_raw = dict(backends.get("cpu") or {})
    mlx_raw = dict(backends.get("mlx_metal") or {})
    widths = dict(mlx_raw.get("qng64_widths") or {})
    supported_n = sorted(
        int(n)
        for n, status in widths.items()
        if str(status) not in {"UNSUPPORTED_BINARY", "UNSUPPORTED"}
    )
    runtime = dict(mlx_raw.get("runtime_control") or {})
    symbols = {
        str(k): bool(v)
        for k, v in sorted((runtime.get("symbols") or {}).items())
    }

    payload = {
        "schema": "beglin-backend-binary-probe-v2",
        "binary_sha256": binary_sha,
        "arch": arch,
        "backends": {
            "cpu": {
                "compiled": bool(cpu_raw.get("compiled", True)),
                "status": (
                    "IMPLEMENTED_UNVERIFIED"
                    if bool(cpu_raw.get("compiled", True))
                    else "UNSUPPORTED"
                ),
                "mutation_modes": ["RESTART_REQUIRED"],
                # Legacy capability report does not prove CPU qNg64 widths.
                "supported_n": [],
            },
            "mlx_metal": {
                "compiled": bool(mlx_raw.get("compiled", False)),
                "status": (
                    "IMPLEMENTED_UNVERIFIED"
                    if bool(mlx_raw.get("compiled", False))
                    else "UNSUPPORTED"
                ),
                "supported_n": supported_n,
                "runtime_control_compiled": bool(runtime.get("compiled", False)),
                "runtime_control_status": str(
                    runtime.get("status", "UNSUPPORTED")
                ),
                "control_symbols": symbols,
                "mutation_modes": (
                    ["HOT_REBIND_SINGLE", "HOT_REBIND_MULTI", "RESTART_REQUIRED"]
                    if bool(runtime.get("compiled", False))
                    else ["RESTART_REQUIRED"]
                ),
            },
        },
    }
    payload["probe_sha256"] = mc.sha256_json(payload)
    return payload


def backend_state_from_ack(
    *,
    backend: str,
    ack: Mapping[str, Any],
    model_id: str | None = None,
) -> bav2.BackendState:
    try:
        epoch = int(ack["weight_epoch"])
        policy = list(ack["active_policy"])
        policy_hash = str(ack["active_policy_hash"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendRuntimeProbeError("runtime state/ACK is incomplete") from exc
    state = bav2.BackendState(
        backend=str(backend),
        epoch=epoch,
        policy=bav2.normalize_policy(policy),
        policy_hash=policy_hash,
        model_id=model_id,
    )
    bav2.validate_backend_state(state)
    return state


class ReadOnlyBackendSurfaceV2:
    """Common CPU/MLX read-only surface used before any mutation bridge."""

    def __init__(
        self,
        *,
        backend: str,
        binary_probe: Mapping[str, Any],
        state_provider: Callable[[], bav2.BackendState | Mapping[str, Any]],
        verified_hot_targets=(),
        max_atomic_targets: int = 8,
    ):
        self.backend = str(backend)
        self.binary_probe = copy.deepcopy(dict(binary_probe))
        self.state_provider = state_provider
        if self.binary_probe.get("schema") != "beglin-backend-binary-probe-v2":
            raise BackendRuntimeProbeError("binary probe is not normalized P12 v2")
        if self.backend not in self.binary_probe.get("backends", {}):
            raise BackendRuntimeProbeError(f"backend missing from probe: {self.backend}")
        if self.backend == "cpu":
            self.adapter = bav2.CpuBackendAdapterV2()
        elif self.backend == "mlx_metal":
            self.adapter = bav2.MlxMetalBackendAdapterV2(
                verified_hot_targets=verified_hot_targets,
                max_atomic_targets=max_atomic_targets,
            )
        else:
            raise BackendRuntimeProbeError(f"unsupported backend={self.backend!r}")

    def probe_capabilities(self) -> dict:
        row = copy.deepcopy(self.binary_probe["backends"][self.backend])
        return {
            "schema": "beglin-backend-adapter-capability-v2",
            "backend": self.backend,
            "probe_sha256": self.binary_probe["probe_sha256"],
            **row,
            "production_write_allowed": False,
        }

    def query_applied_state(self) -> bav2.BackendState:
        raw = self.state_provider()
        if isinstance(raw, bav2.BackendState):
            state = raw
        else:
            state = backend_state_from_ack(
                backend=self.backend,
                ack=raw,
                model_id=raw.get("model_id"),
            )
        if state.backend != self.backend:
            raise BackendRuntimeProbeError(
                f"state backend mismatch: expected={self.backend} actual={state.backend}"
            )
        bav2.validate_backend_state(state)
        return state

    def plan_transition(self, target_policy) -> dict:
        return self.adapter.plan_transition(
            state=self.query_applied_state(),
            target_policy=target_policy,
        )

    def collect_runtime_evidence(self) -> dict:
        state = self.query_applied_state()
        evidence = {
            "schema": "beglin-backend-runtime-evidence-v2",
            "backend": self.backend,
            "probe_sha256": self.binary_probe["probe_sha256"],
            "epoch": int(state.epoch),
            "policy": bav2.normalize_policy(state.policy),
            "policy_hash": state.policy_hash,
            "model_id": state.model_id,
            "production_write_allowed": False,
        }
        evidence["evidence_sha256"] = mc.sha256_json(evidence)
        return evidence

    def apply_transition(self, *args, **kwargs):
        raise bav2.BackendPlanError(
            "P12 ReadOnlyBackendSurfaceV2 does not enable runtime mutation"
        )

    def rollback_transition(self, *args, **kwargs):
        raise bav2.BackendPlanError(
            "P12 ReadOnlyBackendSurfaceV2 does not enable runtime rollback"
        )
