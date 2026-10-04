#!/usr/bin/env python3
"""P12 isolated CPU qNg64 restart-canary controller.

This controller is deliberately model-process agnostic.  It materializes one
exact group-64 arbitrary-n tensor from a BF16/F32 safetensors checkpoint, then
uses the common CpuRestartExecutionAdapterV2 to restart an *isolated* runtime,
validate it, and optionally roll back on failure.

It never accepts production serving paths and never enables live promotion.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

import backend_adapters_v2 as planv2
import backend_execution_v2 as execv2
import model_capability as mc
import precision_context as pc
import quant_sim_n as qsim


class CpuQng64CanaryError(RuntimeError):
    pass


_PRODUCTION_MARKERS = (
    "/vdsp_serving/persistent-workers/",
    "/vdsp_serving/active-route.json",
)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _refuse_production_path(path: str | Path) -> Path:
    value = _resolved(path)
    text = str(value)
    if any(marker in text for marker in _PRODUCTION_MARKERS):
        raise CpuQng64CanaryError(f"production path refused: {value}")
    return value


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with _resolved(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_tensor(
    *,
    checkpoint: str | Path,
    tensor_name: str,
    n: int,
    output: str | Path,
) -> dict:
    checkpoint = _refuse_production_path(checkpoint)
    output = _refuse_production_path(output)
    if int(n) < 2 or int(n) > 16:
        raise CpuQng64CanaryError("qNg64 n must be within [2,16]")
    vals, out_dim, in_dim = qsim.read_tensor_f32(str(checkpoint), str(tensor_name))
    if in_dim % qsim.GROUP:
        raise CpuQng64CanaryError(
            f"tensor input width {in_dim} is not group-{qsim.GROUP} aligned"
        )
    quantized = qsim.quantize_dequantize_n(vals, out_dim, in_dim, int(n))
    rel_l2 = qsim.rel_l2(vals, quantized)
    output.parent.mkdir(parents=True, exist_ok=True)
    qsim.write_single_tensor_safetensors(
        str(output), "sim", (out_dim, in_dim), quantized
    )
    result = {
        "schema": "beglin-p12-cpu-qng64-materialization-v1",
        "status": "MATERIALIZED",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "tensor_name": str(tensor_name),
        "shape": [out_dim, in_dim],
        "group_size": qsim.GROUP,
        "n": int(n),
        "rel_l2": rel_l2,
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "production_write_allowed": False,
        "automatic_live_promotion": False,
    }
    result["materialization_sha256"] = mc.stable_identity_sha256(result)
    return result


class IsolatedCpuQng64RestartCanary:
    def __init__(
        self,
        *,
        bundle: Mapping[str, Any],
        target_key: str,
        role: str,
        layer: int,
        baseline_n: int,
        candidate_n: int,
        query_state: Callable[[], Mapping[str, Any]],
        restart: Callable[[list[dict]], Mapping[str, Any]],
        validate: Callable[[], Mapping[str, Any]],
        rollback: Callable[[list[dict]], Mapping[str, Any]],
    ):
        self.bundle = dict(bundle)
        self.target_key = str(target_key)
        self.role = str(role)
        self.layer = int(layer)
        self.baseline_n = int(baseline_n)
        self.candidate_n = int(candidate_n)
        self.query_state = query_state
        self.restart = restart
        self.validate = validate
        self.rollback = rollback

    def plan(self) -> dict:
        adapter = planv2.adapter_v2("cpu", self.bundle)
        before = execv2._normalize_state(self.query_state(), backend="cpu")
        target_policy = pc.normalize_policy([
            {
                "role": self.role,
                "layer": self.layer,
                "n": self.candidate_n,
            }
        ])
        if before["policy"] != pc.normalize_policy([
            {"role": self.role, "layer": self.layer, "n": self.baseline_n}
        ]):
            raise CpuQng64CanaryError("isolated CPU preimage policy mismatch")
        return adapter.plan_transition(
            state=planv2.BackendStateV2.build(
                backend="cpu",
                epoch=before["epoch"],
                policy=before["policy"],
                model_capability_bundle_sha256=self.bundle["bundle_sha256"],
            ),
            target_policy=target_policy,
            target_keys={(self.role, self.layer): self.target_key},
        )

    def run(self) -> dict:
        plan = self.plan()
        if plan["action"] != "RESTART_REQUIRED":
            raise CpuQng64CanaryError(
                f"CPU qNg64 canary must be restart-only, got {plan['action']}"
            )
        executor = execv2.CpuRestartExecutionAdapterV2(
            model_capability_bundle=self.bundle,
            query_state=self.query_state,
            restart=self.restart,
            validate=self.validate,
            rollback=self.rollback,
        )
        result = executor.execute(plan)
        if result["status"] != "RESTART_VERIFIED":
            raise CpuQng64CanaryError("isolated restart did not verify")
        validation = result.get("validation") or {}
        if validation.get("status") != "PASS":
            raise CpuQng64CanaryError("candidate validation did not PASS")
        out = {
            "schema": "beglin-p12-cpu-qng64-restart-canary-v1",
            "status": "PASS",
            "backend": "cpu",
            "target_key": self.target_key,
            "baseline_n": self.baseline_n,
            "candidate_n": self.candidate_n,
            "transition": result,
            "production_touched": False,
            "production_write_allowed": False,
            "automatic_live_promotion": False,
        }
        out["canary_sha256"] = mc.stable_identity_sha256(out)
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    mat = sub.add_parser("materialize")
    mat.add_argument("--checkpoint", required=True)
    mat.add_argument("--tensor", required=True)
    mat.add_argument("--n", required=True, type=int)
    mat.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.command == "materialize":
        result = materialize_tensor(
            checkpoint=args.checkpoint,
            tensor_name=args.tensor,
            n=args.n,
            output=args.output,
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
