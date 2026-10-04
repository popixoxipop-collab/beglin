#!/usr/bin/env python3
"""Compile a P12 capability bundle and materialize read-only P8->P11 bindings.

This command never mutates a serving worker, route, precision policy, or approval
state. It is an acceptance/materialization tool for capability evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_capability as mc
import p12_pipeline_bridge as bridge


class AcceptanceError(RuntimeError):
    pass


def _load_json(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    try:
        value = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"invalid JSON: {p}") from exc
    if not isinstance(value, dict):
        raise AcceptanceError(f"expected JSON object: {p}")
    return value


def _load_json_list(paths: list[str] | None) -> list[dict]:
    return [_load_json(path) for path in (paths or [])]


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize(
    *,
    model_path: str,
    backend: str,
    runtime_evidence: Mapping[str, Any] | None = None,
    loader_evidence: Mapping[str, Any] | None = None,
    tokenizer_evidence: Mapping[str, Any] | None = None,
    qng64_evidence: list[Mapping[str, Any]] | None = None,
    mutation_evidence: list[Mapping[str, Any]] | None = None,
    target_key: str | None = None,
    requested_n: int | None = None,
    provenance_evidence_sha256: str | None = None,
    runtime_state: Mapping[str, Any] | None = None,
) -> tuple[dict, dict]:
    kwargs = {
        "backend": backend,
        "tokenizer_evidence": tokenizer_evidence,
        "loader_evidence": loader_evidence,
    }
    if backend == "cpu":
        kwargs.update(
            cpu_runtime_evidence=runtime_evidence,
            cpu_qng64_evidence=list(qng64_evidence or []),
            cpu_mutation_evidence=list(mutation_evidence or []),
        )
    elif backend == "mlx_metal":
        kwargs.update(
            mlx_runtime_evidence=runtime_evidence,
            mlx_qng64_evidence=list(qng64_evidence or []),
            mlx_mutation_evidence=list(mutation_evidence or []),
        )
    else:
        raise AcceptanceError(f"unsupported backend: {backend}")

    bundle = mc.compile_model_capabilities(model_path, **kwargs)
    result = {
        "schema": "beglin-p12-capability-acceptance-v1",
        "status": "BUNDLE_COMPILED",
        "model_id": bundle["model_id"],
        "checkpoint_identity_sha256": bundle["checkpoint_identity_sha256"],
        "model_skeleton_sha256": bundle["model_skeleton"]["skeleton_sha256"],
        "model_capability_bundle_sha256": bundle["bundle_sha256"],
        "tensor_count": bundle["tensor_role_graph"]["tensor_count"],
        "mapped_tensor_count": bundle["tensor_role_graph"]["mapped_tensor_count"],
        "unmapped_tensor_count": bundle["tensor_role_graph"]["unmapped_tensor_count"],
        "mapping_coverage": bundle["tensor_role_graph"]["mapping_coverage"],
        "eligibility": bundle["p8_p11_eligibility"],
        "target_key": target_key,
        "requested_n": requested_n,
        "backend": backend,
        "automatic_live_promotion": False,
        "production_write_allowed": False,
    }

    if target_key is None:
        return bundle, result

    p8 = bridge.bind_p8_target(
        bundle,
        target_key=target_key,
        backend=backend,
        requested_n=requested_n,
    )
    result["p8_binding"] = p8
    result["status"] = "P8_BOUND"

    if provenance_evidence_sha256 is None:
        return bundle, result

    p9 = bridge.bind_p9_certification(
        bundle,
        p8_binding=p8,
        provenance_evidence_sha256=provenance_evidence_sha256,
    )
    p10 = bridge.select_p10_canary(bundle, p9_binding=p9)
    result["p9_binding"] = p9
    result["p10_binding"] = p10
    result["status"] = p10["status"]

    if runtime_state is None:
        return bundle, result

    if p10.get("status") != "READY_FOR_P10_CANARY":
        raise AcceptanceError(
            "runtime_state was supplied but P10 still requires capability validation"
        )

    state = dict(runtime_state)
    state.setdefault("model_capability_bundle_sha256", bundle["bundle_sha256"])
    state.setdefault("checkpoint_identity_sha256", bundle["checkpoint_identity_sha256"])
    state.setdefault("backend", backend)
    state.setdefault("target_key", target_key)
    p11 = bridge.build_p11_capability_preimage(
        bundle,
        p10_binding=p10,
        runtime_state=state,
    )
    result["p11_capability_preimage"] = p11
    result["status"] = "P11_CAPABILITY_PREIMAGE_READY"
    return bundle, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--backend", choices=["cpu", "mlx_metal"], required=True)
    ap.add_argument("--runtime-evidence")
    ap.add_argument("--loader-evidence")
    ap.add_argument("--tokenizer-evidence")
    ap.add_argument("--qng64-evidence", action="append", default=[])
    ap.add_argument("--mutation-evidence", action="append", default=[])
    ap.add_argument("--target")
    ap.add_argument("--requested-n", type=int)
    ap.add_argument("--provenance-sha256")
    ap.add_argument("--runtime-state")
    ap.add_argument("--bundle-output")
    ap.add_argument("--result-output")
    args = ap.parse_args()

    try:
        bundle, result = materialize(
            model_path=args.model,
            backend=args.backend,
            runtime_evidence=_load_json(args.runtime_evidence),
            loader_evidence=_load_json(args.loader_evidence),
            tokenizer_evidence=_load_json(args.tokenizer_evidence),
            qng64_evidence=_load_json_list(args.qng64_evidence),
            mutation_evidence=_load_json_list(args.mutation_evidence),
            target_key=args.target,
            requested_n=args.requested_n,
            provenance_evidence_sha256=args.provenance_sha256,
            runtime_state=_load_json(args.runtime_state),
        )
    except (mc.ModelCapabilityError, bridge.PipelineCapabilityError, AcceptanceError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2

    if args.bundle_output:
        Path(args.bundle_output).write_text(
            json.dumps(bundle, sort_keys=True, indent=2) + "\n"
        )
        result["bundle_file_sha256"] = _sha_file(Path(args.bundle_output))
    if args.result_output:
        Path(args.result_output).write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n"
        )
        result["result_file_sha256"] = _sha_file(Path(args.result_output))

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
