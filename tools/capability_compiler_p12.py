#!/usr/bin/env python3
"""P12 model capability compiler.

This is a read-only compiler. It turns the canonical tensor graph plus explicit
backend evidence into backend/quant/mutation matrices and a ModelCapabilityBundle.
No capability is promoted to VERIFIED without checkpoint-bound evidence supplied
by the caller.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import model_capability as mc


QNG64_N = (2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)

_QNG64_ROLE_PREFIXES = {
    "Q_PROJ",
    "K_PROJ",
    "V_PROJ",
    "O_PROJ",
    "Q_A_PROJ",
    "Q_B_PROJ",
    "KV_A_PROJ",
    "KV_B_PROJ",
    "DENSE_GATE",
    "DENSE_UP",
    "DENSE_DOWN",
    "SHARED_GATE",
    "SHARED_UP",
    "SHARED_DOWN",
}


class CapabilityCompilerError(RuntimeError):
    pass


def _verified_evidence(
    target_key: str,
    backend: str,
    evidence_by_target: Mapping[tuple[str, str], Iterable[Mapping]] | None,
) -> list[dict]:
    if not evidence_by_target:
        return []
    return [dict(row) for row in evidence_by_target.get((target_key, backend), [])]


def _role_supports_qng64(role: str) -> bool:
    role = str(role)
    if role.endswith("_BIAS"):
        return False
    return role in _QNG64_ROLE_PREFIXES


def compile_backend_capability_matrix(
    *,
    model_id: str,
    tensor_role_graph: Mapping,
    verified_targets: Mapping[str, Iterable[str]] | None = None,
    verified_hot_targets: Iterable[str] = (),
    evidence_by_target: Mapping[tuple[str, str], Iterable[Mapping]] | None = None,
) -> dict:
    verified_targets = verified_targets or {}
    verified_cpu = set(verified_targets.get("cpu", ()))
    verified_mlx = set(verified_targets.get("mlx_metal", ()))
    hot_mlx = set(verified_hot_targets)

    cells = []
    for node in tensor_role_graph.get("nodes", []):
        if node.get("mapping_status") != "MAPPED":
            continue
        target_key = str(node["canonical_target_key"])
        role = str(node["role"])
        supported_n = list(QNG64_N) if _role_supports_qng64(role) else []
        quant_formats = ["qNg64"] if supported_n else []

        cpu_evidence = _verified_evidence(
            target_key, "cpu", evidence_by_target
        )
        cpu_verified = target_key in verified_cpu and bool(cpu_evidence)
        cells.append({
            "target_key": target_key,
            "backend": "cpu",
            "inference_status": (
                "VERIFIED" if cpu_verified else "IMPLEMENTED_UNVERIFIED"
            ),
            "mutation_mode": "RESTART_REQUIRED",
            "supported_n": supported_n,
            "quant_formats": quant_formats,
            "evidence_refs": cpu_evidence,
            "reason_code": (
                "checkpoint_bound_evidence"
                if cpu_verified
                else "backend_path_known_checkpoint_evidence_missing"
            ),
        })

        mlx_evidence = _verified_evidence(
            target_key, "mlx_metal", evidence_by_target
        )
        mlx_verified = target_key in verified_mlx and bool(mlx_evidence)
        hot_verified = mlx_verified and target_key in hot_mlx and bool(supported_n)
        cells.append({
            "target_key": target_key,
            "backend": "mlx_metal",
            "inference_status": (
                "VERIFIED" if mlx_verified else "IMPLEMENTED_UNVERIFIED"
            ),
            "mutation_mode": (
                "HOT_REBIND_SINGLE" if hot_verified else "RESTART_REQUIRED"
            ),
            "supported_n": supported_n,
            "quant_formats": quant_formats,
            "evidence_refs": mlx_evidence,
            "reason_code": (
                "verified_hot_rebind"
                if hot_verified
                else (
                    "checkpoint_bound_evidence"
                    if mlx_verified
                    else "backend_path_known_checkpoint_evidence_missing"
                )
            ),
        })

    return mc.build_backend_capability_matrix(model_id=model_id, cells=cells)


def compile_quant_matrix(*, tensor_role_graph: Mapping, backend_matrix: Mapping) -> list[dict]:
    by_target: dict[str, list[Mapping]] = {}
    for cell in backend_matrix.get("cells", []):
        by_target.setdefault(str(cell["target_key"]), []).append(cell)

    rows = []
    for node in tensor_role_graph.get("nodes", []):
        if node.get("mapping_status") != "MAPPED":
            continue
        target = str(node["canonical_target_key"])
        cells = by_target.get(target, [])
        supported_n = sorted({
            int(n)
            for cell in cells
            for n in cell.get("supported_n", [])
        })
        verified = any(
            cell.get("inference_status") == "VERIFIED" for cell in cells
        )
        rows.append({
            "target_key": target,
            "role": node["role"],
            "source_dtype": node.get("dtype"),
            "source_quant_format": node.get("source_quant_format"),
            "qng64_status": (
                "VERIFIED"
                if verified and supported_n
                else (
                    "IMPLEMENTED_UNVERIFIED"
                    if supported_n
                    else "UNSUPPORTED_ROLE"
                )
            ),
            "supported_n": supported_n,
            "backends": sorted({str(cell["backend"]) for cell in cells}),
        })
    return sorted(rows, key=lambda row: row["target_key"])


def compile_mutation_matrix(*, backend_matrix: Mapping) -> list[dict]:
    return [
        {
            "target_key": cell["target_key"],
            "backend": cell["backend"],
            "mutation_mode": cell["mutation_mode"],
            "allowed_target_precisions": list(cell.get("supported_n", [])),
            "requires_quiesce": (
                cell["mutation_mode"]
                in {"HOT_REBIND_SINGLE", "HOT_REBIND_MULTI"}
            ),
            "requires_snapshot": (
                cell["mutation_mode"]
                in {"HOT_REBIND_SINGLE", "HOT_REBIND_MULTI"}
            ),
            "rollback_supported": (
                cell["inference_status"] == "VERIFIED"
            ),
            "policy_shape_change_allowed": False,
        }
        for cell in backend_matrix.get("cells", [])
    ]


def compile_capability_report(
    *,
    model_id: str,
    checkpoint_identity: str,
    weight_checkpoint_identity_sha256: str | None = None,
    skeleton_sha256: str,
    architecture_status: str,
    tokenizer_status: str,
    loader_status: str,
    tensor_role_graph: Mapping,
    verified_targets: Mapping[str, Iterable[str]] | None = None,
    verified_hot_targets: Iterable[str] = (),
    evidence_by_target: Mapping[tuple[str, str], Iterable[Mapping]] | None = None,
    bundle_evidence_refs: Iterable[Mapping] = (),
) -> dict:
    backend = compile_backend_capability_matrix(
        model_id=model_id,
        tensor_role_graph=tensor_role_graph,
        verified_targets=verified_targets,
        verified_hot_targets=verified_hot_targets,
        evidence_by_target=evidence_by_target,
    )
    quant = compile_quant_matrix(
        tensor_role_graph=tensor_role_graph,
        backend_matrix=backend,
    )
    mutation = compile_mutation_matrix(backend_matrix=backend)
    unsupported = sorted({
        str(node["canonical_target_key"])
        for node in tensor_role_graph.get("nodes", [])
        if node.get("mapping_status") == "UNSUPPORTED"
    })
    bundle = mc.build_model_capability_bundle(
        model_id=model_id,
        checkpoint_identity=checkpoint_identity,
        weight_checkpoint_identity_sha256=weight_checkpoint_identity_sha256,
        skeleton_sha256=skeleton_sha256,
        architecture_status=architecture_status,
        tokenizer_status=tokenizer_status,
        loader_status=loader_status,
        backend_matrix=backend["cells"],
        quant_matrix=quant,
        mutation_matrix=mutation,
        unsupported_targets=unsupported,
        evidence_refs=bundle_evidence_refs,
    )
    return {
        "schema": "beglin-p12-capability-report-v1",
        "backend_capability": backend,
        "quant_capability": quant,
        "runtime_mutation": mutation,
        "model_capability_bundle": bundle,
    }


def compile_model_capability_bundle(**kwargs) -> dict:
    """Compatibility wrapper returning only the canonical bundle."""
    return compile_capability_report(**kwargs)["model_capability_bundle"]
