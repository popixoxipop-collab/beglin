#!/usr/bin/env python3
"""Immutable verification-evidence registry for P12 model capabilities.

The registry never upgrades a capability by itself. It only selects exact,
already-VERIFIED evidence whose component, architecture, checkpoint identity,
backend, and target (when applicable) match the inspected model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import model_capability as mc


class EvidenceRegistryError(RuntimeError):
    pass


MODEL_COMPONENTS = {"backend_runtime", "tokenizer", "loader"}
TARGET_COMPONENTS = {"qng64_runtime", "mutation_runtime"}
ALL_COMPONENTS = MODEL_COMPONENTS | TARGET_COMPONENTS
BACKEND_COMPONENTS = {"backend_runtime"} | TARGET_COMPONENTS


def _require_object(value: Any, *, source: str) -> dict:
    if not isinstance(value, Mapping):
        raise EvidenceRegistryError(f"evidence must be an object: {source}")
    row = dict(value)
    if row.get("schema") != mc.VERIFICATION_EVIDENCE_SCHEMA:
        raise EvidenceRegistryError(
            f"unsupported evidence schema in {source}: {row.get('schema')!r}"
        )
    if row.get("status") != "VERIFIED":
        raise EvidenceRegistryError(f"evidence is not VERIFIED: {source}")
    component = str(row.get("component") or "")
    if component not in ALL_COMPONENTS:
        raise EvidenceRegistryError(
            f"unsupported evidence component in {source}: {component!r}"
        )
    arch = str(row.get("architecture_id") or "")
    checkpoint = str(row.get("checkpoint_identity_sha256") or "").lower()
    evidence_sha = str(row.get("evidence_sha256") or "").lower()
    if not arch:
        raise EvidenceRegistryError(f"architecture_id is empty: {source}")
    for name, text in (
        ("checkpoint_identity_sha256", checkpoint),
        ("evidence_sha256", evidence_sha),
    ):
        if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
            raise EvidenceRegistryError(f"{name} must be sha256 hex: {source}")

    backend = row.get("backend")
    if component in BACKEND_COMPONENTS:
        if backend not in {"cpu", "mlx_metal"}:
            raise EvidenceRegistryError(
                f"{component} evidence needs cpu/mlx_metal backend: {source}"
            )
    elif backend is not None and backend not in {"cpu", "mlx_metal"}:
        raise EvidenceRegistryError(f"invalid optional backend in {source}")

    if component in TARGET_COMPONENTS:
        target_key = str(row.get("target_key") or "")
        if not target_key:
            raise EvidenceRegistryError(
                f"{component} evidence requires target_key: {source}"
            )
        supported_n = row.get("supported_n")
        if not isinstance(supported_n, list) or not supported_n:
            raise EvidenceRegistryError(
                f"{component} evidence requires supported_n: {source}"
            )
        row["target_key"] = target_key
        row["supported_n"] = sorted({int(x) for x in supported_n})
        if component == "mutation_runtime":
            mode = str(row.get("mutation_mode") or "")
            if not mode:
                raise EvidenceRegistryError(
                    f"mutation_runtime evidence requires mutation_mode: {source}"
                )
            row["mutation_mode"] = mode

    row["checkpoint_identity_sha256"] = checkpoint
    row["evidence_sha256"] = evidence_sha
    row["_registry_source"] = source
    return row


def _load_json_file(path: Path) -> list[dict]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceRegistryError(f"invalid evidence JSON: {path}") from exc
    if isinstance(value, list):
        return [
            _require_object(row, source=f"{path}#{idx}")
            for idx, row in enumerate(value)
        ]
    return [_require_object(value, source=str(path))]


def load_evidence_paths(paths: Iterable[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise EvidenceRegistryError(f"evidence path does not exist: {path}")
        if path.is_symlink():
            raise EvidenceRegistryError(f"evidence symlink is not accepted: {path}")
        if path.is_dir():
            files = sorted(p for p in path.glob("*.json") if p.is_file())
            if not files:
                raise EvidenceRegistryError(
                    f"evidence directory has no JSON files: {path}"
                )
            for file in files:
                rows.extend(_load_json_file(file))
        elif path.is_file():
            rows.extend(_load_json_file(path))
        else:
            raise EvidenceRegistryError(f"unsupported evidence path: {path}")

    dedup: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("component"),
            row.get("architecture_id"),
            row.get("checkpoint_identity_sha256"),
            row.get("backend"),
            row.get("target_key"),
            row.get("evidence_sha256"),
        )
        dedup.setdefault(key, row)
    return sorted(
        dedup.values(),
        key=lambda r: (
            str(r.get("component")),
            str(r.get("architecture_id")),
            str(r.get("checkpoint_identity_sha256")),
            str(r.get("backend") or ""),
            str(r.get("target_key") or ""),
            str(r.get("evidence_sha256")),
        ),
    )


class VerificationEvidenceRegistry:
    def __init__(self, rows: Iterable[Mapping[str, Any]] = ()):
        self.rows = [
            _require_object(
                row, source=str(row.get("_registry_source", "<memory>"))
            )
            for row in rows
        ]

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]):
        return cls(load_evidence_paths(paths))

    def resolve(
        self,
        *,
        component: str,
        architecture_id: str,
        checkpoint_identity_sha256: str,
        backend: str | None = None,
        target_key: str | None = None,
    ) -> dict | None:
        checkpoint = str(checkpoint_identity_sha256).lower()
        if component not in ALL_COMPONENTS:
            raise EvidenceRegistryError(
                f"unsupported evidence component: {component!r}"
            )
        matches = [
            dict(row)
            for row in self.rows
            if row.get("component") == component
            and row.get("architecture_id") == architecture_id
            and row.get("checkpoint_identity_sha256") == checkpoint
            and (
                component not in BACKEND_COMPONENTS
                or row.get("backend") == backend
            )
            and (
                component not in TARGET_COMPONENTS
                or row.get("target_key") == target_key
            )
        ]
        if not matches:
            return None
        hashes = {row["evidence_sha256"] for row in matches}
        if len(hashes) != 1:
            raise EvidenceRegistryError(
                "ambiguous verification evidence for "
                f"component={component} architecture={architecture_id} "
                f"checkpoint={checkpoint} backend={backend} "
                f"target={target_key}: {sorted(hashes)}"
            )
        row = matches[0]
        row.pop("_registry_source", None)
        return row

    def resolve_target_component(
        self,
        *,
        component: str,
        architecture_id: str,
        checkpoint_identity_sha256: str,
        backend: str,
    ) -> list[dict]:
        if component not in TARGET_COMPONENTS:
            raise EvidenceRegistryError(
                f"component is not target-scoped: {component!r}"
            )
        checkpoint = str(checkpoint_identity_sha256).lower()
        targets = sorted(
            {
                str(row["target_key"])
                for row in self.rows
                if row.get("component") == component
                and row.get("architecture_id") == architecture_id
                and row.get("checkpoint_identity_sha256") == checkpoint
                and row.get("backend") == backend
            }
        )
        return [
            self.resolve(
                component=component,
                architecture_id=architecture_id,
                checkpoint_identity_sha256=checkpoint,
                backend=backend,
                target_key=target,
            )
            for target in targets
        ]

    def resolve_model_set(
        self,
        *,
        architecture_id: str,
        checkpoint_identity_sha256: str,
    ) -> dict:
        return {
            "cpu_runtime_evidence": self.resolve(
                component="backend_runtime",
                architecture_id=architecture_id,
                checkpoint_identity_sha256=checkpoint_identity_sha256,
                backend="cpu",
            ),
            "mlx_runtime_evidence": self.resolve(
                component="backend_runtime",
                architecture_id=architecture_id,
                checkpoint_identity_sha256=checkpoint_identity_sha256,
                backend="mlx_metal",
            ),
            "tokenizer_evidence": self.resolve(
                component="tokenizer",
                architecture_id=architecture_id,
                checkpoint_identity_sha256=checkpoint_identity_sha256,
            ),
            "loader_evidence": self.resolve(
                component="loader",
                architecture_id=architecture_id,
                checkpoint_identity_sha256=checkpoint_identity_sha256,
            ),
            "cpu_qng64_evidence": self.resolve_target_component(
                component="qng64_runtime",
                architecture_id=architecture_id,
                checkpoint_identity_sha256=checkpoint_identity_sha256,
                backend="cpu",
            ),
            "mlx_qng64_evidence": self.resolve_target_component(
                component="qng64_runtime",
                architecture_id=architecture_id,
                checkpoint_identity_sha256=checkpoint_identity_sha256,
                backend="mlx_metal",
            ),
            "cpu_mutation_evidence": self.resolve_target_component(
                component="mutation_runtime",
                architecture_id=architecture_id,
                checkpoint_identity_sha256=checkpoint_identity_sha256,
                backend="cpu",
            ),
            "mlx_mutation_evidence": self.resolve_target_component(
                component="mutation_runtime",
                architecture_id=architecture_id,
                checkpoint_identity_sha256=checkpoint_identity_sha256,
                backend="mlx_metal",
            ),
        }


def merge_explicit_and_registry(
    explicit: Mapping[str, Any] | None,
    registry_value: Mapping[str, Any] | None,
    *,
    label: str,
) -> dict | None:
    if explicit is None:
        return dict(registry_value) if registry_value is not None else None
    if registry_value is None:
        return dict(explicit)
    a = dict(explicit)
    b = dict(registry_value)
    if a.get("evidence_sha256") != b.get("evidence_sha256"):
        raise EvidenceRegistryError(
            f"explicit and registry {label} evidence disagree: "
            f"{a.get('evidence_sha256')} != {b.get('evidence_sha256')}"
        )
    return a


def merge_explicit_list_and_registry(
    explicit: Iterable[Mapping[str, Any]],
    registry_values: Iterable[Mapping[str, Any]],
    *,
    label: str,
) -> list[dict]:
    by_target: dict[str, dict] = {}
    for source_name, rows in (
        ("registry", registry_values),
        ("explicit", explicit),
    ):
        for raw in rows:
            row = dict(raw)
            target = str(row.get("target_key") or "")
            if not target:
                raise EvidenceRegistryError(
                    f"{label} {source_name} evidence missing target_key"
                )
            previous = by_target.get(target)
            if previous is not None:
                capability_fields = (
                    "schema", "status", "component", "architecture_id",
                    "checkpoint_identity_sha256", "backend", "target_key",
                    "evidence_sha256", "supported_n", "mutation_mode",
                )
                left = {
                    key: (
                        sorted({int(n) for n in previous.get(key) or []})
                        if key == "supported_n" else previous.get(key)
                    )
                    for key in capability_fields
                }
                right = {
                    key: (
                        sorted({int(n) for n in row.get(key) or []})
                        if key == "supported_n" else row.get(key)
                    )
                    for key in capability_fields
                }
                if left != right:
                    raise EvidenceRegistryError(
                        f"explicit and registry {label} evidence semantics "
                        f"disagree for target={target}: "
                        f"registry={left} explicit={right}"
                    )
            by_target[target] = row
    return [by_target[key] for key in sorted(by_target)]
