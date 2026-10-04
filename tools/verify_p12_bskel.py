#!/usr/bin/env python3
"""Fail-closed verifier for the P12 model BSKEL contract surface."""
from __future__ import annotations

import json
from pathlib import Path

import model_capability as mc

EXPECTED = {
    "model-source-v1.schema.json": "beglin-model-source-v1",
    "architecture-descriptor-v1.schema.json": "beglin-architecture-descriptor-v1",
    "model-skeleton-v1.schema.json": "beglin-model-skeleton-v1",
    "tensor-role-graph-v1.schema.json": "beglin-tensor-role-graph-v1",
    "operator-graph-v1.schema.json": "beglin-operator-graph-v1",
    "tokenizer-contract-v1.schema.json": "beglin-tokenizer-contract-v1",
    "loader-contract-v1.schema.json": "beglin-loader-contract-v1",
    "backend-capability-v1.schema.json": "beglin-backend-capability-v1",
    "quant-capability-v1.schema.json": "beglin-quant-capability-v1",
    "runtime-mutation-v1.schema.json": "beglin-runtime-mutation-v1",
    "model-capability-bundle-v1.schema.json": "beglin-model-capability-bundle-v1",
    "pipeline-eligibility-v1.schema.json": "beglin-pipeline-eligibility-v1",
    "validation-plan-v1.schema.json": "beglin-validation-plan-v1",
}


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "contracts" / "model"
    actual = {p.name for p in root.glob("*.schema.json")}
    missing = sorted(set(EXPECTED) - actual)
    if missing:
        raise SystemExit("P12-BSKEL FAIL missing contracts: " + ",".join(missing))

    ids = set()
    for filename, schema_const in sorted(EXPECTED.items()):
        obj = json.loads((root / filename).read_text())
        if obj.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise SystemExit(f"P12-BSKEL FAIL bad metaschema: {filename}")
        sid = obj.get("$id")
        if not isinstance(sid, str) or not sid.startswith("beglin://contracts/model/"):
            raise SystemExit(f"P12-BSKEL FAIL bad $id: {filename}")
        if sid in ids:
            raise SystemExit(f"P12-BSKEL FAIL duplicate $id: {sid}")
        ids.add(sid)
        got = ((obj.get("properties") or {}).get("schema") or {}).get("const")
        if got != schema_const:
            raise SystemExit(
                f"P12-BSKEL FAIL schema discriminator {filename}: expected={schema_const} actual={got}"
            )

    files = [{"logical_path": "model.safetensors", "sha256": "a" * 64,
              "size_bytes": 7, "kind": "safetensors"}]
    a = mc.build_model_source_manifest(
        model_id="fixture", model_revision="r1", source_format="SAFETENSORS_SINGLE",
        files=files, root_path="/a", discovered_at="one",
    )
    b = mc.build_model_source_manifest(
        model_id="fixture", model_revision="r1", source_format="SAFETENSORS_SINGLE",
        files=files, root_path="/b", discovered_at="two",
    )
    if a["checkpoint_identity_sha256"] != b["checkpoint_identity_sha256"]:
        raise SystemExit("P12-BSKEL FAIL source identity depends on volatile metadata")

    unknown = mc.identify_architecture("totally_unknown")
    if unknown["status"] != "UNKNOWN_ARCHITECTURE" or unknown["inference_allowed"]:
        raise SystemExit("P12-BSKEL FAIL unknown architecture is not fail-closed")

    print("P12-BSKEL VERIFY PASS")
    print(json.dumps({
        "contracts": len(EXPECTED),
        "source_identity_sha256": a["checkpoint_identity_sha256"],
        "unknown_architecture_fail_closed": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
