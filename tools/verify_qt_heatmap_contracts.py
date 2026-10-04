#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

EXPECTED = {
    "observation-identity-v2",
    "numeric-observation-v1",
    "quant-perturbation-v1",
    "training-sensitivity-v1",
    "qheatmap-v2",
    "theatmap-v2",
    "local-precision-policy-v1",
    "selective-training-policy-v1",
    "qt-beval-evidence-v1",
}

def _walk_refs(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "$ref":
                yield v
            yield from _walk_refs(v)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_refs(item)

def verify(root: str | Path | None = None) -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "schemas" / "heatmap"
    found = set()
    refs = []
    for path in sorted(root.glob("*.schema.json")):
        obj = json.loads(path.read_text())
        name = path.name.removesuffix(".schema.json")
        found.add(name)
        if obj.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise AssertionError(f"bad $schema: {path}")
        if obj.get("type") != "object":
            raise AssertionError(f"non-object contract: {path}")
        required = obj.get("required", [])
        if "schema" not in required:
            raise AssertionError(f"missing required schema discriminator: {path}")
        expected_schema = f"beglin-{name}"
        actual_schema = obj.get("properties", {}).get("schema", {}).get("const")
        if actual_schema != expected_schema:
            raise AssertionError(
                f"bad schema discriminator: {path} expected={expected_schema!r} actual={actual_schema!r}"
            )
        for ref in _walk_refs(obj):
            refs.append((path, ref))

    missing = EXPECTED - found
    extra = found - EXPECTED
    if missing or extra:
        raise AssertionError(f"contract set mismatch missing={sorted(missing)} extra={sorted(extra)}")

    for source, ref in refs:
        if "://" in ref or ref.startswith("#"):
            continue
        ref_path = (source.parent / ref).resolve()
        if not ref_path.is_file():
            raise AssertionError(f"unresolved local $ref source={source.name} ref={ref}")

    index = root / "CONTRACT_INDEX.md"
    if not index.is_file():
        raise AssertionError("CONTRACT_INDEX.md missing")
    index_text = index.read_text()
    for name in sorted(EXPECTED):
        if f"`{name}.schema.json`" not in index_text:
            raise AssertionError(f"contract missing from index: {name}")

    return {
        "status": "PASS",
        "schema": "beglin-qt-bskel-verification-v1",
        "schema_count": len(found),
        "local_ref_count": sum(1 for _, ref in refs if "://" not in ref and not ref.startswith("#")),
        "index": str(index),
    }

def main() -> int:
    print(json.dumps(verify(), sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
