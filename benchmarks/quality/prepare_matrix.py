from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.quality.artifact import QualityArtifactError, sha256_json
    from benchmarks.quality.corpus import load_corpus
else:
    from .artifact import QualityArtifactError, sha256_json
    from .corpus import load_corpus

MATRIX_SCHEMA = "beglin-quality-matrix/1"
P8_SCHEMA = "beglin-p8-integration/1"
EXPECTED_P8_GATES = {
    "runtime": "P8_RUNTIME_VALID_8_OF_8",
    "policy": "P8_POLICY_ATTRIBUTION_8_OF_8",
    "evidence": "P8_EVIDENCE_INDEPENDENT_PASS",
}


def load_matrix(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != MATRIX_SCHEMA:
        raise QualityArtifactError(f"matrix.schema must be {MATRIX_SCHEMA}")
    baseline = raw.get("baseline")
    candidates = raw.get("candidates")
    repetitions = raw.get("repetitions_per_prompt")
    if not isinstance(baseline, dict) or baseline.get("precision") not in {"bf16", "fp16", "safe_baseline"}:
        raise QualityArtifactError("baseline precision must be bf16/fp16/safe_baseline")
    if not isinstance(candidates, list) or [row.get("n") for row in candidates] != [4, 5, 6, 7]:
        raise QualityArtifactError("candidates must be ordered n=4,5,6,7")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise QualityArtifactError("repetitions_per_prompt must be >= 2")
    return raw


def load_p8(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != P8_SCHEMA:
        raise QualityArtifactError(f"P8 evidence schema must be {P8_SCHEMA}")
    return raw


def p8_gate(evidence: dict[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {
            "ready": False,
            "status": "WAITING_P8_INTEGRATION",
            "problems": ["P8 integration evidence not supplied"],
        }
    problems = []
    if evidence.get("status") != "P8_INTEGRATION_PASS":
        problems.append("status != P8_INTEGRATION_PASS")
    gates = evidence.get("gates")
    if not isinstance(gates, dict):
        problems.append("gates missing")
    else:
        for key, expected in EXPECTED_P8_GATES.items():
            if gates.get(key) != expected:
                problems.append(f"gates.{key} != {expected}")
    identity = evidence.get("identity")
    required_identity = {
        "source_commit",
        "source_tree",
        "binary_sha256",
        "checkpoint_sha256",
        "model_id",
        "hardware_id",
        "backend",
        "seed",
        "runtime_config_hash",
    }
    if not isinstance(identity, dict):
        problems.append("identity missing")
    else:
        missing = sorted(required_identity - set(identity))
        if missing:
            problems.append(f"identity missing fields: {missing}")
    return {
        "ready": not problems,
        "status": "P8_INTEGRATION_PASS" if not problems else "WAITING_P8_INTEGRATION",
        "problems": problems,
    }


def build_plan(
    matrix_path: Path,
    corpus_path: Path,
    p8_path: Path | None,
) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    corpus = load_corpus(corpus_path)
    evidence = load_p8(p8_path)
    gate = p8_gate(evidence)

    cells = [{
        "variant_id": "reference",
        "kind": "reference",
        "precision": matrix["baseline"]["precision"],
        "n": None,
        "repetitions_per_prompt": matrix["repetitions_per_prompt"],
    }]
    cells.extend({
        "variant_id": f"n{row['n']}",
        "kind": "candidate",
        "precision": f"n{row['n']}",
        "n": row["n"],
        "repetitions_per_prompt": matrix["repetitions_per_prompt"],
    } for row in matrix["candidates"])

    payload = {
        "schema": "beglin-quality-execution-plan/1",
        "status": "P9_EXECUTION_PLAN_READY" if gate["ready"] else "P9_QUALITY_HARNESS_READY",
        "execution_allowed": bool(gate["ready"] and matrix.get("execution_adapter")),
        "p8_gate": gate,
        "matrix_path": str(matrix_path),
        "matrix_hash": sha256_json(matrix),
        "corpus_path": str(corpus_path),
        "corpus_sha256": corpus["corpus_sha256"],
        "prompt_count": len(corpus["entries"]),
        "cells": cells,
        "execution_adapter": matrix.get("execution_adapter"),
        "identity": evidence.get("identity") if evidence else None,
        "notes": [
            "Agent D does not execute physical comparisons before P8 integration passes.",
            "A null execution_adapter is intentional before A0 wires the integrated runner.",
            "All baseline/candidate cells must reuse the same control identity; precision policy is the experimental variable.",
        ],
    }
    payload["plan_hash"] = sha256_json(payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare P9 quality matrix; never bypasses P8.")
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--p8-evidence", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = build_plan(args.matrix, args.corpus, args.p8_evidence)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
