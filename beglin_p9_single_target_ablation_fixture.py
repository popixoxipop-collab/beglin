#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "beglin-p9-single-target-physical-run/1"
RUNNER_ID = "xox-p9.1-single-target-ablation-v1"
REPO = Path("/Users/xox/mcp-sandbox/tailnet-commander/beglin-p9-integration")
SANDBOX = Path("/Users/xox/mcp-sandbox/tailnet-commander")
RUN_ROOT = SANDBOX / "p9_single_target_ablation_runs"
LATEST = SANDBOX / "p9_single_target_ablation_latest.json"
PLAN_PATH = REPO / "configs/quality/p9_single_target_ablation_v1.json"
TOKEN_FIXTURE_PATH = REPO / "configs/quality/p9_token_fixture_v1.json"
CANONICAL_RUNNER = REPO / "beglin_p9_single_target_ablation_fixture.py"

sys.path.insert(0, str(REPO))
import beglin_p9_quality_fixture as base  # noqa: E402
from benchmarks.quality.execution_adapter import parse_engine_log  # noqa: E402


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_plan() -> dict[str, Any]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan.get("schema") != "beglin-p9-single-target-ablation-plan/1":
        raise RuntimeError("single-target ablation plan schema mismatch")
    if plan.get("diagnostic_prompts") != [
        "constraint_words",
        "ppl_short_1",
        "short_reasoning",
    ]:
        raise RuntimeError("diagnostic prompt set or order changed")
    if plan.get("repetitions") != 2 or plan.get("expected_requests_per_cell") != 6:
        raise RuntimeError("ablation request shape changed")
    expected = {
        ("kv_l11_n5", "kv_a_proj_with_mqa", 11, 5),
        ("kv_l11_n6", "kv_a_proj_with_mqa", 11, 6),
        ("kv_l11_n7", "kv_a_proj_with_mqa", 11, 7),
        ("shared_l3_n5", "shared_up_proj", 3, 5),
        ("shared_l3_n6", "shared_up_proj", 3, 6),
        ("shared_l3_n7", "shared_up_proj", 3, 7),
    }
    observed = set()
    for cell in plan.get("cells", []):
        target = cell.get("target") or {}
        row = (
            cell.get("cell_id"),
            target.get("role"),
            target.get("layer"),
            cell.get("n"),
        )
        observed.add(row)
        expected_line = f"{row[1]} {row[2]} {row[3]}"
        if cell.get("promotion_lines") != [expected_line]:
            raise RuntimeError(f"{cell.get('cell_id')}: promotion line is not single-target exact")
    if observed != expected or len(plan.get("cells", [])) != 6:
        raise RuntimeError(f"ablation matrix mismatch: {sorted(observed)}")
    if plan.get("reference", {}).get("promotion_lines") != []:
        raise RuntimeError("ablation reference must not promote any target")
    if plan.get("production_write_allowed") is not False:
        raise RuntimeError("ablation plan must deny production writes")
    return plan


def runner_identity() -> str:
    executing = Path(__file__).resolve()
    if not CANONICAL_RUNNER.is_file():
        raise RuntimeError(f"canonical ablation runner missing: {CANONICAL_RUNNER}")
    executing_sha = sha256_file(executing)
    canonical_sha = sha256_file(CANONICAL_RUNNER)
    if executing_sha != canonical_sha:
        raise RuntimeError(
            "registered ablation runner SHA differs from repository runner: "
            f"{executing_sha} != {canonical_sha}"
        )
    return executing_sha


def source_manifest() -> tuple[str, list[dict[str, Any]]]:
    _, rows = base.source_manifest()
    existing = {row["path"] for row in rows}
    for path in (CANONICAL_RUNNER, PLAN_PATH):
        relative = str(path.relative_to(REPO))
        if relative in existing:
            continue
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    rows.sort(key=lambda row: row["path"])
    return sha256_bytes(stable_json(rows).encode()), rows


def materialize_diagnostic_tokens(
    work: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    full = base.materialize_tokens(work)
    token_fixture = json.loads(TOKEN_FIXTURE_PATH.read_text(encoding="utf-8"))
    max_new = {
        row["prompt_id"]: row["max_new_tokens"]
        for row in token_fixture["entries"]
    }
    materialized = {row["prompt_id"]: row for row in full["files"]}
    selected = []
    manifest_rows = []
    for prompt_id in plan["diagnostic_prompts"]:
        row = materialized.get(prompt_id)
        if row is None:
            raise RuntimeError(f"materialized fixture missing diagnostic prompt: {prompt_id}")
        selected.append(row)
        manifest_rows.append(f"{row['path']} {max_new[prompt_id]}")
    manifest = work / "p9-ablation-prompts.manifest"
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    subset = {
        "schema": "beglin-p9-single-target-materialized-fixture/1",
        "fixture_id": full["fixture_id"],
        "diagnostic_prompts": plan["diagnostic_prompts"],
        "repetitions": plan["repetitions"],
        "expected_requests": plan["expected_requests_per_cell"],
        "files": selected,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "full_materialization_sha256": sha256_file(work / "materialized_fixture.json"),
    }
    atomic_json(work / "ablation_materialized_fixture.json", subset)
    return subset


def engine_env(manifest: Path, promotion: Path | None, requests: int) -> dict[str, str]:
    env = base.engine_env(manifest, promotion)
    env["QWEN_MOE_CB_REQS"] = str(requests)
    return env


def run_variant(
    work: Path,
    name: str,
    policy: dict[str, Any],
    manifest: Path,
    expected_requests: int,
) -> dict[str, Any]:
    promotion = None
    lines = policy["promotion_lines"]
    if lines:
        promotion = work / f"{name}.promotion.txt"
        promotion.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        [str(base.BINARY)],
        cwd=str(REPO),
        env=engine_env(manifest, promotion, expected_requests),
        text=True,
        capture_output=True,
        timeout=21600,
        check=False,
    )
    log = work / f"{name}.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    text = log.read_text(encoding="utf-8")
    validation = base.VALIDATION_RE.search(text)
    request_count = len(base.P9_REQUEST_RE.findall(text))
    parsed_rows = []
    parse_error = None
    try:
        parsed_rows = parse_engine_log(text)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    metrics_complete = (
        len(parsed_rows) == expected_requests
        and all(row["finite_logits"] for row in parsed_rows)
        and all(row["nll_count"] == row["context_tokens"] - 1 for row in parsed_rows)
        and all(row["router_decisions"] > 0 for row in parsed_rows)
        and all(row["nout"] > 0 for row in parsed_rows)
    )
    valid = (
        result.returncode == 0
        and request_count == expected_requests
        and metrics_complete
        and validation is not None
        and validation.group("finite") == "1"
        and int(validation.group("requests")) == expected_requests
        and int(validation.group("count")) > 0
    )
    return {
        "cell_id": name,
        "status": "PASS" if valid else "FAIL",
        "exit_code": result.returncode,
        "request_count": request_count,
        "parsed_request_count": len(parsed_rows),
        "metrics_complete": metrics_complete,
        "parse_error": parse_error,
        "finite_logits": validation.group("finite") == "1" if validation else None,
        "logits_checked": int(validation.group("count")) if validation else None,
        "log_path": str(log),
        "log_sha256": sha256_file(log),
        "policy": policy,
        "policy_hash": sha256_bytes(stable_json(policy).encode()),
        "promotion_path": str(promotion) if promotion else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    plan = load_plan()
    runner_sha = runner_identity()
    head, branch = base.git_identity()
    engine_preflight = base.adapter_preflight()
    cache_projection = base.compact_cache_projection()
    manifest_hash, manifest_rows = source_manifest()
    checkpoint_sha, checkpoint_source = base.checkpoint_identity()

    if args.self_test:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "PASS",
                    "mode": "self-test",
                    "source_commit": head,
                    "runner_sha256": runner_sha,
                    "source_manifest_sha256": manifest_hash,
                    "checkpoint_sha256": checkpoint_sha,
                    "engine_preflight": engine_preflight,
                    "compact_cache_projection": cache_projection,
                    "plan_sha256": sha256_file(PLAN_PATH),
                },
                sort_keys=True,
            )
        )
        return 0

    started = datetime.now(timezone.utc)
    run_id = "p9a-" + started.strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    work = RUN_ROOT / run_id
    work.mkdir(parents=True, exist_ok=False)
    base.configure_and_build()
    materialized = materialize_diagnostic_tokens(work, plan)
    binary_sha = sha256_file(base.BINARY)
    runtime_config = {
        "ablation_id": plan["ablation_id"],
        "diagnostic_prompts": plan["diagnostic_prompts"],
        "repetitions": plan["repetitions"],
        "build": engine_preflight["engine_markers"],
        "compact_cache": cache_projection,
        "materialized_fixture_sha256": materialized["manifest_sha256"],
    }
    identity = {
        "source_commit": head,
        "source_tree": manifest_hash,
        "binary_sha256": binary_sha,
        "checkpoint_sha256": checkpoint_sha,
        "model_id": "deepseek-v2-lite",
        "hardware_id": "xox-apple-silicon",
        "backend": "mlx_metal",
        "seed": 0,
        "diagnostic_corpus_sha256": sha256_bytes(
            stable_json(
                {
                    "prompt_ids": plan["diagnostic_prompts"],
                    "repetitions": plan["repetitions"],
                    "token_manifest_sha256": materialized["manifest_sha256"],
                }
            ).encode()
        ),
        "runtime_config_hash": sha256_bytes(stable_json(runtime_config).encode()),
    }
    atomic_json(work / "identity.json", identity)

    policies = [("reference", plan["reference"])]
    policies.extend((cell["cell_id"], cell) for cell in plan["cells"])
    runs = [
        run_variant(
            work,
            name,
            policy,
            Path(materialized["manifest"]),
            plan["expected_requests_per_cell"],
        )
        for name, policy in policies
    ]
    status = "P9_SINGLE_TARGET_RAW_PASS" if all(row["status"] == "PASS" for row in runs) else "FAIL"
    result = {
        "schema": SCHEMA,
        "runner_id": RUNNER_ID,
        "run_id": run_id,
        "status": status,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "runner_sha256": runner_sha,
        "identity": identity,
        "checkpoint_identity_source": checkpoint_source,
        "source_manifest": manifest_rows,
        "source_manifest_sha256": manifest_hash,
        "engine_preflight": engine_preflight,
        "compact_cache_projection": cache_projection,
        "ablation_plan": plan,
        "ablation_plan_sha256": sha256_file(PLAN_PATH),
        "materialized_fixture": materialized,
        "runs": runs,
        "production_write_allowed": False,
        "note": "Targeted causal diagnosis only; no production release authority.",
    }
    result["result_sha256"] = sha256_bytes(stable_json(result).encode())
    atomic_json(work / "result.json", result)
    atomic_json(LATEST, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "P9_SINGLE_TARGET_RAW_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
