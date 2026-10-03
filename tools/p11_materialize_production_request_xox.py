#!/usr/bin/env python3
"""Materialize a fresh P11 production preimage and exact approval request.

Read-only with respect to the live supervisor/worker state. It writes evidence
only under /Users/xox/vdsp_serving/p11-production-approval-20261004.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import gpu_runtime_control as grc
import precision_p11_production_gate as p11


ROOT = Path("/Users/xox/vdsp_serving/p11-production-approval-20261004")
P10_RESULT = Path("/Users/xox/vdsp_serving/p10-first-real-canary-20261004/result.json")
MANIFEST = Path("/Users/xox/vdsp_serving/active-route.json")
ACK = Path("/Users/xox/vdsp_serving/persistent-workers/candidate/applied_ack.json")
TXN = Path("/Users/xox/vdsp_serving/persistent-workers/candidate/txn.cmd")
PERSISTENT_BINARY = Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
HEALTH_URL = "http://127.0.0.1:18765/healthz"
EXECUTOR = Path(__file__).resolve().parent / "p11_execute_production_cutover_xox.py"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_health() -> dict:
    with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:
        return json.load(response)


def stable_identity(health: dict) -> dict:
    workers = {}
    for route_id, row in sorted((health.get("workers") or {}).items()):
        workers[route_id] = {
            "pid": row.get("pid"),
            "weight_epoch": row.get("weight_epoch"),
            "runtime_policy_hash": row.get("runtime_policy_hash"),
            "ack_sha256": row.get("ack_sha256"),
        }
    return {
        "route_generation": health.get("route_generation"),
        "route_manifest_sha256": health.get("route_manifest_sha256"),
        "active_route": health.get("active_route"),
        "workers": workers,
        "production_write_allowed": health.get("production_write_allowed"),
        "auto_promotion_enabled": health.get("auto_promotion_enabled"),
        "external_network_exposed": health.get("external_network_exposed"),
    }


def write_json(path: Path, value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, indent=2) + "\n"
    path.write_text(raw)
    return hashlib.sha256(raw.encode()).hexdigest()


def main() -> int:
    before = get_health()

    p10_raw = P10_RESULT.read_bytes()
    p10_result = json.loads(p10_raw)
    p10_sha = hashlib.sha256(p10_raw).hexdigest()
    p10_evidence = p11.validate_p10(p10_result, p10_sha)

    manifest = json.loads(MANIFEST.read_text())
    ack = grc.read_runtime_ack(ACK)
    txn_text = TXN.read_text()
    captured_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    preimage = p11.build_preimage(
        captured_at=captured_at,
        health=before,
        manifest=manifest,
        manifest_file_sha256=sha_file(MANIFEST),
        ack=ack,
        ack_file_sha256=sha_file(ACK),
        txn_text=txn_text,
        txn_file_sha256=sha_file(TXN),
        persistent_binary_sha256=sha_file(PERSISTENT_BINARY),
        p10_evidence=p10_evidence,
    )
    executor_source_sha256 = sha_file(EXECUTOR)
    plan = p11.build_plan(preimage, executor_source_sha256=executor_source_sha256)
    approval_request = p11.build_approval_request(plan)

    after = get_health()
    before_identity = stable_identity(before)
    after_identity = stable_identity(after)
    if before_identity != after_identity:
        raise p11.P11Error("production identity changed while P11 request was materialized")

    ROOT.mkdir(parents=True, exist_ok=True)
    preimage_file_sha = write_json(ROOT / "production-preimage.json", preimage)
    plan_file_sha = write_json(ROOT / "cutover-plan.json", plan)
    request_file_sha = write_json(ROOT / "approval-request.json", approval_request)

    result = {
        "schema": "beglin-precision-p11-production-request-materialization-v1",
        "status": "PASS",
        "production_touched": False,
        "production_unchanged": True,
        "production_cutover_allowed": False,
        "automatic_live_promotion": False,
        "trusted_production_approval_present": False,
        "preimage_sha256": preimage["preimage_sha256"],
        "cutover_plan_sha256": plan["cutover_plan_sha256"],
        "approval_request_sha256": approval_request["approval_request_sha256"],
        "p10_result_sha256": p10_evidence["p10_result_sha256"],
        "executor_source_sha256": executor_source_sha256,
        "expected_worker_pid": plan["expected_live_preimage"]["worker_pid"],
        "expected_weight_epoch": plan["expected_live_preimage"]["weight_epoch"],
        "expected_after_epoch": plan["target"]["expected_after_epoch"],
        "baseline_policy_hash": p11.BASELINE_POLICY_HASH,
        "candidate_policy_hash": p11.TARGET_POLICY_HASH,
        "files": {
            "production_preimage": str(ROOT / "production-preimage.json"),
            "production_preimage_file_sha256": preimage_file_sha,
            "cutover_plan": str(ROOT / "cutover-plan.json"),
            "cutover_plan_file_sha256": plan_file_sha,
            "approval_request": str(ROOT / "approval-request.json"),
            "approval_request_file_sha256": request_file_sha,
        },
    }
    result_file_sha = write_json(ROOT / "result.json", result)
    summary = dict(result)
    summary["result_file_sha256"] = result_file_sha
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
