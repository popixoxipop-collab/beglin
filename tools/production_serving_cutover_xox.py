#!/usr/bin/env python3
"""Apply one exact GitHub-approved localhost Beglin cutover on XOX.

Safety properties:
- verifies the cutover approval with GitHub OIDC/Sigstore before mutation
- rebuilds the exact approved cutover plan and checks its SHA
- performs generation-based CAS on the supervisor route manifest
- immediately probes 12 requests through the installed localhost supervisor
- leaves the candidate active only on a clean health verdict
- rolls back to the exact baseline on any failed health check or exception
- never enables automatic cutover or external network exposure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import production_routing_cutover as routing
import production_serving_supervisor as supervisor


ROUTE_MANIFEST = Path("/Users/xox/vdsp_serving/active-route.json")
APPROVAL_PATH = Path(
    "/Users/xox/mcp-sandbox/tailnet-commander/"
    "BEGLIN_PRODUCTION_CUTOVER_APPROVAL_2026-10-02.json"
)
EXPECTED_PLAN_SHA = "d398f2c56181f7ef819bbdce7418e00a73571687027bbc00e67e477246585c3d"
EXPECTED_APPROVAL_SHA = "2ab873a7b6ad34c15faac0d3344933551dcd8f00a25e804e68895b24c1be7bb9"
EXPECTED_SIGNER_WORKFLOW = (
    "popixoxipop-collab/beglin/.github/workflows/"
    "production-cutover-approval-attest.yml"
)
EXPECTED_SOURCE_REF = "refs/heads/production-serving-supervisor-20261002"
EXPECTED_REPO = "popixoxipop-collab/beglin"
PORT = 18765
REQUESTS = 12
EXPECTED_REFERENCE_TOKEN = 1224
REFERENCE_GEN_INDEX = 8


class CutoverExecutionError(RuntimeError):
    pass


def bridge_evidence() -> dict:
    return {
        "schema": "beglin-production-adapter-bridge-execution/1",
        "plan": {
            "plan_sha256": "d26bcc0273a887f4341581536e41dcb4398c319138107d12c9061aa3d07bc1dc"
        },
        "result": {
            "schema": "mlx-isolated-restart-canary-result-v1",
            "source_commit": supervisor.EXPECTED_HEAD,
            "binary_sha256": supervisor.EXPECTED_BINARY_SHA,
            "checkpoint_sha256": supervisor.EXPECTED_CHECKPOINT_SHA,
            "candidate_policy_hash": supervisor.CANDIDATE_POLICY_HASH,
            "requested_policy_applied": True,
            "finite_logits": True,
            "reference_token_match": True,
            "candidate_worker_exited": True,
            "baseline_worker_mutated": False,
            "production_routing_switched": False,
            "result_sha256": "37227ea66f6dc5766e879eca383e41494860bd7d5496334de5bc8ecd90f3202e",
        },
        "verdict": {
            "schema": "beglin-production-adapter-bridge-verdict/1",
            "status": "CANARY_PASS_REVIEW_REQUIRED",
            "decision": "NO_AUTO_PROMOTION",
            "production_routing_switch_allowed": False,
            "verdict_sha256": "1cc814c61e886befd30d1b9549d47ab4a53186a16d4f81ce82f609e9ccf8a1ea",
        },
    }


def build_exact_plan() -> dict:
    cap = supervisor.router_capability(
        route_manifest=ROUTE_MANIFEST,
        port=PORT,
    )
    plan = routing.build_cutover_plan(
        bridge_status=bridge_evidence(),
        baseline_route=supervisor.baseline_route(),
        candidate_route=supervisor.candidate_route(),
        router_capability=cap,
    )
    if plan["status"] != "READY_FOR_EXPLICIT_CUTOVER_APPROVAL":
        raise CutoverExecutionError(
            f"cutover plan is not ready: {plan['status']}"
        )
    if plan["plan_sha256"] != EXPECTED_PLAN_SHA:
        raise CutoverExecutionError(
            "cutover plan SHA mismatch: "
            f"expected={EXPECTED_PLAN_SHA} actual={plan['plan_sha256']}"
        )
    return plan


def verify_github_approval(path: Path = APPROVAL_PATH) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise CutoverExecutionError("cutover approval is not a regular file")
    actual_sha = routing.hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual_sha != EXPECTED_APPROVAL_SHA:
        raise CutoverExecutionError(
            f"approval file SHA mismatch: expected={EXPECTED_APPROVAL_SHA} actual={actual_sha}"
        )
    gh = shutil.which("gh")
    if not gh:
        raise CutoverExecutionError("GitHub CLI 'gh' is unavailable")
    cmd = [
        gh,
        "attestation",
        "verify",
        str(resolved),
        "--repo",
        EXPECTED_REPO,
        "--signer-workflow",
        EXPECTED_SIGNER_WORKFLOW,
        "--source-ref",
        EXPECTED_SOURCE_REF,
        "--deny-self-hosted-runners",
        "--format",
        "json",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise CutoverExecutionError(
            "GitHub cutover approval attestation verification failed"
        )
    try:
        verified = json.loads(proc.stdout)
    except Exception as exc:
        raise CutoverExecutionError(
            "invalid gh attestation verification JSON"
        ) from exc
    if not isinstance(verified, list) or not verified:
        raise CutoverExecutionError("no verified cutover attestation returned")
    approval = json.loads(resolved.read_text())
    return {
        "approval": approval,
        "verified_attestations": len(verified),
        "approval_sha256": actual_sha,
    }


def health_from_response(response: dict) -> tuple[dict, dict]:
    rows = response.get("responses")
    if not isinstance(rows, list):
        raise CutoverExecutionError("supervisor response list is missing")
    values = []
    for row in rows:
        tokens = row.get("generated_tokens")
        if not isinstance(tokens, list):
            raise CutoverExecutionError("generated_tokens is missing")
        values.append(
            tokens[REFERENCE_GEN_INDEX]
            if len(tokens) > REFERENCE_GEN_INDEX
            else None
        )
    reference_hits = sum(
        value == EXPECTED_REFERENCE_TOKEN for value in values
    )
    parity = (
        len(values) == REQUESTS
        and reference_hits == REQUESTS
        and response.get("finite_logits") is True
        and response.get("route", {}).get("route_id")
        == supervisor.candidate_route()["route_id"]
        and response.get("route", {}).get("policy_hash")
        == supervisor.CANDIDATE_POLICY_HASH
    )
    metrics = {
        "requests": len(rows),
        "errors": 0 if parity else 1,
        "duration_ms": int(response.get("duration_ms", 0)),
        "reference_parity": parity,
    }
    detail = {
        "reference_hits": reference_hits,
        "expected_reference_token": EXPECTED_REFERENCE_TOKEN,
        "distinct_reference_tokens": sorted(
            {value for value in values if value is not None}
        ),
        "finite_logits": response.get("finite_logits") is True,
        "route_id": response.get("route", {}).get("route_id"),
        "policy_hash": response.get("route", {}).get("policy_hash"),
        "duration_ms": int(response.get("duration_ms", 0)),
        "peak_child_rss_bytes": int(
            response.get("peak_child_rss_bytes", 0)
        ),
        "worker_instance_id": response.get("worker_instance_id"),
    }
    return metrics, detail


def execute() -> dict:
    supervisor.verify_runtime_identity()
    plan = build_exact_plan()
    verified = verify_github_approval()
    approval = routing.validate_cutover_approval(
        verified["approval"],
        cutover_plan=plan,
    )
    store = routing.AtomicRouteManifestStore(ROUTE_MANIFEST)
    before = store.read()
    if before["active_route"] != plan["baseline_route"]:
        raise CutoverExecutionError(
            "active route is not the exact approved baseline"
        )

    switched = None
    try:
        switched = routing.execute_cutover(
            store=store,
            cutover_plan=plan,
            approval=approval,
        )
        prompt = supervisor._read_first_certified_prompt()
        response = supervisor._post_json(
            PORT,
            "/v1/batch_generate",
            {
                "requests": [
                    {
                        "prompt_tokens": prompt,
                        "max_new_tokens": 10,
                    }
                    for _ in range(REQUESTS)
                ]
            },
        )
        metrics, detail = health_from_response(response)
        health = routing.evaluate_health(
            cutover_plan=plan,
            metrics=metrics,
        )
        if health["status"] != "CUTOVER_HEALTH_PASS_REVIEW_REQUIRED":
            rolled = routing.rollback(
                store=store,
                cutover_plan=plan,
                cutover_result=switched,
                reason="candidate health failed",
            )
            return {
                "schema": "beglin-production-serving-cutover-execution/1",
                "status": "ROLLED_BACK_AFTER_HEALTH_FAILURE",
                "production_routing_switched": False,
                "candidate_left_active": False,
                "automatic_cutover_used": False,
                "external_network_exposed": False,
                "plan_sha256": plan["plan_sha256"],
                "approval_sha256": verified["approval_sha256"],
                "verified_attestations": verified["verified_attestations"],
                "health": health,
                "probe": detail,
                "rollback": rolled,
            }
        current = store.read()
        if current["active_route"] != plan["candidate_route"]:
            raise CutoverExecutionError(
                "candidate route disappeared after passing health"
            )
        result = {
            "schema": "beglin-production-serving-cutover-execution/1",
            "status": "CUTOVER_HEALTH_PASS_REVIEW_REQUIRED",
            "production_routing_switched": True,
            "candidate_left_active": True,
            "automatic_cutover_used": False,
            "external_network_exposed": False,
            "plan_sha256": plan["plan_sha256"],
            "approval_sha256": verified["approval_sha256"],
            "verified_attestations": verified["verified_attestations"],
            "cutover_generation": switched["generation"],
            "active_route": current["active_route"],
            "route_manifest_sha256": current["manifest_sha256"],
            "health": health,
            "probe": detail,
        }
        result["result_sha256"] = routing.sha256_json(result)
        return result
    except Exception:
        if switched is not None:
            try:
                current = store.read()
                if current["active_route"] == plan["candidate_route"]:
                    routing.rollback(
                        store=store,
                        cutover_plan=plan,
                        cutover_result=switched,
                        reason="exception during cutover health verification",
                    )
            except Exception:
                pass
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", required=True)
    args = ap.parse_args()
    result = execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "CUTOVER_HEALTH_PASS_REVIEW_REQUIRED":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
