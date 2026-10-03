#!/usr/bin/env python3
"""Reviewed bridge from an approved Agent E proposal to an isolated MLX canary.

This module deliberately does NOT switch production routing. It verifies:
- the normalized proposal and measured runtime preimage,
- a GitHub-account + OIDC approval seal,
- the one-target candidate policy,
- bounded isolated-canary results.

Only an isolated restart canary may be launched by a caller. A successful
canary ends in CANARY_PASS_REVIEW_REQUIRED. Production routing remains a
separate disabled gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping

import manual_canary_contract as mc


APPROVAL_SCHEMA = "beglin-agent-e-github-oidc-seal/1"
APPROVAL_STATUS = "VERIFIED_GITHUB_OIDC_ATTESTATION_SUBJECT"
BRIDGE_SCHEMA = "beglin-production-adapter-bridge-plan/1"
RESULT_SCHEMA = "mlx-isolated-restart-canary-result-v1"
EXPECTED_REPO = "popixoxipop-collab/beglin"
EXPECTED_ACTOR = "popixoxipop-collab"
EXPECTED_ACTOR_ID = "261965176"
EXPECTED_SIGNER_WORKFLOW = (
    "popixoxipop-collab/beglin/.github/workflows/"
    "agent-e-github-approval-attest.yml"
)
EXPECTED_SOURCE_REF = "refs/heads/agent-e-github-approval-seal-20261002"


class ProductionAdapterBridgeError(RuntimeError):
    pass


class ProductionRoutingDisabled(ProductionAdapterBridgeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _policy_map(rows: list[dict]) -> dict[tuple[str, int], int]:
    out: dict[tuple[str, int], int] = {}
    for row in rows:
        try:
            key = (str(row["role"]), int(row["layer"]))
            n = int(row["n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionAdapterBridgeError("invalid policy row") from exc
        if not key[0] or key[1] < 0 or n <= 0:
            raise ProductionAdapterBridgeError("invalid policy target")
        if key in out:
            raise ProductionAdapterBridgeError("duplicate role/layer in policy")
        out[key] = n
    return out


def verify_github_attestation(
    artifact_path: str | Path,
    *,
    repo: str = EXPECTED_REPO,
    signer_workflow: str = EXPECTED_SIGNER_WORKFLOW,
    source_ref: str = EXPECTED_SOURCE_REF,
    gh_path: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    """Cryptographically verify the local seal artifact using GitHub attestations."""
    artifact = Path(artifact_path).expanduser().resolve(strict=True)
    if not artifact.is_file():
        raise ProductionAdapterBridgeError("attestation subject is not a regular file")

    if gh_path is None:
        resolved = shutil.which("gh")
        if not resolved:
            raise ProductionAdapterBridgeError("GitHub CLI 'gh' is unavailable")
        gh = Path(resolved).resolve(strict=True)
    else:
        gh = Path(gh_path).expanduser().resolve(strict=True)
    if gh.name != "gh":
        raise ProductionAdapterBridgeError("attestation verifier executable must be gh")

    cmd = [
        str(gh),
        "attestation",
        "verify",
        str(artifact),
        "--repo",
        str(repo),
        "--signer-workflow",
        str(signer_workflow),
        "--source-ref",
        str(source_ref),
        "--deny-self-hosted-runners",
        "--format",
        "json",
    ]
    proc = runner(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if int(proc.returncode) != 0:
        raise ProductionAdapterBridgeError("GitHub OIDC attestation verification failed")
    try:
        verified = json.loads(proc.stdout)
    except Exception as exc:
        raise ProductionAdapterBridgeError("invalid gh attestation JSON output") from exc
    if not isinstance(verified, list) or not verified:
        raise ProductionAdapterBridgeError("no verified GitHub attestations returned")
    return {
        "schema": "beglin-github-attestation-verification/1",
        "status": "VERIFIED",
        "artifact_sha256": sha256_file(artifact),
        "repository": str(repo),
        "signer_workflow": str(signer_workflow),
        "source_ref": str(source_ref),
        "verified_attestations": len(verified),
    }


def validate_github_approval_seal(
    seal: Mapping[str, Any],
    *,
    proposal_digest: str,
    measured_capture_sha256: str,
    expected_repo: str = EXPECTED_REPO,
    expected_actor: str = EXPECTED_ACTOR,
    expected_actor_id: str = EXPECTED_ACTOR_ID,
    attestation_verification: Mapping[str, Any] | None = None,
) -> dict:
    value = dict(seal)
    if value.get("schema") != APPROVAL_SCHEMA:
        raise ProductionAdapterBridgeError("unsupported GitHub approval seal schema")
    if value.get("status") != APPROVAL_STATUS:
        raise ProductionAdapterBridgeError("GitHub approval seal is not verified")
    if value.get("approval_path") != "github_account_plus_github_oidc":
        raise ProductionAdapterBridgeError("unexpected approval path")
    github = value.get("github")
    if not isinstance(github, dict):
        raise ProductionAdapterBridgeError("GitHub approval identity is missing")
    if github.get("repository") != expected_repo:
        raise ProductionAdapterBridgeError("GitHub approval repository mismatch")
    if github.get("actor") != expected_actor:
        raise ProductionAdapterBridgeError("GitHub approval actor mismatch")
    if str(github.get("actor_id")) != str(expected_actor_id):
        raise ProductionAdapterBridgeError("GitHub approval actor id mismatch")
    if value.get("proposal_digest") != proposal_digest:
        raise ProductionAdapterBridgeError("GitHub approval proposal digest mismatch")
    if value.get("measured_capture_sha256") != measured_capture_sha256:
        raise ProductionAdapterBridgeError("GitHub approval capture hash mismatch")
    for key in (
        "execution_enabled",
        "production_write_allowed",
        "auto_promotion_enabled",
        "production_bridge_reviewed",
    ):
        if value.get(key) is not False:
            raise ProductionAdapterBridgeError(f"GitHub approval guard {key} must be false")

    if attestation_verification is None:
        raise ProductionAdapterBridgeError("cryptographic GitHub attestation verification required")
    av = dict(attestation_verification)
    if av.get("schema") != "beglin-github-attestation-verification/1":
        raise ProductionAdapterBridgeError("unsupported attestation verification schema")
    if av.get("status") != "VERIFIED":
        raise ProductionAdapterBridgeError("GitHub attestation is not VERIFIED")
    if av.get("repository") != expected_repo:
        raise ProductionAdapterBridgeError("attestation verification repository mismatch")
    if int(av.get("verified_attestations", 0)) < 1:
        raise ProductionAdapterBridgeError("no verified attestation evidence")
    return value


def build_bridge_plan(
    *,
    proposal: Mapping[str, Any],
    runtime_preimage: Mapping[str, Any],
    candidate_policy: list[dict],
    github_seal: Mapping[str, Any],
    measured_capture_sha256: str,
    attestation_verification: Mapping[str, Any],
) -> dict:
    p = mc.normalize_proposal(proposal)
    if p["backend"] != "mlx_metal":
        raise ProductionAdapterBridgeError("bridge currently supports mlx_metal only")
    if p["production_write_allowed"] is not False:
        raise ProductionAdapterBridgeError("proposal must remain production-write disabled")

    digest = mc.sha256_json(p)
    validate_github_approval_seal(
        github_seal,
        proposal_digest=digest,
        measured_capture_sha256=str(measured_capture_sha256),
        attestation_verification=attestation_verification,
    )

    try:
        active_policy = list(runtime_preimage["active_policy"])
        active_policy_hash = str(runtime_preimage["active_policy_hash"])
        weight_epoch = int(runtime_preimage["weight_epoch"])
        ack_sha256 = str(runtime_preimage["ack_sha256"])
        worker_instance_id = str(runtime_preimage["worker_instance_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionAdapterBridgeError("runtime preimage is incomplete") from exc

    baseline = mc.normalize_proposal({**p})["baseline_policy_hash"]
    if active_policy_hash != baseline:
        raise ProductionAdapterBridgeError("runtime baseline policy hash mismatch")
    if mc.sha256_json(active_policy) != baseline:
        raise ProductionAdapterBridgeError("runtime active policy canonical hash mismatch")
    if weight_epoch != int(p["expected_epoch"]):
        raise ProductionAdapterBridgeError("runtime weight epoch mismatch")
    if worker_instance_id != p["restart_instance_id"]:
        raise ProductionAdapterBridgeError("runtime worker instance mismatch")
    if not ack_sha256:
        raise ProductionAdapterBridgeError("runtime ACK hash is empty")

    before = _policy_map(active_policy)
    after = _policy_map(candidate_policy)
    target = p["single_target"]
    key = (str(target["role"]), int(target["layer"]))
    expected_before = target.get("before_n")
    if expected_before is None:
        if key in before:
            raise ProductionAdapterBridgeError("target unexpectedly exists in baseline policy")
    elif before.get(key) != int(expected_before):
        raise ProductionAdapterBridgeError("target baseline precision mismatch")
    if after.get(key) != int(target["after_n"]):
        raise ProductionAdapterBridgeError("candidate target precision mismatch")
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if changed != {key}:
        raise ProductionAdapterBridgeError("bridge requires exactly one policy target change")
    if mc.sha256_json(candidate_policy) != p["candidate_policy_hash"]:
        raise ProductionAdapterBridgeError("candidate policy hash mismatch")

    if int(p["budget"]["max_requests"]) < 12:
        raise ProductionAdapterBridgeError("approved request budget is too small for the certified canary corpus")

    plan = {
        "schema": BRIDGE_SCHEMA,
        "status": "READY_FOR_ISOLATED_RESTART_CANARY",
        "mode": "isolated_restart",
        "backend": "mlx_metal",
        "proposal_digest": digest,
        "measured_capture_sha256": str(measured_capture_sha256),
        "source_commit": p["source_commit"],
        "binary_sha256": p["binary_sha256"],
        "checkpoint_sha256": p["checkpoint_sha256"],
        "runtime_preimage": {
            "active_policy": active_policy,
            "active_policy_hash": active_policy_hash,
            "weight_epoch": weight_epoch,
            "ack_sha256": ack_sha256,
            "worker_instance_id": worker_instance_id,
        },
        "candidate_policy": list(candidate_policy),
        "candidate_policy_hash": p["candidate_policy_hash"],
        "target": dict(target),
        "budget": dict(p["budget"]),
        "canary_requests": 12,
        "candidate_worker_launch_allowed": True,
        "production_routing_switch_allowed": False,
        "production_write_allowed": False,
        "auto_promotion_enabled": False,
        "baseline_worker_mutation_allowed": False,
        "rollback_strategy": "terminate isolated candidate; baseline worker remains untouched",
        "github_approval": {
            "actor": github_seal["github"]["actor"],
            "actor_id": str(github_seal["github"]["actor_id"]),
            "repository": github_seal["github"]["repository"],
            "run_id": str(github_seal["github"]["run_id"]),
            "source_sha": github_seal["github"]["sha"],
        },
        "attestation": dict(attestation_verification),
    }
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def evaluate_isolated_candidate(plan: Mapping[str, Any], result: Mapping[str, Any]) -> dict:
    p = dict(plan)
    r = dict(result)
    if p.get("schema") != BRIDGE_SCHEMA:
        raise ProductionAdapterBridgeError("invalid bridge plan schema")
    if p.get("production_routing_switch_allowed") is not False:
        raise ProductionAdapterBridgeError("bridge plan unexpectedly permits production routing")
    if r.get("schema") != RESULT_SCHEMA:
        raise ProductionAdapterBridgeError("invalid isolated candidate result schema")

    reasons: list[str] = []
    if r.get("source_commit") != p.get("source_commit"):
        reasons.append("source_commit")
    if r.get("binary_sha256") != p.get("binary_sha256"):
        reasons.append("binary_sha256")
    if r.get("checkpoint_sha256") != p.get("checkpoint_sha256"):
        reasons.append("checkpoint_sha256")
    if r.get("candidate_policy_hash") != p.get("candidate_policy_hash"):
        reasons.append("candidate_policy_hash")
    if r.get("requested_policy_applied") is not True:
        reasons.append("requested_policy_not_applied")
    if int(r.get("returncode", -1)) != 0:
        reasons.append("worker_returncode")
    if r.get("finite_logits") is not True:
        reasons.append("finite_logits")
    if r.get("reference_token_match") is not True:
        reasons.append("reference_token_match")

    budget = p["budget"]
    for metric, limit in (
        ("requests", "max_requests"),
        ("tokens", "max_tokens"),
        ("duration_ms", "max_duration_ms"),
        ("memory_bytes", "max_memory_bytes"),
    ):
        try:
            value = int(r[metric])
        except (KeyError, TypeError, ValueError):
            reasons.append(metric + "_missing")
            continue
        if value < 0 or value > int(budget[limit]):
            reasons.append(metric + "_budget")

    if reasons:
        out = {
            "schema": "beglin-production-adapter-bridge-verdict/1",
            "status": "ROLLBACK_REQUIRED",
            "decision": "REJECT_ISOLATED_CANDIDATE",
            "reasons": sorted(set(reasons)),
            "candidate_worker_must_exit": True,
            "baseline_remains_active": True,
            "production_routing_switch_allowed": False,
            "production_write_allowed": False,
            "auto_promotion_enabled": False,
            "plan_sha256": p.get("plan_sha256"),
        }
    else:
        out = {
            "schema": "beglin-production-adapter-bridge-verdict/1",
            "status": "CANARY_PASS_REVIEW_REQUIRED",
            "decision": "NO_AUTO_PROMOTION",
            "reasons": [],
            "candidate_worker_must_exit": True,
            "baseline_remains_active": True,
            "production_routing_switch_allowed": False,
            "production_write_allowed": False,
            "auto_promotion_enabled": False,
            "plan_sha256": p.get("plan_sha256"),
        }
    out["verdict_sha256"] = sha256_json(out)
    return out


def execute_production_routing_switch(*args, **kwargs):
    raise ProductionRoutingDisabled(
        "production routing switch is not implemented in the reviewed bridge; "
        "a separate cutover gate is required after isolated canary review"
    )
