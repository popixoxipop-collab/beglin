#!/usr/bin/env python3
"""Execute one exact, attested P11 production precision cutover on XOX.

This is deliberately narrower than the HTTP serving path. The live candidate
worker has an L26 adaptive wrapper; P11 must reproduce the fixed-policy P10
canary, so the cutover trigger is one direct persistent-worker admission.
No route-manifest mutation and no automatic promotion are performed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import urllib.request
import uuid

import gpu_runtime_control as grc
import precision_p11_production_gate as p11
import production_serving_supervisor_persistent as ps


SERVING = Path("/Users/xox/vdsp_serving")
EVIDENCE_ROOT = SERVING / "p11-production-approval-20261004"
REQUEST_PATH = EVIDENCE_ROOT / "approval-request.json"
PLAN_PATH = EVIDENCE_ROOT / "cutover-plan.json"
PREIMAGE_PATH = EVIDENCE_ROOT / "production-preimage.json"
P10_RAW = (
    SERVING
    / "p8-l3-provenance-recapture-f7f7532"
    / "provenance"
    / "9a0dfec8bb5d047acb340a7d61fa2467624f2f2b5b95398e2a3d60211357aab7"[:24]
    / "prompt.i32"
)
MANIFEST_PATH = SERVING / "active-route.json"
ACK_PATH = SERVING / "persistent-workers/candidate/applied_ack.json"
TXN_PATH = SERVING / "persistent-workers/candidate/txn.cmd"
QUEUE_DIR = SERVING / "persistent-workers/candidate/queue"
REQUESTS_DIR = SERVING / "persistent-workers/candidate/requests"
APPROVAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "attestations"
    / "BEGLIN_PRECISION_P11_PRODUCTION_APPROVAL_2026-10-04.json"
)
WORKFLOW = (
    "popixoxipop-collab/beglin/.github/workflows/"
    "precision-p11-production-approval-attest.yml"
)
SOURCE_REF = "refs/heads/precision-p11-production-approval-v1-20261004"
REPO = "popixoxipop-collab/beglin"
HEALTH_URL = "http://127.0.0.1:18765/healthz"
EXPECTED_GOOD = [55222, 1]
EXPECTED_BAD = [55222, 372]


class P11ExecutionError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def health() -> dict:
    with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:
        return json.load(response)


def tokens_from_raw(path: Path = P10_RAW) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise P11ExecutionError("P10 raw token file has invalid byte length")
    return list(struct.unpack("<" + "i" * (len(raw) // 4), raw))


def verify_attestation() -> dict:
    if not APPROVAL_PATH.is_file():
        raise P11ExecutionError("P11 approval artifact is missing")
    gh = shutil.which("gh")
    if not gh:
        raise P11ExecutionError("GitHub CLI is unavailable")
    proc = subprocess.run(
        [
            gh,
            "attestation",
            "verify",
            str(APPROVAL_PATH),
            "--repo",
            REPO,
            "--signer-workflow",
            WORKFLOW,
            "--source-ref",
            SOURCE_REF,
            "--deny-self-hosted-runners",
            "--format",
            "json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise P11ExecutionError("GitHub OIDC approval attestation verification failed")
    try:
        rows = json.loads(proc.stdout)
    except Exception as exc:
        raise P11ExecutionError("invalid gh attestation verification output") from exc
    if not isinstance(rows, list) or not rows:
        raise P11ExecutionError("no verified P11 approval attestation found")
    return {"verified_attestations": len(rows), "approval_sha256": sha_file(APPROVAL_PATH)}


def verify_approval(request: dict, plan: dict) -> dict:
    approval = read_json(APPROVAL_PATH)
    if approval.get("schema") != "beglin-precision-p11-production-approval-v1":
        raise P11ExecutionError("unexpected approval schema")
    if approval.get("status") != "VERIFIED_GITHUB_OIDC":
        raise P11ExecutionError("approval is not VERIFIED_GITHUB_OIDC")
    exact = {
        "approval_request_sha256": request["approval_request_sha256"],
        "cutover_plan_sha256": request["cutover_plan_sha256"],
        "preimage_sha256": request["preimage_sha256"],
        "p10_result_sha256": request["p10_result_sha256"],
        "executor_source_sha256": request["executor_source_sha256"],
        "route_generation": request["route_generation"],
        "route_manifest_sha256": request["route_manifest_sha256"],
        "worker_pid": request["worker_pid"],
        "weight_epoch": request["weight_epoch"],
        "ack_sha256": request["ack_sha256"],
        "candidate_policy_hash": request["candidate_policy_hash"],
    }
    for key, value in exact.items():
        if approval.get(key) != value:
            raise P11ExecutionError(f"approval binding mismatch: {key}")
    if approval.get("one_shot_precision_cutover_allowed") is not True:
        raise P11ExecutionError("approval does not authorize one-shot cutover")
    if approval.get("automatic_live_promotion") is not False:
        raise P11ExecutionError("approval unexpectedly enables auto promotion")
    if approval.get("external_network_exposed") is not False:
        raise P11ExecutionError("approval unexpectedly enables external exposure")
    if plan.get("cutover_plan_sha256") != request["cutover_plan_sha256"]:
        raise P11ExecutionError("request/plan SHA binding mismatch")
    if plan.get("executor_source_sha256") != request["executor_source_sha256"]:
        raise P11ExecutionError("request/plan executor SHA mismatch")
    if sha_file(Path(__file__).resolve()) != request["executor_source_sha256"]:
        raise P11ExecutionError("executor source SHA differs from approved plan")

    now = datetime.now(timezone.utc)
    issued = datetime.fromisoformat(approval["issued_at"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(approval["expires_at"].replace("Z", "+00:00"))
    if not issued <= now < expires:
        raise P11ExecutionError("P11 approval is expired or not yet valid")
    nonce = str(approval.get("nonce") or "")
    if not nonce:
        raise P11ExecutionError("approval nonce is empty")
    return approval


def verify_live_preimage(request: dict) -> tuple[dict, dict]:
    live = health()
    if live.get("status") != "ok":
        raise P11ExecutionError("production supervisor is not healthy")
    if live.get("route_generation") != request["route_generation"]:
        raise P11ExecutionError("route generation drifted")
    if live.get("route_manifest_sha256") != request["route_manifest_sha256"]:
        raise P11ExecutionError("route manifest drifted")
    if live.get("production_write_allowed") is not False:
        raise P11ExecutionError("production_write_allowed boundary changed")
    if live.get("auto_promotion_enabled") is not False:
        raise P11ExecutionError("auto promotion boundary changed")
    if live.get("external_network_exposed") is not False:
        raise P11ExecutionError("external exposure boundary changed")

    worker = (live.get("workers") or {}).get(p11.ACTIVE_ROUTE_ID) or {}
    if worker.get("alive") is not True:
        raise P11ExecutionError("candidate worker is not alive")
    if int(worker.get("pid")) != int(request["worker_pid"]):
        raise P11ExecutionError("candidate worker PID drifted")
    if int(worker.get("weight_epoch")) != int(request["weight_epoch"]):
        raise P11ExecutionError("candidate worker epoch drifted")
    if worker.get("runtime_policy_hash") != p11.BASELINE_POLICY_HASH:
        raise P11ExecutionError("candidate runtime policy drifted")
    if worker.get("ack_sha256") != request["ack_sha256"]:
        raise P11ExecutionError("candidate ACK drifted")

    ack = grc.read_runtime_ack(ACK_PATH)
    if int(ack["weight_epoch"]) != int(request["weight_epoch"]):
        raise P11ExecutionError("runtime ACK epoch drifted")
    if ack["active_policy_hash"] != p11.BASELINE_POLICY_HASH:
        raise P11ExecutionError("runtime ACK policy drifted")
    if ack["ack_sha256"] != request["ack_sha256"]:
        raise P11ExecutionError("runtime ACK identity drifted")
    return live, ack


def require_queue_idle() -> None:
    busy = [
        path
        for path in (QUEUE_DIR / "request.txt", QUEUE_DIR / "request.processing")
        if path.exists()
    ]
    if busy:
        raise P11ExecutionError(
            "candidate persistent queue is busy: " + ",".join(str(p) for p in busy)
        )


def direct_admission(tokens: list[int], *, max_new_tokens: int = 2) -> dict:
    """One direct persistent-engine admission, bypassing the adaptive L26 wrapper."""
    require_queue_idle()
    request_id = "p11-" + uuid.uuid4().hex
    root = REQUESTS_DIR / request_id
    root.mkdir(parents=True, exist_ok=False)
    raw = root / "req-0.i32"
    manifest = root / "manifest.txt"
    result_path = root / "result.txt"
    tmp = QUEUE_DIR / f"request.txt.tmp.{request_id}"
    ready = QUEUE_DIR / "request.txt"
    processing = QUEUE_DIR / "request.processing"

    try:
        raw.write_bytes(struct.pack("<" + "i" * len(tokens), *tokens))
        manifest.write_text(f"{raw} {int(max_new_tokens)}\n")
        if ready.exists() or processing.exists():
            raise P11ExecutionError("candidate queue became busy before admission")
        tmp.write_text(
            f"BEGLIN_GPU_PERSISTENT_REQUEST_V1 {request_id} {manifest} {result_path}\n"
        )
        started = time.monotonic()
        os.replace(tmp, ready)
        deadline = time.time() + 60
        while time.time() < deadline:
            if result_path.is_file():
                parsed = ps.parse_persistent_result(
                    result_path.read_text(), request_id=request_id, expected_requests=1
                )
                parsed["roundtrip_ms"] = (time.monotonic() - started) * 1000.0
                return parsed
            time.sleep(0.01)
        raise P11ExecutionError("direct persistent admission timed out")
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        if result_path.exists():
            shutil.rmtree(root, ignore_errors=True)


def cancel_pending_txn(txn_id: str) -> bool:
    ack = grc.read_runtime_ack(ACK_PATH)
    if ack.get("txn_id") == txn_id:
        return False
    if ack["active_policy_hash"] != p11.BASELINE_POLICY_HASH:
        return False
    require_queue_idle()
    try:
        text = TXN_PATH.read_text().strip()
    except FileNotFoundError:
        return True
    parts = text.split()
    if len(parts) >= 2 and parts[1] == txn_id:
        TXN_PATH.unlink()
        return True
    return False


def rollback_if_needed(*, approval: dict, request: dict, tokens: list[int]) -> dict:
    ack = grc.read_runtime_ack(ACK_PATH)
    if ack["active_policy_hash"] == p11.BASELINE_POLICY_HASH:
        return {
            "status": "BASELINE_ALREADY_ACTIVE",
            "weight_epoch": int(ack["weight_epoch"]),
            "runtime_policy_hash": ack["active_policy_hash"],
        }
    if ack["active_policy_hash"] != p11.TARGET_POLICY_HASH:
        raise P11ExecutionError(
            "cannot auto-rollback unknown runtime policy "
            + str(ack["active_policy_hash"])
        )
    rollback_txn = "p11-rollback-" + approval["nonce"]
    requested = grc.prepare_rebind_set(
        ack_path=ACK_PATH,
        txn_path=TXN_PATH,
        txn_id=rollback_txn,
        expected_epoch=int(ack["weight_epoch"]),
        expected_policy_hash=p11.TARGET_POLICY_HASH,
        changes=[
            {
                "role": "shared_up_proj",
                "layer": 3,
                "expected_n": 5,
                "target_n": 6,
            }
        ],
    )
    if requested["target_policy_hash"] != p11.BASELINE_POLICY_HASH:
        raise P11ExecutionError("rollback target policy hash mismatch")
    result = direct_admission(tokens)
    terminal = grc.verify_terminal_ack(
        ack_path=ACK_PATH,
        txn_id=rollback_txn,
        allowed_statuses={"REBIND_SET_APPLIED"},
    )
    if terminal["active_policy_hash"] != p11.BASELINE_POLICY_HASH:
        raise P11ExecutionError("rollback terminal policy mismatch")
    if result.get("finite_logits") is not True or result.get("responses") != [EXPECTED_BAD]:
        raise P11ExecutionError("rollback reference verification failed")
    return {
        "status": "ROLLBACK_VERIFIED",
        "txn_id": rollback_txn,
        "after_epoch": int(terminal["weight_epoch"]),
        "runtime_policy_hash": terminal["active_policy_hash"],
        "response": result["responses"][0],
        "roundtrip_ms": float(result.get("roundtrip_ms", 0.0)),
    }


def execute() -> dict:
    request = read_json(REQUEST_PATH)
    plan = read_json(PLAN_PATH)
    preimage = read_json(PREIMAGE_PATH)
    attestation = verify_attestation()
    approval = verify_approval(request, plan)

    if preimage.get("preimage_sha256") != request["preimage_sha256"]:
        raise P11ExecutionError("request/preimage SHA binding mismatch")
    live_before, ack_before = verify_live_preimage(request)
    require_queue_idle()
    tokens = tokens_from_raw()
    raw_sha = hashlib.sha256(P10_RAW.read_bytes()).hexdigest()
    if raw_sha != plan["evidence"]["raw_token_sha256"]:
        raise P11ExecutionError("P10 raw input SHA mismatch")

    nonce_dir = EVIDENCE_ROOT / "consumed-nonces"
    nonce_dir.mkdir(parents=True, exist_ok=True)
    nonce_path = nonce_dir / (approval["nonce"] + ".json")
    try:
        fd = os.open(nonce_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise P11ExecutionError("approval nonce was already consumed") from exc
    with os.fdopen(fd, "w") as handle:
        json.dump(
            {
                "nonce": approval["nonce"],
                "approval_request_sha256": request["approval_request_sha256"],
                "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")

    txn_id = "p11-prod-" + approval["nonce"]
    published = False
    try:
        requested = grc.prepare_rebind_set(
            ack_path=ACK_PATH,
            txn_path=TXN_PATH,
            txn_id=txn_id,
            expected_epoch=int(request["weight_epoch"]),
            expected_policy_hash=p11.BASELINE_POLICY_HASH,
            changes=plan["target"]["changes"],
        )
        published = True
        if requested["target_policy_hash"] != p11.TARGET_POLICY_HASH:
            raise P11ExecutionError("cutover target policy hash mismatch")

        result = direct_admission(tokens)
        terminal = grc.verify_terminal_ack(
            ack_path=ACK_PATH,
            txn_id=txn_id,
            allowed_statuses={"REBIND_SET_APPLIED"},
        )
        if int(terminal["weight_epoch"]) != int(request["weight_epoch"]) + 1:
            raise P11ExecutionError("cutover did not advance exactly one epoch")
        if terminal["active_policy_hash"] != p11.TARGET_POLICY_HASH:
            raise P11ExecutionError("cutover terminal policy mismatch")
        if result.get("finite_logits") is not True:
            raise P11ExecutionError("cutover produced non-finite logits")
        if result.get("responses") != [EXPECTED_GOOD]:
            raise P11ExecutionError(
                f"cutover reference mismatch: {result.get('responses')!r}"
            )

        live_after = health()
        worker = (live_after.get("workers") or {}).get(p11.ACTIVE_ROUTE_ID) or {}
        if int(worker.get("pid")) != int(request["worker_pid"]):
            raise P11ExecutionError("worker PID changed during cutover")
        if int(worker.get("weight_epoch")) != int(request["weight_epoch"]) + 1:
            raise P11ExecutionError("health epoch differs from terminal ACK")
        if worker.get("runtime_policy_hash") != p11.TARGET_POLICY_HASH:
            raise P11ExecutionError("health runtime policy differs from target")
        if live_after.get("route_generation") != request["route_generation"]:
            raise P11ExecutionError("route generation changed during precision cutover")
        if live_after.get("route_manifest_sha256") != request["route_manifest_sha256"]:
            raise P11ExecutionError("route manifest changed during precision cutover")

        out = {
            "schema": "beglin-precision-p11-production-cutover-result-v1",
            "status": "CUTOVER_VERIFIED",
            "production_touched": True,
            "automatic_live_promotion": False,
            "external_network_exposed": False,
            "approval_request_sha256": request["approval_request_sha256"],
            "cutover_plan_sha256": request["cutover_plan_sha256"],
            "approval_sha256": attestation["approval_sha256"],
            "verified_attestations": attestation["verified_attestations"],
            "approval_nonce": approval["nonce"],
            "worker_pid": int(request["worker_pid"]),
            "before_epoch": int(ack_before["weight_epoch"]),
            "after_epoch": int(terminal["weight_epoch"]),
            "before_policy_hash": p11.BASELINE_POLICY_HASH,
            "after_policy_hash": terminal["active_policy_hash"],
            "txn_id": txn_id,
            "response": result["responses"][0],
            "engine_wall_ms": float(result.get("engine_wall_ms", 0.0)),
            "roundtrip_ms": float(result.get("roundtrip_ms", 0.0)),
            "route_generation": live_after["route_generation"],
            "route_manifest_sha256": live_after["route_manifest_sha256"],
            "rollback": None,
        }
        out["result_sha256"] = p11.sha256_json(out)
        (EVIDENCE_ROOT / "cutover-result.json").write_text(
            json.dumps(out, sort_keys=True, indent=2) + "\n"
        )
        return out
    except Exception as exc:
        cancel = False
        if published:
            try:
                cancel = cancel_pending_txn(txn_id)
            except Exception:
                cancel = False
        try:
            rollback = rollback_if_needed(
                approval=approval, request=request, tokens=tokens
            )
        except Exception as rollback_exc:
            failure = {
                "schema": "beglin-precision-p11-production-cutover-result-v1",
                "status": "CUTOVER_FAILED_ROLLBACK_FAILED",
                "cutover_error": f"{type(exc).__name__}: {exc}",
                "rollback_error": f"{type(rollback_exc).__name__}: {rollback_exc}",
                "pending_txn_cancelled": cancel,
                "approval_nonce": approval["nonce"],
            }
            (EVIDENCE_ROOT / "cutover-result.json").write_text(
                json.dumps(failure, sort_keys=True, indent=2) + "\n"
            )
            raise P11ExecutionError(
                "cutover failed and rollback also failed"
            ) from rollback_exc
        failure = {
            "schema": "beglin-precision-p11-production-cutover-result-v1",
            "status": "CUTOVER_FAILED_ROLLED_BACK",
            "cutover_error": f"{type(exc).__name__}: {exc}",
            "pending_txn_cancelled": cancel,
            "approval_nonce": approval["nonce"],
            "rollback": rollback,
        }
        failure["result_sha256"] = p11.sha256_json(failure)
        (EVIDENCE_ROOT / "cutover-result.json").write_text(
            json.dumps(failure, sort_keys=True, indent=2) + "\n"
        )
        raise


def main() -> int:
    result = execute()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
