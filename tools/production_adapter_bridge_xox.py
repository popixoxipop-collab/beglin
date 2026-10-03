#!/usr/bin/env python3
"""Run the reviewed Agent E bridge as an isolated XOX restart canary.

This executable NEVER switches production routing and NEVER mutates the
baseline worker. It launches one owned qwen worker with exactly the approved
single-target promotion, observes 12 replay requests, records bounded resource
usage, and exits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shlex
import subprocess
import sys
import tempfile
import time

import manual_canary_contract as mc
import production_adapter_bridge as bridge


REPO = Path("/Users/xox/vdsp-engine-gpu-precision")
BINARY = REPO / "build-gpu-precision/qwen_infer_gpu"
MOE_BASE = Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
SAFETENSORS = Path(
    "/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/"
    "model.safetensors.index.json"
)
SOURCE_MANIFEST = REPO / "g5_drill_manifest.txt"
CHECKPOINT_EVIDENCE = Path(
    "/Users/xox/vdsp_shadow_runs/checkpoint_identity/checkpoint_identity.json"
)
OUTPUT_ROOT = Path("/Users/xox/vdsp_shadow_runs/production_adapter_bridge")

EXPECTED_HEAD = "330954b27f146b8a17db2cb353c3e620968bad5e"
EXPECTED_BINARY_SHA = "6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392"
EXPECTED_CHECKPOINT_SHA = "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898"
EXPECTED_TARGET = {"role": "shared_up_proj", "layer": 3, "before_n": None, "after_n": 6}
REFERENCE_TOKEN = 1224
REFERENCE_POS = 16
PROMPT_LEN = 9
REQUESTS = 12
SLOTS = 4
TIMEOUT_SECONDS = 180

REQUEST_RE = re.compile(
    r"\[moe gpu cb online\] req (\d+) .*? tokens:\s*([^\n\r]*)"
)
VALIDATION_RE = re.compile(
    r"GPU_VALIDATION_V1 .*?finite_logits=(?P<finite>[01]).*?requests=(?P<requests>\d+)"
)


class XoxBridgeError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return proc.stdout.strip()


def _minimal_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _manifest_input() -> tuple[Path, int]:
    rows = [
        line.strip()
        for line in SOURCE_MANIFEST.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise XoxBridgeError("source manifest has no active rows")
    parts = shlex.split(rows[0])
    if len(parts) != 2:
        raise XoxBridgeError("invalid source manifest row")
    raw = Path(parts[0])
    max_new = int(parts[1])
    if not raw.is_file() or max_new <= 0:
        raise XoxBridgeError("invalid replay input")
    return raw, max_new


def _parse_requests(output: str) -> dict[int, list[int]]:
    reqs: dict[int, list[int]] = {}
    for match in REQUEST_RE.finditer(output):
        reqs[int(match.group(1))] = [int(x) for x in match.group(2).split()]
    return reqs


def _reference_stats(requests: dict[int, list[int]]) -> dict:
    gen_idx = REFERENCE_POS - (PROMPT_LEN - 1)
    values = [
        tokens[gen_idx]
        for tokens in requests.values()
        if 0 <= gen_idx < len(tokens)
    ]
    return {
        "gen_idx": gen_idx,
        "eligible_requests": len(values),
        "reference_token": REFERENCE_TOKEN,
        "reference_hits": sum(token == REFERENCE_TOKEN for token in values),
        "distinct_tokens": sorted(set(values)),
        "reference_token_match": (
            len(values) == REQUESTS
            and all(token == REFERENCE_TOKEN for token in values)
        ),
    }


def _load_plan(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise XoxBridgeError("bridge plan must be a JSON object")
    if value.get("schema") != bridge.BRIDGE_SCHEMA:
        raise XoxBridgeError("unsupported bridge plan schema")
    if value.get("status") != "READY_FOR_ISOLATED_RESTART_CANARY":
        raise XoxBridgeError("bridge plan is not ready for isolated canary")
    if value.get("candidate_worker_launch_allowed") is not True:
        raise XoxBridgeError("bridge plan does not permit candidate launch")
    if value.get("production_routing_switch_allowed") is not False:
        raise XoxBridgeError("bridge plan unexpectedly permits production routing")
    if value.get("production_write_allowed") is not False:
        raise XoxBridgeError("bridge plan unexpectedly permits production writes")
    if value.get("baseline_worker_mutation_allowed") is not False:
        raise XoxBridgeError("bridge plan unexpectedly permits baseline mutation")
    if value.get("target") != EXPECTED_TARGET:
        raise XoxBridgeError("bridge plan target is outside the reviewed scope")
    if value.get("candidate_policy") != [
        {"role": "shared_up_proj", "layer": 3, "n": 6}
    ]:
        raise XoxBridgeError("candidate policy is outside the reviewed scope")
    if int(value.get("canary_requests", 0)) != REQUESTS:
        raise XoxBridgeError("bridge plan request count mismatch")
    return value


def run(plan: dict) -> dict:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise XoxBridgeError("XOX bridge requires Darwin arm64")
    if _git_head() != EXPECTED_HEAD:
        raise XoxBridgeError("certified runtime source HEAD mismatch")
    if _sha256_file(BINARY) != EXPECTED_BINARY_SHA:
        raise XoxBridgeError("certified runtime binary SHA mismatch")

    checkpoint = json.loads(CHECKPOINT_EVIDENCE.read_text())
    if checkpoint.get("status") != "VERIFIED":
        raise XoxBridgeError("checkpoint identity evidence is not VERIFIED")
    if checkpoint.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise XoxBridgeError("checkpoint content identity mismatch")

    if plan.get("source_commit") != EXPECTED_HEAD:
        raise XoxBridgeError("plan source identity mismatch")
    if plan.get("binary_sha256") != EXPECTED_BINARY_SHA:
        raise XoxBridgeError("plan binary identity mismatch")
    if plan.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise XoxBridgeError("plan checkpoint identity mismatch")

    raw, max_new = _manifest_input()
    run_id = datetime.now(timezone.utc).strftime("bridge-%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = run_dir / "manifest.txt"
    manifest.write_text((f"{raw} {max_new}\n") * REQUESTS)
    ack_path = run_dir / "applied_ack.json"
    txn_path = run_dir / "txn.cmd"
    promo_path = run_dir / "promotion_nq.txt"
    promo_path.write_text("shared_up_proj 3 6\n")

    env = _minimal_env()
    env.update(
        {
            "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
            "QWEN_MOE_BASE": str(MOE_BASE),
            "QWEN_MOE_NEARTIE_CORRECT": "0",
            "QWEN_MOE_CB_PROMPT_MANIFEST": str(manifest),
            "QWEN_MOE_CB_SLOTS": str(SLOTS),
            "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
            "QWEN_MOE_GPU_APPLIED_ACK": str(ack_path),
            "QWEN_MOE_GPU_TXN_FILE": str(txn_path),
            "QWEN_MOE_PROMOTION_FILE_NQ": str(promo_path),
            "QWEN_MOE_PROMOTION_SAFETENSORS": str(SAFETENSORS),
        }
    )

    sys.path.insert(0, str(REPO / "tools"))
    import gpu_runtime_control as grc
    import precision_context as pc

    started_ns = time.monotonic_ns()
    proc = subprocess.Popen(
        [str(BINARY)],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    worker_pid = int(proc.pid)
    try:
        output, _ = proc.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate(timeout=10)
        raise XoxBridgeError(f"candidate worker timed out and was stopped: pid={worker_pid}")
    finished_ns = time.monotonic_ns()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    (run_dir / "worker.log").write_text(output)
    if not ack_path.is_file():
        raise XoxBridgeError("candidate worker ACK is missing")
    ack = grc.read_runtime_ack(ack_path)
    requests = _parse_requests(output)
    reference = _reference_stats(requests)
    validation = VALIDATION_RE.search(output)

    result = {
        "schema": bridge.RESULT_SCHEMA,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_commit": EXPECTED_HEAD,
        "binary_sha256": EXPECTED_BINARY_SHA,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "candidate_policy_hash": str(ack["active_policy_hash"]),
        "requested_policy_applied": (
            ack.get("status") == "PROMOTION_APPLIED"
            and ack.get("active_policy_hash") == plan["candidate_policy_hash"]
            and ack.get("active_policy") == pc.normalize_policy(plan["candidate_policy"])
        ),
        "ack_status": ack.get("status"),
        "ack_sha256": ack.get("ack_sha256"),
        "weight_epoch": int(ack.get("weight_epoch", -1)),
        "worker_instance_id": f"pid-{worker_pid}",
        "returncode": int(proc.returncode),
        "finite_logits": (
            validation is not None
            and validation.group("finite") == "1"
            and int(validation.group("requests")) == REQUESTS
        ),
        "reference_token_match": bool(reference["reference_token_match"]),
        "reference": reference,
        "requests": len(requests),
        "tokens": sum(len(tokens) for tokens in requests.values()),
        "duration_ms": max(1, (finished_ns - started_ns) // 1_000_000),
        "memory_bytes": int(usage.ru_maxrss),
        "production_routing_switched": False,
        "production_write_allowed": False,
        "baseline_worker_mutated": False,
        "candidate_worker_exited": True,
        "worker_log_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "manifest_sha256": _sha256_file(manifest),
        "promotion_file_sha256": _sha256_file(promo_path),
    }
    result["result_sha256"] = bridge.sha256_json(result)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    verdict = bridge.evaluate_isolated_candidate(plan, result)
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return {"result": result, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    args = ap.parse_args()
    plan = _load_plan(Path(args.plan).expanduser().resolve(strict=True))
    bundle = run(plan)
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
