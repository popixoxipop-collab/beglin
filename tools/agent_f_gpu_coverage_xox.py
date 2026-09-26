#!/usr/bin/env python3
"""Bounded real-hardware Agent F coverage for the certified XOX GPU worktree.

Default invocation launches one detached coverage worker. --status reports the
current/terminal state. --worker is private to the local launcher and should not
be exposed by Tailnet Commander.

Scope:
- kv_a_proj_with_mqa/L11 bounded n={5,6,7} bad-candidate search
- G5 durable rollback when a real bad kv_a candidate is found
- G6 positive/regression/budget/inconclusive for kv_a/L11 and shared_up/L3
- custom-Metal batch regression: kv_a/L11 n=7 with slots 1 vs 2 vs 4

All artifacts live under /Users/xox/vdsp_shadow_runs/agent_f_coverage.
No Supabase write, production control directory, or production promotion path is
used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


REPO = Path("/Users/xox/vdsp-engine-gpu-precision")
BINARY = REPO / "build-gpu-precision/qwen_infer_gpu"
MOE_BASE = Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
SAFETENSORS = Path(
    "/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/"
    "model.safetensors.index.json"
)
CHECKPOINT_EVIDENCE = Path(
    "/Users/xox/vdsp_shadow_runs/checkpoint_identity/checkpoint_identity.json"
)
EXPECTED_CHECKPOINT_SHA = (
    "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898"
)
EXPECTED_CHECKPOINT_RESULT_SHA = (
    "7c2553ae70203f554e5a5a9d50764a78c816ad7c6226dafc8370aceb61f3a0ad"
)
# The C/C++ GPU binary was certified before the subsequent Python-only
# shadow/checkpoint commits. Refuse to run if that binary changed.
EXPECTED_BINARY_SHA = (
    "6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392"
)
# Historical auxiliary identity used by the already-certified G4/G5/G6 drills.
# It is not upgraded here into a newly verified tokenizer/moe-base identity.
LEGACY_AUX_SHA = (
    "43f205631553478b204f761d5894aaeec67100ed276aca762a24c62e63a91c50"
)

KVA_SOURCE_MANIFEST = REPO / "g6_drill_kva11_manifest.txt"
SHARED_SOURCE_MANIFEST = REPO / "g5_drill_manifest.txt"

OUTPUT_ROOT = Path("/Users/xox/vdsp_shadow_runs/agent_f_coverage")
STATUS_FILE = OUTPUT_ROOT / "status.json"
LATEST_RESULT = OUTPUT_ROOT / "latest_coverage_matrix.json"
WORKER_LOG = OUTPUT_ROOT / "worker.log"

MAX_PROCESS_LAUNCHES = 10
SCAN_REQUESTS = 10
ROLLBACK_REQUESTS = 50
WORKER_TIMEOUT = 180

KVA = {
    "role": "kv_a_proj_with_mqa",
    "layer": 11,
    "orig_token": 8713,
    "corrected_token": 4794,
    "pos": 14,
    "prompt_len": 15,
    "good_n": 7,
}
SHARED_UP = {
    "role": "shared_up_proj",
    "layer": 3,
    "orig_token": 3268,
    "corrected_token": 1224,
    "pos": 16,
    "prompt_len": 9,
    "good_n": 6,
    "bad_n": 7,
}


class CoverageError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _minimal_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _git_head() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    head = p.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise CoverageError(f"invalid git HEAD: {head!r}")
    return head


def _read_checkpoint_evidence(
    path: Path = CHECKPOINT_EVIDENCE,
    *,
    expected_checkpoint_sha: str = EXPECTED_CHECKPOINT_SHA,
    expected_result_sha: str = EXPECTED_CHECKPOINT_RESULT_SHA,
) -> dict:
    if not path.is_file():
        raise CoverageError(f"checkpoint evidence is missing: {path}")
    actual_result_sha = _sha256_file(path)
    if actual_result_sha != expected_result_sha:
        raise CoverageError(
            "checkpoint evidence artifact SHA mismatch: "
            f"expected={expected_result_sha} actual={actual_result_sha}"
        )
    try:
        obj = json.loads(path.read_text())
    except Exception as exc:
        raise CoverageError(f"checkpoint evidence JSON is invalid: {exc}") from exc
    if obj.get("status") != "VERIFIED":
        raise CoverageError(
            f"checkpoint evidence status is not VERIFIED: {obj.get('status')!r}"
        )
    if obj.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise CoverageError(
            "checkpoint content identity mismatch: "
            f"expected={expected_checkpoint_sha} "
            f"actual={obj.get('checkpoint_sha256')}"
        )
    passes = obj.get("verification_passes")
    if not isinstance(passes, list) or len(passes) != 2:
        raise CoverageError("checkpoint evidence must contain exactly two passes")
    if any(p.get("checkpoint_sha256") != expected_checkpoint_sha for p in passes):
        raise CoverageError("checkpoint verification passes disagree")
    return obj


def _first_manifest_entry(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise CoverageError(f"source manifest is missing: {path}")
    rows = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise CoverageError(f"source manifest has no active rows: {path}")
    parts = shlex.split(rows[0])
    if len(parts) != 2:
        raise CoverageError(f"invalid manifest row: {rows[0]!r}")
    raw = Path(parts[0])
    try:
        max_new = int(parts[1])
    except ValueError as exc:
        raise CoverageError("manifest max_new_tokens is not an integer") from exc
    if max_new <= 0:
        raise CoverageError("manifest max_new_tokens must be positive")
    if not raw.is_file():
        raise CoverageError(f"replay token file is missing: {raw}")
    return str(raw), max_new


def _make_manifest(source: Path, dest: Path, count: int) -> dict:
    if count < 1 or count > 100:
        raise CoverageError("manifest repeat count must be in [1,100]")
    raw, max_new = _first_manifest_entry(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text((f"{raw} {max_new}\n") * count)
    return {
        "path": str(dest),
        "sha256": _sha256_file(dest),
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256_file(source),
        "raw_token_file": raw,
        "raw_token_sha256": _sha256_file(Path(raw)),
        "max_new_tokens": max_new,
        "requests": count,
    }


def classify_candidate(
    *,
    returncode: int,
    requested_policy_applied: bool,
    request_count: int,
    corrected_hits: int,
    min_requests: int = 10,
) -> str:
    if returncode != 0 or not requested_policy_applied:
        return "UNSUPPORTED_OR_RUNTIME_ERROR"
    if request_count < min_requests:
        return "INCONCLUSIVE"
    if corrected_hits == request_count:
        return "GOOD"
    return "BAD"


def choose_bad_candidate(rows: list[dict]) -> dict | None:
    for row in rows:
        if row.get("classification") == "BAD":
            return row
    return None


def _parse_requests(output: str) -> dict[int, list[int]]:
    reqs: dict[int, list[int]] = {}
    for m in re.finditer(
        r"\[moe gpu cb online\] req (\d+) .*? tokens:\s*([^\n\r]*)",
        output,
    ):
        reqs[int(m.group(1))] = [int(x) for x in m.group(2).split()]
    return reqs


def _token_counts(reqs: dict[int, list[int]], target: dict) -> dict:
    idx = int(target["pos"]) - (int(target["prompt_len"]) - 1)
    if idx < 0:
        raise CoverageError("target event position precedes prompt generation boundary")
    values = [
        toks[idx]
        for toks in reqs.values()
        if 0 <= idx < len(toks)
    ]
    return {
        "gen_idx": idx,
        "eligible_requests": len(values),
        "orig_hits": sum(v == int(target["orig_token"]) for v in values),
        "corrected_hits": sum(v == int(target["corrected_token"]) for v in values),
        "distinct_tokens": sorted(set(values)),
    }


class Progress:
    def __init__(self, run_id: str, run_root: Path):
        self.run_id = run_id
        self.run_root = run_root
        self.launches = 0
        self.completed: list[dict] = []

    def update(self, step: str, **extra) -> None:
        _atomic_json(
            STATUS_FILE,
            {
                "schema": "agent-f-gpu-coverage-status-v1",
                "status": "RUNNING",
                "production_write_allowed": False,
                "pid": os.getpid(),
                "run_id": self.run_id,
                "run_root": str(self.run_root),
                "step": step,
                "process_launches": self.launches,
                "max_process_launches": MAX_PROCESS_LAUNCHES,
                "completed": self.completed,
                "updated_at": _now(),
                **extra,
            },
        )

    def consume_launch(self, label: str) -> None:
        if self.launches >= MAX_PROCESS_LAUNCHES:
            raise CoverageError(
                f"process launch budget exceeded before {label}: "
                f"{self.launches}/{MAX_PROCESS_LAUNCHES}"
            )
        self.launches += 1
        self.update("launching", launch_label=label)


def _load_gpu_modules():
    sys.path.insert(0, str(REPO / "tools"))
    import precision_context as pc
    import precision_control_state as pcs
    import backend_adapters as ba
    import autopilot_observer_v3 as observer
    import gpu_observer_control as goc
    import gpu_runtime_control as grc
    import gpu_restart_canary as canary

    return pc, pcs, ba, observer, goc, grc, canary


def _policy_has_target(policy, target: dict, n: int) -> bool:
    for row in policy or []:
        try:
            if (
                row.get("role") == target["role"]
                and int(row.get("layer")) == int(target["layer"])
                and int(row.get("n")) == int(n)
            ):
                return True
        except Exception:
            continue
    return False


def _worker_env(
    *,
    manifest: Path,
    ack_path: Path,
    txn_path: Path,
    promo_path: Path,
    slots: int,
) -> dict[str, str]:
    env = _minimal_env()
    env.update(
        {
            "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
            "QWEN_MOE_BASE": str(MOE_BASE),
            "QWEN_MOE_NEARTIE_CORRECT": "0",
            "QWEN_MOE_CB_PROMPT_MANIFEST": str(manifest),
            "QWEN_MOE_CB_SLOTS": str(int(slots)),
            "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
            "QWEN_MOE_GPU_APPLIED_ACK": str(ack_path),
            "QWEN_MOE_GPU_TXN_FILE": str(txn_path),
            "QWEN_MOE_PROMOTION_FILE_NQ": str(promo_path),
            "QWEN_MOE_PROMOTION_SAFETENSORS": str(SAFETENSORS),
        }
    )
    return env


def _prepare_run_files(
    run_dir: Path,
    *,
    target: dict | None,
    n: int | None,
) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    ack = run_dir / "applied_ack.json"
    txn = run_dir / "txn.cmd"
    promo = run_dir / "promotion_nq.txt"
    for p in (ack, txn):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    if target is not None and n is not None:
        promo.write_text(
            f"{target['role']} {int(target['layer'])} {int(n)}\n"
        )
    else:
        promo.write_text("")
    return ack, txn, promo


def _launch_worker(
    progress: Progress,
    *,
    label: str,
    run_dir: Path,
    manifest: Path,
    target: dict | None = None,
    n: int | None = None,
    slots: int = 4,
):
    progress.consume_launch(label)
    ack, txn, promo = _prepare_run_files(run_dir, target=target, n=n)
    proc = subprocess.Popen(
        [str(BINARY)],
        cwd=str(REPO),
        env=_worker_env(
            manifest=manifest,
            ack_path=ack,
            txn_path=txn,
            promo_path=promo,
            slots=slots,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, ack, txn


def _communicate_owned(proc, timeout: int = WORKER_TIMEOUT) -> tuple[str, int]:
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate(timeout=10)
        raise CoverageError(
            f"owned qwen worker timed out and was stopped: pid={proc.pid}"
        )
    return output, int(proc.returncode)


def _wait_for_startup_ack(grc, ack_path: Path, timeout: int = 60) -> dict:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if ack_path.is_file() and ack_path.stat().st_size > 0:
            try:
                ack = grc.read_runtime_ack(str(ack_path))
                if ack.get("status") in ("PROMOTION_APPLIED", "STARTUP_STATE"):
                    return ack
            except Exception as exc:
                last_error = exc
        time.sleep(0.05)
    raise CoverageError(
        f"timed out waiting for startup ACK: {ack_path}; last_error={last_error}"
    )


def _finish_sync_run(
    *,
    grc,
    progress: Progress,
    label: str,
    proc,
    ack_path: Path,
    run_dir: Path,
    target: dict,
    n: int | None,
    slots: int,
) -> dict:
    output, rc = _communicate_owned(proc)
    (run_dir / "worker.log").write_text(output)
    reqs = _parse_requests(output)
    counts = _token_counts(reqs, target)
    ack = None
    ack_error = None
    try:
        ack = grc.read_runtime_ack(str(ack_path))
    except Exception as exc:
        ack_error = str(exc)
    applied = (
        n is None
        or (
            isinstance(ack, dict)
            and _policy_has_target(ack.get("active_policy"), target, int(n))
        )
    )
    classification = (
        "BASELINE"
        if n is None
        else classify_candidate(
            returncode=rc,
            requested_policy_applied=applied,
            request_count=len(reqs),
            corrected_hits=counts["corrected_hits"],
            min_requests=SCAN_REQUESTS,
        )
    )
    row = {
        "label": label,
        "pid": proc.pid,
        "returncode": rc,
        "target": {
            "role": target["role"],
            "layer": int(target["layer"]),
        },
        "n": n,
        "slots": int(slots),
        "requests_completed": len(reqs),
        "tokens_evaluated": sum(len(v) for v in reqs.values()),
        "counts": counts,
        "requested_policy_applied": bool(applied),
        "classification": classification,
        "ack_status": ack.get("status") if isinstance(ack, dict) else None,
        "active_policy_hash": (
            ack.get("active_policy_hash") if isinstance(ack, dict) else None
        ),
        "weight_epoch": (
            ack.get("weight_epoch") if isinstance(ack, dict) else None
        ),
        "ack_sha256": (
            ack.get("ack_sha256") if isinstance(ack, dict) else None
        ),
        "ack_error": ack_error,
        "run_dir": str(run_dir),
        "_requests": reqs,
        "_ack": ack,
    }
    progress.completed.append(
        {
            k: v
            for k, v in row.items()
            if not k.startswith("_")
        }
    )
    progress.update("run_complete", last_run=progress.completed[-1])
    return row


def _run_sync(
    *,
    grc,
    progress: Progress,
    label: str,
    run_root: Path,
    manifest: Path,
    target: dict,
    n: int | None,
    slots: int = 4,
) -> dict:
    run_dir = run_root / label
    proc, ack, _ = _launch_worker(
        progress,
        label=label,
        run_dir=run_dir,
        manifest=manifest,
        target=target if n is not None else None,
        n=n,
        slots=slots,
    )
    return _finish_sync_run(
        grc=grc,
        progress=progress,
        label=label,
        proc=proc,
        ack_path=ack,
        run_dir=run_dir,
        target=target,
        n=n,
        slots=slots,
    )


def _execution_context(pc, *, binary_sha: str, head: str):
    return pc.ExecutionContext(
        schema="precision-context-v1",
        model_id="deepseek-v2-lite",
        architecture="mla",
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA,
        tokenizer_sha256=LEGACY_AUX_SHA,
        base_artifact_sha256=LEGACY_AUX_SHA,
        backend="mlx_metal",
        device_fingerprint="xox-apple-silicon",
        binary_sha256=binary_sha,
        build_manifest_sha256=binary_sha,
        kernel_revision=f"gpu-precision-g1-g3@{head}",
        execution_mode="online_cbatch",
        runtime_config_sha256=pc.sha256_json(
            {
                "agent": "F",
                "schema": "agent-f-coverage-v1",
                "default_slots": 4,
                "scan_requests": SCAN_REQUESTS,
                "rollback_requests": ROLLBACK_REQUESTS,
            }
        ),
        quant_format="qng64",
        group_size=64,
        correction_mode="off",
    )


def _observation(observer, run: dict, context_hash: str, target: dict, *, baseline: bool):
    ack = run.get("_ack")
    if not isinstance(ack, dict):
        raise CoverageError(f"run lacks runtime ACK: {run['label']}")
    reqs = run["_requests"]
    misses = len(reqs) - int(run["counts"]["corrected_hits"])
    return observer.ObservationEvidence(
        context_hash=context_hash,
        backend="mlx_metal",
        policy_hash=ack["active_policy_hash"],
        weight_epoch=int(ack["weight_epoch"]),
        requests_completed=len(reqs),
        tokens_evaluated=sum(len(v) for v in reqs.values()),
        near_tie_events=len(reqs),
        reference_checks_attempted=len(reqs),
        effective_attribution_checks=len(reqs),
        attribution_hits=misses,
        run_errors=0 if int(run["returncode"]) == 0 else 1,
        median_margin=0.0,
        target_replay_pass=(
            None
            if baseline
            else (
                int(run["counts"]["corrected_hits"]) == len(reqs)
                and len(reqs) > 0
            )
        ),
        worker_instance_id=f"pid-{run['pid']}",
    )


def _canary_admission(pc, context_hash: str, binary_sha: str, post_run: dict, target: dict, n: int):
    baseline_policy = []
    candidate_policy = [
        {"role": target["role"], "layer": int(target["layer"]), "n": int(n)}
    ]
    return baseline_policy, candidate_policy, {
        "action": "ADMIT_ONE_TARGET_RESTART_CANARY",
        "backend": "mlx_metal",
        "evidence_mode": "isolated_restart",
        "context_hash": context_hash,
        "binary_sha256": binary_sha,
        "expected_epoch": int(post_run["_ack"]["weight_epoch"]),
        "baseline_policy_hash": pc.policy_hash(baseline_policy),
        "candidate_policy_hash": pc.policy_hash(candidate_policy),
        "candidate_policy": candidate_policy,
    }


def _evaluate_g6_matrix(
    *,
    pc,
    observer,
    canary,
    context_hash: str,
    binary_sha: str,
    target: dict,
    baseline_run: dict,
    good_run: dict,
    bad_run: dict | None,
) -> dict:
    baseline_ev = _observation(
        observer, baseline_run, context_hash, target, baseline=True
    )
    good_ev = _observation(
        observer, good_run, context_hash, target, baseline=False
    )
    baseline_policy, _, admission_good = _canary_admission(
        pc,
        context_hash,
        binary_sha,
        good_run,
        target,
        int(good_run["n"]),
    )
    positive_plan = canary.build_restart_canary(
        admission_good,
        baseline_policy=baseline_policy,
        min_requests=SCAN_REQUESTS,
        max_requests=100,
        min_effective_checks=1,
    )
    positive = canary.evaluate_restart_canary(
        positive_plan, baseline_ev, good_ev
    )

    source_for_negative = bad_run if bad_run is not None else good_run
    negative_ev = _observation(
        observer, source_for_negative, context_hash, target, baseline=False
    )
    baseline_policy, _, admission_neg = _canary_admission(
        pc,
        context_hash,
        binary_sha,
        source_for_negative,
        target,
        int(source_for_negative["n"]),
    )
    regression_plan = canary.build_restart_canary(
        admission_neg,
        baseline_policy=baseline_policy,
        min_requests=SCAN_REQUESTS,
        max_requests=100,
        min_effective_checks=1,
    )
    regression = canary.evaluate_restart_canary(
        regression_plan, baseline_ev, negative_ev
    )

    actual_requests = int(source_for_negative["requests_completed"])
    budget_plan = canary.build_restart_canary(
        admission_neg,
        baseline_policy=baseline_policy,
        min_requests=1,
        max_requests=max(1, actual_requests - 1),
        min_effective_checks=1,
    )
    budget = canary.evaluate_restart_canary(
        budget_plan, baseline_ev, negative_ev
    )

    inconclusive_plan = canary.build_restart_canary(
        admission_neg,
        baseline_policy=baseline_policy,
        min_requests=actual_requests + 1,
        max_requests=1000,
        min_effective_checks=1,
    )
    inconclusive = canary.evaluate_restart_canary(
        inconclusive_plan, baseline_ev, negative_ev
    )

    return {
        "positive": {
            "result": positive,
            "reused_hardware_run": False,
            "source_run": good_run["label"],
        },
        "regression": {
            "result": regression,
            "reused_hardware_run": False,
            "source_run": source_for_negative["label"],
            "real_bad_candidate": bad_run is not None,
        },
        "budget_exceeded": {
            "result": budget,
            "reused_hardware_run": True,
            "source_run": source_for_negative["label"],
            "note": "same hardware evidence; controller max_requests changed",
        },
        "inconclusive": {
            "result": inconclusive,
            "reused_hardware_run": True,
            "source_run": source_for_negative["label"],
            "note": "same hardware evidence; controller min_requests changed",
        },
    }


def _run_g5_rollback(
    *,
    pc,
    pcs,
    ba,
    observer,
    goc,
    grc,
    context,
    progress: Progress,
    run_root: Path,
    manifest_50: Path,
    baseline_run: dict,
    bad_run: dict,
) -> dict:
    target = KVA
    bad_n = int(bad_run["n"])
    context_hash = context.context_hash

    baseline_ack = baseline_run["_ack"]
    if not isinstance(baseline_ack, dict):
        raise CoverageError("kv_a baseline lacks ACK")
    baseline_evidence = goc.evidence_from_runtime_ack(
        context_hash=context_hash,
        runtime_ack=baseline_ack,
        metrics={
            "requests_completed": baseline_run["requests_completed"],
            "tokens_evaluated": baseline_run["tokens_evaluated"],
            "near_tie_events": baseline_run["requests_completed"],
            "reference_checks_attempted": baseline_run["requests_completed"],
            "effective_attribution_checks": baseline_run["requests_completed"],
            "attribution_hits": 0,
            "run_errors": 0,
            "median_margin": 0.0,
        },
        target_replay_pass=None,
    )
    post_evidence = goc.evidence_from_runtime_ack(
        context_hash=context_hash,
        runtime_ack=bad_run["_ack"],
        metrics={
            "requests_completed": bad_run["requests_completed"],
            "tokens_evaluated": bad_run["tokens_evaluated"],
            "near_tie_events": bad_run["requests_completed"],
            "reference_checks_attempted": bad_run["requests_completed"],
            "effective_attribution_checks": bad_run["requests_completed"],
            "attribution_hits": bad_run["counts"]["corrected_hits"],
            "run_errors": 0,
            "median_margin": 0.0,
        },
        target_replay_pass=False,
    )
    verdict = observer.evaluate(
        baseline_evidence,
        post_evidence,
        min_requests=SCAN_REQUESTS,
        min_effective_checks=1,
    )
    if verdict.get("status") != "REGRESSION_DETECTED":
        raise CoverageError(
            f"real bad candidate did not produce REGRESSION_DETECTED: {verdict}"
        )

    live_dir = run_root / "g5_kva_live_rollback"
    proc, ack_path, txn_path = _launch_worker(
        progress,
        label="g5_kva_live_rollback",
        run_dir=live_dir,
        manifest=manifest_50,
        target=target,
        n=bad_n,
        slots=4,
    )
    live_ack = _wait_for_startup_ack(grc, ack_path)
    if not _policy_has_target(live_ack.get("active_policy"), target, bad_n):
        proc.terminate()
        _communicate_owned(proc, timeout=20)
        raise CoverageError("live rollback worker did not apply bad kv_a candidate")

    store = pcs.ControlStore(
        root=str(run_root / "g5_control"),
        model_revision_id="deepseek-v2-lite",
        backend="mlx_metal",
    )
    failed_policy = [
        {"role": target["role"], "layer": target["layer"], "n": bad_n}
    ]
    store.set_desired(
        context_hash=context_hash,
        policy=failed_policy,
        reason="Agent F bad kv_a candidate under isolated rollback test",
        source="agent-f",
    )
    store.set_applied(
        context_hash=context_hash,
        policy=live_ack["active_policy"],
        epoch=live_ack["weight_epoch"],
        txn_id="agent-f-initial-bad",
        ack_sha256=live_ack["ack_sha256"],
    )
    adapter = ba.MlxMetalBackendAdapter(
        verified=True,
        context=context,
        ack_path=str(ack_path),
        txn_path=str(txn_path),
    )
    actual_state = adapter.query_applied_state()

    live_post_evidence = goc.evidence_from_runtime_ack(
        context_hash=context_hash,
        runtime_ack=live_ack,
        metrics={
            "requests_completed": bad_run["requests_completed"],
            "tokens_evaluated": bad_run["tokens_evaluated"],
            "near_tie_events": bad_run["requests_completed"],
            "reference_checks_attempted": bad_run["requests_completed"],
            "effective_attribution_checks": bad_run["requests_completed"],
            "attribution_hits": bad_run["counts"]["corrected_hits"],
            "run_errors": 0,
            "median_margin": 0.0,
        },
        target_replay_pass=False,
    )
    txn_id = f"{progress.run_id}-rollback"
    requested = goc.request_regression_rollback(
        adapter=adapter,
        store=store,
        baseline=baseline_evidence,
        post=live_post_evidence,
        baseline_policy=[],
        failed_policy=failed_policy,
        role=target["role"],
        layer=target["layer"],
        n=bad_n,
        txn_id=txn_id,
        min_requests=SCAN_REQUESTS,
        min_effective_checks=1,
    )
    if requested.get("action") != "ROLLBACK_REQUESTED":
        proc.terminate()
        _communicate_owned(proc, timeout=20)
        raise CoverageError(
            f"rollback request was not accepted: {requested}"
        )

    output, rc = _communicate_owned(proc)
    (live_dir / "worker.log").write_text(output)
    if rc != 0:
        raise CoverageError(f"live rollback worker exited nonzero: {rc}")

    complete = goc.complete_regression_rollback(
        adapter=adapter,
        store=store,
        txn_id=txn_id,
        context_hash=context_hash,
        baseline_policy=[],
        failed_epoch=live_post_evidence.weight_epoch,
    )
    store2 = pcs.ControlStore(
        root=str(run_root / "g5_control"),
        model_revision_id="deepseek-v2-lite",
        backend="mlx_metal",
    )
    quarantined = store2.is_quarantined(
        context_hash=context_hash,
        role=target["role"],
        layer=target["layer"],
        n=bad_n,
    )
    desired = store2.desired()
    applied = store2.applied()
    reconcile = store2.reconcile()
    ok = (
        complete.get("status") == "ROLLBACK_APPLIED"
        and complete.get("rollback_complete") is True
        and quarantined
        and desired.get("policy") == []
        and applied.get("policy") == []
        and reconcile.get("status") == "IN_SYNC"
    )
    if not ok:
        raise CoverageError(
            "G5 rollback terminal invariants failed: "
            + json.dumps(
                {
                    "complete": complete,
                    "quarantined": quarantined,
                    "desired": desired,
                    "applied": applied,
                    "reconcile": reconcile,
                },
                default=str,
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "bad_n": bad_n,
        "observer_verdict": verdict,
        "runtime_before_rollback": actual_state,
        "rollback_request": requested,
        "rollback_complete": complete,
        "quarantined": quarantined,
        "desired_policy_after_restart": desired.get("policy"),
        "applied_policy_after_restart": applied.get("policy"),
        "reconcile_after_restart": reconcile.get("status"),
        "run_dir": str(live_dir),
    }


def _strip_private(run: dict) -> dict:
    return {k: v for k, v in run.items() if not k.startswith("_")}


def _worker_once(run_id: str) -> dict:
    pc, pcs, ba, observer, goc, grc, canary = _load_gpu_modules()

    if not REPO.is_dir() or not BINARY.is_file() or not MOE_BASE.is_dir():
        raise CoverageError("certified XOX engine/binary/moe_base is missing")
    if not SAFETENSORS.is_file():
        raise CoverageError("certified safetensors index is missing")

    checkpoint_evidence = _read_checkpoint_evidence()
    binary_sha = _sha256_file(BINARY)
    if binary_sha != EXPECTED_BINARY_SHA:
        raise CoverageError(
            f"GPU binary SHA changed: expected={EXPECTED_BINARY_SHA} actual={binary_sha}"
        )
    head = _git_head()
    context = _execution_context(pc, binary_sha=binary_sha, head=head)

    run_root = OUTPUT_ROOT / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    progress = Progress(run_id, run_root)
    progress.update(
        "preflight",
        source_head=head,
        binary_sha256=binary_sha,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA,
    )

    kva10_meta = _make_manifest(
        KVA_SOURCE_MANIFEST, run_root / "manifests/kva10.txt", SCAN_REQUESTS
    )
    kva50_meta = _make_manifest(
        KVA_SOURCE_MANIFEST, run_root / "manifests/kva50.txt", ROLLBACK_REQUESTS
    )
    sup10_meta = _make_manifest(
        SHARED_SOURCE_MANIFEST,
        run_root / "manifests/shared_up10.txt",
        SCAN_REQUESTS,
    )
    kva10 = Path(kva10_meta["path"])
    kva50 = Path(kva50_meta["path"])
    sup10 = Path(sup10_meta["path"])

    # 1-4: kv_a baseline + bounded n scan.
    kva_base = _run_sync(
        grc=grc,
        progress=progress,
        label="kva_baseline",
        run_root=run_root,
        manifest=kva10,
        target=KVA,
        n=None,
        slots=4,
    )
    if (
        kva_base["returncode"] != 0
        or kva_base["requests_completed"] < SCAN_REQUESTS
        or kva_base["counts"]["orig_hits"] != kva_base["requests_completed"]
    ):
        raise CoverageError(
            "kv_a baseline no longer reproduces orig token on all requests"
        )

    kva_candidates = []
    for n in (5, 6, 7):
        row = _run_sync(
            grc=grc,
            progress=progress,
            label=f"kva_n{n}_slots4",
            run_root=run_root,
            manifest=kva10,
            target=KVA,
            n=n,
            slots=4,
        )
        kva_candidates.append(row)

    kva_good = next((r for r in kva_candidates if int(r["n"]) == 7), None)
    if kva_good is None or kva_good["classification"] != "GOOD":
        raise CoverageError(
            f"historically-good kv_a/L11 n=7 regressed: "
            f"{_strip_private(kva_good or {})}"
        )
    kva_bad = choose_bad_candidate(
        [r for r in kva_candidates if int(r["n"]) in (5, 6)]
    )

    # 5 (if real bad found): G5 durable rollback.
    if kva_bad is not None:
        g5 = _run_g5_rollback(
            pc=pc,
            pcs=pcs,
            ba=ba,
            observer=observer,
            goc=goc,
            grc=grc,
            context=context,
            progress=progress,
            run_root=run_root,
            manifest_50=kva50,
            baseline_run=kva_base,
            bad_run=kva_bad,
        )
    else:
        g5 = {
            "status": "NO_BAD_CANDIDATE_WITHIN_BUDGET",
            "tested_n": [5, 6],
        }

    kva_g6 = _evaluate_g6_matrix(
        pc=pc,
        observer=observer,
        canary=canary,
        context_hash=context.context_hash,
        binary_sha=binary_sha,
        target=KVA,
        baseline_run=kva_base,
        good_run=kva_good,
        bad_run=kva_bad,
    )

    # 6-8 (or 5-7 if no G5): shared_up positive + known bad.
    sup_base = _run_sync(
        grc=grc,
        progress=progress,
        label="shared_up_baseline",
        run_root=run_root,
        manifest=sup10,
        target=SHARED_UP,
        n=None,
        slots=4,
    )
    if (
        sup_base["returncode"] != 0
        or sup_base["requests_completed"] < SCAN_REQUESTS
        or sup_base["counts"]["orig_hits"] != sup_base["requests_completed"]
    ):
        raise CoverageError(
            "shared_up baseline no longer reproduces orig token on all requests"
        )
    sup_good = _run_sync(
        grc=grc,
        progress=progress,
        label="shared_up_n6_slots4",
        run_root=run_root,
        manifest=sup10,
        target=SHARED_UP,
        n=6,
        slots=4,
    )
    sup_bad = _run_sync(
        grc=grc,
        progress=progress,
        label="shared_up_n7_slots4",
        run_root=run_root,
        manifest=sup10,
        target=SHARED_UP,
        n=7,
        slots=4,
    )
    if sup_good["classification"] != "GOOD":
        raise CoverageError(
            f"historically-good shared_up/L3 n=6 regressed: {_strip_private(sup_good)}"
        )
    if sup_bad["classification"] != "BAD":
        raise CoverageError(
            f"historically-bad shared_up/L3 n=7 is not BAD: {_strip_private(sup_bad)}"
        )

    sup_g6 = _evaluate_g6_matrix(
        pc=pc,
        observer=observer,
        canary=canary,
        context_hash=context.context_hash,
        binary_sha=binary_sha,
        target=SHARED_UP,
        baseline_run=sup_base,
        good_run=sup_good,
        bad_run=sup_bad,
    )

    # 9-10: custom-Metal scheduling regression with the known-good kv_a n=7.
    kva_slots1 = _run_sync(
        grc=grc,
        progress=progress,
        label="kva_n7_slots1",
        run_root=run_root,
        manifest=kva10,
        target=KVA,
        n=7,
        slots=1,
    )
    kva_slots2 = _run_sync(
        grc=grc,
        progress=progress,
        label="kva_n7_slots2",
        run_root=run_root,
        manifest=kva10,
        target=KVA,
        n=7,
        slots=2,
    )
    if kva_slots1["classification"] != "GOOD" or kva_slots2["classification"] != "GOOD":
        raise CoverageError("kv_a n=7 slots1/2 did not stay GOOD")
    batch_equal_1_2 = kva_slots1["_requests"] == kva_slots2["_requests"]
    batch_equal_2_4 = kva_slots2["_requests"] == kva_good["_requests"]
    if not (batch_equal_1_2 and batch_equal_2_4):
        raise CoverageError(
            "kv_a n=7 request-token output differs across slots 1/2/4"
        )

    def g6_expect(matrix: dict, *, require_regression: bool):
        pos = matrix["positive"]["result"]
        reg = matrix["regression"]["result"]
        budget = matrix["budget_exceeded"]["result"]
        inc = matrix["inconclusive"]["result"]
        if not (
            pos.get("status") == "CANARY_PASS"
            and pos.get("decision") == "CANARY_PASS_NO_AUTO_EXPANSION"
            and pos.get("rollback_required") is False
            and pos.get("auto_expand") is False
        ):
            raise CoverageError(f"G6 positive invariant failed: {pos}")
        if require_regression and not (
            reg.get("status") == "REGRESSION_DETECTED"
            and reg.get("decision") == "ROLLBACK_REQUIRED"
            and reg.get("rollback_required") is True
            and reg.get("auto_expand") is False
        ):
            raise CoverageError(f"G6 regression invariant failed: {reg}")
        if not (
            budget.get("status") == "CANARY_BUDGET_EXCEEDED"
            and budget.get("decision") == "ROLLBACK_REQUIRED"
            and budget.get("rollback_required") is True
            and budget.get("auto_expand") is False
        ):
            raise CoverageError(f"G6 budget invariant failed: {budget}")
        if not (
            inc.get("decision") == "ROLLBACK_REQUIRED_INCONCLUSIVE"
            and inc.get("rollback_required") is True
            and inc.get("auto_expand") is False
        ):
            raise CoverageError(f"G6 inconclusive invariant failed: {inc}")

    g6_expect(kva_g6, require_regression=kva_bad is not None)
    g6_expect(sup_g6, require_regression=True)

    if progress.launches > MAX_PROCESS_LAUNCHES:
        raise CoverageError("internal process launch budget accounting failed")

    result = {
        "schema": "agent-f-gpu-coverage-v1",
        "status": (
            "VERIFIED"
            if kva_bad is not None and g5.get("status") == "PASS"
            else "PARTIAL_NO_KVA_BAD_CANDIDATE"
        ),
        "production_write_allowed": False,
        "run_id": run_id,
        "run_root": str(run_root),
        "source_head": head,
        "binary_sha256": binary_sha,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "checkpoint_evidence_sha256": EXPECTED_CHECKPOINT_RESULT_SHA,
        "checkpoint_evidence_status": checkpoint_evidence["status"],
        "legacy_aux_identity_sha256": LEGACY_AUX_SHA,
        "legacy_aux_identity_verified_by_agent_f": False,
        "process_launches": progress.launches,
        "max_process_launches": MAX_PROCESS_LAUNCHES,
        "manifests": {
            "kva10": kva10_meta,
            "kva50": kva50_meta,
            "shared_up10": sup10_meta,
        },
        "kva_candidate_scan": [
            _strip_private(row) for row in kva_candidates
        ],
        "kva_bad_candidate": (
            _strip_private(kva_bad) if kva_bad is not None else None
        ),
        "g5_kva_rollback": g5,
        "g6_kva": kva_g6,
        "g6_shared_up": sup_g6,
        "batch_regression": {
            "target": "kv_a_proj_with_mqa/L11 n=7",
            "slots": [1, 2, 4],
            "slots1_classification": kva_slots1["classification"],
            "slots2_classification": kva_slots2["classification"],
            "slots4_classification": kva_good["classification"],
            "slots1_equals_slots2": batch_equal_1_2,
            "slots2_equals_slots4": batch_equal_2_4,
            "requests": len(kva_slots1["_requests"]),
            "finite_logits_proxy": "worker completed successfully; token output parsed",
            "note": "same fixed replay manifest; exact request-token arrays compared",
        },
        "runs": [
            _strip_private(kva_base),
            *[_strip_private(x) for x in kva_candidates],
            _strip_private(sup_base),
            _strip_private(sup_good),
            _strip_private(sup_bad),
            _strip_private(kva_slots1),
            _strip_private(kva_slots2),
        ],
        "limitations": [
            "Agent F reuses the historical auxiliary context SHA for non-checkpoint components; only checkpoint and binary content identities are revalidated here.",
            "Budget/inconclusive G6 rows intentionally reuse the same real hardware evidence with different controller limits and are labeled reused_hardware_run.",
            "No production promotion, Supabase write, or production candidate-history mutation is performed.",
        ],
        "finished_at": _now(),
    }
    _atomic_json(run_root / "coverage_matrix.json", result)
    _atomic_json(LATEST_RESULT, result)
    return result


def _load_status() -> dict | None:
    if not STATUS_FILE.is_file():
        return None
    try:
        value = json.loads(STATUS_FILE.read_text())
    except Exception as exc:
        raise CoverageError(f"coverage status unreadable: {exc}") from exc
    return value


def launch() -> dict:
    old = _load_status()
    if old and old.get("status") == "RUNNING" and _pid_alive(old.get("pid")):
        raise CoverageError(
            f"Agent F coverage already running with pid={old.get('pid')}"
        )
    run_id = datetime.now(timezone.utc).strftime("f-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log = open(WORKER_LOG, "ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--run-id", run_id],
        cwd=str(REPO),
        env=_minimal_env(),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    return {
        "schema": "agent-f-gpu-coverage-status-v1",
        "status": "LAUNCHED",
        "production_write_allowed": False,
        "worker_pid": proc.pid,
        "run_id": run_id,
        "status_file": str(STATUS_FILE),
        "latest_result": str(LATEST_RESULT),
    }


def worker(run_id: str) -> int:
    try:
        result = _worker_once(run_id)
        final = {
            "schema": "agent-f-gpu-coverage-status-v1",
            "status": "COMPLETE",
            "coverage_status": result["status"],
            "production_write_allowed": False,
            "pid": os.getpid(),
            "run_id": run_id,
            "process_launches": result["process_launches"],
            "source_head": result["source_head"],
            "binary_sha256": result["binary_sha256"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "kva_bad_n": (
                result["kva_bad_candidate"].get("n")
                if result.get("kva_bad_candidate")
                else None
            ),
            "g5_status": result["g5_kva_rollback"].get("status"),
            "finished_at": result["finished_at"],
            "latest_result": str(LATEST_RESULT),
        }
        _atomic_json(STATUS_FILE, final)
        print(json.dumps(final, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "agent-f-gpu-coverage-status-v1",
            "status": "FAILED",
            "production_write_allowed": False,
            "pid": os.getpid(),
            "run_id": run_id,
            "error": str(exc),
            "finished_at": _now(),
        }
        _atomic_json(STATUS_FILE, failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


def _public_case(case: dict | None) -> dict | None:
    if not isinstance(case, dict):
        return None
    result = case.get("result")
    if not isinstance(result, dict):
        result = {}
    return {
        "status": result.get("status"),
        "decision": result.get("decision"),
        "rollback_required": result.get("rollback_required"),
        "auto_expand": result.get("auto_expand"),
        "reused_hardware_run": case.get("reused_hardware_run"),
        "source_run": case.get("source_run"),
        "real_bad_candidate": case.get("real_bad_candidate"),
    }


def _latest_public_summary() -> dict | None:
    if not LATEST_RESULT.is_file():
        return None
    try:
        obj = json.loads(LATEST_RESULT.read_text())
    except Exception as exc:
        return {
            "status": "MATRIX_UNREADABLE",
            "error": str(exc),
        }
    if not isinstance(obj, dict):
        return {"status": "MATRIX_UNREADABLE"}
    kva_scan = []
    for row in obj.get("kva_candidate_scan") or []:
        if not isinstance(row, dict):
            continue
        kva_scan.append({
            "n": row.get("n"),
            "classification": row.get("classification"),
            "requests_completed": row.get("requests_completed"),
            "counts": row.get("counts"),
            "requested_policy_applied": row.get("requested_policy_applied"),
        })
    g6_kva = obj.get("g6_kva") if isinstance(obj.get("g6_kva"), dict) else {}
    g6_shared = (
        obj.get("g6_shared_up")
        if isinstance(obj.get("g6_shared_up"), dict)
        else {}
    )
    return {
        "status": obj.get("status"),
        "run_id": obj.get("run_id"),
        "source_head": obj.get("source_head"),
        "binary_sha256": obj.get("binary_sha256"),
        "checkpoint_sha256": obj.get("checkpoint_sha256"),
        "process_launches": obj.get("process_launches"),
        "matrix_sha256": _sha256_file(LATEST_RESULT),
        "kva_candidate_scan": kva_scan,
        "g5_kva_status": (
            obj.get("g5_kva_rollback", {}).get("status")
            if isinstance(obj.get("g5_kva_rollback"), dict)
            else None
        ),
        "g6_kva": {
            name: _public_case(g6_kva.get(name))
            for name in (
                "positive",
                "regression",
                "budget_exceeded",
                "inconclusive",
            )
        },
        "g6_shared_up": {
            name: _public_case(g6_shared.get(name))
            for name in (
                "positive",
                "regression",
                "budget_exceeded",
                "inconclusive",
            )
        },
        "batch_regression": obj.get("batch_regression"),
        "production_write_allowed": obj.get("production_write_allowed"),
    }


def status() -> dict:
    value = _load_status()
    if value is None:
        return {
            "schema": "agent-f-gpu-coverage-status-v1",
            "status": "NOT_STARTED",
            "production_write_allowed": False,
        }
    result = dict(value)
    if result.get("status") == "RUNNING":
        result["pid_alive"] = _pid_alive(result.get("pid"))
    if result.get("status") in {"COMPLETE", "FAILED"}:
        result["coverage_summary"] = _latest_public_summary()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-id", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.status and args.worker:
        print(json.dumps({"status": "ERROR", "error": "conflicting mode flags"}))
        return 2
    try:
        if args.worker:
            if not args.run_id:
                raise CoverageError("--worker requires run id")
            return worker(args.run_id)
        result = status() if args.status else launch()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": "agent-f-gpu-coverage-status-v1",
                    "status": "ERROR",
                    "production_write_allowed": False,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
