#!/usr/bin/env python3
"""Fixed XOX manual-canary baseline preimage capture.

This helper is deliberately narrow:
- no user-supplied paths or target arguments
- certified DeepSeek-V2-Lite / MLX-Metal tuple only
- baseline policy only (no promotion rows)
- writes scratch evidence under XOX Tailnet sandbox
- never touches production control roots or Supabase
- reports a measured resource envelope for later human approval

Default invocation performs one bounded baseline capture. --status only reads
the last sanitized result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_runtime_control as grc


ENGINE_ROOT = Path("/Users/xox/vdsp-engine-gpu-precision")
BINARY = ENGINE_ROOT / "build-gpu-precision/qwen_infer_gpu"
SOURCE_MANIFEST = ENGINE_ROOT / "g5_drill_manifest.txt"
MOE_BASE = Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
SAFETENSORS = Path(
    "/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/"
    "model.safetensors.index.json"
)
CHECKPOINT_EVIDENCE = Path(
    "/Users/xox/vdsp_shadow_runs/checkpoint_identity/checkpoint_identity.json"
)
OUTPUT_ROOT = Path(
    "/Users/xox/mcp-sandbox/tailnet-commander/beglin_manual_canary_preimage"
)
STATUS_FILE = OUTPUT_ROOT / "status.json"

EXPECTED_ENGINE_HEAD = "330954b27f146b8a17db2cb353c3e620968bad5e"
EXPECTED_BINARY_SHA256 = (
    "6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898"
)
EXPECTED_CHECKPOINT_EVIDENCE_SHA256 = (
    "7c2553ae70203f554e5a5a9d50764a78c816ad7c6226dafc8370aceb61f3a0ad"
)

REQUESTS = 12
SLOTS = 4
TIMEOUT_SECONDS = 180
TARGET = {
    "role": "shared_up_proj",
    "layer": 3,
    "orig_token": 3268,
    "reference_token": 1224,
    "pos": 16,
    "prompt_len": 9,
    "after_n": 6,
}


class PreimageCaptureError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _minimal_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            out[key] = value
    return out


def _git_head() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ENGINE_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    head = p.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise PreimageCaptureError(f"invalid engine HEAD: {head!r}")
    return head


def _first_manifest_entry(path: Path) -> tuple[Path, int]:
    rows = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise PreimageCaptureError("source manifest has no active rows")
    parts = shlex.split(rows[0])
    if len(parts) != 2:
        raise PreimageCaptureError("invalid source manifest first row")
    raw = Path(parts[0])
    try:
        max_new = int(parts[1])
    except ValueError as exc:
        raise PreimageCaptureError("source manifest max_new_tokens is invalid") from exc
    if max_new <= 0 or not raw.is_file():
        raise PreimageCaptureError("source manifest input is unavailable")
    return raw, max_new


def _materialize_manifest(dest: Path) -> dict:
    raw, max_new = _first_manifest_entry(SOURCE_MANIFEST)
    dest.write_text((f"{raw} {max_new}\n") * REQUESTS)
    return {
        "path": str(dest),
        "sha256": _sha256_file(dest),
        "raw_token_file": str(raw),
        "raw_token_sha256": _sha256_file(raw),
        "max_new_tokens": max_new,
        "requests": REQUESTS,
    }


def _parse_requests(output: str) -> dict[int, list[int]]:
    reqs: dict[int, list[int]] = {}
    for m in re.finditer(
        r"\[moe gpu cb online\] req (\d+) .*? tokens:\s*([^\n\r]*)",
        output,
    ):
        reqs[int(m.group(1))] = [int(x) for x in m.group(2).split()]
    return reqs


def _target_counts(reqs: dict[int, list[int]], target: dict = TARGET) -> dict:
    idx = int(target["pos"]) - (int(target["prompt_len"]) - 1)
    if idx < 0:
        raise PreimageCaptureError("target precedes generation boundary")
    values = [tokens[idx] for tokens in reqs.values() if idx < len(tokens)]
    return {
        "gen_idx": idx,
        "eligible_requests": len(values),
        "orig_hits": sum(v == int(target["orig_token"]) for v in values),
        "reference_hits": sum(v == int(target["reference_token"]) for v in values),
        "distinct_tokens": sorted(set(values)),
    }


def _rss_to_bytes(raw: int, platform: str | None = None) -> int:
    platform = sys.platform if platform is None else platform
    raw = int(raw)
    if raw <= 0:
        return 0
    # Darwin reports ru_maxrss in bytes; Linux reports KiB.
    return raw if platform == "darwin" else raw * 1024


def _align_up(value: int, quantum: int) -> int:
    return ((int(value) + int(quantum) - 1) // int(quantum)) * int(quantum)


def recommend_budget(*, requests: int, tokens: int, duration_ms: int, peak_rss_bytes: int) -> dict:
    values = {
        "requests": int(requests),
        "tokens": int(tokens),
        "duration_ms": int(duration_ms),
        "peak_rss_bytes": int(peak_rss_bytes),
    }
    if any(v <= 0 for v in values.values()):
        raise PreimageCaptureError(f"invalid measured budget input: {values}")
    mib = 1024 * 1024
    return {
        "max_requests": max(values["requests"] + 4, math.ceil(values["requests"] * 1.5)),
        "max_tokens": max(values["tokens"] + 32, math.ceil(values["tokens"] * 1.5)),
        "max_duration_ms": _align_up(max(10_000, values["duration_ms"] * 2), 1000),
        "max_memory_bytes": _align_up(
            max(values["peak_rss_bytes"] + 256 * mib, math.ceil(values["peak_rss_bytes"] * 1.5)),
            256 * mib,
        ),
        "basis": "measured_baseline_with_headroom",
        "requires_human_review": True,
    }


def _verify_checkpoint_evidence() -> dict:
    if _sha256_file(CHECKPOINT_EVIDENCE) != EXPECTED_CHECKPOINT_EVIDENCE_SHA256:
        raise PreimageCaptureError("checkpoint evidence artifact SHA mismatch")
    value = json.loads(CHECKPOINT_EVIDENCE.read_text())
    if value.get("status") != "VERIFIED":
        raise PreimageCaptureError("checkpoint evidence is not VERIFIED")
    if value.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise PreimageCaptureError("checkpoint SHA mismatch")
    return value


def capture() -> dict:
    head = _git_head()
    if head != EXPECTED_ENGINE_HEAD:
        raise PreimageCaptureError(
            f"engine HEAD mismatch: expected={EXPECTED_ENGINE_HEAD} actual={head}"
        )
    if _sha256_file(BINARY) != EXPECTED_BINARY_SHA256:
        raise PreimageCaptureError("GPU binary SHA mismatch")
    _verify_checkpoint_evidence()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_meta = _materialize_manifest(OUTPUT_ROOT / "baseline.manifest")
    ack_path = OUTPUT_ROOT / "baseline.applied_ack.json"
    txn_path = OUTPUT_ROOT / "baseline.txn.cmd"
    promo_path = OUTPUT_ROOT / "baseline.promotion_nq.txt"
    worker_log = OUTPUT_ROOT / "baseline.worker.log"
    for path in (ack_path, txn_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    promo_path.write_text("")

    env = _minimal_env()
    env.update(
        {
            "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
            "QWEN_MOE_BASE": str(MOE_BASE),
            "QWEN_MOE_NEARTIE_CORRECT": "0",
            "QWEN_MOE_CB_PROMPT_MANIFEST": manifest_meta["path"],
            "QWEN_MOE_CB_SLOTS": str(SLOTS),
            "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
            "QWEN_MOE_GPU_APPLIED_ACK": str(ack_path),
            "QWEN_MOE_GPU_TXN_FILE": str(txn_path),
            "QWEN_MOE_PROMOTION_FILE_NQ": str(promo_path),
            "QWEN_MOE_PROMOTION_SAFETENSORS": str(SAFETENSORS),
        }
    )

    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.monotonic()
    proc = subprocess.Popen(
        [str(BINARY)],
        cwd=ENGINE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = proc.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate(timeout=10)
        worker_log.write_text(output)
        raise PreimageCaptureError(f"baseline worker timed out: pid={proc.pid}")
    duration_ms = max(1, math.ceil((time.monotonic() - started) * 1000))
    worker_log.write_text(output)
    if proc.returncode != 0:
        raise PreimageCaptureError(f"baseline worker failed rc={proc.returncode}")
    if not ack_path.is_file():
        raise PreimageCaptureError("baseline ACK missing")

    ack = grc.read_runtime_ack(ack_path)
    if ack.get("status") != "STARTUP_STATE":
        raise PreimageCaptureError(f"unexpected baseline ACK status: {ack.get('status')!r}")
    if ack["active_policy"] != []:
        raise PreimageCaptureError("baseline policy is not empty")
    if ack["weight_epoch"] != 0:
        raise PreimageCaptureError("baseline epoch is not zero")
    if ack.get("correction_mode") != "off":
        raise PreimageCaptureError("baseline correction mode is not off")

    reqs = _parse_requests(output)
    counts = _target_counts(reqs)
    if len(reqs) != REQUESTS:
        raise PreimageCaptureError(
            f"unexpected completed request count: expected={REQUESTS} actual={len(reqs)}"
        )
    if counts["eligible_requests"] != REQUESTS or counts["orig_hits"] != REQUESTS:
        raise PreimageCaptureError(f"baseline target reproduction failed: {counts}")

    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_raw = max(int(before_usage), int(after_usage))
    peak_bytes = _rss_to_bytes(peak_raw)
    tokens = sum(len(v) for v in reqs.values())
    budget = recommend_budget(
        requests=len(reqs),
        tokens=tokens,
        duration_ms=duration_ms,
        peak_rss_bytes=peak_bytes,
    )

    result = {
        "schema": "manual-canary-runtime-preimage-capture-v1",
        "status": "CAPTURED_REVIEW_REQUIRED",
        "production_write_allowed": False,
        "engine_source_commit": head,
        "binary_sha256": EXPECTED_BINARY_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "runtime_preimage": {
            "active_policy": ack["active_policy"],
            "active_policy_hash": ack["active_policy_hash"],
            "weight_epoch": ack["weight_epoch"],
            "ack_sha256": ack["ack_sha256"],
            "worker_instance_id": f"pid-{proc.pid}",
        },
        "target": {
            "role": TARGET["role"],
            "layer": TARGET["layer"],
            "before_n": None,
            "after_n": TARGET["after_n"],
        },
        "measurement": {
            "requests": len(reqs),
            "tokens": tokens,
            "duration_ms": duration_ms,
            "peak_rss_bytes": peak_bytes,
            "baseline_token_counts": counts,
        },
        "recommended_budget": budget,
        "manifest": manifest_meta,
        "worker_log_sha256": _sha256_file(worker_log),
        "note": (
            "Measured baseline preimage only. Candidate was not applied. "
            "Budget is a measured recommendation and still requires human review/signature."
        ),
    }
    _atomic_json(STATUS_FILE, result)
    return result


def status() -> dict:
    if not STATUS_FILE.is_file():
        return {
            "schema": "manual-canary-runtime-preimage-capture-v1",
            "status": "NOT_CAPTURED",
            "production_write_allowed": False,
        }
    value = json.loads(STATUS_FILE.read_text())
    if not isinstance(value, dict):
        raise PreimageCaptureError("status is invalid")
    return value


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--status":
        try:
            result = status()
        except Exception as exc:
            print(json.dumps({"status": "ERROR", "production_write_allowed": False, "error": str(exc)}, sort_keys=True))
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    if len(sys.argv) != 1:
        print(json.dumps({"status": "ARGUMENTS_DENIED", "production_write_allowed": False}, sort_keys=True))
        return 2
    try:
        result = capture()
    except Exception as exc:
        failure = {
            "schema": "manual-canary-runtime-preimage-capture-v1",
            "status": "FAILED",
            "production_write_allowed": False,
            "error": str(exc),
        }
        try:
            _atomic_json(STATUS_FILE, failure)
        except Exception:
            pass
        print(json.dumps(failure, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
