#!/usr/bin/env python3
"""Capture one real XOX baseline runtime preimage for Agent E.

Fixed-scope, baseline-only:
- certified runtime source/binary/checkpoint identities are verified first
- empty promotion policy only
- 12 replay requests
- no candidate application, production mutation, Supabase write, or auto-promotion
- duration, peak RSS and physical memory are measured on the actual XOX host
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import resource
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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

EXPECTED_HEAD = "330954b27f146b8a17db2cb353c3e620968bad5e"
EXPECTED_BINARY_SHA = "6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392"
EXPECTED_CHECKPOINT_SHA = "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898"
BASELINE_POLICY_HASH = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
REQUESTS = 12
SLOTS = 4
TIMEOUT_SECONDS = 180

sys.path.insert(0, str(REPO / "tools"))
import gpu_runtime_control as grc

REQUEST_RE = re.compile(
    r"\[moe gpu cb online\] req (\d+) .*? tokens:\s*([^\n\r]*)"
)
VALIDATION_RE = re.compile(
    r"GPU_VALIDATION_V1 .*?finite_logits=(?P<finite>[01]).*?requests=(?P<requests>\d+)"
)


class BaselineCaptureError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_head() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return p.stdout.strip()


def _physical_memory_bytes() -> int:
    p = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "hw.memsize"],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    value = int(p.stdout.strip())
    if value <= 0:
        raise BaselineCaptureError("invalid physical memory")
    return value


def _manifest_input() -> tuple[Path, int]:
    rows = [
        line.strip()
        for line in SOURCE_MANIFEST.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise BaselineCaptureError("source manifest has no active rows")
    parts = shlex.split(rows[0])
    if len(parts) != 2:
        raise BaselineCaptureError("invalid source manifest row")
    raw = Path(parts[0])
    max_new = int(parts[1])
    if not raw.is_file() or max_new <= 0:
        raise BaselineCaptureError("invalid replay input")
    return raw, max_new


def _minimal_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            out[key] = value
    return out


def capture() -> dict:
    if platform.system() != "Darwin":
        raise BaselineCaptureError("XOX baseline capture requires Darwin")

    head = _git_head()
    if head != EXPECTED_HEAD:
        raise BaselineCaptureError(
            f"unexpected certified runtime HEAD: expected={EXPECTED_HEAD} actual={head}"
        )
    binary_sha = _sha256_file(BINARY)
    if binary_sha != EXPECTED_BINARY_SHA:
        raise BaselineCaptureError(
            f"binary SHA mismatch: expected={EXPECTED_BINARY_SHA} actual={binary_sha}"
        )

    checkpoint = json.loads(CHECKPOINT_EVIDENCE.read_text())
    if checkpoint.get("status") != "VERIFIED":
        raise BaselineCaptureError("checkpoint evidence is not VERIFIED")
    if checkpoint.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise BaselineCaptureError("checkpoint content identity mismatch")

    physical_memory = _physical_memory_bytes()
    raw, max_new = _manifest_input()

    with tempfile.TemporaryDirectory(prefix="beglin-agent-e-baseline-") as td:
        root = Path(td)
        manifest = root / "manifest.txt"
        manifest.write_text((f"{raw} {max_new}\n") * REQUESTS)
        ack_path = root / "applied_ack.json"
        txn_path = root / "txn.cmd"
        promo_path = root / "promotion_nq.txt"
        promo_path.write_text("")

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
            worker_output, _ = proc.communicate(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                worker_output, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                worker_output, _ = proc.communicate(timeout=10)
            raise BaselineCaptureError(
                f"baseline worker timed out and was stopped: pid={worker_pid}"
            )
        finished_ns = time.monotonic_ns()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)

        if proc.returncode != 0:
            raise BaselineCaptureError(f"baseline worker failed rc={proc.returncode}")
        if not ack_path.is_file():
            raise BaselineCaptureError("baseline runtime ACK is missing")

        ack = grc.read_runtime_ack(ack_path)
        if ack.get("status") != "STARTUP_STATE":
            raise BaselineCaptureError(f"unexpected baseline ACK status: {ack.get('status')!r}")
        if ack.get("active_policy") != []:
            raise BaselineCaptureError("baseline active policy is not empty")
        if ack.get("active_policy_hash") != BASELINE_POLICY_HASH:
            raise BaselineCaptureError(
                "baseline policy hash mismatch after runtime ACK normalization: "
                f"expected={BASELINE_POLICY_HASH} actual={ack.get('active_policy_hash')}"
            )
        if _stable_hash([]) != BASELINE_POLICY_HASH:
            raise BaselineCaptureError("local canonical empty-policy hash mismatch")
        if int(ack.get("weight_epoch", -1)) != 0:
            raise BaselineCaptureError("baseline weight epoch is not zero")
        if not str(ack.get("ack_sha256", "")):
            raise BaselineCaptureError("baseline ACK SHA is missing")

        requests: dict[int, list[int]] = {}
        for match in REQUEST_RE.finditer(worker_output):
            requests[int(match.group(1))] = [int(x) for x in match.group(2).split()]
        if len(requests) != REQUESTS:
            raise BaselineCaptureError(
                f"request count mismatch: expected={REQUESTS} actual={len(requests)}"
            )

        validation = VALIDATION_RE.search(worker_output)
        if validation is None:
            raise BaselineCaptureError("GPU validation report is missing")
        if validation.group("finite") != "1":
            raise BaselineCaptureError("GPU validation reports non-finite logits")
        if int(validation.group("requests")) != REQUESTS:
            raise BaselineCaptureError("GPU validation request count mismatch")

        # CPython on Darwin reports ru_maxrss in bytes. This process launches only
        # small identity helpers plus one qwen worker; the qwen child dominates the max.
        peak_rss = int(usage.ru_maxrss)
        if peak_rss <= 0:
            raise BaselineCaptureError("invalid peak RSS")
        if peak_rss >= physical_memory:
            raise BaselineCaptureError(
                "peak RSS is not below physical memory; refuse to materialize a budget"
            )

        duration_ms = max(1, (finished_ns - started_ns) // 1_000_000)
        return {
            "schema": "beglin-agent-e-baseline-preimage-capture/1",
            "status": "VERIFIED",
            "production_write_allowed": False,
            "runtime_mutation_requested": False,
            "candidate_applied": False,
            "source_head": head,
            "binary_sha256": binary_sha,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "active_policy": [],
            "active_policy_hash": str(ack["active_policy_hash"]),
            "weight_epoch": 0,
            "ack_sha256": str(ack["ack_sha256"]),
            "worker_instance_id": f"pid-{worker_pid}",
            "worker_exit_code": int(proc.returncode),
            "requests": len(requests),
            "tokens": sum(len(tokens) for tokens in requests.values()),
            "duration_ms": int(duration_ms),
            "peak_rss_bytes": peak_rss,
            "physical_memory_bytes": physical_memory,
            "slots": SLOTS,
            "max_new_tokens": max_new,
            "replay_manifest_sha256": _sha256_file(manifest),
            "source_manifest_sha256": _sha256_file(SOURCE_MANIFEST),
            "raw_token_sha256": _sha256_file(raw),
            "worker_log_sha256": hashlib.sha256(worker_output.encode()).hexdigest(),
            "runner_hostname": subprocess.check_output(["hostname"], text=True).strip(),
            "runner_arch": platform.machine(),
            "rss_source": "resource.getrusage(RUSAGE_CHILDREN).ru_maxrss on Darwin",
        }


def main() -> int:
    result = capture()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
