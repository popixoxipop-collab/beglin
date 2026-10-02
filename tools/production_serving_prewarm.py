#!/usr/bin/env python3
"""Prewarmed one-shot worker pool for the Beglin localhost supervisor.

The certified native binary is unchanged. A worker is started early with its
model/precision state loaded and blocks in the existing manifest loader on a
private FIFO. When one HTTP request arrives, the supervisor writes that
request's manifest to the FIFO, closes it, and the already-warm worker runs to
completion. A replacement is then prewarmed in the background.

This removes request-path model loading without inventing a new native IPC
protocol or changing the certified GPU binary.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid


REQUEST_RE = re.compile(
    r"\[moe gpu cb online\] req (\d+) .*? tokens:\s*([^\n\r]*)"
)
VALIDATION_RE = re.compile(
    r"GPU_VALIDATION_V1 .*?finite_logits=(?P<finite>[01]).*?requests=(?P<requests>\d+)"
)


class PrewarmError(RuntimeError):
    pass


def _minimal_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _write_i32(path: Path, values: list[int]) -> None:
    import struct
    with path.open("wb") as handle:
        for value in values:
            handle.write(struct.pack("<i", int(value)))


def _parse_generated(output: str) -> dict[int, list[int]]:
    generated: dict[int, list[int]] = {}
    for match in REQUEST_RE.finditer(output):
        generated[int(match.group(1))] = [
            int(value) for value in match.group(2).split()
        ]
    return generated


class WarmWorker:
    def __init__(
        self,
        *,
        state_root: Path,
        repo: Path,
        binary: Path,
        moe_base: Path,
        safetensors: Path,
        route_snapshot: dict,
        batch_size: int,
        timeout_seconds: int,
    ):
        self.state_root = state_root
        self.repo = repo
        self.binary = binary
        self.moe_base = moe_base
        self.safetensors = safetensors
        self.route_snapshot = json.loads(json.dumps(route_snapshot))
        self.batch_size = int(batch_size)
        self.timeout_seconds = int(timeout_seconds)
        self.root = state_root / (
            f"worker-g{route_snapshot['generation']}-b{batch_size}-"
            + uuid.uuid4().hex[:12]
        )
        self.root.mkdir(parents=True, exist_ok=False)
        self.fifo = self.root / "manifest.fifo"
        self.promo = self.root / "promotion_nq.txt"
        self.ack = self.root / "applied_ack.json"
        self.txn = self.root / "txn.cmd"
        self.log = self.root / "worker.log"
        os.mkfifo(self.fifo, 0o600)
        route = route_snapshot["route"]
        n = route.get("n")
        self.promo.write_text(
            "" if n is None else f"{route['role']} {int(route['layer'])} {int(n)}\n"
        )

        env = _minimal_env()
        env.update(
            {
                "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
                "QWEN_MOE_BASE": str(moe_base),
                "QWEN_MOE_NEARTIE_CORRECT": "0",
                "QWEN_MOE_CB_PROMPT_MANIFEST": str(self.fifo),
                "QWEN_MOE_CB_SLOTS": str(min(4, self.batch_size)),
                "QWEN_MOE_CB_REQS": str(self.batch_size),
                "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
                "QWEN_MOE_GPU_APPLIED_ACK": str(self.ack),
                "QWEN_MOE_GPU_TXN_FILE": str(self.txn),
                "QWEN_MOE_PROMOTION_FILE_NQ": str(self.promo),
                "QWEN_MOE_PROMOTION_SAFETENSORS": str(safetensors),
            }
        )
        self._log_handle = self.log.open("w")
        self.spawn_started_ns = time.monotonic_ns()
        self.proc = subprocess.Popen(
            [str(binary)],
            cwd=repo,
            env=env,
            text=True,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )
        self.pid = int(self.proc.pid)
        self.writer_fd = self._wait_until_manifest_reader_ready()
        self.ready_ns = time.monotonic_ns()
        self._validate_startup_ack()

    @property
    def startup_ms(self) -> int:
        return max(1, (self.ready_ns - self.spawn_started_ns) // 1_000_000)

    def _wait_until_manifest_reader_ready(self) -> int:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                self._log_handle.flush()
                output = self.log.read_text(errors="replace")
                raise PrewarmError(
                    f"prewarm worker exited before manifest wait rc={rc}: "
                    + output[-2000:]
                )
            try:
                return os.open(
                    self.fifo,
                    os.O_WRONLY | os.O_NONBLOCK,
                )
            except OSError as exc:
                if exc.errno != errno.ENXIO:
                    raise
                time.sleep(0.05)
        self.terminate()
        raise PrewarmError("prewarm worker did not reach manifest reader before timeout")

    def _validate_startup_ack(self) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not self.ack.is_file():
            if self.proc.poll() is not None:
                break
            time.sleep(0.02)
        if not self.ack.is_file():
            self.terminate()
            raise PrewarmError("prewarm worker startup ACK is missing")
        sys.path.insert(0, str(self.repo / "tools"))
        import gpu_runtime_control as grc
        ack = grc.read_runtime_ack(self.ack)
        expected = self.route_snapshot["route"]["policy_hash"]
        if ack["active_policy_hash"] != expected:
            self.terminate()
            raise PrewarmError(
                "prewarm worker policy mismatch: "
                f"expected={expected} actual={ack['active_policy_hash']}"
            )

    def serve(self, parsed: list[tuple[list[int], int]]) -> dict:
        if len(parsed) != self.batch_size:
            raise PrewarmError(
                f"prewarm batch size mismatch expected={self.batch_size} actual={len(parsed)}"
            )
        request_started_ns = time.monotonic_ns()
        lines = []
        for idx, (tokens, max_new) in enumerate(parsed):
            raw = self.root / f"req-{idx}.i32"
            _write_i32(raw, tokens)
            lines.append(f"{raw} {int(max_new)}\n")
        payload = "".join(lines).encode()
        try:
            with os.fdopen(self.writer_fd, "wb", closefd=True) as writer:
                self.writer_fd = -1
                writer.write(payload)
                writer.flush()
        except Exception:
            self.terminate()
            raise

        try:
            self.proc.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            self.terminate()
            raise PrewarmError(f"prewarmed worker timed out pid={self.pid}")
        self._log_handle.flush()
        self._log_handle.close()
        output = self.log.read_text(errors="replace")
        if self.proc.returncode != 0:
            raise PrewarmError(
                f"prewarmed worker failed rc={self.proc.returncode}: "
                + output[-2000:]
            )
        sys.path.insert(0, str(self.repo / "tools"))
        import gpu_runtime_control as grc
        ack = grc.read_runtime_ack(self.ack)
        route = self.route_snapshot["route"]
        if ack["active_policy_hash"] != route["policy_hash"]:
            raise PrewarmError("prewarmed worker ACK policy drift")

        generated = _parse_generated(output)
        if len(generated) < len(parsed):
            raise PrewarmError(
                f"prewarmed response count mismatch expected_at_least={len(parsed)} "
                f"actual={len(generated)}"
            )
        validation = VALIDATION_RE.search(output)
        if validation is None or validation.group("finite") != "1":
            raise PrewarmError("prewarmed validation report failed")
        validation_requests = int(validation.group("requests"))
        if validation_requests < len(parsed):
            raise PrewarmError("prewarmed validation request count below admitted count")
        duration_ms = max(
            1, (time.monotonic_ns() - request_started_ns) // 1_000_000
        )
        result = {
            "schema": "beglin-supervisor-batch-response-v1",
            "status": "OK",
            "route_generation": self.route_snapshot["generation"],
            "route_manifest_sha256": self.route_snapshot["manifest_sha256"],
            "route": route,
            "worker_instance_id": f"pid-{self.pid}",
            "worker_ack_sha256": ack["ack_sha256"],
            "worker_epoch": int(ack["weight_epoch"]),
            "finite_logits": True,
            "duration_ms": duration_ms,
            "peak_child_rss_bytes": 0,
            "engine_requests_completed": len(generated),
            "engine_validation_requests": validation_requests,
            "engine_cycles_manifest_when_underfilled": len(generated) > len(parsed),
            "prewarmed_worker": True,
            "prewarm_startup_ms": self.startup_ms,
            "responses": [
                {
                    "request_index": idx,
                    "generated_tokens": generated[idx],
                }
                for idx in range(len(parsed))
            ],
            "production_write_allowed": False,
            "auto_promotion_enabled": False,
        }
        return result

    def terminate(self) -> None:
        if getattr(self, "writer_fd", -1) >= 0:
            try:
                os.close(self.writer_fd)
            except OSError:
                pass
            self.writer_fd = -1
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if getattr(self, "_log_handle", None) is not None and not self._log_handle.closed:
            self._log_handle.close()

    def cleanup(self) -> None:
        self.terminate()
        shutil.rmtree(self.root, ignore_errors=True)


class PrewarmPool:
    """One ready R=1 worker, replenished after each use."""

    def __init__(
        self,
        *,
        state_root: Path,
        repo: Path,
        binary: Path,
        moe_base: Path,
        safetensors: Path,
        timeout_seconds: int,
    ):
        self.state_root = state_root
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.repo = repo
        self.binary = binary
        self.moe_base = moe_base
        self.safetensors = safetensors
        self.timeout_seconds = int(timeout_seconds)
        self.lock = threading.Lock()
        self.target_snapshot: dict | None = None
        self.ready_worker: WarmWorker | None = None
        self.warming = False
        self.last_error: str | None = None
        self.last_startup_ms: int | None = None
        self.generation_token = 0

    @staticmethod
    def _key(snapshot: dict) -> tuple[int, str]:
        return (
            int(snapshot["generation"]),
            str(snapshot["route"]["policy_hash"]),
        )

    def configure(self, snapshot: dict) -> None:
        stale: WarmWorker | None = None
        should_fill = False
        with self.lock:
            if (
                self.target_snapshot is None
                or self._key(self.target_snapshot) != self._key(snapshot)
            ):
                self.target_snapshot = json.loads(json.dumps(snapshot))
                self.generation_token += 1
                stale = self.ready_worker
                self.ready_worker = None
                self.last_error = None
                should_fill = not self.warming
            elif self.ready_worker is None and not self.warming:
                should_fill = True
        if stale is not None:
            stale.cleanup()
        if should_fill:
            self._start_fill()

    def _start_fill(self) -> None:
        with self.lock:
            if self.warming or self.target_snapshot is None or self.ready_worker is not None:
                return
            self.warming = True
            token = self.generation_token
            snapshot = json.loads(json.dumps(self.target_snapshot))
        thread = threading.Thread(
            target=self._fill,
            args=(token, snapshot),
            daemon=True,
            name="beglin-prewarm-fill",
        )
        thread.start()

    def _fill(self, token: int, snapshot: dict) -> None:
        worker: WarmWorker | None = None
        error: str | None = None
        try:
            worker = WarmWorker(
                state_root=self.state_root,
                repo=self.repo,
                binary=self.binary,
                moe_base=self.moe_base,
                safetensors=self.safetensors,
                route_snapshot=snapshot,
                batch_size=1,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        keep = False
        with self.lock:
            if (
                worker is not None
                and token == self.generation_token
                and self.target_snapshot is not None
                and self._key(snapshot) == self._key(self.target_snapshot)
                and self.ready_worker is None
            ):
                self.ready_worker = worker
                self.last_startup_ms = worker.startup_ms
                self.last_error = None
                keep = True
            else:
                self.last_error = error
            self.warming = False
        if worker is not None and not keep:
            worker.cleanup()

    def acquire(self, snapshot: dict, batch_size: int) -> WarmWorker | None:
        if int(batch_size) != 1:
            return None
        self.configure(snapshot)
        with self.lock:
            if self.ready_worker is None:
                return None
            if self._key(self.ready_worker.route_snapshot) != self._key(snapshot):
                return None
            worker = self.ready_worker
            self.ready_worker = None
            return worker

    def consumed(self, worker: WarmWorker) -> None:
        worker.cleanup()
        self._start_fill()

    def status(self) -> dict:
        with self.lock:
            worker = self.ready_worker
            target = json.loads(json.dumps(self.target_snapshot)) if self.target_snapshot else None
            return {
                "schema": "beglin-prewarm-pool-status-v1",
                "enabled": True,
                "batch_size": 1,
                "depth": 1,
                "ready": worker is not None,
                "warming": self.warming,
                "ready_worker_pid": worker.pid if worker is not None else None,
                "ready_worker_startup_ms": worker.startup_ms if worker is not None else None,
                "last_startup_ms": self.last_startup_ms,
                "last_error": self.last_error,
                "target_generation": target["generation"] if target else None,
                "target_policy_hash": target["route"]["policy_hash"] if target else None,
            }

    def shutdown(self) -> None:
        worker = None
        with self.lock:
            self.generation_token += 1
            worker = self.ready_worker
            self.ready_worker = None
            self.target_snapshot = None
        if worker is not None:
            worker.cleanup()
