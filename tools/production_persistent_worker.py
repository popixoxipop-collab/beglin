#!/usr/bin/env python3
"""Long-lived Beglin GPU worker client.

The native qwen_infer_gpu persistent mode owns model/MLX state.  This Python
client only manages its file-queue protocol and validates its fixed precision
identity at startup.  One client instance corresponds to exactly one immutable
route policy (baseline or one reviewed candidate).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import threading
import time
from typing import Any, Mapping

import production_routing_cutover as routing


DEFAULT_BINARY = Path("/Users/xox/vdsp_serving/bin/qwen_infer_gpu_persistent")
DEFAULT_REPO = Path("/Users/xox/mcp-sandbox/tailnet-commander/beglin-persistent-worker-20261003")
DEFAULT_MOE_BASE = Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
DEFAULT_CHECKPOINT = Path(
    "/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/"
    "model.safetensors.index.json"
)


class PersistentWorkerError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _wait_file(path: Path, *, timeout: float, process: subprocess.Popen | None = None) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if path.is_file() and path.stat().st_size > 0:
            return
        if process is not None and process.poll() is not None:
            raise PersistentWorkerError(
                f"persistent worker exited before publishing {path.name}: rc={process.returncode}"
            )
        time.sleep(0.01)
    raise PersistentWorkerError(f"timed out waiting for {path}")


def _write_i32(path: Path, values: list[int]) -> None:
    with path.open("wb") as handle:
        for value in values:
            handle.write(struct.pack("<i", int(value)))


def _expected_active_policy(route: Mapping[str, Any]) -> list[dict]:
    r = routing.normalize_route(route)
    if r["n"] is None:
        return []
    return [{"role": r["role"], "layer": int(r["layer"]), "n": int(r["n"])}]


def _minimal_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


class PersistentGpuWorker:
    def __init__(
        self,
        *,
        name: str,
        route: Mapping[str, Any],
        queue_dir: str | Path,
        binary: str | Path = DEFAULT_BINARY,
        repo: str | Path = DEFAULT_REPO,
        moe_base: str | Path = DEFAULT_MOE_BASE,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        request_timeout: float = 90.0,
    ):
        self.name = str(name)
        self.route = routing.normalize_route(route)
        self.queue_dir = Path(queue_dir)
        self.binary = Path(binary)
        self.repo = Path(repo)
        self.moe_base = Path(moe_base)
        self.checkpoint = Path(checkpoint)
        self.request_timeout = float(request_timeout)
        self.process: subprocess.Popen | None = None
        self.log_handle = None
        self.ack: dict | None = None
        self.ack_sha256: str | None = None
        self._lock = threading.Lock()
        self._counter = 0
        self.warmed = False

    @property
    def pid(self) -> int | None:
        return None if self.process is None else int(self.process.pid)

    def _prepare_queue(self) -> None:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        requests = self.queue_dir / "requests"
        requests.mkdir(parents=True, exist_ok=True)
        for path in (self.queue_dir / "request.ready", self.queue_dir / "stop"):
            if path.exists():
                path.unlink()
        for path in self.queue_dir.glob("response.*.json"):
            path.unlink()
        promotion = self.queue_dir / "promotion_nq.txt"
        expected = _expected_active_policy(self.route)
        if expected:
            row = expected[0]
            promotion.write_text(f"{row['role']} {row['layer']} {row['n']}\n")
        else:
            promotion.write_text("")
        for path in (self.queue_dir / "applied_ack.json", self.queue_dir / "txn.cmd"):
            if path.exists():
                path.unlink()

    def start(self) -> dict:
        if self.process is not None and self.process.poll() is None:
            return self.health()
        for path, label in (
            (self.binary, "persistent binary"),
            (self.repo, "persistent source repo"),
            (self.moe_base, "MoE base"),
            (self.checkpoint, "checkpoint index"),
        ):
            if not path.exists():
                raise PersistentWorkerError(f"{label} missing: {path}")
        self._prepare_queue()
        ack_path = self.queue_dir / "applied_ack.json"
        txn_path = self.queue_dir / "txn.cmd"
        promotion = self.queue_dir / "promotion_nq.txt"
        log_path = self.queue_dir / "worker.log"
        self.log_handle = log_path.open("a", buffering=1)
        env = _minimal_env()
        env.update(
            {
                "QWEN_MOE_BASE": str(self.moe_base),
                "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
                "QWEN_MOE_CB_SLOTS": "4",
                "QWEN_MOE_CB_REQS": "1",
                "QWEN_MOE_CB_PREFILL_BUDGET": "16",
                "QWEN_MOE_GPU_CB_CHECK": "1",
                "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
                "QWEN_MOE_NEARTIE_CORRECT": "0",
                "QWEN_MOE_PROMOTION_FILE_NQ": str(promotion),
                "QWEN_MOE_PROMOTION_SAFETENSORS": str(self.checkpoint),
                "QWEN_MOE_GPU_APPLIED_ACK": str(ack_path),
                "QWEN_MOE_GPU_TXN_FILE": str(txn_path),
                "QWEN_MOE_GPU_PERSISTENT_DIR": str(self.queue_dir),
            }
        )
        self.process = subprocess.Popen(
            [str(self.binary)],
            cwd=str(self.repo),
            env=env,
            text=True,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
        )
        _wait_file(ack_path, timeout=120.0, process=self.process)
        raw = ack_path.read_bytes()
        ack = json.loads(raw)
        expected_policy = _expected_active_policy(self.route)
        if ack.get("active_policy") != expected_policy:
            self.stop()
            raise PersistentWorkerError(
                f"{self.name} active policy mismatch: "
                f"expected={expected_policy!r} actual={ack.get('active_policy')!r}"
            )
        expected_epoch = 0 if not expected_policy else 1
        if int(ack.get("weight_epoch", -1)) != expected_epoch:
            self.stop()
            raise PersistentWorkerError(
                f"{self.name} weight epoch mismatch: "
                f"expected={expected_epoch} actual={ack.get('weight_epoch')}"
            )
        expected_status = "STARTUP_STATE" if not expected_policy else "PROMOTION_APPLIED"
        if ack.get("status") != expected_status:
            self.stop()
            raise PersistentWorkerError(
                f"{self.name} startup ACK mismatch: "
                f"expected={expected_status} actual={ack.get('status')}"
            )
        self.ack = ack
        self.ack_sha256 = hashlib.sha256(raw).hexdigest()
        return self.health()

    def health(self) -> dict:
        alive = self.process is not None and self.process.poll() is None
        return {
            "schema": "beglin-persistent-worker-health-v1",
            "name": self.name,
            "alive": alive,
            "pid": self.pid,
            "route_id": self.route["route_id"],
            "policy_hash": self.route["policy_hash"],
            "weight_epoch": None if self.ack is None else int(self.ack["weight_epoch"]),
            "ack_status": None if self.ack is None else self.ack["status"],
            "ack_sha256": self.ack_sha256,
            "warmed": bool(self.warmed),
            "rss_bytes": self.rss_bytes() if alive else 0,
        }

    def rss_bytes(self) -> int:
        if self.process is None or self.process.poll() is not None:
            return 0
        proc = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(self.process.pid)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0
        return int(proc.stdout.strip().split()[0]) * 1024

    def submit(self, requests: list[dict]) -> dict:
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                raise PersistentWorkerError(f"{self.name} worker is not alive")
            if not isinstance(requests, list) or not requests:
                raise PersistentWorkerError("requests must be a non-empty list")
            if len(requests) > 18:
                raise PersistentWorkerError("persistent request batch exceeds 18")
            self._counter += 1
            rid = f"{self.name}-{os.getpid()}-{self._counter:08d}"
            req_dir = self.queue_dir / "requests" / rid
            req_dir.mkdir(parents=True, exist_ok=False)
            manifest = req_dir / "manifest.txt"
            lines: list[str] = []
            for index, request in enumerate(requests):
                tokens = request.get("prompt_tokens")
                if not isinstance(tokens, list) or not tokens:
                    raise PersistentWorkerError("prompt_tokens must be a non-empty list")
                max_new = int(request.get("max_new_tokens", 10))
                if max_new <= 0 or max_new > 256:
                    raise PersistentWorkerError("max_new_tokens outside [1,256]")
                values = [int(value) for value in tokens]
                raw = req_dir / f"request-{index}.i32"
                _write_i32(raw, values)
                lines.append(f"{raw} {max_new}\n")
            manifest.write_text("".join(lines))
            response = self.queue_dir / f"response.{rid}.json"
            if response.exists():
                response.unlink()
            request_ready = self.queue_dir / "request.ready"
            if request_ready.exists():
                raise PersistentWorkerError("persistent queue already has request.ready")
            started = time.monotonic()
            _atomic_text(request_ready, f"{rid} {manifest}\n")
            _wait_file(response, timeout=self.request_timeout, process=self.process)
            roundtrip_ms = (time.monotonic() - started) * 1000.0
            value = json.loads(response.read_text())
            if value.get("schema") != "beglin-gpu-persistent-response-v1":
                raise PersistentWorkerError("unexpected persistent response schema")
            if value.get("status") != "OK" or value.get("request_id") != rid:
                raise PersistentWorkerError("persistent response identity/status mismatch")
            if int(value.get("requests", -1)) != len(requests):
                raise PersistentWorkerError("persistent response count mismatch")
            if value.get("finite_logits") is not True:
                raise PersistentWorkerError("persistent response reports non-finite logits")
            value["roundtrip_ms"] = round(roundtrip_ms, 3)
            value["worker_pid"] = int(self.process.pid)
            value["worker_rss_bytes"] = self.rss_bytes()
            response.unlink(missing_ok=True)
            shutil.rmtree(req_dir, ignore_errors=True)
            return value

    def warm(self, *, prompt_tokens: list[int], expected_token: int) -> dict:
        value = self.submit(
            [{"prompt_tokens": list(prompt_tokens), "max_new_tokens": 10}]
        )
        tokens = value["responses"][0]["generated_tokens"]
        if len(tokens) <= 8 or int(tokens[8]) != int(expected_token):
            raise PersistentWorkerError(
                f"{self.name} warmup reference mismatch: "
                f"expected={expected_token} actual={tokens[8] if len(tokens)>8 else None}"
            )
        self.warmed = True
        value["reference_token_at_gen_idx_8"] = int(tokens[8])
        return value

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                (self.queue_dir / "stop").write_text("stop\n")
                process.wait(timeout=20)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except Exception:
                pass
        self.process = None
        self.log_handle = None


class PersistentWorkerPool:
    def __init__(self, *, root: str | Path, binary: str | Path = DEFAULT_BINARY):
        self.root = Path(root)
        self.binary = Path(binary)
        self.workers: dict[str, PersistentGpuWorker] = {}
        self._gpu_lock = threading.BoundedSemaphore(1)

    def add(self, *, name: str, route: Mapping[str, Any]) -> PersistentGpuWorker:
        worker = PersistentGpuWorker(
            name=name,
            route=route,
            queue_dir=self.root / name,
            binary=self.binary,
        )
        self.workers[route["route_id"]] = worker
        return worker

    def start_all(self) -> dict:
        status = {}
        for route_id, worker in self.workers.items():
            status[route_id] = worker.start()
        return status

    def warm_all(self, *, prompt_tokens: list[int], expected_by_route: Mapping[str, int]) -> dict:
        out = {}
        for route_id, worker in self.workers.items():
            if route_id not in expected_by_route:
                raise PersistentWorkerError(f"missing warmup reference for {route_id}")
            with self._gpu_lock:
                out[route_id] = worker.warm(
                    prompt_tokens=prompt_tokens,
                    expected_token=int(expected_by_route[route_id]),
                )
        return out

    def submit(self, route_id: str, requests: list[dict]) -> dict:
        worker = self.workers.get(str(route_id))
        if worker is None:
            raise PersistentWorkerError(f"no persistent worker for route {route_id!r}")
        with self._gpu_lock:
            return worker.submit(requests)

    def health(self) -> dict:
        return {route_id: worker.health() for route_id, worker in self.workers.items()}

    def stop_all(self) -> None:
        for worker in self.workers.values():
            worker.stop()
