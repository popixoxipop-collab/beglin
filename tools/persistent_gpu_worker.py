#!/usr/bin/env python3
"""Long-lived qwen_infer_gpu client for the Beglin serving supervisor."""
from __future__ import annotations

import json
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Callable

import production_routing_cutover as routing


class PersistentWorkerError(RuntimeError):
    pass


class PersistentGpuWorker:
    def __init__(
        self,
        *,
        binary: str | Path,
        cwd: str | Path,
        base_env: dict[str, str],
        route: dict,
        generation: int,
        response_timeout_seconds: int = 90,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        self.binary = Path(binary)
        self.cwd = Path(cwd)
        self.base_env = dict(base_env)
        self.route = routing.normalize_route(route)
        self.generation = int(generation)
        self.response_timeout_seconds = int(response_timeout_seconds)
        self._popen_factory = popen_factory
        self._proc: subprocess.Popen | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        return int(self._proc.pid) if self._proc is not None else None

    def _drain_stderr(self) -> None:
        assert self._proc is not None
        stream = self._proc.stderr
        if stream is None:
            return
        for line in stream:
            self._stderr_lines.append(line.rstrip("\n"))
            if len(self._stderr_lines) > 4000:
                del self._stderr_lines[:2000]

    def start(self) -> None:
        if self._proc is not None:
            raise PersistentWorkerError("persistent worker already started")
        if not self.binary.is_file():
            raise PersistentWorkerError(f"persistent binary missing: {self.binary}")
        env = dict(self.base_env)
        env["QWEN_MOE_GPU_PERSIST_STDIN"] = "1"
        self._proc = self._popen_factory(
            [str(self.binary)],
            cwd=self.cwd,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name=f"beglin-persistent-stderr-{self._proc.pid}",
            daemon=True,
        )
        self._stderr_thread.start()

    def _ensure_alive(self) -> subprocess.Popen:
        if self._proc is None:
            raise PersistentWorkerError("persistent worker not started")
        rc = self._proc.poll()
        if rc is not None:
            tail="\n".join(self._stderr_lines[-40:])
            raise PersistentWorkerError(
                f"persistent worker exited rc={rc}; stderr tail:\n{tail}"
            )
        return self._proc

    def request(self, *, manifest_path: Path, request_id: str | None = None) -> dict:
        with self._lock:
            proc=self._ensure_alive()
            rid=request_id or uuid.uuid4().hex
            if any(ch.isspace() for ch in rid) or not rid:
                raise PersistentWorkerError("request_id must be one non-empty token")
            stdin=proc.stdin
            stdout=proc.stdout
            if stdin is None or stdout is None:
                raise PersistentWorkerError("persistent worker stdio unavailable")
            started=time.monotonic()
            stdin.write(f"{rid} {manifest_path}\n")
            stdin.flush()

            # stdout is line-oriented. select() makes the timeout real even when
            # the engine is silent while executing a long GPU batch.
            selector=selectors.DefaultSelector()
            selector.register(stdout,selectors.EVENT_READ)
            try:
                while True:
                    remaining=self.response_timeout_seconds-(time.monotonic()-started)
                    if remaining <= 0:
                        raise PersistentWorkerError(
                            f"persistent response timeout request_id={rid}"
                        )
                    events=selector.select(timeout=remaining)
                    if not events:
                        raise PersistentWorkerError(
                            f"persistent response timeout request_id={rid}"
                        )
                    line=stdout.readline()
                    if line == "":
                        self._ensure_alive()
                        raise PersistentWorkerError("persistent worker stdout closed")
                    if not line.startswith("PERSIST_RESPONSE "):
                        continue
                    try:
                        value=json.loads(line[len("PERSIST_RESPONSE "):])
                    except json.JSONDecodeError as exc:
                        raise PersistentWorkerError("invalid persistent response JSON") from exc
                    if value.get("status") != "OK":
                        raise PersistentWorkerError(
                            f"persistent worker rejected request: {value.get('status')}"
                        )
                    if value.get("request_id") != rid:
                        raise PersistentWorkerError(
                            "persistent response request_id mismatch"
                        )
                    return value
            finally:
                selector.close()

    def close(self) -> None:
        proc=self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        finally:
            self._proc=None


class PersistentWorkerManager:
    def __init__(
        self,
        *,
        binary: str | Path,
        cwd: str | Path,
        env_builder: Callable[[dict, Path, Path, Path], dict[str,str]],
        ack_reader: Callable[[Path], dict],
        timeout_seconds: int = 90,
        worker_factory=PersistentGpuWorker,
    ):
        self.binary=Path(binary)
        self.cwd=Path(cwd)
        self.env_builder=env_builder
        self.ack_reader=ack_reader
        self.timeout_seconds=int(timeout_seconds)
        self.worker_factory=worker_factory
        self.worker: PersistentGpuWorker | None=None
        self.generation: int | None=None
        self.route_id: str | None=None
        self.policy_hash: str | None=None
        self.worker_root: tempfile.TemporaryDirectory | None=None
        self.startup_ack: dict | None=None

    def _shutdown(self):
        if self.worker is not None:
            self.worker.close()
        self.worker=None
        self.generation=None
        self.route_id=None
        self.policy_hash=None
        self.startup_ack=None
        if self.worker_root is not None:
            self.worker_root.cleanup()
        self.worker_root=None

    def ensure(self, snapshot: dict) -> PersistentGpuWorker:
        generation=int(snapshot["generation"])
        route=routing.normalize_route(snapshot["route"])
        if (
            self.worker is not None
            and self.generation == generation
            and self.route_id == route["route_id"]
            and self.policy_hash == route["policy_hash"]
        ):
            self.worker._ensure_alive()
            return self.worker

        self._shutdown()
        root=tempfile.TemporaryDirectory(prefix="beglin-persistent-worker-")
        rp=Path(root.name)
        promo=rp/"promotion_nq.txt"
        ack=rp/"applied_ack.json"
        txn=rp/"txn.cmd"
        env=self.env_builder(route,promo,ack,txn)
        worker=self.worker_factory(
            binary=self.binary,
            cwd=self.cwd,
            base_env=env,
            route=route,
            generation=generation,
            response_timeout_seconds=self.timeout_seconds,
        )
        worker.start()

        deadline=time.monotonic()+self.timeout_seconds
        while time.monotonic() < deadline:
            if ack.is_file():
                state=self.ack_reader(ack)
                if state["active_policy_hash"] != route["policy_hash"]:
                    worker.close()
                    root.cleanup()
                    raise PersistentWorkerError(
                        "persistent startup ACK policy hash mismatch"
                    )
                self.worker=worker
                self.worker_root=root
                self.generation=generation
                self.route_id=route["route_id"]
                self.policy_hash=route["policy_hash"]
                self.startup_ack=dict(state)
                return worker
            worker._ensure_alive()
            time.sleep(0.05)
        worker.close()
        root.cleanup()
        raise PersistentWorkerError("persistent startup ACK timeout")

    def close(self):
        self._shutdown()
