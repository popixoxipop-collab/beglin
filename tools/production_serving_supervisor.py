#!/usr/bin/env python3
"""Local Beglin serving supervisor for XOX.

This is the first real ingress layer around the native CLI engine.  The native
DeepSeek/MoE path does not yet expose a text tokenizer API, so this supervisor
accepts token ids, not text.

Contract:
- binds only to 127.0.0.1
- reads one atomic active-route manifest at request admission
- launches an owned MLX worker with that exact route's precision policy
- returns generated token ids plus route generation/identity
- never mutates the route manifest itself
- never enables auto-promotion
- serializes GPU work to avoid accidental memory-contention fanout
"""
from __future__ import annotations

import argparse
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import production_routing_cutover as routing


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
DEFAULT_STATE_ROOT = Path("/Users/xox/vdsp_serving")
DEFAULT_ROUTE_MANIFEST = DEFAULT_STATE_ROOT / "active-route.json"
DEFAULT_PORT = 18765

EXPECTED_HEAD = "330954b27f146b8a17db2cb353c3e620968bad5e"
EXPECTED_BINARY_SHA = "6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392"
EXPECTED_CHECKPOINT_SHA = "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898"
BASELINE_POLICY_HASH = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
CANDIDATE_POLICY_HASH = "0e048b8c5c50a1c005ea573caadd6ccdf99797c24b9c443e993fe0d7b0b27141"
ALLOWED_TARGET = ("shared_up_proj", 3)
ALLOWED_CANDIDATE_N = 6

MAX_BATCH_REQUESTS = 12
MAX_PROMPT_TOKENS = 4096
MAX_NEW_TOKENS = 256
WORKER_TIMEOUT_SECONDS = 30
MAX_BODY_BYTES = 2 * 1024 * 1024

REQUEST_RE = re.compile(
    r"\[moe gpu cb online\] req (\d+) .*? tokens:\s*([^\n\r]*)"
)
VALIDATION_RE = re.compile(
    r"GPU_VALIDATION_V1 .*?finite_logits=(?P<finite>[01]).*?requests=(?P<requests>\d+)"
)


class SupervisorError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def verify_runtime_identity() -> dict:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SupervisorError("Beglin XOX supervisor requires Darwin arm64")
    head = _git_head()
    if head != EXPECTED_HEAD:
        raise SupervisorError(
            f"runtime source HEAD mismatch: expected={EXPECTED_HEAD} actual={head}"
        )
    binary_sha = _sha256_file(BINARY)
    if binary_sha != EXPECTED_BINARY_SHA:
        raise SupervisorError(
            f"runtime binary mismatch: expected={EXPECTED_BINARY_SHA} actual={binary_sha}"
        )
    checkpoint = json.loads(CHECKPOINT_EVIDENCE.read_text())
    if checkpoint.get("status") != "VERIFIED":
        raise SupervisorError("checkpoint evidence is not VERIFIED")
    if checkpoint.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise SupervisorError("checkpoint content identity mismatch")
    return {
        "source_commit": head,
        "binary_sha256": binary_sha,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
    }


def baseline_route() -> dict:
    return {
        "route_id": "beglin-baseline-q4g64",
        "worker_instance_id": "spawn-per-admission",
        "endpoint": "local-supervisor://baseline",
        "source_commit": EXPECTED_HEAD,
        "binary_sha256": EXPECTED_BINARY_SHA,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "policy_hash": BASELINE_POLICY_HASH,
        "role": ALLOWED_TARGET[0],
        "layer": ALLOWED_TARGET[1],
        "n": None,
    }


def candidate_route() -> dict:
    return {
        "route_id": "beglin-candidate-shared-up-l3-n6",
        "worker_instance_id": "spawn-per-admission",
        "endpoint": "local-supervisor://candidate",
        "source_commit": EXPECTED_HEAD,
        "binary_sha256": EXPECTED_BINARY_SHA,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "policy_hash": CANDIDATE_POLICY_HASH,
        "role": ALLOWED_TARGET[0],
        "layer": ALLOWED_TARGET[1],
        "n": ALLOWED_CANDIDATE_N,
    }


def router_capability(*, route_manifest: str | Path, port: int) -> dict:
    return {
        "schema": routing.ROUTER_CAPABILITY_SCHEMA,
        "status": "VERIFIED",
        "router_id": "beglin-local-http-supervisor-v1",
        "implementation_verified": True,
        "atomic_cas": True,
        "health_observation": True,
        "rollback": True,
        "request_drain": True,
        "route_store_kind": "atomic-json-manifest",
        "listen_host": "127.0.0.1",
        "listen_port": int(port),
        "route_manifest": str(Path(route_manifest)),
        "request_contract": "prompt_tokens-v1",
        "external_network_exposed": False,
        "automatic_cutover_allowed": False,
    }


def _minimal_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _route_policy_line(route: dict) -> str:
    normalized = routing.normalize_route(route)
    key = (normalized["role"], int(normalized["layer"]))
    if key != ALLOWED_TARGET:
        raise SupervisorError("route target is outside reviewed supervisor scope")
    n = normalized.get("n")
    if n is None:
        if normalized["policy_hash"] != BASELINE_POLICY_HASH:
            raise SupervisorError("baseline route policy hash mismatch")
        return ""
    if int(n) != ALLOWED_CANDIDATE_N:
        raise SupervisorError("candidate precision is outside reviewed scope")
    if normalized["policy_hash"] != CANDIDATE_POLICY_HASH:
        raise SupervisorError("candidate route policy hash mismatch")
    return f"{key[0]} {key[1]} {int(n)}\n"


def _validate_request(req: Any) -> tuple[list[int], int]:
    if not isinstance(req, dict):
        raise SupervisorError("request must be a JSON object")
    tokens = req.get("prompt_tokens")
    if not isinstance(tokens, list) or not tokens:
        raise SupervisorError("prompt_tokens must be a non-empty list")
    if len(tokens) > MAX_PROMPT_TOKENS:
        raise SupervisorError("prompt_tokens exceeds supervisor limit")
    out: list[int] = []
    for value in tokens:
        try:
            token = int(value)
        except (TypeError, ValueError) as exc:
            raise SupervisorError("prompt_tokens must contain integers") from exc
        if token < 0 or token > 2**31 - 1:
            raise SupervisorError("prompt token is outside int32 range")
        out.append(token)
    try:
        max_new = int(req.get("max_new_tokens", 10))
    except (TypeError, ValueError) as exc:
        raise SupervisorError("max_new_tokens must be an integer") from exc
    if max_new <= 0 or max_new > MAX_NEW_TOKENS:
        raise SupervisorError("max_new_tokens is outside supervisor limit")
    return out, max_new


def _write_i32(path: Path, values: list[int]) -> None:
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


class EngineExecutor:
    def __init__(self, route_manifest: str | Path):
        self.route_manifest = Path(route_manifest)
        self.store = routing.AtomicRouteManifestStore(self.route_manifest)
        self.gpu_lock = threading.BoundedSemaphore(1)

    def read_route_snapshot(self) -> dict:
        manifest = self.store.read()
        route = routing.normalize_route(manifest["active_route"])
        _route_policy_line(route)
        return {
            "generation": int(manifest["generation"]),
            "manifest_sha256": str(manifest["manifest_sha256"]),
            "route": route,
        }

    def generate_batch(self, requests: list[dict]) -> dict:
        if not isinstance(requests, list) or not requests:
            raise SupervisorError("requests must be a non-empty list")
        if len(requests) > MAX_BATCH_REQUESTS:
            raise SupervisorError("batch exceeds supervisor request limit")
        parsed = [_validate_request(row) for row in requests]
        snapshot = self.read_route_snapshot()

        if not self.gpu_lock.acquire(blocking=False):
            raise SupervisorError("GPU worker is busy")
        started_ns = time.monotonic_ns()
        before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            with tempfile.TemporaryDirectory(prefix="beglin-supervisor-") as td:
                root = Path(td)
                manifest_path = root / "manifest.txt"
                promo_path = root / "promotion_nq.txt"
                ack_path = root / "applied_ack.json"
                txn_path = root / "txn.cmd"

                lines = []
                for idx, (tokens, max_new) in enumerate(parsed):
                    raw = root / f"req-{idx}.i32"
                    _write_i32(raw, tokens)
                    lines.append(f"{raw} {max_new}\n")
                manifest_path.write_text("".join(lines))
                promo_path.write_text(_route_policy_line(snapshot["route"]))

                env = _minimal_env()
                env.update(
                    {
                        "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
                        "QWEN_MOE_BASE": str(MOE_BASE),
                        "QWEN_MOE_NEARTIE_CORRECT": "0",
                        "QWEN_MOE_CB_PROMPT_MANIFEST": str(manifest_path),
                        "QWEN_MOE_CB_SLOTS": str(min(4, len(parsed))),
                        "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
                        "QWEN_MOE_GPU_APPLIED_ACK": str(ack_path),
                        "QWEN_MOE_GPU_TXN_FILE": str(txn_path),
                        "QWEN_MOE_PROMOTION_FILE_NQ": str(promo_path),
                        "QWEN_MOE_PROMOTION_SAFETENSORS": str(SAFETENSORS),
                    }
                )
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
                    output, _ = proc.communicate(timeout=WORKER_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        output, _ = proc.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        output, _ = proc.communicate(timeout=5)
                    raise SupervisorError(
                        f"engine worker timed out and was stopped: pid={worker_pid}"
                    )
                if proc.returncode != 0:
                    raise SupervisorError(
                        f"engine worker failed rc={proc.returncode}"
                    )
                if not ack_path.is_file():
                    raise SupervisorError("engine worker ACK is missing")

                sys.path.insert(0, str(REPO / "tools"))
                import gpu_runtime_control as grc
                ack = grc.read_runtime_ack(ack_path)
                if ack["active_policy_hash"] != snapshot["route"]["policy_hash"]:
                    raise SupervisorError("worker self-report policy hash mismatch")

                generated = _parse_generated(output)
                if len(generated) < len(parsed):
                    raise SupervisorError(
                        f"worker response count mismatch: expected_at_least={len(parsed)} "
                        f"actual={len(generated)}"
                    )
                validation = VALIDATION_RE.search(output)
                if validation is None or validation.group("finite") != "1":
                    raise SupervisorError("worker validation report failed")
                validation_requests = int(validation.group("requests"))
                if validation_requests < len(parsed):
                    raise SupervisorError(
                        "worker validation request count is below admitted API request count"
                    )
                after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                duration_ms = max(
                    1, (time.monotonic_ns() - started_ns) // 1_000_000
                )
                return {
                    "schema": "beglin-supervisor-batch-response-v1",
                    "status": "OK",
                    "route_generation": snapshot["generation"],
                    "route_manifest_sha256": snapshot["manifest_sha256"],
                    "route": snapshot["route"],
                    "worker_instance_id": f"pid-{worker_pid}",
                    "worker_ack_sha256": ack["ack_sha256"],
                    "worker_epoch": int(ack["weight_epoch"]),
                    "finite_logits": True,
                    "duration_ms": duration_ms,
                    "peak_child_rss_bytes": int(after_usage.ru_maxrss),
                    "engine_requests_completed": len(generated),
                    "engine_validation_requests": validation_requests,
                    "engine_cycles_manifest_when_underfilled": len(generated) > len(parsed),
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
        finally:
            self.gpu_lock.release()


class SupervisorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_cls, executor: EngineExecutor):
        super().__init__(server_address, handler_cls)
        self.executor = executor
        self.started_at = time.time()


class Handler(BaseHTTPRequestHandler):
    server_version = "BeglinSupervisor/1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(
            json.dumps(
                {
                    "kind": "http_access",
                    "client": self.client_address[0],
                    "message": fmt % args,
                },
                sort_keys=True,
            )
            + "\n"
        )

    def _send(self, status: int, value: Any) -> None:
        payload = (json.dumps(value, sort_keys=True) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        try:
            if self.path == "/healthz":
                snap = self.server.executor.read_route_snapshot()
                self._send(
                    HTTPStatus.OK,
                    {
                        "schema": "beglin-supervisor-health-v1",
                        "status": "ok",
                        "router_id": "beglin-local-http-supervisor-v1",
                        "listen_host": self.server.server_address[0],
                        "listen_port": self.server.server_address[1],
                        "route_generation": snap["generation"],
                        "active_route": snap["route"],
                        "route_manifest_sha256": snap["manifest_sha256"],
                        "uptime_seconds": int(time.time() - self.server.started_at),
                        "external_network_exposed": False,
                        "production_write_allowed": False,
                        "auto_promotion_enabled": False,
                    },
                )
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            self._send(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": type(exc).__name__, "message": str(exc)},
            )

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise SupervisorError("invalid request body size")
            body = self.rfile.read(length)
            value = json.loads(body)
            if self.path == "/v1/generate":
                batch = [value]
            elif self.path == "/v1/batch_generate":
                if not isinstance(value, dict):
                    raise SupervisorError("batch body must be an object")
                batch = value.get("requests")
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            result = self.server.executor.generate_batch(batch)
            self._send(HTTPStatus.OK, result)
        except json.JSONDecodeError:
            self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
        except SupervisorError as exc:
            code = (
                HTTPStatus.SERVICE_UNAVAILABLE
                if "busy" in str(exc).lower()
                else HTTPStatus.BAD_REQUEST
            )
            self._send(code, {"error": "supervisor_error", "message": str(exc)})
        except Exception as exc:
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": type(exc).__name__, "message": str(exc)},
            )


def ensure_route_manifest(path: Path) -> dict:
    store = routing.AtomicRouteManifestStore(path)
    if path.exists():
        current = store.read()
        _route_policy_line(current["active_route"])
        return current
    path.parent.mkdir(parents=True, exist_ok=True)
    return store.initialize(baseline_route())


def make_server(
    *,
    route_manifest: str | Path,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
) -> SupervisorHTTPServer:
    if host != "127.0.0.1":
        raise SupervisorError("supervisor currently refuses non-loopback binding")
    verify_runtime_identity()
    manifest = Path(route_manifest)
    ensure_route_manifest(manifest)
    executor = EngineExecutor(manifest)
    return SupervisorHTTPServer((host, int(port)), Handler, executor)


def _post_json(port: int, path: str, value: dict) -> dict:
    payload = json.dumps(value).encode()
    req = urlrequest.Request(
        f"http://127.0.0.1:{port}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=WORKER_TIMEOUT_SECONDS + 5) as resp:
            return json.loads(resp.read())
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SupervisorError(
            f"HTTP self-test request failed status={exc.code} body={body}"
        ) from exc


def _get_json(port: int, path: str) -> dict:
    with urlrequest.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=5
    ) as resp:
        return json.loads(resp.read())


def _read_first_certified_prompt() -> list[int]:
    source = REPO / "g5_drill_manifest.txt"
    rows = [
        line.strip()
        for line in source.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise SupervisorError("certified prompt manifest is empty")
    raw_path = Path(rows[0].split()[0])
    data = raw_path.read_bytes()
    if len(data) % 4:
        raise SupervisorError("certified prompt token file is not int32-aligned")
    return [
        struct.unpack_from("<i", data, offset)[0]
        for offset in range(0, len(data), 4)
    ]


def self_test() -> dict:
    verify_runtime_identity()
    tokens = _read_first_certified_prompt()
    with tempfile.TemporaryDirectory(prefix="beglin-supervisor-selftest-") as td:
        root = Path(td)
        route_path = root / "active-route.json"
        store = routing.AtomicRouteManifestStore(route_path)
        initial = store.initialize(baseline_route())
        server = make_server(route_manifest=route_path, port=0)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            health1 = _get_json(port, "/healthz")
            baseline = _post_json(
                port,
                "/v1/generate",
                {"prompt_tokens": tokens, "max_new_tokens": 10},
            )
            b_tokens = baseline["responses"][0]["generated_tokens"]
            if len(b_tokens) <= 8 or int(b_tokens[8]) != 3268:
                raise SupervisorError("baseline reference token mismatch")

            after_cutover = store.compare_and_swap(
                expected_generation=int(initial["generation"]),
                expected_route_id=baseline_route()["route_id"],
                new_route=candidate_route(),
                reason="supervisor localhost self-test candidate",
                authorization_digest="a" * 64,
            )
            candidate = _post_json(
                port,
                "/v1/generate",
                {"prompt_tokens": tokens, "max_new_tokens": 10},
            )
            c_tokens = candidate["responses"][0]["generated_tokens"]
            if len(c_tokens) <= 8 or int(c_tokens[8]) != 1224:
                raise SupervisorError("candidate reference token mismatch")

            after_rollback = store.compare_and_swap(
                expected_generation=int(after_cutover["generation"]),
                expected_route_id=candidate_route()["route_id"],
                new_route=baseline_route(),
                reason="supervisor localhost self-test rollback",
                authorization_digest="b" * 64,
            )
            restored = _post_json(
                port,
                "/v1/generate",
                {"prompt_tokens": tokens, "max_new_tokens": 10},
            )
            r_tokens = restored["responses"][0]["generated_tokens"]
            if len(r_tokens) <= 8 or int(r_tokens[8]) != 3268:
                raise SupervisorError("rollback reference token mismatch")
            health2 = _get_json(port, "/healthz")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        out = {
            "schema": "beglin-serving-supervisor-selftest-v1",
            "status": "PASS",
            "listen_host": "127.0.0.1",
            "listen_port": port,
            "external_network_exposed": False,
            "route_generation_path": [
                int(initial["generation"]),
                int(after_cutover["generation"]),
                int(after_rollback["generation"]),
            ],
            "baseline_reference_token": int(b_tokens[8]),
            "candidate_reference_token": int(c_tokens[8]),
            "rollback_reference_token": int(r_tokens[8]),
            "baseline_route_id": baseline["route"]["route_id"],
            "candidate_route_id": candidate["route"]["route_id"],
            "restored_route_id": restored["route"]["route_id"],
            "health_before": health1["status"],
            "health_after": health2["status"],
            "production_write_allowed": False,
            "auto_promotion_enabled": False,
            "network_ingress_verified": True,
        }
        out["result_sha256"] = routing.sha256_json(out)
        return out


def install_user_launchd(
    *,
    script_path: Path,
    route_manifest: Path,
    port: int,
) -> dict:
    if platform.system() != "Darwin":
        raise SupervisorError("launchd installation requires Darwin")
    state_root = route_manifest.parent
    state_root.mkdir(parents=True, exist_ok=True)
    ensure_route_manifest(route_manifest)
    logs = state_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    label = "com.beglin.production-supervisor"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    program = str(Path(sys.executable).resolve())
    script = str(script_path.resolve(strict=True))
    route = str(route_manifest.resolve())
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>{label}</string>
<key>ProgramArguments</key><array>
<string>{program}</string>
<string>{script}</string>
<string>--serve</string>
<string>--route-manifest</string><string>{route}</string>
<string>--port</string><string>{int(port)}</string>
</array>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
<key>ProcessType</key><string>Background</string>
<key>StandardOutPath</key><string>{logs / "supervisor.stdout.log"}</string>
<key>StandardErrorPath</key><string>{logs / "supervisor.stderr.log"}</string>
</dict></plist>
"""
    tmp = plist.with_name(plist.name + f".tmp.{os.getpid()}")
    tmp.write_text(xml)
    os.replace(tmp, plist)
    uid = os.getuid()
    domain = f"gui/{uid}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    proc = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SupervisorError(
            f"launchctl bootstrap failed: {proc.stderr.strip()}"
        )
    return {
        "schema": "beglin-supervisor-launchd-install-v1",
        "status": "INSTALLED",
        "label": label,
        "plist": str(plist),
        "python": program,
        "script": script,
        "route_manifest": route,
        "listen_host": "127.0.0.1",
        "listen_port": int(port),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--install-user-launchd", action="store_true")
    mode.add_argument("--probe-health", action="store_true")
    mode.add_argument("--probe-reference", action="store_true")
    mode.add_argument("--probe-batch-reference", action="store_true")
    ap.add_argument(
        "--route-manifest",
        default=str(DEFAULT_ROUTE_MANIFEST),
    )
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    if args.probe_health:
        print(json.dumps(_get_json(args.port, "/healthz"), indent=2, sort_keys=True))
        return 0
    if args.probe_reference:
        tokens = _read_first_certified_prompt()
        result = _post_json(
            args.port,
            "/v1/generate",
            {"prompt_tokens": tokens, "max_new_tokens": 10},
        )
        generated = result["responses"][0]["generated_tokens"]
        print(json.dumps({
            "schema": "beglin-supervisor-reference-probe-v1",
            "status": "PASS",
            "route_generation": result["route_generation"],
            "route_id": result["route"]["route_id"],
            "policy_hash": result["route"]["policy_hash"],
            "reference_token_at_gen_idx_8": generated[8] if len(generated) > 8 else None,
            "finite_logits": result["finite_logits"],
            "duration_ms": result["duration_ms"],
            "worker_instance_id": result["worker_instance_id"],
        }, indent=2, sort_keys=True))
        return 0
    if args.probe_batch_reference:
        tokens = _read_first_certified_prompt()
        result = _post_json(
            args.port,
            "/v1/batch_generate",
            {
                "requests": [
                    {"prompt_tokens": tokens, "max_new_tokens": 10}
                    for _ in range(12)
                ]
            },
        )
        values = []
        for row in result["responses"]:
            generated = row["generated_tokens"]
            values.append(generated[8] if len(generated) > 8 else None)
        route_n = result["route"].get("n")
        expected = 3268 if route_n is None else 1224
        matched = sum(value == expected for value in values)
        status = "PASS" if matched == 12 and result["finite_logits"] else "FAIL"
        print(json.dumps({
            "schema": "beglin-supervisor-batch-reference-probe-v1",
            "status": status,
            "route_generation": result["route_generation"],
            "route_id": result["route"]["route_id"],
            "policy_hash": result["route"]["policy_hash"],
            "route_n": route_n,
            "expected_reference_token": expected,
            "reference_hits": matched,
            "requests": len(result["responses"]),
            "finite_logits": result["finite_logits"],
            "duration_ms": result["duration_ms"],
            "peak_child_rss_bytes": result["peak_child_rss_bytes"],
            "worker_instance_id": result["worker_instance_id"],
        }, indent=2, sort_keys=True))
        return 0 if status == "PASS" else 2
    if args.install_user_launchd:
        result = install_user_launchd(
            script_path=Path(__file__),
            route_manifest=Path(args.route_manifest),
            port=args.port,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    server = make_server(
        route_manifest=args.route_manifest,
        host="127.0.0.1",
        port=args.port,
    )
    capability = router_capability(
        route_manifest=args.route_manifest,
        port=args.port,
    )
    print(
        json.dumps(
            {
                "schema": "beglin-serving-supervisor-start-v1",
                "status": "SERVING",
                "capability": capability,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
