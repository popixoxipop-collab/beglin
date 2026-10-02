#!/usr/bin/env python3
"""Persistent Beglin serving supervisor.

Reuses the certified localhost HTTP contract/handler and atomic route manifest
from production_serving_supervisor.py, but serves requests through two resident
GPU workers: the exact baseline and the reviewed shared_up_proj/L3 n=6 candidate.

A single cross-worker GPU admission semaphore prevents baseline/candidate GPU
work from overlapping. Route selection is read from the existing atomic route
manifest at request admission; rollback is therefore only a route CAS, not a
model reload.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import threading
import time

import production_routing_cutover as routing
import production_serving_supervisor as legacy
from production_persistent_worker import (
    DEFAULT_BINARY,
    PersistentWorkerError,
    PersistentWorkerPool,
)


STATE_ROOT = Path("/Users/xox/vdsp_serving")
ROUTE_MANIFEST = STATE_ROOT / "active-route.json"
WORKER_ROOT = STATE_ROOT / "persistent-workers"
ADMISSION_LOCK = STATE_ROOT / "admission.lock"
DEFAULT_PORT = 18765


class PersistentEngineExecutor:
    def __init__(self, route_manifest: str | Path, pool: PersistentWorkerPool):
        self.route_manifest = Path(route_manifest)
        self.store = routing.AtomicRouteManifestStore(self.route_manifest)
        self.pool = pool
        ADMISSION_LOCK.parent.mkdir(parents=True, exist_ok=True)
        self._admission_handle = ADMISSION_LOCK.open("a+")

    def read_route_snapshot(self) -> dict:
        manifest = self.store.read()
        route = routing.normalize_route(manifest["active_route"])
        legacy._route_policy_line(route)
        return {
            "generation": int(manifest["generation"]),
            "manifest_sha256": str(manifest["manifest_sha256"]),
            "route": route,
        }

    def generate_batch(self, requests: list[dict]) -> dict:
        if not isinstance(requests, list) or not requests:
            raise legacy.SupervisorError("requests must be a non-empty list")
        if len(requests) > legacy.MAX_BATCH_REQUESTS:
            raise legacy.SupervisorError("batch exceeds supervisor request limit")
        parsed = []
        for request in requests:
            tokens, max_new = legacy._validate_request(request)
            parsed.append({"prompt_tokens": tokens, "max_new_tokens": max_new})

        fcntl.flock(self._admission_handle.fileno(), fcntl.LOCK_EX)
        try:
            snapshot = self.read_route_snapshot()
            persistent = self.pool.submit(snapshot["route"]["route_id"], parsed)
        except PersistentWorkerError as exc:
            raise legacy.SupervisorError(str(exc)) from exc
        finally:
            fcntl.flock(self._admission_handle.fileno(), fcntl.LOCK_UN)

        worker_health = self.pool.health()[snapshot["route"]["route_id"]]
        return {
            "schema": "beglin-supervisor-batch-response-v1",
            "status": "OK",
            "execution_mode": "persistent-worker",
            "route_generation": snapshot["generation"],
            "route_manifest_sha256": snapshot["manifest_sha256"],
            "route": snapshot["route"],
            "worker_instance_id": f"pid-{persistent['worker_pid']}",
            "worker_ack_sha256": worker_health["ack_sha256"],
            "worker_epoch": int(worker_health["weight_epoch"]),
            "finite_logits": True,
            "duration_ms": float(persistent["duration_ms"]),
            "roundtrip_ms": float(persistent["roundtrip_ms"]),
            "peak_child_rss_bytes": int(persistent["worker_rss_bytes"]),
            "responses": [
                {
                    "request_index": int(row["request_index"]),
                    "generated_tokens": list(row["generated_tokens"]),
                }
                for row in persistent["responses"]
            ],
            "production_write_allowed": False,
            "auto_promotion_enabled": False,
        }

    def close(self) -> None:
        try:
            self._admission_handle.close()
        except Exception:
            pass


class PersistentSupervisorHTTPServer(legacy.SupervisorHTTPServer):
    def __init__(self, server_address, handler_cls, executor, pool):
        super().__init__(server_address, handler_cls, executor)
        self.pool = pool

    def server_close(self) -> None:
        try:
            self.pool.stop_all()
        finally:
            try:
                self.executor.close()
            finally:
                super().server_close()


def build_pool(*, binary: str | Path = DEFAULT_BINARY) -> PersistentWorkerPool:
    pool = PersistentWorkerPool(root=WORKER_ROOT, binary=binary)
    pool.add(name="baseline", route=legacy.baseline_route())
    pool.add(name="candidate", route=legacy.candidate_route())
    return pool


def start_pool(*, binary: str | Path = DEFAULT_BINARY) -> tuple[PersistentWorkerPool, dict]:
    pool = build_pool(binary=binary)
    status = pool.start_all()
    prompt = legacy._read_first_certified_prompt()
    warm = pool.warm_all(
        prompt_tokens=prompt,
        expected_by_route={
            legacy.baseline_route()["route_id"]: 3268,
            legacy.candidate_route()["route_id"]: 1224,
        },
    )
    return pool, {"startup": status, "warmup": warm}


def make_server(
    *,
    route_manifest: str | Path = ROUTE_MANIFEST,
    binary: str | Path = DEFAULT_BINARY,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
):
    if host != "127.0.0.1":
        raise legacy.SupervisorError("persistent supervisor refuses non-loopback binding")
    legacy.verify_runtime_identity()
    legacy.ensure_route_manifest(Path(route_manifest))
    pool, evidence = start_pool(binary=binary)
    executor = PersistentEngineExecutor(route_manifest, pool)
    server = PersistentSupervisorHTTPServer(
        (host, int(port)),
        legacy.Handler,
        executor,
        pool,
    )
    server.persistent_startup_evidence = evidence
    return server


def persistent_health(port: int) -> dict:
    health = legacy._get_json(port, "/healthz")
    return {
        "schema": "beglin-persistent-supervisor-health-v1",
        "status": health["status"],
        "router_id": health["router_id"],
        "listen_host": health["listen_host"],
        "listen_port": health["listen_port"],
        "route_generation": health["route_generation"],
        "active_route": health["active_route"],
        "route_manifest_sha256": health["route_manifest_sha256"],
        "external_network_exposed": False,
        "execution_mode": "persistent-worker",
    }


def self_test(*, binary: str | Path) -> dict:
    legacy.verify_runtime_identity()
    prompt = legacy._read_first_certified_prompt()
    import tempfile
    with tempfile.TemporaryDirectory(prefix="beglin-persistent-supervisor-test-") as td:
        root = Path(td)
        route_path = root / "active-route.json"
        store = routing.AtomicRouteManifestStore(route_path)
        initial = store.initialize(legacy.baseline_route())
        pool = PersistentWorkerPool(root=root / "workers", binary=binary)
        pool.add(name="baseline", route=legacy.baseline_route())
        pool.add(name="candidate", route=legacy.candidate_route())
        pool.start_all()
        warm = pool.warm_all(
            prompt_tokens=prompt,
            expected_by_route={
                legacy.baseline_route()["route_id"]: 3268,
                legacy.candidate_route()["route_id"]: 1224,
            },
        )
        executor = PersistentEngineExecutor(route_path, pool)
        server = PersistentSupervisorHTTPServer(("127.0.0.1", 0), legacy.Handler, executor, pool)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            b = legacy._post_json(port, "/v1/generate", {"prompt_tokens": prompt, "max_new_tokens": 10})
            bt = b["responses"][0]["generated_tokens"][8]
            switched = store.compare_and_swap(
                expected_generation=int(initial["generation"]),
                expected_route_id=legacy.baseline_route()["route_id"],
                new_route=legacy.candidate_route(),
                reason="persistent supervisor self-test",
                authorization_digest="a" * 64,
            )
            c = legacy._post_json(port, "/v1/generate", {"prompt_tokens": prompt, "max_new_tokens": 10})
            ct = c["responses"][0]["generated_tokens"][8]
            rolled = store.compare_and_swap(
                expected_generation=int(switched["generation"]),
                expected_route_id=legacy.candidate_route()["route_id"],
                new_route=legacy.baseline_route(),
                reason="persistent supervisor self-test rollback",
                authorization_digest="b" * 64,
            )
            rb = legacy._post_json(port, "/v1/generate", {"prompt_tokens": prompt, "max_new_tokens": 10})
            rt = rb["responses"][0]["generated_tokens"][8]
            if (bt, ct, rt) != (3268, 1224, 3268):
                raise legacy.SupervisorError(f"persistent self-test token mismatch {(bt,ct,rt)}")
            out = {
                "schema": "beglin-persistent-supervisor-selftest-v1",
                "status": "PASS",
                "route_generation_path": [initial["generation"], switched["generation"], rolled["generation"]],
                "baseline_token": bt,
                "candidate_token": ct,
                "rollback_token": rt,
                "baseline_roundtrip_ms": b.get("roundtrip_ms"),
                "candidate_roundtrip_ms": c.get("roundtrip_ms"),
                "rollback_roundtrip_ms": rb.get("roundtrip_ms"),
                "warmup": warm,
                "same_baseline_pid_after_rollback": b["worker_instance_id"] == rb["worker_instance_id"],
                "candidate_pid": c["worker_instance_id"],
                "external_network_exposed": False,
                "production_write_allowed": False,
                "auto_promotion_enabled": False,
            }
            out["result_sha256"] = routing.sha256_json(out)
            return out
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def install_launchd(
    *,
    script_path: Path,
    binary: Path,
    route_manifest: Path,
    port: int,
) -> dict:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    legacy.ensure_route_manifest(route_manifest)
    logs = STATE_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    label = "com.beglin.production-supervisor"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    python = str(Path(os.sys.executable).resolve())
    script = str(script_path.resolve(strict=True))
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>{label}</string>
<key>ProgramArguments</key><array>
<string>{python}</string><string>{script}</string><string>--serve</string>
<string>--binary</string><string>{binary}</string>
<string>--route-manifest</string><string>{route_manifest}</string>
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
    import subprocess
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    proc = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if proc.returncode != 0:
        raise legacy.SupervisorError(f"launchctl bootstrap failed: {proc.stderr.strip()}")
    return {
        "schema": "beglin-persistent-supervisor-launchd-install-v1",
        "status": "INSTALLED",
        "label": label,
        "plist": str(plist),
        "script": script,
        "binary": str(binary),
        "route_manifest": str(route_manifest),
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
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--route-manifest", default=str(ROUTE_MANIFEST))
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    if args.self_test:
        print(json.dumps(self_test(binary=args.binary), indent=2, sort_keys=True))
        return 0
    if args.install_user_launchd:
        print(json.dumps(install_launchd(
            script_path=Path(__file__),
            binary=Path(args.binary),
            route_manifest=Path(args.route_manifest),
            port=args.port,
        ), indent=2, sort_keys=True))
        return 0
    if args.probe_health:
        print(json.dumps(persistent_health(args.port), indent=2, sort_keys=True))
        return 0
    if args.probe_reference:
        prompt = legacy._read_first_certified_prompt()
        result = legacy._post_json(
            args.port, "/v1/generate",
            {"prompt_tokens": prompt, "max_new_tokens": 10},
        )
        tokens = result["responses"][0]["generated_tokens"]
        print(json.dumps({
            "schema": "beglin-persistent-supervisor-reference-probe-v1",
            "status": "PASS",
            "execution_mode": result.get("execution_mode"),
            "route_generation": result["route_generation"],
            "route_id": result["route"]["route_id"],
            "reference_token_at_gen_idx_8": tokens[8] if len(tokens)>8 else None,
            "duration_ms": result["duration_ms"],
            "roundtrip_ms": result.get("roundtrip_ms"),
            "worker_instance_id": result["worker_instance_id"],
        }, indent=2, sort_keys=True))
        return 0

    server = make_server(
        route_manifest=args.route_manifest,
        binary=args.binary,
        port=args.port,
    )
    print(json.dumps({
        "schema": "beglin-persistent-supervisor-start-v1",
        "status": "SERVING",
        "listen_host": "127.0.0.1",
        "listen_port": int(server.server_address[1]),
        "route_manifest": str(args.route_manifest),
        "binary": str(args.binary),
        "startup": server.persistent_startup_evidence,
        "external_network_exposed": False,
        "auto_promotion_enabled": False,
    }, sort_keys=True), flush=True)
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
