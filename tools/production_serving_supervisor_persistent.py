#!/usr/bin/env python3
"""Beglin persistent-worker serving supervisor.

Runs alongside the existing spawn-per-admission supervisor until certified.
It keeps one baseline and one candidate MLX worker alive, warmed, and isolated.
The active atomic route manifest selects which persistent worker receives each
HTTP batch. No public/external bind is supported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import persistent_gpu_worker as pgw
import production_routing_cutover as routing
import production_serving_supervisor as legacy


PERSISTENT_REPO=Path("/Users/xox/vdsp-engine-persistent-serving")
PERSISTENT_BINARY=PERSISTENT_REPO/"build-gpu-persistent/qwen_infer_gpu"
BUILD_STATUS=PERSISTENT_REPO/"PERSISTENT_GPU_BUILD_2026-10-02.txt"
STATE_ROOT=Path("/Users/xox/vdsp_serving_persistent")
DEFAULT_ROUTE_MANIFEST=STATE_ROOT/"active-route.json"
DEFAULT_PORT=18766

MOE_BASE=legacy.MOE_BASE
SAFETENSORS=legacy.SAFETENSORS
CHECKPOINT_EVIDENCE=legacy.CHECKPOINT_EVIDENCE
BASELINE_POLICY_HASH=legacy.BASELINE_POLICY_HASH
CANDIDATE_POLICY_HASH=legacy.CANDIDATE_POLICY_HASH


class PersistentSupervisorError(RuntimeError):
    pass


def read_build_identity(path: Path=BUILD_STATUS) -> dict:
    if not path.is_file():
        raise PersistentSupervisorError(f"persistent build status missing: {path}")
    values={}
    for line in path.read_text().splitlines():
        if "=" in line:
            key,value=line.split("=",1)
            values[key.strip()]=value.strip()
    required=("status","source_commit","qwen_infer_c_sha256","binary_path","binary_sha256")
    missing=[key for key in required if not values.get(key)]
    if missing:
        raise PersistentSupervisorError(f"persistent build status missing fields: {missing}")
    if values["status"]!="BUILT":
        raise PersistentSupervisorError("persistent build is not BUILT")
    binary=Path(values["binary_path"])
    if binary.resolve()!=PERSISTENT_BINARY.resolve():
        raise PersistentSupervisorError("persistent binary path mismatch")
    if pgw.sha256_file(binary)!=values["binary_sha256"]:
        raise PersistentSupervisorError("persistent binary SHA mismatch")
    if values.get("production_binary_replaced")!="false":
        raise PersistentSupervisorError("persistent build unexpectedly replaced production binary")
    return values


def verify_checkpoint() -> dict:
    obj=json.loads(CHECKPOINT_EVIDENCE.read_text())
    if obj.get("status")!="VERIFIED":
        raise PersistentSupervisorError("checkpoint evidence is not VERIFIED")
    if obj.get("checkpoint_sha256")!=legacy.EXPECTED_CHECKPOINT_SHA:
        raise PersistentSupervisorError("checkpoint SHA mismatch")
    return obj


def persistent_route(*, candidate: bool, build: dict) -> dict:
    if candidate:
        return {
            "route_id":"beglin-candidate-shared-up-l3-n6-persistent-v1",
            "worker_instance_id":"persistent-candidate-v1",
            "endpoint":"local-supervisor://persistent/candidate",
            "source_commit":build["source_commit"],
            "binary_sha256":build["binary_sha256"],
            "checkpoint_sha256":legacy.EXPECTED_CHECKPOINT_SHA,
            "policy_hash":CANDIDATE_POLICY_HASH,
            "role":"shared_up_proj","layer":3,"n":6,
        }
    return {
        "route_id":"beglin-baseline-q4g64-persistent-v1",
        "worker_instance_id":"persistent-baseline-v1",
        "endpoint":"local-supervisor://persistent/baseline",
        "source_commit":build["source_commit"],
        "binary_sha256":build["binary_sha256"],
        "checkpoint_sha256":legacy.EXPECTED_CHECKPOINT_SHA,
        "policy_hash":BASELINE_POLICY_HASH,
        "role":"shared_up_proj","layer":3,"n":None,
    }


class PersistentEngineExecutor:
    def __init__(self, route_manifest: str|Path, *, state_root: Path=STATE_ROOT):
        self.route_manifest=Path(route_manifest)
        self.store=routing.AtomicRouteManifestStore(self.route_manifest)
        self.state_root=Path(state_root)
        self.build=read_build_identity()
        verify_checkpoint()
        self.gpu_lock=threading.BoundedSemaphore(1)
        tools=PERSISTENT_REPO/"tools"
        common={
            "binary":PERSISTENT_BINARY,
            "repo":PERSISTENT_REPO,
            "moe_base":MOE_BASE,
            "safetensors":SAFETENSORS,
            "tools_dir":tools,
            "slots":4,
            "startup_timeout":180.0,
            "request_timeout":90.0,
        }
        baseline=persistent_route(candidate=False,build=self.build)
        candidate=persistent_route(candidate=True,build=self.build)
        self.routes={
            baseline["route_id"]:baseline,
            candidate["route_id"]:candidate,
        }
        self.pool=pgw.PersistentWorkerPool(workers={
            baseline["route_id"]:pgw.PersistentGpuWorker(
                name="baseline",
                state_dir=self.state_root/"workers/baseline",
                promotion_line="",
                expected_policy_hash=BASELINE_POLICY_HASH,
                **common,
            ),
            candidate["route_id"]:pgw.PersistentGpuWorker(
                name="candidate",
                state_dir=self.state_root/"workers/candidate",
                promotion_line="shared_up_proj 3 6\n",
                expected_policy_hash=CANDIDATE_POLICY_HASH,
                **common,
            ),
        })
        self.start_evidence=self.pool.start_all()
        prompt=legacy._read_first_certified_prompt()
        self.warmup={}
        for route_id in (baseline["route_id"],candidate["route_id"]):
            self.warmup[route_id]=self.pool.get(route_id).submit([
                {"prompt_tokens":prompt,"max_new_tokens":10}
            ])

    def close(self) -> None:
        self.pool.stop_all()

    def read_route_snapshot(self) -> dict:
        manifest=self.store.read()
        route=routing.normalize_route(manifest["active_route"])
        if route["route_id"] not in self.routes:
            raise PersistentSupervisorError(
                f"active route is not backed by persistent pool: {route['route_id']}"
            )
        expected=self.routes[route["route_id"]]
        if route!=expected:
            raise PersistentSupervisorError("active persistent route identity mismatch")
        worker=self.pool.get(route["route_id"])
        st=worker.status()
        if not st["alive"]:
            raise PersistentSupervisorError("active persistent worker is not alive")
        return {
            "generation":int(manifest["generation"]),
            "manifest_sha256":str(manifest["manifest_sha256"]),
            "route":route,
            "persistent_worker":st,
        }

    def generate_batch(self, requests: list[dict]) -> dict:
        parsed=pgw.validate_batch(requests,max_requests=12)
        snapshot=self.read_route_snapshot()
        if not self.gpu_lock.acquire(blocking=False):
            raise legacy.SupervisorError("GPU worker is busy")
        started=time.monotonic()
        try:
            result=self.pool.get(snapshot["route"]["route_id"]).submit(requests)
            return {
                "schema":"beglin-supervisor-batch-response-v1",
                "status":"OK",
                "route_generation":snapshot["generation"],
                "route_manifest_sha256":snapshot["manifest_sha256"],
                "route":snapshot["route"],
                "worker_instance_id":f"pid-{result['pid']}",
                "worker_epoch":result["weight_epoch"],
                "finite_logits":result["finite_logits"],
                "duration_ms":result["roundtrip_duration_ms"],
                "engine_duration_ms":result["engine_duration_ms"],
                "persistent_worker_reused":True,
                "persistent_worker_seq":result["seq"],
                "responses":result["responses"],
                "production_write_allowed":False,
                "auto_promotion_enabled":False,
            }
        finally:
            self.gpu_lock.release()


class PersistentHTTPServer(legacy.SupervisorHTTPServer):
    def server_close(self):
        try:
            if hasattr(self.executor,"close"):
                self.executor.close()
        finally:
            super().server_close()


class PersistentHandler(legacy.Handler):
    def do_GET(self) -> None:
        if self.path!="/healthz":
            return super().do_GET()
        try:
            snap=self.server.executor.read_route_snapshot()
            workers={
                route_id:worker.status()
                for route_id,worker in self.server.executor.pool.workers.items()
            }
            self._send(200,{
                "schema":"beglin-persistent-supervisor-health-v1",
                "status":"ok",
                "router_id":"beglin-local-persistent-supervisor-v1",
                "listen_host":self.server.server_address[0],
                "listen_port":self.server.server_address[1],
                "route_generation":snap["generation"],
                "active_route":snap["route"],
                "route_manifest_sha256":snap["manifest_sha256"],
                "persistent_workers":workers,
                "warmup":self.server.executor.warmup,
                "external_network_exposed":False,
                "production_write_allowed":False,
                "auto_promotion_enabled":False,
            })
        except Exception as exc:
            self._send(503,{"error":type(exc).__name__,"message":str(exc)})


def ensure_manifest(path: Path, *, build: dict, candidate: bool=True) -> dict:
    store=routing.AtomicRouteManifestStore(path)
    route=persistent_route(candidate=candidate,build=build)
    if path.exists():
        current=store.read()
        if current["active_route"]!=route:
            raise PersistentSupervisorError(
                "existing persistent route manifest does not match requested initial route"
            )
        return current
    path.parent.mkdir(parents=True,exist_ok=True)
    return store.initialize(route)


def make_server(*, route_manifest: str|Path, host: str="127.0.0.1", port: int=DEFAULT_PORT) -> PersistentHTTPServer:
    if host!="127.0.0.1":
        raise PersistentSupervisorError("persistent supervisor refuses non-loopback binding")
    build=read_build_identity()
    ensure_manifest(Path(route_manifest),build=build,candidate=True)
    executor=PersistentEngineExecutor(route_manifest)
    return PersistentHTTPServer((host,int(port)),PersistentHandler,executor)


def self_test() -> dict:
    build=read_build_identity()
    prompt=legacy._read_first_certified_prompt()
    import tempfile
    with tempfile.TemporaryDirectory(prefix="beglin-persistent-supervisor-") as td:
        root=Path(td)
        route=root/"active-route.json"
        store=routing.AtomicRouteManifestStore(route)
        candidate=persistent_route(candidate=True,build=build)
        baseline=persistent_route(candidate=False,build=build)
        first=store.initialize(candidate)
        server=make_server(route_manifest=route,port=0)
        port=int(server.server_address[1])
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            r1=legacy._post_json(port,"/v1/generate",{"prompt_tokens":prompt,"max_new_tokens":10})
            r2=legacy._post_json(port,"/v1/generate",{"prompt_tokens":prompt,"max_new_tokens":10})
            t1=r1["responses"][0]["generated_tokens"]
            t2=r2["responses"][0]["generated_tokens"]
            if t1[8]!=1224 or t2[8]!=1224:
                raise PersistentSupervisorError("candidate persistent reference mismatch")
            if r1["worker_instance_id"]!=r2["worker_instance_id"]:
                raise PersistentSupervisorError("candidate PID changed across requests")

            switched=store.compare_and_swap(
                expected_generation=first["generation"],
                expected_route_id=candidate["route_id"],
                new_route=baseline,
                reason="persistent selftest baseline",
                authorization_digest="c"*64,
            )
            rb=legacy._post_json(port,"/v1/generate",{"prompt_tokens":prompt,"max_new_tokens":10})
            bt=rb["responses"][0]["generated_tokens"]
            if bt[8]!=3268:
                raise PersistentSupervisorError("baseline persistent reference mismatch")
            store.compare_and_swap(
                expected_generation=switched["generation"],
                expected_route_id=baseline["route_id"],
                new_route=candidate,
                reason="persistent selftest restore candidate",
                authorization_digest="d"*64,
            )
            r3=legacy._post_json(port,"/v1/generate",{"prompt_tokens":prompt,"max_new_tokens":10})
            t3=r3["responses"][0]["generated_tokens"]
            if t3[8]!=1224:
                raise PersistentSupervisorError("candidate restore reference mismatch")
            health=legacy._get_json(port,"/healthz")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)

        out={
            "schema":"beglin-persistent-supervisor-selftest-v1",
            "status":"PASS",
            "binary_sha256":build["binary_sha256"],
            "source_commit":build["source_commit"],
            "candidate_pid":r1["worker_instance_id"],
            "candidate_pid_reused":r1["worker_instance_id"]==r2["worker_instance_id"]==r3["worker_instance_id"],
            "candidate_seq_path":[r1["persistent_worker_seq"],r2["persistent_worker_seq"],r3["persistent_worker_seq"]],
            "candidate_reference_tokens":[t1[8],t2[8],t3[8]],
            "baseline_reference_token":bt[8],
            "candidate_roundtrip_ms":[r1["duration_ms"],r2["duration_ms"],r3["duration_ms"]],
            "candidate_engine_ms":[r1["engine_duration_ms"],r2["engine_duration_ms"],r3["engine_duration_ms"]],
            "baseline_roundtrip_ms":rb["duration_ms"],
            "health_status":health["status"],
            "route_generation_path":[1,2,3],
            "external_network_exposed":False,
            "production_write_allowed":False,
            "auto_promotion_enabled":False,
        }
        out["result_sha256"]=routing.sha256_json(out)
        return out


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--serve",action="store_true")
    group.add_argument("--self-test",action="store_true")
    group.add_argument("--probe-health",action="store_true")
    ap.add_argument("--route-manifest",default=str(DEFAULT_ROUTE_MANIFEST))
    ap.add_argument("--port",type=int,default=DEFAULT_PORT)
    args=ap.parse_args()

    if args.self_test:
        print(json.dumps(self_test(),indent=2,sort_keys=True))
        return 0
    if args.probe_health:
        print(json.dumps(legacy._get_json(args.port,"/healthz"),indent=2,sort_keys=True))
        return 0

    server=make_server(route_manifest=args.route_manifest,host="127.0.0.1",port=args.port)
    print(json.dumps({
        "schema":"beglin-persistent-supervisor-start-v1",
        "status":"SERVING",
        "listen_host":"127.0.0.1",
        "listen_port":args.port,
        "external_network_exposed":False,
        "binary_sha256":server.executor.build["binary_sha256"],
        "source_commit":server.executor.build["source_commit"],
    },sort_keys=True),flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
