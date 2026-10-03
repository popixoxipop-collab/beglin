#!/usr/bin/env python3
"""Scratch-only persistent-worker benchmark for allocator candidates.

Uses the certified persistent native worker protocol but never opens a serving
port, never reads/writes the production route manifest, and never writes a
production promotion/control file. v1 is intentionally scoped to
shared_up_proj/L3, the target with real G4/G6 evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

import backend_capabilities as bc
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as persistent


ALLOWED_TARGET = ("shared_up_proj", 3)
SUPPORTED = tuple(sorted(set(bc.GPU_NATIVE_QNG64) | set(bc.GPU_CUSTOM_QNG64)))
DEFAULT_ROOT = Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench")


class BenchmarkError(RuntimeError):
    pass


def percentile(values, p):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    k = (len(values) - 1) * float(p)
    lo = math.floor(k); hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def validate_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve(strict=False)
    allowed = DEFAULT_ROOT.resolve(strict=False)
    if root != allowed and allowed not in root.parents:
        raise BenchmarkError(f"benchmark root must stay under {allowed}")
    for forbidden in (
        Path("/Users/xox/vdsp_serving"),
        Path("/Users/xox/mcp-sandbox/tailnet-commander"),
    ):
        f = forbidden.resolve(strict=False)
        if root == f or f in root.parents:
            raise BenchmarkError("benchmark root overlaps production/control path")
    return root


def scratch_route(n: int) -> dict:
    n = int(n)
    if n not in SUPPORTED:
        raise BenchmarkError(f"unsupported qNg64 width: {n}")
    role, layer = ALLOWED_TARGET
    route = dict(base.candidate_route())
    route.update({
        "route_id": f"beglin-scratch-{role}-l{layer}-n{n}",
        "worker_instance_id": "scratch-persistent-benchmark",
        "endpoint": "scratch-persistent://allocator-benchmark",
        "policy_hash": pc.policy_hash([{"role": role, "layer": layer, "n": n}]),
        "role": role,
        "layer": layer,
        "n": n,
    })
    return route


class ScratchPersistentWorker(persistent.PersistentRouteWorker):
    def _prepare_files(self) -> None:
        role = str(self.route["role"])
        layer = int(self.route["layer"])
        n = int(self.route["n"])
        if (role, layer) != ALLOWED_TARGET:
            raise BenchmarkError("scratch benchmark target is outside v1 scope")
        if n not in SUPPORTED:
            raise BenchmarkError("scratch benchmark width is unsupported")
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        for stale in ("request.txt", "request.processing", "shutdown"):
            try:
                (self.queue_dir / stale).unlink()
            except FileNotFoundError:
                pass
        self.promotion_path.write_text(f"{role} {layer} {n}\n")
        for path in (self.ack_path, self.txn_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _check_result(result: dict, expected=1224) -> int:
    if result.get("finite_logits") is not True:
        raise BenchmarkError("non-finite logits")
    hits = sum(
        len(tokens) > 8 and int(tokens[8]) == int(expected)
        for tokens in result["responses"]
    )
    if hits != len(result["responses"]):
        raise BenchmarkError(
            f"reference mismatch: {hits}/{len(result['responses'])}"
        )
    return hits


def benchmark_one(n: int, root: Path, *, warmups=3, iterations=30, batch_iterations=10) -> dict:
    route = scratch_route(n)
    worker_root = root / f"n{n}"
    if worker_root.exists():
        shutil.rmtree(worker_root)
    worker = ScratchPersistentWorker(route=route, root=worker_root)
    prompt = base._read_first_certified_prompt()
    try:
        worker.start()
        pid = worker.health()["pid"]
        for _ in range(int(warmups)):
            _check_result(worker.submit([(prompt, 10)]))

        singles = []
        for _ in range(int(iterations)):
            result = worker.submit([(prompt, 10)])
            _check_result(result)
            singles.append(result)

        batches = []
        for _ in range(int(batch_iterations)):
            result = worker.submit([(prompt, 10) for _ in range(12)])
            _check_result(result)
            batches.append(result)

        health = worker.health()
        if health["pid"] != pid or not health["alive"]:
            raise BenchmarkError("persistent worker PID changed during benchmark")

        se = [r["engine_wall_ms"] for r in singles]
        sr = [r["roundtrip_ms"] for r in singles]
        be = [r["engine_wall_ms"] for r in batches]
        br = [r["roundtrip_ms"] for r in batches]
        return {
            "role": ALLOWED_TARGET[0],
            "layer": ALLOWED_TARGET[1],
            "n": int(n),
            "status": "PASS",
            "persistent_worker": True,
            "same_pid": True,
            "pid": int(pid),
            "rss_bytes": int(health["rss_bytes"]),
            "iterations": len(singles),
            "batch12_iterations": len(batches),
            "p50_engine_ms": percentile(se, .50),
            "p95_engine_ms": percentile(se, .95),
            "mean_engine_ms": statistics.mean(se),
            "p50_roundtrip_ms": percentile(sr, .50),
            "p95_roundtrip_ms": percentile(sr, .95),
            "batch12_p50_engine_ms": percentile(be, .50),
            "batch12_p95_engine_ms": percentile(be, .95),
            "batch12_p50_roundtrip_ms": percentile(br, .50),
            "batch12_p95_roundtrip_ms": percentile(br, .95),
            "reference_token": 1224,
        }
    finally:
        worker.stop(force=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[5, 6, 9])
    ap.add_argument("--root", default=str(DEFAULT_ROOT / "shared_up_l3_20261003"))
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--batch-iterations", type=int, default=10)
    ap.add_argument("--output")
    args = ap.parse_args()

    persistent.verify_persistent_artifact()
    root = validate_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    unavailable = 0
    for n in args.n:
        try:
            rows.append(benchmark_one(
                n, root,
                warmups=args.warmups,
                iterations=args.iterations,
                batch_iterations=args.batch_iterations,
            ))
        except (persistent.PersistentSupervisorError, BenchmarkError) as exc:
            unavailable += 1
            rows.append({
                "role": ALLOWED_TARGET[0],
                "layer": ALLOWED_TARGET[1],
                "n": int(n),
                "status": "STARTUP_UNAVAILABLE",
                "persistent_worker": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "production_write_allowed": False,
            })
    out = {
        "schema": "beglin-persistent-precision-benchmark-v1",
        "status": "PASS" if unavailable == 0 else "PARTIAL_UNAVAILABLE",
        "production_write_allowed": False,
        "production_route_manifest_touched": False,
        "target": {"role": ALLOWED_TARGET[0], "layer": ALLOWED_TARGET[1]},
        "rows": rows,
    }
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
