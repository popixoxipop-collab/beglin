#!/usr/bin/env python3
"""Research-only BEGLIN v11 sealed-prequant -> Metal GPU LRU stress.

Build with build_and_test_gpuowned_v11_candidate.sh first. The coordinator
quantizes three SYNTHETIC matrices once with original native C and seals them
on disk; TWO concurrent independent child processes then repeatedly open the
same read-only prequant artifact. Each child serves multiple request threads
through its per-instance serialized GPU LRU, checks CPU-C parity, forces LRU
eviction, verifies stale-handle rejection, and fully releases GPU handles.
No original model is used; no model TTFT/TPOT or production safety claim.
"""
from __future__ import annotations

import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

from prequant_n7_sealed_v1 import PackWriter, PackReader, role_files
from research_n7_device_lru_cache_v3_GPUonly_candidate import ResidentN7Cache

SOURCE_SHA = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
HERE = Path(__file__).resolve().parent
GPU = HERE / "build" / "libbeglin_n7_multi_projection_GPU_resident_v11_candidate.dylib"
REF = HERE / "build" / "libbeglin_qng64_reference.dylib"
U = ctypes.POINTER(ctypes.c_uint8)
F = ctypes.POINTER(ctypes.c_float)
SHAPES = ((128, 512), (512, 512), (256, 1024))
ROLES = tuple("model.layers.%d.self_attn.q_proj.weight" % i for i in range(3))


def must(ok, why):
    if not ok:
        raise AssertionError("BEGLIN_V11_PREQUANT_METAL_STRESS_FAIL: " + why)


def ptr(a, kind):
    return a.ctypes.data_as(kind)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref_library():
    must(REF.is_file(), "native reference C library missing")
    lib = ctypes.CDLL(str(REF))
    q = lib.qng64_quantize_f32
    q.argtypes = (F, ctypes.c_int, ctypes.c_int, ctypes.c_int, U, F)
    q.restype = ctypes.c_int
    mm = lib.qng64_matmul_f32
    mm.argtypes = (U, F, ctypes.c_int, F, F, F,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int)
    mm.restype = None
    return lib


def create_fixture(root):
    rng = np.random.default_rng(20261019)
    c = ref_library()
    writer = PackWriter(root, SOURCE_SHA)
    for role, (out, in_features) in zip(ROLES, SHAPES):
        w = np.ascontiguousarray(rng.normal(0, 0.04, (out, in_features)), dtype=np.float32)
        groups = out * in_features // 64
        planes = np.empty((groups * 56,), dtype=np.uint8)
        scales = np.empty((groups,), dtype=np.float32)
        rc = c.qng64_quantize_f32(ptr(w, F), out, in_features, 7,
                                  ptr(planes, U), ptr(scales, F))
        must(rc == 0, "source C n7 quantization failed")
        writer.add(role, out, in_features, planes, scales)
    return writer.seal()


def rss_bytes():
    try:
        return int(subprocess.check_output(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True, timeout=5).strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def child_worker(root, manifest_sha, worker_id):
    import torch
    import threading

    read = PackReader(root, manifest_sha, SOURCE_SHA)
    modules = {r: read.load(r) for r in ROLES}
    ref = ref_library()
    must(GPU.is_file(), "physical Metal native bridge missing")
    lib = ctypes.CDLL(str(GPU))
    lib.beglin_multi_n7_gpu_available_v1.restype = ctypes.c_int
    must(lib.beglin_multi_n7_gpu_available_v1() == 1, "physical GPU unavailable")
    cache = ResidentN7Cache(lib, bytes_budget=512 * 1024, max_handles=2)
    tensors = {}
    expected = {}
    canonical = {}
    for role in ROLES:
        m = modules[role]
        x = torch.linspace(-0.7, 0.7, m.in_features, dtype=torch.float32).to(torch.bfloat16)[None, :]
        f32 = np.ascontiguousarray(x.float().numpy())
        y = np.empty((1, m.out), dtype=np.float32)
        ref.qng64_matmul_f32(ptr(m.packed, U), ptr(m.scales, F), 7,
                            ptr(f32, F), None, ptr(y, F),
                            m.out, m.in_features, 1)
        tensors[role] = x
        expected[role] = torch.from_numpy(y).to(torch.bfloat16)
        got = cache.call(role, m, x)
        must(got.shape == expected[role].shape, "BF16 projected shape mismatch")
        denom = torch.linalg.vector_norm(expected[role].float()).item()
        diff = torch.linalg.vector_norm((got.float() - expected[role].float())).item()
        must(denom > 0 and diff / denom < 3e-4, "native C/Metal parity mismatch")
        canonical[role] = got.clone()
    # Warm all three geometries before concurrent dispatch and Metal compilation.
    epochs = int(os.environ.get("BEGLIN_V11_STRESS_EPOCHS", "3"))
    repeats = int(os.environ.get("BEGLIN_V11_STRESS_REPEATS", "10"))
    must(1 <= epochs <= 200 and 1 <= repeats <= 100 and
         epochs * repeats <= 10000, "bounded stress profile required")
    counts = []
    failures = []
    failures_lock = threading.Lock()
    after_warm = rss_bytes()
    epoch_rss = []
    started = time.monotonic()

    def one_thread():
        local = 0
        # Adjacent repeated roles guarantee both LRU hits and forced evictions.
        for _ in range(repeats):
            for role in ROLES:
                for _same_role_hit in range(2):
                    answer = cache.call(role, modules[role], tensors[role])
                    if not torch.equal(answer, canonical[role]):
                        with failures_lock:
                            failures.append("non-deterministic BF16 GPU output")
                    local += 1
        return local

    try:
        for _epoch in range(epochs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                counts.extend(pool.map(lambda unused: one_thread(), range(3)))
            epoch_rss.append(rss_bytes())
        state = cache.get_state()
        must(not failures, "concurrent request output changed")
        expected_calls = epochs * 3 * repeats * len(ROLES) * 2
        must(sum(counts) == expected_calls, "worker request accounting drift")
        must(state["actual_GPU_forward_calls"] == expected_calls + len(ROLES),
             "native physical GPU invocation count drift")
        must(state["GPU_reuse_hits"] > 0 and state["GPU_LRU_evictions"] > 0,
             "bounded LRU reuse/eviction not exercised")
        must(state["active_GPU_n7_role_handles"] <= 2 and
             state["active_GPU_packed_planes_scales_bytes"] <= 512*1024,
             "GPU cache capacity exceeded")
        stale = next(iter(cache.lru.values()))["handle"]
    finally:
        cache.close()
    must(lib.beglin_multi_n7_active_count_v1() == 0 and
         lib.beglin_multi_n7_payload_bytes_v1() == 0,
         "unreleased native GPU handle after threaded loads")
    lib.beglin_multi_n7_forward_v1.argtypes = (ctypes.c_uint64, F, ctypes.c_int, F)
    lib.beglin_multi_n7_forward_v1.restype = ctypes.c_int
    m = modules[ROLES[0]]
    testx = np.ones((1, m.in_features), dtype=np.float32)
    testy = np.empty((1, m.out), dtype=np.float32)
    must(lib.beglin_multi_n7_forward_v1(stale, ptr(testx, F), 1, ptr(testy, F)) == 0,
         "stale GPU weight handle remained callable")
    final_rss = rss_bytes()
    must(after_warm is not None and final_rss is not None and
         all(v is not None for v in epoch_rss), "RSS measurements required")
    drift = max(epoch_rss) - epoch_rss[0]
    must(drift <= 64 * 1024 * 1024,
         "post-warm RSS grew by more than 64 MiB")
    return {
        "status": "PASS_V11_PREQUANT_GPU_LRU_MULTI_REQUEST_WORKER",
        "worker_id": worker_id,
        "source_original_C_quantize_calls_in_worker": 0,
        "original_BF16_weights_loaded_in_worker": False,
        "read_only_mmap": True,
        "physical_Metal_forward_calls": state["actual_GPU_forward_calls"],
        "cache_hits": state["GPU_reuse_hits"],
        "cache_evictions": state["GPU_LRU_evictions"],
        "GPU_opaque_handles_after_cleanup": 0,
        "GPU_compressed_payload_bytes_after_cleanup": 0,
        "epoch_rss_bytes": epoch_rss,
        "postwarm_rss_drift_bytes": drift,
        "stress_epochs": epochs,
        "stress_repeats_per_thread": repeats,
        "extended_soak_profile": epochs >= 25,
        "after_warm_rss_bytes": after_warm,
        "final_rss_bytes": final_rss,
        "MLX_active_allocator_at_last_snapshot_B": state["active_MLX_allocator_bytes"],
        "runtime_seconds": round(time.monotonic()-started, 3),
        "canonical_BF16_output_SHA256":
            sha(b"".join(canonical[k].view(torch.int16).numpy().tobytes()
                         for k in ROLES)),
    }


def orchestrate():
    must(sys.platform == "darwin" and GPU.is_file() and REF.is_file(),
         "physical Apple Metal C++ build required")
    with tempfile.TemporaryDirectory(prefix="beglin_v11_prequant_metal_") as scratch:
        root = Path(scratch) / "frozen"
        manifest_sha = create_fixture(root)
        children = []
        for worker in range(2):
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--worker", str(root), manifest_sha, str(worker)]
            children.append(subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, text=True))
        reports = []
        for index, proc in enumerate(children):
            try:
                stdout, stderr = proc.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                must(False, "GPU worker exceeded 120s")
            must(proc.returncode == 0, "GPU worker %d failed: %s" %
                 (index, stderr[-1800:]))
            report = json.loads(stdout.strip().splitlines()[-1])
            must(report["status"] == "PASS_V11_PREQUANT_GPU_LRU_MULTI_REQUEST_WORKER"
                 and report["GPU_opaque_handles_after_cleanup"] == 0,
                 "process GPU teardown failed")
            reports.append(report)
        must(reports[0]["canonical_BF16_output_SHA256"] ==
             reports[1]["canonical_BF16_output_SHA256"],
             "independent GPU process output hash drift")
        result = {
            "schema": "beglin.v11.sealed-prequant-concurrent-metal-stress/1",
            "status": "PASS_V11_TWO_CONCURRENT_METAL_PROCESSES_PREQUANT_LRU_STRESS",
            "true_process_concurrency": True,
            "threaded_single_cache_calls_are_mutex_serialized": True,
            "workers": reports,
            "prequant_manifest_sha256": manifest_sha,
            "original_model_checkpoint_loaded": False,
            "full27_model_stress_verified": False,
            "TTFT_TPOT_certified": False,
            "production_promotion_allowed": False,
            "source_files": ["prequant_n7_sealed_v1.py",
                             "research_n7_device_lru_cache_v3_GPUonly_candidate.py",
                             "n7_multi_projection_gpu_residency_v11_gpuowned_candidate.cpp"],
        }
        raw = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
        evidence_root = HERE / "build" / "research_stress_receipts"
        evidence_root.mkdir(exist_ok=True, parents=True)
        name = "v11_prequant_2proc_" + str(time.time_ns()) + ".json"
        with (evidence_root / name).open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({
            "status": result["status"],
            "physical_GPU_GEMVs": sum(w["physical_Metal_forward_calls"] for w in reports),
            "GPU_LRU_evictions": sum(w["cache_evictions"] for w in reports),
            "all_handles_released": True,
            "outputs_SHA_identical": True,
            "receipt_SHA256": sha(raw),
            "receipt_file_basename": name,
            "workers": [
                {"rss": w["epoch_rss_bytes"], "seconds": w["runtime_seconds"]}
                for w in reports
            ],
        }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        print(json.dumps(child_worker(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])),
                         sort_keys=True))
    elif len(sys.argv) == 1:
        orchestrate()
    else:
        raise SystemExit("usage: test_v11_prequant_metal_concurrency_v1.py [--worker root manifest_sha worker_id]")
