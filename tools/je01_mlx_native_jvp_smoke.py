#!/usr/bin/env python3
"""Bounded native MLX 0.32.1 custom-JVP operator proof using generated data.

This compiles a separate executable against the existing wheel, without editing
MLX, loading a model, or touching an ordinary/fixed-route JE01 experiment.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time


SCHEMA = "beglin-je01-mlx-native-custom-jvp-smoke/2"
RECEIPT_SCHEMA = "beglin-je01-mlx-native-custom-jvp-smoke-receipt/2"
VERSION = "0.32.1"
PREFIX = "je01_native_jvp_smoke_"
EXPECTED_CASES = {
    "gather_gate_q4": ("gather_qmm", 384),
    "gather_down_q4": ("gather_qmm", 384),
    "gather_gate_q5": ("gather_qmm", 384),
    "gather_down_q5": ("gather_qmm", 384),
    "scatter_both": ("scatter", 64),
    "scatter_cache_only": ("scatter", 64),
    "scatter_update_only": ("scatter", 64),
    "kv_two_step": ("kv_recurrence", 66),
}
LIMITS = {
    "gather_qmm": (1e-5, 2e-5, 2e-4, 2e-4),
    "scatter": (0.0, 0.0, 0.0, 0.0),
    "kv_recurrence": (1e-7, 1e-6, 1e-7, 1e-6),
}
EXPECTED_GUARDS = {
    "gather_qmm_vjp": "JE01_JVP_ADAPTER_REVERSE_MODE_NOT_SUPPORTED",
    "gather_qmm_direct_vmap": "JE01_JVP_ADAPTER_DIRECT_VMAP_NOT_SUPPORTED",
    "scatter_vjp": "JE01_JVP_ADAPTER_REVERSE_MODE_NOT_SUPPORTED",
    "scatter_direct_vmap": "JE01_JVP_ADAPTER_DIRECT_VMAP_NOT_SUPPORTED",
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class NativeCommandFailure(ValueError):
    """A failed native command with its original bounded diagnostics preserved."""

    def __init__(self, label: str, details: dict):
        self.details = details
        report = details.get("native_failure_report", {})
        message = report.get("error") or details["stderr_tail"] or "NO_STDERR"
        super().__init__(f"JE01_JVP_{label.upper()}_FAILED:{details['returncode']}:{message}")


def safe_output_root(value: str | Path) -> Path:
    path = Path(value)
    need(path.is_absolute() and path.name.startswith(PREFIX),
         "JE01_JVP_OUTPUT_ABSOLUTE_RESEARCH_PREFIX_REQUIRED")
    need(".." not in path.parts and path.name != PREFIX, "JE01_JVP_OUTPUT_PATH_INVALID")
    for parent in (path, *path.parents):
        need(not parent.is_symlink(), "JE01_JVP_OUTPUT_SYMLINK_FORBIDDEN")
    need(not path.exists(), "JE01_JVP_OUTPUT_MUST_BE_FRESH")
    need(path.parent.is_dir(), "JE01_JVP_OUTPUT_PARENT_MISSING")
    return path


def metrics(actual: list[float], reference: list[float]) -> dict:
    need(len(actual) == len(reference) and bool(actual), "JE01_JVP_ARRAY_LENGTH_MISMATCH")
    need(all(math.isfinite(x) for x in (*actual, *reference)), "JE01_JVP_NONFINITE_ARRAY")
    delta = [a - b for a, b in zip(actual, reference)]
    return {
        "max_abs": max(abs(d) for d in delta),
        "rms": math.sqrt(math.fsum(d * d for d in delta) / len(delta)),
        "reference_rms": math.sqrt(math.fsum(v * v for v in reference) / len(reference)),
    }


def numeric_gate(value: dict, atol: float, rtol: float, name: str) -> None:
    bound = atol + rtol * value["reference_rms"]
    need(value["rms"] <= bound and value["max_abs"] <= 8 * bound,
         f"JE01_JVP_NUMERIC_GATE:{name}")


def independent_reference(case: dict, values: dict[str, list[float]]) -> list[float]:
    name = case["name"]
    if case["kind"] == "gather_qmm":
        decoded, dx = values[case["decoded_weights"]], values[case["activation_tangent"]]
        need(len(decoded) == 8 * 64 * 128, "JE01_JVP_DECODED_SHAPE_MISMATCH")
        down = "_down_" in name
        need(len(dx) == (6 if down else 1) * 128, "JE01_JVP_ACTIVATION_SHAPE_MISMATCH")
        return [math.fsum(dx[(k if down else 0) * 128 + j] * decoded[(expert * 64 + row) * 128 + j]
                              for j in range(128))
                for k, expert in enumerate((7, 0, 3, 5, 1, 6)) for row in range(64)]
    if case["kind"] == "scatter":
        ref = [0.0 if name == "scatter_update_only" else ((i % 17) - 8) / 32 for i in range(64)]
        for head in range(2):
            for j in range(8):
                ref[(head * 4 + 2) * 8 + j] = (
                    0.0 if name == "scatter_cache_only" else (((head * 8 + j) % 7) - 3) / 16)
        return ref
    need(name == "kv_two_step", "JE01_JVP_UNKNOWN_REFERENCE")
    u0 = [.5 + (i % 5) * .0625 for i in range(16)]
    later = [-.25 + (i % 7) * .03125 + .5 * u0[i] for i in range(16)]
    ref = [0.0] * 66
    for head in range(2):
        for j in range(8):
            i = head * 8 + j
            ref[head * 32 + j] = u0[i]
            ref[head * 32 + 8 + j] = later[i]
    ref[64] = .5 * math.fsum(v * v for v in u0)
    ref[65] = .5 * (math.fsum(v * v for v in u0) + math.fsum(v * v for v in later))
    return ref


def validate_smoke_report(report: dict, data_root: Path) -> dict:
    need(report.get("schema") == SCHEMA and report.get("status") == "PASS",
         "JE01_JVP_CPP_REPORT_NOT_PASS")
    need(report.get("expected_mlx_version") == VERSION and report.get("device") == "gpu",
         "JE01_JVP_RUNTIME_IDENTITY_MISMATCH")
    need(report.get("proof_scope") == "tiny-native-primitives-and-two-step-KV-only",
         "JE01_JVP_PROOF_SCOPE_MISMATCH")
    need((report.get("fd_h"), report.get("fd_h2")) == (.0625, .03125),
         "JE01_JVP_FD_STEPS_MISMATCH")
    errors = report.get("raw_jvp_errors", {})
    need("GatherQMM::jvp NYI" in errors.get("gather_qmm", "") and
         "[scatter] JVP not yet implemented" in errors.get("scatter", ""),
         "JE01_JVP_RAW_UNSUPPORTED_CONTROLS_MISSING")
    guard_errors = report.get("transform_guard_errors", {})
    need(set(guard_errors) == set(EXPECTED_GUARDS) and
         all(isinstance(guard_errors[k], str) and expected == guard_errors[k]
             for k, expected in EXPECTED_GUARDS.items()),
         "JE01_JVP_TRANSFORM_REFUSAL_CONTROLS_MISSING_OR_WRONG")
    need(isinstance(report.get("arrays"), list) and 1 <= len(report["arrays"]) <= 128,
         "JE01_JVP_ARRAY_MANIFEST_COUNT_INVALID")
    raw, values, manifests = {}, {}, {}
    for entry in report["arrays"]:
        name, dtype, elements, shape = (entry.get(k) for k in ("file", "dtype", "elements", "shape"))
        need(isinstance(name, str) and re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.(?:f32|u32)", name)
             and name not in manifests, "JE01_JVP_UNSAFE_OR_DUPLICATE_ARRAY_FILE")
        need(dtype in ("float32", "uint32") and name.endswith(".f32" if dtype == "float32" else ".u32"),
             "JE01_JVP_ARRAY_DTYPE_MISMATCH")
        need(type(elements) is int and 0 < elements <= 1000000 and isinstance(shape, list)
             and all(type(x) is int and x > 0 for x in shape) and math.prod(shape) == elements,
             "JE01_JVP_ARRAY_SHAPE_INVALID")
        path = data_root / name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size == elements * 4,
             "JE01_JVP_ARRAY_FILE_INVALID")
        raw[name] = path.read_bytes()
        if dtype == "float32":
            val = array.array("f")
            val.frombytes(raw[name])
            need(val.itemsize == 4 and all(math.isfinite(v) for v in val), "JE01_JVP_NONFINITE_ARRAY")
            values[name] = list(val)
        manifests[name] = {**entry, "bytes": len(raw[name]), "sha256": hashlib.sha256(raw[name]).hexdigest()}
    cases = report.get("cases", [])
    need(len(cases) == len(EXPECTED_CASES) and {c.get("name") for c in cases} == set(EXPECTED_CASES),
         "JE01_JVP_CASE_SET_MISMATCH")
    verified = []
    for c in cases:
        name, kind = c["name"], c["kind"]
        need((kind, c.get("elements")) == EXPECTED_CASES[name], "JE01_JVP_CASE_SHAPE_MISMATCH")
        for key in ("native", "wrapped", "jvp_primal", "tangent", "reference", "fd_coarse", "fd_fine"):
            need(c.get(key) == f"{name}.{key}.f32" and c[key] in values
                 and len(values[c[key]]) == c["elements"], "JE01_JVP_CASE_ARRAY_BINDING_INVALID")
        need(raw[c["native"]] == raw[c["wrapped"]] == raw[c["jvp_primal"]],
             "JE01_JVP_NATIVE_PRIMAL_BYTE_MISMATCH")
        t, r = values[c["tangent"]], values[c["reference"]]
        computed = {
            "analytic": metrics(t, r),
            "fd_coarse": metrics(values[c["fd_coarse"]], t),
            "fd_fine": metrics(values[c["fd_fine"]], t),
            "reference_independent": metrics(r, independent_reference(c, values)),
        }
        aa, ar, fa, fr = LIMITS[kind]
        need(tuple(c.get(k) for k in ("analytic_atol", "analytic_rtol", "fd_atol", "fd_rtol"))
             == (aa, ar, fa, fr), "JE01_JVP_TOLERANCE_CHANGED")
        numeric_gate(computed["analytic"], aa, ar, name + ":analytic")
        numeric_gate(computed["fd_coarse"], fa, fr, name + ":fd_coarse")
        numeric_gate(computed["fd_fine"], fa, fr, name + ":fd_fine")
        numeric_gate(computed["reference_independent"], 1e-7, 1e-7, name + ":reference")
        detail = {"name": name, "kind": kind, "native_primal_exact": True,
                  "jvp_primal_exact": True, "errors": computed}
        if kind == "gather_qmm":
            need(c.get("group_size") == 64 and c.get("bits") == int(name[-1]), "JE01_JVP_QUANTIZATION_MISMATCH")
            need(c.get("native_tangent") == f"{name}.native_tangent.f32" and
                 raw[c["native_tangent"]] == raw[c["tangent"]], "JE01_JVP_NATIVE_TANGENT_BYTE_MISMATCH")
            for key, suffix in (("packed", "packed.u32"), ("scales", "scales.f32"), ("biases", "biases.f32")):
                need(c.get(key) == f"{name}.{suffix}" and c[key] in manifests, "JE01_JVP_QUANTIZED_ARRAY_MISSING")
            need(manifests[c["packed"]]["shape"] == [8, 64, 128 * c["bits"] // 32]
                 and manifests[c["scales"]]["shape"] == [8, 64, 2]
                 and manifests[c["biases"]]["shape"] == [8, 64, 2], "JE01_JVP_QUANTIZED_SHAPE_INVALID")
            detail["native_tangent_exact"] = True
        elif kind == "scatter":
            need(raw[c["tangent"]] == raw[c["reference"]], "JE01_JVP_SCATTER_REPLACEMENT_MISMATCH")
            detail["cpu_replacement_tangent_exact"] = True
        else:
            need(c.get("streamed_tangent") == "kv_two_step.streamed_tangent.f32" and
                 c.get("detached_tangent") == "kv_two_step.detached_tangent.f32", "JE01_JVP_HISTORY_ARRAY_BINDING_INVALID")
            streaming = metrics(values[c["streamed_tangent"]], t)
            numeric_gate(streaming, 1e-7, 1e-6, "kv_two_step:streaming")
            detached_error = abs(values[c["detached_tangent"]][-1] - t[-1])
            need(detached_error > .1, "JE01_JVP_DETACHED_HISTORY_NOT_DETECTED")
            # Independent closed-form negative control: only alpha*U1 is
            # differentiated in step 1 when the incoming cache tangent is lost.
            need(abs(values[c["detached_tangent"]][-1] - (-.1572265625)) <= 1e-7,
                 "JE01_JVP_DETACHED_HISTORY_CONTROL_ANALYTIC_MISMATCH")
            need(c.get("loss_is_toy_quadratic_not_token_nll") is True, "JE01_JVP_TOY_SCOPE_MISSING")
            detail.update(streaming_error=streaming, detached_loss1_error=detached_error,
                          loss1_derivative=t[-1])
        verified.append(detail)
    return {"cases": verified, "artifacts": list(manifests.values()),
            "executed_transform_refusal_controls": dict(guard_errors),
            "native_primal_sha_pairs": [{"case": c["name"],
                "native_sha256": manifests[c["native"]]["sha256"],
                "wrapped_sha256": manifests[c["wrapped"]]["sha256"],
                "jvp_primal_sha256": manifests[c["jvp_primal"]]["sha256"]} for c in cases]}


def bounded_run(command: list[str], root: Path, label: str, timeout: int, env: dict) -> dict:
    stdout, stderr = root / f"{label}.stdout.log", root / f"{label}.stderr.log"
    start = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   cwd=root, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise ValueError(f"JE01_JVP_{label.upper()}_TIMEOUT:{timeout}") from None
    details = {"command": command, "returncode": code, "elapsed_seconds": time.monotonic() - start,
               "timeout_seconds": timeout, "stdout": str(stdout), "stderr": str(stderr)}
    if code != 0:
        details["label"] = label
        for key, path in (("stdout", stdout), ("stderr", stderr)):
            size = path.stat().st_size
            with path.open("rb") as stream:
                stream.seek(max(0, size - 8192))
                details[key + "_tail"] = stream.read().decode("utf-8", errors="replace")
            details[key + "_tail_truncated"] = size > 8192
            details[key + "_sha256"] = sha_file(path)
        if stdout.stat().st_size <= 1024 * 1024:
            try:
                failed = json.loads(stdout.read_text())
            except (ValueError, UnicodeError):
                failed = None
            if isinstance(failed, dict) and failed.get("schema") == SCHEMA and failed.get("status") == "FAIL":
                details["native_failure_report"] = failed
        raise NativeCommandFailure(label, details)
    return details


def discover_runtime() -> tuple[Path, dict]:
    need(platform.system() == "Darwin" and platform.machine() == "arm64", "JE01_JVP_APPLE_NATIVE_RUNTIME_REQUIRED")
    need(importlib.metadata.version("mlx") == VERSION, "JE01_JVP_EXACT_MLX_0_32_1_REQUIRED")
    import mlx
    import mlx.core as mx
    need(getattr(mx, "__version__", VERSION) == VERSION, "JE01_JVP_CORE_VERSION_MISMATCH")
    candidates = [Path(p).resolve() for p in mlx.__path__]
    available = [p for p in candidates if (p / "include/mlx/mlx.h").is_file() and (p / "lib/libmlx.dylib").is_file()]
    need(len(available) == 1, "JE01_JVP_ONE_MLX_HEADER_LIBRARY_ROOT_REQUIRED")
    root = available[0]
    return root, {"mlx_version": VERSION, "mlx_root": str(root), "platform": platform.platform(),
                  "python": sys.executable, "dylib_sha256": sha_file(root / "lib/libmlx.dylib"),
                  "header_sha256": {f: sha_file(root / "include/mlx" / f)
                                    for f in ("array.h", "ops.h", "transforms.h")}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--run-timeout", type=int, default=90)
    args = parser.parse_args()
    root = None
    receipt = {"schema": RECEIPT_SCHEMA, "status": "FAIL", "proof_scope": "operator-smoke-only",
               "model_loaded": False, "installed_mlx_modified": False, "full_model_fisher_proven": False}
    try:
        need(1 <= args.compile_timeout <= 300 and 1 <= args.run_timeout <= 180, "JE01_JVP_TIMEOUT_BOUNDS")
        requested = safe_output_root(args.output_root)
        runtime, identity = discover_runtime()
        compiler = shutil.which("clang++")
        need(compiler is not None, "JE01_JVP_CLANG_UNAVAILABLE")
        requested.mkdir(mode=0o700)
        root = requested
        source, data = root / "source", root / "arrays"
        source.mkdir()
        data.mkdir()
        tool_dir = Path(__file__).resolve().parent
        names = ("je01_mlx_native_jvp_adapters.h", "je01_mlx_native_jvp_smoke.cpp", Path(__file__).name)
        source_pins = {name: sha_file(tool_dir / name) for name in names}
        for name in names:
            shutil.copyfile(tool_dir / name, source / name)
        executable = root / "je01_mlx_native_jvp_smoke"
        command = [compiler, "-std=c++20", "-O2", "-fno-fast-math", "-I", str(runtime / "include"),
                   str(source / "je01_mlx_native_jvp_smoke.cpp"), "-L", str(runtime / "lib"), "-lmlx",
                   "-Wl,-rpath," + str(runtime / "lib"), "-o", str(executable)]
        env = dict(os.environ)
        env["DYLD_LIBRARY_PATH"] = str(runtime / "lib") + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
        receipt.update(output_root=str(root), runtime=identity, source_sha256=source_pins)
        receipt["compile"] = bounded_run(command, root, "compile", args.compile_timeout, env)
        receipt["executable_sha256"] = sha_file(executable)
        receipt["run"] = bounded_run([str(executable), str(data)], root, "run", args.run_timeout, env)
        raw = (root / "run.stdout.log").read_text()
        need(len(raw) <= 1024 * 1024, "JE01_JVP_REPORT_TOO_LARGE")
        report = json.loads(raw)
        receipt["verified"] = validate_smoke_report(report, data)
        receipt["native_report_sha256"] = sha_file(root / "run.stdout.log")
        receipt["status"] = "PASS"
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, NativeCommandFailure):
            receipt["failed_command"] = exc.details
    receipt["canonical_sha256"] = canonical_sha(receipt)
    if root is not None:
        (root / "JE01_NATIVE_JVP_SMOKE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "canonical_sha256", "output_root", "error") if k in receipt}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
