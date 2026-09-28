#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "beglin-p9-physical-quality-run/1"
RUNNER_ID = "xox-p9-quality-v1"
REPO = Path("/Users/xox/mcp-sandbox/tailnet-commander/beglin-p9-integration")
SANDBOX = Path("/Users/xox/mcp-sandbox/tailnet-commander")
BUILD = REPO / "build-gpu-p9-quality"
BINARY = BUILD / "qwen_infer_gpu"
MOE_BASE = Path("/Users/xox/vdsp_local_data/moe_base_deepseek")
CHECKPOINT_INDEX = Path(
    "/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/"
    "model.safetensors.index.json"
)
CHECKPOINT_EVIDENCE = (
    Path("/Users/xox/vdsp_shadow_runs/checkpoint_identity/status.json"),
    Path("/Users/xox/vdsp_shadow_runs/checkpoint_identity/checkpoint_identity.json"),
)
RUN_ROOT = SANDBOX / "p9_quality_runs"
LATEST = SANDBOX / "p9_quality_latest.json"
ADAPTER_PATH = REPO / "configs/quality/p9_execution_adapter_v1.json"
CORPUS_PATH = REPO / "configs/quality/deterministic_corpus_v1.json"
TOKEN_FIXTURE_PATH = REPO / "configs/quality/p9_token_fixture_v1.json"
CANONICAL_RUNNER = REPO / "beglin_p9_quality_fixture.py"
MODEL_CONFIG = MOE_BASE / "weights_moe/arch_config_moe.txt"
P9_MAX_CONTEXT = 16384
P9_REQUEST_RE = re.compile(r"^P9_QUALITY_REQUEST_V1 ", re.MULTILINE)
VALIDATION_RE = re.compile(
    r"GPU_VALIDATION_V1 backend=mlx_metal arch=deepseek-v2-lite correction=off "
    r"finite_logits=(?P<finite>[01]) logits_checked=(?P<count>\d+) requests=(?P<requests>\d+)"
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_checked(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {' '.join(argv)}\n"
            f"stdout={result.stdout[-8000:]}\nstderr={result.stderr[-8000:]}"
        )
    return result


def nested_values(value: Any, key: str) -> list[Any]:
    out: list[Any] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                out.append(child)
            out.extend(nested_values(child, key))
    elif isinstance(value, list):
        for child in value:
            out.extend(nested_values(child, key))
    return out


def checkpoint_identity() -> tuple[str, str]:
    found: list[tuple[str, str]] = []
    for path in CHECKPOINT_EVIDENCE:
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        for candidate in nested_values(value, "checkpoint_sha256"):
            if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{64}", candidate):
                found.append((candidate, str(path)))
    unique = sorted({value for value, _ in found})
    if len(unique) != 1:
        raise RuntimeError(f"checkpoint identity is missing or ambiguous: {unique}")
    return unique[0], next(path for value, path in found if value == unique[0])


def source_manifest() -> tuple[str, list[dict[str, Any]]]:
    fixed = [
        "beglin_p9_quality_fixture.py",
        "CMakeLists.txt",
        "qwen_infer.c",
        "hf_config.c",
        "safetensors_load.c",
        "safetensors_quants.c",
        "gguf_load.c",
        "gguf_quants.c",
        "gguf_transcode.c",
        "gguf_cache.c",
        "bpe_tokenizer.c",
        "bpe_tokenizer.h",
        "gguf_write.c",
        "gguf_write.h",
        "gguf_write_quants.c",
        "gguf_write_quants.h",
        "sme2_kai.c",
        "mlx_moe.cpp",
        "mlx_moe.h",
        "benchmarks/quality/execution_adapter.py",
        "benchmarks/quality/reference_evaluator.py",
        "configs/quality/p9_execution_adapter_v1.json",
        "configs/quality/p9_token_fixture_v1.json",
        "configs/quality/deterministic_corpus_v1.json",
    ]
    paths = [REPO / relative for relative in fixed]
    excluded = ("kai_test", "thread_scaling", "prototype", "vdsp_ref", "verify_gemm", "verify_real_repack")
    for path in sorted((REPO / "kleidiai").iterdir()):
        if path.suffix in {".c", ".S"} and not any(marker in path.name for marker in excluded):
            paths.append(path)
    rows = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"build input missing: {path}")
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return sha256_bytes(stable_json(rows).encode()), rows


def runner_identity() -> str:
    executing = Path(__file__).resolve()
    if not CANONICAL_RUNNER.is_file():
        raise RuntimeError(f"canonical runner missing: {CANONICAL_RUNNER}")
    executing_sha = sha256_file(executing)
    canonical_sha = sha256_file(CANONICAL_RUNNER)
    if executing_sha != canonical_sha:
        raise RuntimeError(
            f"registered runner SHA differs from repository runner: {executing_sha} != {canonical_sha}"
        )
    return executing_sha


def compact_cache_projection() -> dict[str, Any]:
    if not MODEL_CONFIG.is_file():
        raise RuntimeError(f"model config missing: {MODEL_CONFIG}")
    values: dict[str, float] = {}
    for raw_line in MODEL_CONFIG.read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = raw_line.partition("=")
        if separator:
            try:
                values[key] = float(raw_value)
            except ValueError:
                continue
    required = ["NL", "N_HEADS", "QK_ROPE_HEAD_DIM", "QK_NOPE_HEAD_DIM", "V_HEAD_DIM"]
    missing = [key for key in required if key not in values]
    if missing:
        raise RuntimeError(f"model config missing cache dimensions: {missing}")
    layers = int(values["NL"])
    heads = int(values["N_HEADS"])
    key_width = int(values["QK_ROPE_HEAD_DIM"] + values["QK_NOPE_HEAD_DIM"])
    value_width = int(values["V_HEAD_DIM"])
    element_count = layers * heads * P9_MAX_CONTEXT * (key_width + value_width)
    compact_bytes = element_count + layers * heads * P9_MAX_CONTEXT * 2 * 2
    float32_bytes = element_count * 4
    if compact_bytes >= 3_000_000_000:
        raise RuntimeError(f"projected compact K/V cache is unexpectedly large: {compact_bytes}")
    return {
        "format": "symmetric_int8_per_head_position",
        "scale_dtype": "float16",
        "positions": P9_MAX_CONTEXT,
        "layers": layers,
        "heads": heads,
        "key_width": key_width,
        "value_width": value_width,
        "projected_bytes": compact_bytes,
        "projected_float32_bytes": float32_bytes,
        "reduction_ratio": compact_bytes / float32_bytes,
        "model_config": str(MODEL_CONFIG),
        "model_config_sha256": sha256_file(MODEL_CONFIG),
    }


def git_identity() -> tuple[str, str]:
    head = run_checked(["git", "rev-parse", "HEAD"], cwd=REPO, timeout=30).stdout.strip()
    branch = run_checked(["git", "branch", "--show-current"], cwd=REPO, timeout=30).stdout.strip()
    if branch != "prod/p9-integration":
        raise RuntimeError(f"expected prod/p9-integration, got {branch!r}")
    for argv in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        result = subprocess.run(argv, cwd=REPO, timeout=30, check=False)
        if result.returncode != 0:
            raise RuntimeError("tracked repository state is dirty")
    return head, branch


def adapter_preflight() -> dict[str, Any]:
    result = run_checked(
        [
            sys.executable,
            "-m",
            "benchmarks.quality.execution_adapter",
            "preflight",
            "--repo",
            str(REPO),
            "--adapter",
            str(ADAPTER_PATH),
            "--corpus",
            str(CORPUS_PATH),
        ],
        cwd=REPO,
        timeout=60,
    )
    value = json.loads(result.stdout)
    blockers = value.get("blockers", [])
    if [row.get("code") for row in blockers] not in ([], ["TAILNET_FIXTURE_UNREGISTERED"]):
        raise RuntimeError(f"adapter preflight failed: {blockers}")
    return value


def configure_and_build() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "cmake",
            "-S",
            str(REPO),
            "-B",
            str(BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBEGLIN_P9_LONG_CONTEXT=ON",
        ],
        timeout=300,
    )
    run_checked(
        ["cmake", "--build", str(BUILD), "--target", "qwen_infer_gpu", "-j2"],
        timeout=1800,
    )
    if not BINARY.is_file() or not os.access(BINARY, os.X_OK):
        raise RuntimeError("P9 GPU binary was not produced")


def materialize_tokens(work: Path) -> dict[str, Any]:
    output = work / "materialized_fixture.json"
    run_checked(
        [
            sys.executable,
            "-m",
            "benchmarks.quality.execution_adapter",
            "materialize",
            "--fixture",
            str(TOKEN_FIXTURE_PATH),
            "--output-dir",
            str(work / "tokens"),
            "--output",
            str(output),
        ],
        cwd=REPO,
        timeout=60,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def engine_env(manifest: Path, promotion: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("QWEN_MOE_") or key.startswith("BEGLIN_"):
            env.pop(key, None)
    env.update(
        {
            "QWEN_MOE_BASE": str(MOE_BASE),
            "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
            "QWEN_MOE_CB_PROMPT_MANIFEST": str(manifest),
            "QWEN_MOE_CB_SLOTS": "1",
            "QWEN_MOE_CB_REQS": "16",
            "QWEN_MOE_CB_PREFILL_BUDGET": "64",
            "QWEN_MOE_GPU_CB_CHECK": "1",
            "QWEN_MOE_NEARTIE_CORRECT": "0",
            "QWEN_MOE_ATTRIB": "0",
            "BEGLIN_P9_QUALITY_METRICS": "1",
            "BEGLIN_P9_ROUTER_NEAR_TIE_THRESHOLD": "0.001",
            "BEGLIN_P9_WARMUP": "0",
        }
    )
    if promotion is not None:
        env["QWEN_MOE_PROMOTION_FILE_NQ"] = str(promotion)
        env["QWEN_MOE_PROMOTION_SAFETENSORS"] = str(CHECKPOINT_INDEX)
    return env


def run_variant(work: Path, name: str, policy: dict[str, Any], manifest: Path) -> dict[str, Any]:
    promotion = None
    lines = policy["promotion_lines"]
    if lines:
        promotion = work / f"{name}.promotion.txt"
        promotion.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        [str(BINARY)],
        cwd=REPO,
        env=engine_env(manifest, promotion),
        text=True,
        capture_output=True,
        timeout=21600,
        check=False,
    )
    log = work / f"{name}.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    text = log.read_text(encoding="utf-8")
    validation = VALIDATION_RE.search(text)
    request_count = len(P9_REQUEST_RE.findall(text))
    sys.path.insert(0, str(REPO))
    from benchmarks.quality.execution_adapter import parse_engine_log

    parsed_rows = []
    parse_error = None
    try:
        parsed_rows = parse_engine_log(text)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    metrics_complete = (
        len(parsed_rows) == 16
        and all(row["finite_logits"] for row in parsed_rows)
        and all(row["nll_count"] == row["context_tokens"] - 1 for row in parsed_rows)
        and all(row["router_decisions"] > 0 for row in parsed_rows)
        and all(row["nout"] > 0 for row in parsed_rows)
    )
    valid = (
        result.returncode == 0
        and request_count == 16
        and metrics_complete
        and validation is not None
        and validation.group("finite") == "1"
        and int(validation.group("requests")) == 16
        and int(validation.group("count")) > 0
    )
    return {
        "variant": name,
        "status": "PASS" if valid else "FAIL",
        "exit_code": result.returncode,
        "request_count": request_count,
        "parsed_request_count": len(parsed_rows),
        "metrics_complete": metrics_complete,
        "parse_error": parse_error,
        "finite_logits": validation.group("finite") == "1" if validation else None,
        "logits_checked": int(validation.group("count")) if validation else None,
        "log_path": str(log),
        "log_sha256": sha256_file(log),
        "policy": policy,
        "policy_hash": sha256_bytes(stable_json(policy).encode()),
        "promotion_path": str(promotion) if promotion else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    runner_sha = runner_identity()
    head, branch = git_identity()
    preflight = adapter_preflight()
    cache_projection = compact_cache_projection()
    manifest_hash, manifest_rows = source_manifest()
    checkpoint_sha, checkpoint_source = checkpoint_identity()
    adapter = json.loads(ADAPTER_PATH.read_text(encoding="utf-8"))

    if args.self_test:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "PASS",
                    "mode": "self-test",
                    "source_commit": head,
                    "runner_sha256": runner_sha,
                    "source_manifest_sha256": manifest_hash,
                    "checkpoint_sha256": checkpoint_sha,
                    "adapter_preflight": preflight,
                    "compact_cache_projection": cache_projection,
                },
                sort_keys=True,
            )
        )
        return 0

    started = datetime.now(timezone.utc)
    run_id = "p9-" + started.strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    work = RUN_ROOT / run_id
    work.mkdir(parents=True, exist_ok=False)
    configure_and_build()
    materialized = materialize_tokens(work)
    binary_sha = sha256_file(BINARY)
    runtime_config = {
        "adapter_id": adapter["adapter_id"],
        "build": adapter["engine"]["build"],
        "runtime": adapter["engine"]["runtime"],
        "token_fixture_sha256": preflight["token_fixture_sha256"],
    }
    identity = {
        "source_commit": head,
        "source_tree": manifest_hash,
        "binary_sha256": binary_sha,
        "checkpoint_sha256": checkpoint_sha,
        "model_id": "deepseek-v2-lite",
        "hardware_id": "xox-apple-silicon",
        "backend": "mlx_metal",
        "seed": 0,
        "prompt_corpus_sha256": preflight["corpus_sha256"],
        "runtime_config_hash": sha256_bytes(stable_json(runtime_config).encode()),
    }
    atomic_json(work / "identity.json", identity)

    policies = [("reference", adapter["reference"])]
    policies.extend((f"n{row['n']}", row) for row in adapter["candidates"])
    runs = [run_variant(work, name, policy, Path(materialized["manifest"])) for name, policy in policies]
    status = "P9_PHYSICAL_RAW_PASS" if all(row["status"] == "PASS" for row in runs) else "FAIL"
    result = {
        "schema": SCHEMA,
        "runner_id": RUNNER_ID,
        "run_id": run_id,
        "status": status,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "runner_sha256": runner_sha,
        "identity": identity,
        "checkpoint_identity_source": checkpoint_source,
        "source_manifest": manifest_rows,
        "source_manifest_sha256": manifest_hash,
        "adapter_preflight": preflight,
        "compact_cache_projection": cache_projection,
        "materialized_fixture": materialized,
        "runs": runs,
        "production_write_allowed": False,
        "note": "Raw physical evidence only; EOE normalization, Policy Freeze, and P9 final audit remain required.",
    }
    result["result_sha256"] = sha256_bytes(stable_json(result).encode())
    atomic_json(work / "result.json", result)
    atomic_json(LATEST, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "P9_PHYSICAL_RAW_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
