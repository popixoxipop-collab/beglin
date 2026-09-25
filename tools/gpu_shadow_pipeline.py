#!/usr/bin/env python3
"""One-cycle GPU production-shadow pipeline.

Flow:
  read-only production evidence discovery
  -> pick at most one READY candidate
  -> materialize local replay manifests/spec from an explicit local mirror
  -> run the certified GPU autopilot through the scratch-only shadow wrapper
  -> persist a shadow cycle result

No production promotion/control/Supabase writes are performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import gpu_shadow_discovery as discovery
import gpu_shadow_materialize as materialize
import gpu_shadow_runner as runner


SCHEMA = "gpu-shadow-pipeline-v1"
HISTORY_SCHEMA = "gpu-shadow-history-v1"

CONTROL_IDENTITY_FILES = (
    "tools/gpu_autopilot.py",
    "tools/gpu_shadow_runner.py",
    "tools/gpu_shadow_materialize.py",
    "tools/gpu_shadow_pipeline.py",
    "tools/gpu_isolated_preflight.py",
    "tools/precision_planner_v3.py",
    "tools/gpu_restart_canary.py",
    "tools/gpu_observer_control.py",
    "tools/gpu_runtime_control.py",
    "tools/autopilot_observer_v3.py",
    "tools/precision_control_state.py",
    "tools/backend_adapters.py",
    "tools/precision_context.py",
)


class ShadowPipelineError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(discovery._json_safe(value), f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _runtime_identity(config: dict) -> dict:
    cwd = Path(_require(config, "cwd")).expanduser().resolve(strict=False)
    binary = Path(_require(config, "binary")).expanduser().resolve(strict=False)
    autopilot = Path(_require(config, "autopilot")).expanduser().resolve(strict=False)
    if not binary.is_file():
        raise ShadowPipelineError(f"GPU binary does not exist: {binary}")
    if not autopilot.is_file():
        raise ShadowPipelineError(f"certified autopilot does not exist: {autopilot}")

    checkpoint_input = str(_require(config, "checkpoint_sha256")).strip().lower()
    if checkpoint_input == "auto":
        checkpoint_sha256, checkpoint_manifest = (
            materialize.checkpoint_identity_from_safetensors(
                str(_require(config, "safetensors"))
            )
        )
    else:
        if len(checkpoint_input) != 64 or any(
            ch not in "0123456789abcdef" for ch in checkpoint_input
        ):
            raise ShadowPipelineError(
                'checkpoint_sha256 must be 64 lowercase hex chars or "auto"'
            )
        checkpoint_sha256 = checkpoint_input
        checkpoint_manifest = {
            "schema": "checkpoint-identity-v1",
            "kind": "manual",
            "sha256": checkpoint_sha256,
        }

    code_sha256 = {}
    for rel in CONTROL_IDENTITY_FILES:
        path = cwd / rel
        if not path.is_file():
            raise ShadowPipelineError(
                f"control-plane identity file is missing: {path}"
            )
        code_sha256[rel] = _sha256_file(path)

    return {
        "schema": "gpu-shadow-runtime-identity-v1",
        "model": str(config.get("model", "deepseek-v2-lite")),
        "binary_sha256": _sha256_file(binary),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity": checkpoint_manifest,
        "autopilot_sha256": _sha256_file(autopilot),
        "control_plane_sha256": code_sha256,
        "moe_base": str(
            Path(_require(config, "moe_base")).expanduser().resolve(strict=False)
        ),
        "safetensors": str(
            Path(_require(config, "safetensors")).expanduser().resolve(strict=False)
        ),
        "path_maps": list(config.get("path_maps", [])),
        "g6_repeats": int(config.get("g6_repeats", 64)),
        "timeout": int(config.get("timeout", 3600)),
    }


def _candidate_fingerprint(row: dict, runtime_identity: dict) -> str:
    return discovery.sha256_json({
        "candidate": row,
        "runtime_identity_sha256": discovery.sha256_json(runtime_identity),
    })


def _load_history(path: Path) -> dict:
    value = _read_json(path, {"schema": HISTORY_SCHEMA, "candidates": {}})
    if not isinstance(value, dict) or value.get("schema") != HISTORY_SCHEMA:
        raise ShadowPipelineError("invalid shadow candidate history")
    if not isinstance(value.get("candidates"), dict):
        raise ShadowPipelineError("invalid shadow candidate history candidates map")
    return value


def _record_history(
    path: Path,
    history: dict,
    row: dict,
    run: dict,
    *,
    fingerprint: str,
    runtime_identity_sha256: str,
) -> None:
    candidate_id = str(row["candidate_id"])
    next_history = {
        "schema": HISTORY_SCHEMA,
        "updated_at": _now(),
        "candidates": dict(history.get("candidates", {})),
    }
    next_history["candidates"][candidate_id] = {
        "fingerprint": fingerprint,
        "runtime_identity_sha256": runtime_identity_sha256,
        "shadow_status": run.get("shadow_status"),
        "result_sha256": run.get("result_sha256"),
        "observed_at": _now(),
    }
    _atomic_json(path, next_history)


@contextmanager
def _cycle_lock(shadow_root: Path):
    path = shadow_root / ".cycle.lock"
    handle = open(path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ShadowPipelineError(
                f"another shadow cycle already holds {path}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def load_config(path):
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ShadowPipelineError("pipeline config must be a JSON object")
    return value


def _require(config, key):
    value = config.get(key)
    if value is None or value == "":
        raise ShadowPipelineError(f"pipeline config missing {key}")
    return value


def _run_cycle_locked(
    config: dict,
    *,
    discover_fn=discovery.discover,
    materialize_fn=materialize.materialize,
    run_shadow_fn=runner.run_shadow,
) -> dict:
    model = str(config.get("model", "deepseek-v2-lite"))
    limit = int(config.get("limit", 100))
    shadow_root = Path(_require(config, "shadow_root")).expanduser().resolve(strict=False)
    cwd = str(_require(config, "cwd"))
    autopilot = str(_require(config, "autopilot"))

    started_at = _now()
    discovered = discover_fn(model, limit)
    discovery_path = shadow_root / "discovery.json"
    _atomic_json(discovery_path, discovered)

    ready = discovered.get("ready") or []
    history_path = shadow_root / "candidate_history.json"
    history = _load_history(history_path)

    if not ready:
        result = {
            "schema": SCHEMA,
            "mode": "shadow",
            "production_write_allowed": False,
            "status": "NO_READY_CANDIDATE",
            "model": model,
            "ready_count": 0,
            "already_observed_candidate_ids": [],
            "discovery_path": str(discovery_path),
            "history_path": str(history_path),
            "started_at": started_at,
            "finished_at": _now(),
        }
        _atomic_json(shadow_root / "last_cycle.json", result)
        return result

    runtime_identity = _runtime_identity(config)
    runtime_identity_sha256 = discovery.sha256_json(runtime_identity)

    already_observed = []
    eligible = []
    for row in ready:
        candidate_id = str(row.get("candidate_id"))
        previous = history["candidates"].get(candidate_id)
        fingerprint = _candidate_fingerprint(row, runtime_identity)
        if previous and previous.get("fingerprint") == fingerprint:
            already_observed.append(candidate_id)
        else:
            eligible.append(row)

    if not eligible:
        result = {
            "schema": SCHEMA,
            "mode": "shadow",
            "production_write_allowed": False,
            "status": "NO_NEW_READY_CANDIDATE",
            "model": model,
            "ready_count": len(ready),
            "already_observed_candidate_ids": already_observed,
            "runtime_identity_sha256": runtime_identity_sha256,
            "discovery_path": str(discovery_path),
            "history_path": str(history_path),
            "started_at": started_at,
            "finished_at": _now(),
        }
        _atomic_json(shadow_root / "last_cycle.json", result)
        return result

    # Deliberately serialize one candidate per shadow cycle. This preserves the
    # same one-target-at-a-time evidence discipline as the real canary path.
    selected = eligible[0]
    deferred = [row.get("candidate_id") for row in eligible[1:]]

    mappings = materialize.parse_maps(config.get("path_maps", []))
    spec_result = materialize_fn(
        discovered,
        candidate_id=selected["candidate_id"],
        path_mappings=mappings,
        output_dir=str(shadow_root / "inputs"),
        cwd=cwd,
        binary=str(_require(config, "binary")),
        checkpoint_sha256=str(_require(config, "checkpoint_sha256")).lower(),
        moe_base=str(_require(config, "moe_base")),
        safetensors=str(_require(config, "safetensors")),
        g6_repeats=int(config.get("g6_repeats", 64)),
    )

    spec = materialize.load_json(spec_result["candidate_spec"])
    run = run_shadow_fn(
        spec,
        shadow_root=str(shadow_root / "executions"),
        autopilot=autopilot,
        python_bin=str(config.get("python", os.environ.get("PYTHON", "python3"))),
        timeout=int(config.get("timeout", 3600)),
        forbidden_roots=config.get("forbidden_roots", []),
    )

    selected_fingerprint = _candidate_fingerprint(selected, runtime_identity)
    _record_history(
        history_path,
        history,
        selected,
        run,
        fingerprint=selected_fingerprint,
        runtime_identity_sha256=runtime_identity_sha256,
    )
    result = {
        "schema": SCHEMA,
        "mode": "shadow",
        "production_write_allowed": False,
        "status": "SHADOW_CYCLE_COMPLETE",
        "model": model,
        "selected_candidate_id": selected["candidate_id"],
        "deferred_candidate_ids": deferred,
        "already_observed_candidate_ids": already_observed,
        "runtime_identity_sha256": runtime_identity_sha256,
        "ready_count": len(ready),
        "discovery_path": str(discovery_path),
        "candidate_spec": spec_result["candidate_spec"],
        "history_path": str(history_path),
        "shadow_run": run,
        "started_at": started_at,
        "finished_at": _now(),
    }
    _atomic_json(shadow_root / "last_cycle.json", result)
    return result


def run_cycle(
    config: dict,
    *,
    discover_fn=discovery.discover,
    materialize_fn=materialize.materialize,
    run_shadow_fn=runner.run_shadow,
) -> dict:
    shadow_root = Path(_require(config, "shadow_root")).expanduser().resolve(strict=False)
    cwd = str(_require(config, "cwd"))

    # Validate and create the scratch root before acquiring the process lock.
    runner.validate_shadow_root(
        str(shadow_root),
        candidate_cwd=cwd,
        forbidden_roots=config.get("forbidden_roots", []),
    )
    shadow_root.mkdir(parents=True, exist_ok=True)

    with _cycle_lock(shadow_root):
        return _run_cycle_locked(
            config,
            discover_fn=discover_fn,
            materialize_fn=materialize_fn,
            run_shadow_fn=run_shadow_fn,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    try:
        result = run_cycle(load_config(args.config))
    except Exception as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "mode": "shadow",
            "production_write_allowed": False,
            "status": "SHADOW_PIPELINE_ERROR",
            "error": str(exc),
        }, sort_keys=True))
        return 2
    print(json.dumps(discovery._json_safe(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
