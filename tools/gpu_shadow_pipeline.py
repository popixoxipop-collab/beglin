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
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import gpu_shadow_discovery as discovery
import gpu_shadow_materialize as materialize
import gpu_shadow_runner as runner


SCHEMA = "gpu-shadow-pipeline-v1"


class ShadowPipelineError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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


def run_cycle(
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

    # Reuse the runner's stronger root guard before any local writes.
    runner.validate_shadow_root(
        str(shadow_root),
        candidate_cwd=cwd,
        forbidden_roots=config.get("forbidden_roots", []),
    )
    shadow_root.mkdir(parents=True, exist_ok=True)

    started_at = _now()
    discovered = discover_fn(model, limit)
    discovery_path = shadow_root / "discovery.json"
    _atomic_json(discovery_path, discovered)

    ready = discovered.get("ready") or []
    if not ready:
        result = {
            "schema": SCHEMA,
            "mode": "shadow",
            "production_write_allowed": False,
            "status": "NO_READY_CANDIDATE",
            "model": model,
            "ready_count": 0,
            "discovery_path": str(discovery_path),
            "started_at": started_at,
            "finished_at": _now(),
        }
        _atomic_json(shadow_root / "last_cycle.json", result)
        return result

    # Deliberately serialize one candidate per shadow cycle. This preserves the
    # same one-target-at-a-time evidence discipline as the real canary path.
    selected = ready[0]
    deferred = [row.get("candidate_id") for row in ready[1:]]

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

    result = {
        "schema": SCHEMA,
        "mode": "shadow",
        "production_write_allowed": False,
        "status": "SHADOW_CYCLE_COMPLETE",
        "model": model,
        "selected_candidate_id": selected["candidate_id"],
        "deferred_candidate_ids": deferred,
        "ready_count": len(ready),
        "discovery_path": str(discovery_path),
        "candidate_spec": spec_result["candidate_spec"],
        "shadow_run": run,
        "started_at": started_at,
        "finished_at": _now(),
    }
    _atomic_json(shadow_root / "last_cycle.json", result)
    return result


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
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
