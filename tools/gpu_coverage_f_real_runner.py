#!/usr/bin/env python3
"""Agent-F bounded real-GPU candidate runner.

Uses only the documented gpu_autopilot.py CLI contract from the certified GPU
handoff.  It deliberately does not import or guess private gpu_observer_control
function signatures.

This runner can:
- execute bounded n candidates for one target through the real GPU autopilot,
- keep one scratch control root per candidate,
- capture raw stdout/stderr and parse the final JSON object,
- label G4 rejection / admitted / canary outcomes without promoting them to
  production status.

It cannot claim G5 rollback for a bad candidate that G4 rejects.  That path
remains blocked until the current certified gpu_observer_control API is
available to Agent F.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCHEMA = "gpu-coverage-f-real-run-v1"


class RunnerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise RunnerError(f"{path} must contain a JSON object")
    return value


def _last_json(text: str):
    lines = text.splitlines()
    for i in range(len(lines)):
        candidate = "\n".join(lines[i:]).strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_config(cfg: dict, repo_root: Path) -> dict:
    required = [
        "role", "layer", "candidate_ns", "event", "reference", "prompt_len",
        "g4_manifest", "g6_manifest", "binary", "binary_sha256",
        "checkpoint_sha256", "moe_base", "safetensors", "control_root",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise RunnerError("missing config key(s): " + ", ".join(missing))

    role = str(cfg["role"])
    layer = int(cfg["layer"])
    candidate_ns = [int(x) for x in cfg["candidate_ns"]]
    if not candidate_ns or len(candidate_ns) > 3:
        raise RunnerError("candidate_ns must contain 1..3 values")
    if len(set(candidate_ns)) != len(candidate_ns):
        raise RunnerError("candidate_ns contains duplicates")
    if any(n not in (5, 6, 7) for n in candidate_ns):
        raise RunnerError("Agent-F bounded search allows only n in {5,6,7}")

    if not isinstance(cfg["event"], dict) or not isinstance(cfg["reference"], dict):
        raise RunnerError("event/reference must be objects")
    for k in ("orig_token", "corrected_token", "pos"):
        int(cfg["event"][k])
    int(cfg["reference"]["emitted_token"])
    prompt_len = int(cfg["prompt_len"])
    if prompt_len <= 0:
        raise RunnerError("prompt_len must be positive")

    checkpoint_sha = str(cfg["checkpoint_sha256"]).lower()
    if len(checkpoint_sha) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha):
        raise RunnerError("checkpoint_sha256 must be verified 64-hex")
    binary_sha = str(cfg["binary_sha256"]).lower()
    if len(binary_sha) != 64 or any(c not in "0123456789abcdef" for c in binary_sha):
        raise RunnerError("binary_sha256 must be 64-hex")

    paths = {}
    for key in ("g4_manifest", "g6_manifest", "binary", "moe_base", "safetensors", "control_root"):
        paths[key] = Path(cfg[key]).expanduser().resolve(strict=False)

    if not paths["g4_manifest"].is_file():
        raise RunnerError(f"missing g4_manifest: {paths['g4_manifest']}")
    if not paths["g6_manifest"].is_file():
        raise RunnerError(f"missing g6_manifest: {paths['g6_manifest']}")
    if not paths["binary"].is_file():
        raise RunnerError(f"missing GPU binary: {paths['binary']}")
    if _sha256_file(paths["binary"]) != binary_sha:
        raise RunnerError("GPU binary SHA does not match config")
    if not paths["moe_base"].exists():
        raise RunnerError(f"missing moe_base: {paths['moe_base']}")
    if not paths["safetensors"].is_file():
        raise RunnerError(f"missing safetensors index: {paths['safetensors']}")

    control = paths["control_root"]
    if repo_root == control or repo_root in control.parents or control in repo_root.parents:
        raise RunnerError("control_root must not overlap repo_root")

    return {
        "role": role,
        "layer": layer,
        "candidate_ns": candidate_ns,
        "event": {
            "orig_token": int(cfg["event"]["orig_token"]),
            "corrected_token": int(cfg["event"]["corrected_token"]),
            "pos": int(cfg["event"]["pos"]),
        },
        "reference": {"emitted_token": int(cfg["reference"]["emitted_token"])},
        "prompt_len": prompt_len,
        "paths": paths,
        "binary_sha256": binary_sha,
        "checkpoint_sha256": checkpoint_sha,
        "timeout_seconds": min(max(int(cfg.get("timeout_seconds", 3600)), 1), 3600),
        "max_candidates": 3,
        "production_write_allowed": False,
    }


def _classify(returncode: int, payload: dict | None) -> str:
    if returncode != 0:
        return "RUN_ERROR"
    if not isinstance(payload, dict):
        return "UNCLASSIFIED"
    raw = str(
        payload.get("final_status")
        or payload.get("status")
        or payload.get("action")
        or payload.get("decision")
        or ""
    ).upper()
    if raw in {"ADMITTED", "CANARY_PASS", "CANARY_PASS_NO_AUTO_EXPANSION"}:
        return "ADMITTED"
    if "REJECTED_AT_G4" in raw or raw.startswith("REJECT"):
        return "REJECTED_AT_G4"
    if "ROLLBACK" in raw or "REGRESSION" in raw:
        return "ROLLBACK_OR_REGRESSION"
    return "UNCLASSIFIED"


def run_candidate(repo_root: Path, cfg: dict, n: int, run_root: Path) -> dict:
    autopilot = repo_root / "tools" / "gpu_autopilot.py"
    if autopilot.is_symlink() or not autopilot.is_file():
        raise RunnerError(f"certified gpu_autopilot.py missing/non-regular: {autopilot}")

    candidate_root = run_root / f"n{n}"
    control_root = candidate_root / "control"
    candidate_root.mkdir(parents=True, exist_ok=False)

    argv = [
        sys.executable, str(autopilot),
        "--role", cfg["role"],
        "--layer", str(cfg["layer"]),
        "--n", str(n),
        "--event-json", json.dumps(cfg["event"], sort_keys=True, separators=(",", ":")),
        "--reference-json", json.dumps(cfg["reference"], sort_keys=True, separators=(",", ":")),
        "--prompt-len", str(cfg["prompt_len"]),
        "--g4-manifest", str(cfg["paths"]["g4_manifest"]),
        "--g6-manifest", str(cfg["paths"]["g6_manifest"]),
        "--cwd", str(repo_root),
        "--binary", str(cfg["paths"]["binary"]),
        "--binary-sha256", cfg["binary_sha256"],
        "--checkpoint-sha256", cfg["checkpoint_sha256"],
        "--moe-base", str(cfg["paths"]["moe_base"]),
        "--safetensors", str(cfg["paths"]["safetensors"]),
        "--control-root", str(control_root),
    ]
    started = _now()
    proc = subprocess.run(
        argv,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=cfg["timeout_seconds"],
        env=os.environ.copy(),
    )
    (candidate_root / "stdout.log").write_text(proc.stdout or "")
    (candidate_root / "stderr.log").write_text(proc.stderr or "")
    payload = _last_json((proc.stdout or "") + "\n" + (proc.stderr or ""))
    result = {
        "schema": SCHEMA,
        "run_kind": "real_gpu_candidate",
        "production_write_allowed": False,
        "role": cfg["role"],
        "layer": cfg["layer"],
        "n": n,
        "returncode": int(proc.returncode),
        "classification": _classify(proc.returncode, payload),
        "payload": payload,
        "started_at": started,
        "finished_at": _now(),
        "binary_sha256": cfg["binary_sha256"],
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "stdout_sha256": hashlib.sha256((proc.stdout or "").encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256((proc.stderr or "").encode()).hexdigest(),
        "candidate_root": str(candidate_root),
    }
    _atomic_json(candidate_root / "result.json", result)
    return result


def run_all(config_path: Path, output_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = validate_config(_load(config_path), repo_root)
    root = cfg["paths"]["control_root"]
    root.mkdir(parents=True, exist_ok=True)
    run_root = root / ("f-coverage-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_root.mkdir(parents=True, exist_ok=False)

    results = []
    for n in cfg["candidate_ns"][:cfg["max_candidates"]]:
        results.append(run_candidate(repo_root, cfg, n, run_root))

    bad_candidates = [
        r["n"] for r in results
        if r["classification"] in {"REJECTED_AT_G4", "ROLLBACK_OR_REGRESSION"}
    ]
    value = {
        "schema": SCHEMA,
        "production_write_allowed": False,
        "role": cfg["role"],
        "layer": cfg["layer"],
        "candidate_ns": cfg["candidate_ns"],
        "results": results,
        "bad_candidates": bad_candidates,
        "g5_status": (
            "BAD_CANDIDATE_FOUND_G5_NOT_RUN"
            if bad_candidates
            else "NO_BAD_CANDIDATE_WITHIN_BUDGET"
        ),
        "note": (
            "A G4 rejection is a genuine bad candidate signal, but this runner does "
            "not claim G5 rollback. G5 must use the current certified observer/control API."
        ),
        "finished_at": _now(),
    }
    _atomic_json(output_path, value)
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    try:
        value = run_all(Path(args.config), Path(args.output))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "ok": True,
        "bad_candidates": value["bad_candidates"],
        "g5_status": value["g5_status"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
