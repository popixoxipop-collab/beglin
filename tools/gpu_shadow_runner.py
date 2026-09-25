#!/usr/bin/env python3
"""Run the certified GPU precision autopilot in production-shadow mode.

This wrapper deliberately has no production mutation path. It consumes an
explicit candidate spec, launches tools/gpu_autopilot.py with a scratch-only
control root, scrubs inherited mutation/control environment variables, and
persists only shadow evidence/results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping, Optional


SCHEMA = "gpu-shadow-run-v1"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

REQUIRED_SPEC = (
    "role",
    "layer",
    "n",
    "event",
    "reference",
    "prompt_len",
    "g4_manifest",
    "g6_manifest",
    "cwd",
    "binary",
    "binary_sha256",
    "checkpoint_sha256",
    "moe_base",
    "safetensors",
)

# Never inherit a live control path into the child process. gpu_autopilot.py is
# itself scratch-root scoped, but shadow mode adds a second independent guard.
MUTATION_ENV = (
    "QWEN_MOE_PROMOTION_FILE",
    "QWEN_MOE_PROMOTION_FILE_NQ",
    "QWEN_MOE_DEMOTION_FILE",
    "QWEN_MOE_DEMOTION_FILE_NQ",
    "QWEN_MOE_GPU_TXN_FILE",
    "QWEN_MOE_GPU_APPLIED_ACK",
    "QWEN_PRECISION_CONTROL_DIR",
    "QWEN_AUTOPILOT_GPU",
    "QWEN_AUTOPILOT_P3",
    "QWEN_AUTOPILOT_P5",
)


class ShadowModeError(RuntimeError):
    pass


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(tmp, "w") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_spec(path: str) -> dict:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ShadowModeError("candidate spec must be a JSON object")
    validate_spec(value)
    return value


def validate_spec(spec: Mapping) -> None:
    missing = [key for key in REQUIRED_SPEC if key not in spec]
    if missing:
        raise ShadowModeError("candidate spec missing: " + ", ".join(missing))
    if not isinstance(spec["event"], dict) or not isinstance(spec["reference"], dict):
        raise ShadowModeError("event/reference must be JSON objects")
    for key in ("layer", "n", "prompt_len"):
        try:
            value = int(spec[key])
        except (TypeError, ValueError) as exc:
            raise ShadowModeError(f"{key} must be an integer") from exc
        if value < 0:
            raise ShadowModeError(f"{key} must be non-negative")
    for key in ("binary_sha256", "checkpoint_sha256"):
        value = str(spec[key]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ShadowModeError(f"{key} must be 64 lowercase/uppercase hex chars")


def _resolved(path: str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_shadow_root(
    shadow_root: str,
    *,
    candidate_cwd: str,
    forbidden_roots: Iterable[str] = (),
) -> Path:
    root = _resolved(shadow_root)
    if root == Path("/"):
        raise ShadowModeError("shadow root may not be filesystem root")

    repo = _resolved(candidate_cwd)
    if _is_within(root, repo) or _is_within(repo, root):
        raise ShadowModeError(
            "shadow root must be outside the engine checkout/worktree"
        )

    defaults = ["/private/tmp/qng64_ctl"]
    env_control = os.environ.get("QWEN_PRECISION_CONTROL_DIR")
    if env_control:
        defaults.append(env_control)
    env_forbidden = os.environ.get("GPU_SHADOW_FORBID_ROOTS")
    if env_forbidden:
        defaults.extend(x for x in env_forbidden.split(os.pathsep) if x)
    defaults.extend(forbidden_roots)

    for item in defaults:
        forbidden = _resolved(str(item))
        if root == forbidden or _is_within(root, forbidden):
            raise ShadowModeError(
                f"shadow root overlaps forbidden production/control root: {forbidden}"
            )
    return root


def build_child_env(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if base is None else base)
    for key in MUTATION_ENV:
        env.pop(key, None)
    env["QWEN_GPU_SHADOW_MODE"] = "1"
    env["QWEN_GPU_PRODUCTION_WRITE"] = "0"
    return env


def build_autopilot_command(
    spec: Mapping,
    *,
    autopilot: str,
    control_root: Path,
    python_bin: str,
):
    validate_spec(spec)
    return [
        python_bin,
        str(autopilot),
        "--role",
        str(spec["role"]),
        "--layer",
        str(int(spec["layer"])),
        "--n",
        str(int(spec["n"])),
        "--event-json",
        canonical_json(spec["event"]),
        "--reference-json",
        canonical_json(spec["reference"]),
        "--prompt-len",
        str(int(spec["prompt_len"])),
        "--g4-manifest",
        str(spec["g4_manifest"]),
        "--g6-manifest",
        str(spec["g6_manifest"]),
        "--cwd",
        str(spec["cwd"]),
        "--binary",
        str(spec["binary"]),
        "--binary-sha256",
        str(spec["binary_sha256"]).lower(),
        "--checkpoint-sha256",
        str(spec["checkpoint_sha256"]).lower(),
        "--moe-base",
        str(spec["moe_base"]),
        "--safetensors",
        str(spec["safetensors"]),
        "--control-root",
        str(control_root),
    ]


def _extract_last_json(text: str):
    for line in reversed(text.splitlines()):
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


def _shadow_status(returncode: int, child_payload) -> str:
    if returncode != 0:
        return "SHADOW_ERROR"
    if not isinstance(child_payload, dict):
        return "SHADOW_COMPLETED_UNCLASSIFIED"
    raw = str(
        child_payload.get("status")
        or child_payload.get("action")
        or child_payload.get("decision")
        or ""
    ).upper()
    if raw in {"ADMITTED", "CANARY_PASS", "CANARY_PASS_NO_AUTO_EXPANSION"}:
        return "SHADOW_ADMITTED"
    if raw.startswith("REJECT") or raw.startswith("BLOCK"):
        return "SHADOW_REJECTED"
    if "ROLLBACK" in raw or "REGRESSION" in raw:
        return "SHADOW_ROLLBACK_OR_REGRESSION"
    return "SHADOW_COMPLETED_UNCLASSIFIED"


def run_shadow(
    spec: Mapping,
    *,
    shadow_root: str,
    autopilot: str,
    python_bin: str = sys.executable,
    run_id: Optional[str] = None,
    timeout: int = 3600,
    forbidden_roots: Iterable[str] = (),
    base_env: Optional[Mapping[str, str]] = None,
) -> dict:
    validate_spec(spec)
    root = validate_shadow_root(
        shadow_root,
        candidate_cwd=str(spec["cwd"]),
        forbidden_roots=forbidden_roots,
    )
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid())
    if not SAFE_RUN_ID.fullmatch(run_id):
        raise ShadowModeError("run_id contains unsupported characters")

    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise ShadowModeError(f"shadow run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    spec_copy = dict(spec)
    command = build_autopilot_command(
        spec_copy,
        autopilot=autopilot,
        control_root=run_dir / "control",
        python_bin=python_bin,
    )
    env = build_child_env(base_env)
    started_at = _utc_now()

    input_record = {
        "schema": SCHEMA,
        "mode": "shadow",
        "production_write_allowed": False,
        "run_id": run_id,
        "candidate_spec": spec_copy,
        "candidate_spec_sha256": sha256_json(spec_copy),
        "command": command,
        "scrubbed_env_keys": sorted(MUTATION_ENV),
        "started_at": started_at,
    }
    _atomic_json(run_dir / "shadow_input.json", input_record)

    proc = subprocess.run(
        command,
        cwd=str(spec["cwd"]),
        env=env,
        capture_output=True,
        text=True,
        timeout=int(timeout),
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    _atomic_text(run_dir / "stdout.log", stdout)
    _atomic_text(run_dir / "stderr.log", stderr)

    child_payload = _extract_last_json(stdout + "\n" + stderr)
    result = {
        "schema": SCHEMA,
        "mode": "shadow",
        "production_write_allowed": False,
        "run_id": run_id,
        "shadow_status": _shadow_status(proc.returncode, child_payload),
        "child_returncode": int(proc.returncode),
        "child_payload": child_payload,
        "candidate_spec_sha256": input_record["candidate_spec_sha256"],
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        "control_root": str(run_dir / "control"),
        "started_at": started_at,
        "finished_at": _utc_now(),
    }
    result["result_sha256"] = sha256_json(result)
    _atomic_json(run_dir / "shadow_result.json", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-spec", required=True)
    ap.add_argument("--shadow-root", required=True)
    ap.add_argument("--autopilot", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--run-id")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--forbid-root", action="append", default=[])
    args = ap.parse_args()

    try:
        result = run_shadow(
            load_spec(args.candidate_spec),
            shadow_root=args.shadow_root,
            autopilot=args.autopilot,
            python_bin=args.python,
            run_id=args.run_id,
            timeout=args.timeout,
            forbidden_roots=args.forbid_root,
        )
    except (ShadowModeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "mode": "shadow",
            "production_write_allowed": False,
            "shadow_status": "SHADOW_ERROR",
            "error": str(exc),
        }, sort_keys=True))
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
