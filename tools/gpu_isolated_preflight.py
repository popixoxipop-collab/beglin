#!/usr/bin/env python3
"""Isolated MLX/Metal A/B preflight using startup-only qNg64 promotion.

This runner deliberately does *not* hot-promote a live worker. Baseline and
candidate each run in a fresh process, with correction forced OFF. The worker
must self-report its exact applied policy through QWEN_MOE_GPU_APPLIED_ACK,
and that ACK must match the policy the controller requested before token
results can be admitted as GPU evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import backend_capabilities as bc
import gpu_runtime_control as grc
import precision_context as pc


ACK_SCHEMA = grc.ACK_SCHEMA
BACKEND = "mlx_metal"
SUPPORTED_QNG64_NS = set(bc.GPU_NATIVE_QNG64) | set(bc.GPU_CUSTOM_QNG64)
ARCH_GATE = {
    "mla": "QWEN_MOE_GPU_CBATCH_ONLINE",
    "gqa": "QWEN_MOE_GPU_GQA_CBATCH_ONLINE",
}
ARCH_PREFIX = {
    "mla": "moe gpu cb online",
    "gqa": "moe gpu gqa cb online",
}
VALIDATION_RE = re.compile(
    r"GPU_VALIDATION_V1 backend=(\S+) arch=(\S+) correction=(\S+) "
    r"finite_logits=(\d+) logits_checked=(\d+) requests=(\d+)"
)


class GpuPreflightError(RuntimeError):
    pass


def normalize_policy(policy) -> list[dict]:
    rows = pc.normalize_policy(policy)
    seen = set()
    for row in rows:
        key = (row["role"], int(row["layer"]))
        if key in seen:
            raise GpuPreflightError(
                f"duplicate policy target: {row['role']}/L{row['layer']}"
            )
        seen.add(key)
        if int(row["n"]) not in SUPPORTED_QNG64_NS:
            raise GpuPreflightError(
                f"n={row['n']} is outside current GPU qNg64 ladder "
                f"{sorted(SUPPORTED_QNG64_NS)}"
            )
    return rows


def render_promotion_file(policy) -> str:
    rows = normalize_policy(policy)
    return "".join(
        f"{row['role']} {int(row['layer'])} {int(row['n'])}\n"
        for row in rows
    )


def _run(host, command, *, input_text=None, timeout=180):
    argv = ["ssh", host, command] if host else ["sh", "-lc", command]
    return subprocess.run(
        argv,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _require_ok(proc, what):
    if proc.returncode != 0:
        raise GpuPreflightError(
            f"{what} failed rc={proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return proc


def _write_text(host, path, text):
    path = str(path)
    directory = str(Path(path).parent)
    tmp = f"{path}.tmp.{os.getpid()}"
    cmd = (
        f"mkdir -p {shlex.quote(directory)} && "
        f"cat > {shlex.quote(tmp)} && "
        f"mv {shlex.quote(tmp)} {shlex.quote(path)}"
    )
    _require_ok(
        _run(host, cmd, input_text=text, timeout=30),
        f"write {path}",
    )


def _read_text(host, path):
    p = _require_ok(
        _run(host, f"cat {shlex.quote(str(path))}", timeout=30),
        f"read {path}",
    )
    return p.stdout


def _remove(host, path):
    _require_ok(
        _run(host, f"rm -f {shlex.quote(str(path))}", timeout=30),
        f"remove {path}",
    )


def _binary_identity(host, binary):
    q = shlex.quote(str(binary))
    p = _require_ok(
        _run(
            host,
            (
                f"set -e; shasum -a 256 {q}; "
                f"stat -f '%z' {q}; hostname; uname -m"
            ),
            timeout=30,
        ),
        "collect GPU worker identity",
    )
    lines = p.stdout.splitlines()
    if len(lines) < 4:
        raise GpuPreflightError(f"incomplete worker identity: {lines!r}")
    sha = lines[0].split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise GpuPreflightError(f"invalid binary sha256: {lines[0]!r}")
    return {
        "host": lines[2].strip(),
        "arch": lines[3].strip(),
        "binary_path": str(binary),
        "binary_sha256": sha,
        "binary_size": int(lines[1].strip()),
    }


def _require_same_worker_identity(identity, capability):
    worker = capability.get("worker")
    if not isinstance(worker, dict):
        raise GpuPreflightError("capability artifact is missing worker identity")
    for key in ("host", "arch", "binary_sha256", "binary_size"):
        if str(worker.get(key)) != str(identity.get(key)):
            raise GpuPreflightError(
                f"binary identity changed between probes for {key}: "
                f"identity={identity.get(key)!r} capability={worker.get(key)!r}"
            )


def _require_binary_capability(host, binary, policy, identity=None):
    try:
        capability = bc.collect(host, str(binary))
    except Exception as exc:
        raise GpuPreflightError(
            f"could not collect GPU binary capability: {exc}"
        ) from exc
    gpu = capability["backends"]["mlx_metal"]
    runtime_control = gpu["runtime_control"]
    if not gpu.get("compiled"):
        raise GpuPreflightError("worker binary has no MLX/Metal backend")
    if not runtime_control.get("compiled"):
        missing = [
            symbol
            for symbol, present in runtime_control.get("symbols", {}).items()
            if not present
        ]
        raise GpuPreflightError(
            f"worker binary has incomplete GPU control plane: missing={missing}"
        )
    if capability["worker"].get("arch") not in {"arm64", "aarch64"}:
        raise GpuPreflightError(
            "MLX/Metal preflight requires Apple arm64 worker"
        )
    widths = gpu.get("qng64_widths", {})
    for row in normalize_policy(policy):
        status = widths.get(str(int(row["n"])))
        if status in {None, "UNSUPPORTED_BINARY", "FAILED"}:
            raise GpuPreflightError(
                f"GPU binary does not support {row['role']}/L{row['layer']} n={row['n']}: {status}"
            )
    if identity is not None:
        _require_same_worker_identity(identity, capability)
    return capability


def _remote_sha256(host, path):
    p = _require_ok(
        _run(host, f"shasum -a 256 {shlex.quote(str(path))}", timeout=30),
        f"hash {path}",
    )
    sha = p.stdout.split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise GpuPreflightError(f"invalid SHA-256 for {path}: {p.stdout!r}")
    return sha


def _write_json(host, path, value):
    _write_text(
        host,
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def _sha256_text(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


def _parse_ack_text(text: str) -> dict:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GpuPreflightError(f"invalid runtime ACK JSON: {exc}") from exc
    try:
        return grc.normalize_runtime_ack(raw)
    except grc.RuntimeControlError as exc:
        raise GpuPreflightError(str(exc)) from exc


def parse_emitted_tokens(output: str, architecture: str, req: int = 0) -> list[int]:
    if architecture not in ARCH_PREFIX:
        raise GpuPreflightError(
            f"unsupported architecture mode {architecture!r}"
        )
    prefix = re.escape(ARCH_PREFIX[architecture])
    matches = re.findall(
        rf"\[{prefix}\] req {int(req)} .*? tokens:\s*([^\n\r]*)",
        output,
    )
    if not matches:
        return []
    toks = []
    for part in matches[-1].strip().split():
        try:
            toks.append(int(part))
        except ValueError:
            break
    return toks


def emitted_token_at_position(
    output: str,
    *,
    architecture: str,
    prompt_len: int,
    pos: int,
    req: int = 0,
):
    gen_idx = int(pos) - (int(prompt_len) - 1)
    tokens = parse_emitted_tokens(output, architecture, req=req)
    if 0 <= gen_idx < len(tokens):
        return tokens[gen_idx]
    return None


def parse_validation_report(output: str, architecture: str) -> dict:
    if architecture not in ARCH_GATE:
        raise GpuPreflightError(
            f"unsupported architecture mode {architecture!r}"
        )
    matches = VALIDATION_RE.findall(output)
    if not matches:
        raise GpuPreflightError("GPU_VALIDATION_V1 report missing")
    backend, arch, correction, finite, checked, requests = matches[-1]
    if backend != BACKEND:
        raise GpuPreflightError(
            f"validation backend mismatch: {backend!r}"
        )
    if arch != architecture:
        raise GpuPreflightError(
            f"validation architecture mismatch: expected={architecture!r} actual={arch!r}"
        )
    return {
        "backend": backend,
        "architecture": arch,
        "correction_mode": correction,
        "finite_logits": finite == "1",
        "logits_checked": int(checked),
        "requests": int(requests),
    }


def _worker_env(
    *,
    architecture,
    moe_base,
    manifest,
    promotion_file,
    safetensors,
    ack_path,
    requests=1,
    slots=1,
    prefill_budget=16,
    stop_extra=None,
):
    if architecture not in ARCH_GATE:
        raise GpuPreflightError(
            f"unsupported architecture mode {architecture!r}"
        )
    env = {
        "QWEN_MOE_BASE": str(moe_base),
        ARCH_GATE[architecture]: "1",
        "QWEN_MOE_CB_PROMPT_MANIFEST": str(manifest),
        "QWEN_MOE_CB_REQS": str(int(requests)),
        "QWEN_MOE_CB_SLOTS": str(int(slots)),
        "QWEN_MOE_CB_PREFILL_BUDGET": str(int(prefill_budget)),
        "QWEN_MOE_PROMOTION_FILE_NQ": str(promotion_file),
        "QWEN_MOE_PROMOTION_SAFETENSORS": str(safetensors),
        "QWEN_MOE_GPU_APPLIED_ACK": str(ack_path),
        "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
        # G4 hard requirement: candidate quality is measured without the
        # correction/oracle path mutating output.
        "QWEN_MOE_NEARTIE_CORRECT": "0",
    }
    if stop_extra is not None:
        env["QWEN_MOE_CB_STOP_EXTRA"] = str(int(stop_extra))
    return env


def _command(cwd, binary, env):
    env_text = " ".join(
        f"{key}={shlex.quote(str(value))}"
        for key, value in sorted(env.items())
    )
    return (
        f"cd {shlex.quote(str(cwd))} && "
        "env "
        "-u QWEN_MOE_PROMOTION_FILE "
        "-u QWEN_MOE_DEMOTION_FILE_NQ "
        "-u QWEN_MOE_GPU_TXN_FILE "
        "-u QWEN_MOE_NEARTIE_CORRECT_SAFETENSORS "
        f"{env_text} {shlex.quote(str(binary))}"
    )


def run_isolated_worker(
    *,
    host,
    cwd,
    binary,
    architecture,
    moe_base,
    manifest,
    safetensors,
    run_dir,
    policy,
    timeout=180,
    requests=1,
    slots=1,
    prefill_budget=16,
    stop_extra=None,
):
    rows = normalize_policy(policy)
    expected_hash = pc.policy_hash(rows)
    run_dir = str(run_dir).rstrip("/")
    promotion_file = f"{run_dir}/promotion_nq.txt"
    ack_path = f"{run_dir}/applied_ack.json"

    _write_text(host, promotion_file, render_promotion_file(rows))
    _remove(host, ack_path)

    env = _worker_env(
        architecture=architecture,
        moe_base=moe_base,
        manifest=manifest,
        promotion_file=promotion_file,
        safetensors=safetensors,
        ack_path=ack_path,
        requests=requests,
        slots=slots,
        prefill_budget=prefill_budget,
        stop_extra=stop_extra,
    )
    identity = _binary_identity(host, binary)
    capability = _require_binary_capability(
        host, binary, rows, identity=identity
    )
    proc = _run(
        host,
        _command(cwd, binary, env),
        timeout=timeout,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    worker_log_path = f"{run_dir}/worker.log"
    _write_text(host, worker_log_path, output)
    worker_log_sha256 = _sha256_text(output)
    if proc.returncode != 0:
        raise GpuPreflightError(
            f"isolated GPU worker failed rc={proc.returncode}: "
            f"{output[-4000:]}"
        )

    validation = parse_validation_report(output, architecture)
    if validation["correction_mode"] != "off":
        raise GpuPreflightError(
            "GPU validation report says correction is not OFF"
        )
    if not validation["finite_logits"]:
        raise GpuPreflightError("GPU validation reported non-finite logits")
    if validation["logits_checked"] <= 0:
        raise GpuPreflightError("GPU validation checked zero logits")
    if validation["requests"] != int(requests):
        raise GpuPreflightError(
            f"GPU validation request count mismatch: expected={requests} "
            f"actual={validation['requests']}"
        )

    ack = _parse_ack_text(_read_text(host, ack_path))
    if ack["correction_mode"] != "off":
        raise GpuPreflightError(
            f"runtime correction mode is {ack['correction_mode']!r}, expected 'off'"
        )
    if ack["active_policy_hash"] != expected_hash:
        raise GpuPreflightError(
            "applied startup policy mismatch: "
            f"expected={expected_hash} actual={ack['active_policy_hash']}"
        )
    if ack["active_policy"] != rows:
        raise GpuPreflightError(
            "applied startup policy rows differ from requested policy"
        )
    expected_status = "PROMOTION_APPLIED" if rows else "STARTUP_STATE"
    if ack.get("status") != expected_status:
        raise GpuPreflightError(
            f"unexpected startup ACK status: {ack.get('status')!r}, "
            f"expected {expected_status!r}"
        )
    if int(ack.get("changed_targets", -1)) != len(rows):
        raise GpuPreflightError(
            "runtime changed_targets does not match requested policy size"
        )
    # Fresh process starts at epoch zero and increments once per successfully
    # applied promotion row. This is a second independent proof that all rows
    # were actually committed, not merely echoed by the controller.
    if ack["weight_epoch"] != len(rows):
        raise GpuPreflightError(
            f"unexpected startup weight_epoch={ack['weight_epoch']} "
            f"for {len(rows)} policy rows"
        )

    return {
        "backend": BACKEND,
        "architecture": architecture,
        "correction_mode": "off",
        "policy": rows,
        "applied_policy_hash": ack["active_policy_hash"],
        "weight_epoch": ack["weight_epoch"],
        "worker": identity,
        "capability": capability,
        "promotion_file_sha256": hashlib.sha256(
            render_promotion_file(rows).encode()
        ).hexdigest(),
        "ack": ack,
        "validation": validation,
        "output": output,
        "returncode": proc.returncode,
        "worker_log_path": worker_log_path,
        "worker_log_sha256": worker_log_sha256,
        "manifest_sha256": _remote_sha256(host, manifest),
        "safetensors_sha256": _remote_sha256(host, safetensors),
        "ack_sha256": ack["ack_sha256"],
    }


def _worker_bundle_view(run):
    return {
        "backend": run["backend"],
        "architecture": run["architecture"],
        "correction_mode": run["correction_mode"],
        "policy": run["policy"],
        "applied_policy_hash": run["applied_policy_hash"],
        "weight_epoch": run["weight_epoch"],
        "worker": run["worker"],
        "promotion_file_sha256": run["promotion_file_sha256"],
        "ack_sha256": run["ack_sha256"],
        "validation": run["validation"],
        "worker_log_path": run["worker_log_path"],
        "worker_log_sha256": run["worker_log_sha256"],
        "manifest_sha256": run["manifest_sha256"],
        "safetensors_sha256": run["safetensors_sha256"],
    }


def run_ab_preflight(
    *,
    host,
    cwd,
    binary,
    architecture,
    moe_base,
    manifest,
    safetensors,
    root,
    baseline_policy,
    candidate_policy,
    event,
    reference,
    prompt_len,
    timeout=180,
    persist_bundle=True,
):
    for key in ("orig_token", "corrected_token", "pos"):
        if key not in event:
            raise GpuPreflightError(f"event missing {key}")
    if "emitted_token" not in reference:
        raise GpuPreflightError("reference missing emitted_token")
    if int(reference["emitted_token"]) != int(event["corrected_token"]):
        raise GpuPreflightError(
            "reference token does not match recorded corrected token"
        )

    baseline = run_isolated_worker(
        host=host,
        cwd=cwd,
        binary=binary,
        architecture=architecture,
        moe_base=moe_base,
        manifest=manifest,
        safetensors=safetensors,
        run_dir=f"{str(root).rstrip('/')}/baseline",
        policy=baseline_policy,
        timeout=timeout,
    )
    baseline_token = emitted_token_at_position(
        baseline["output"],
        architecture=architecture,
        prompt_len=prompt_len,
        pos=int(event["pos"]),
    )
    if baseline_token != int(event["orig_token"]):
        raise GpuPreflightError(
            "GPU baseline no longer reproduces recorded original token: "
            f"expected={event['orig_token']} actual={baseline_token}"
        )

    candidate = run_isolated_worker(
        host=host,
        cwd=cwd,
        binary=binary,
        architecture=architecture,
        moe_base=moe_base,
        manifest=manifest,
        safetensors=safetensors,
        run_dir=f"{str(root).rstrip('/')}/candidate",
        policy=candidate_policy,
        timeout=timeout,
    )
    candidate_token = emitted_token_at_position(
        candidate["output"],
        architecture=architecture,
        prompt_len=prompt_len,
        pos=int(event["pos"]),
    )
    if candidate_token != int(reference["emitted_token"]):
        raise GpuPreflightError(
            "GPU candidate token does not match immutable reference: "
            f"expected={reference['emitted_token']} actual={candidate_token}"
        )

    if (
        baseline["worker"]["binary_sha256"]
        != candidate["worker"]["binary_sha256"]
    ):
        raise GpuPreflightError(
            "baseline/candidate ran different GPU binaries"
        )

    for field in ("manifest_sha256", "safetensors_sha256"):
        if (
            baseline.get(field)
            and candidate.get(field)
            and baseline[field] != candidate[field]
        ):
            raise GpuPreflightError(
                f"baseline/candidate input artifact mismatch for {field}"
            )
    baseline_cap = baseline.get("capability")
    candidate_cap = candidate.get("capability")
    if baseline_cap and candidate_cap:
        if (
            baseline_cap.get("worker_identity_sha256")
            != candidate_cap.get("worker_identity_sha256")
        ):
            raise GpuPreflightError(
                "baseline/candidate capability worker identity mismatch"
            )

    result = {
        "status": "passed",
        "backend": BACKEND,
        "architecture": architecture,
        "binary_sha256": candidate["worker"]["binary_sha256"],
        "baseline_policy_hash": baseline["applied_policy_hash"],
        "candidate_policy_hash": candidate["applied_policy_hash"],
        "baseline_epoch": baseline["weight_epoch"],
        "candidate_epoch": candidate["weight_epoch"],
        "baseline_emitted_token": baseline_token,
        "candidate_emitted_token": candidate_token,
        "reference_emitted_token": int(reference["emitted_token"]),
        "correction_mode": "off",
        "baseline": baseline,
        "candidate": candidate,
    }
    if persist_bundle:
        if not candidate_cap:
            raise GpuPreflightError(
                "cannot persist evidence bundle without binary capability artifact"
            )
        root_path = str(root).rstrip("/")
        inputs = {
            "schema": "gpu-preflight-inputs-v1",
            "architecture": architecture,
            "manifest_path": str(manifest),
            "manifest_sha256": candidate["manifest_sha256"],
            "safetensors_path": str(safetensors),
            "safetensors_sha256": candidate["safetensors_sha256"],
            "prompt_len": int(prompt_len),
            "event": event,
            "reference": reference,
            "baseline_policy": normalize_policy(baseline_policy),
            "candidate_policy": normalize_policy(candidate_policy),
        }
        verdict = {
            "schema": "gpu-isolated-preflight-verdict-v1",
            "status": "passed",
            "backend": BACKEND,
            "architecture": architecture,
            "binary_sha256": result["binary_sha256"],
            "worker_identity_sha256": candidate_cap["worker_identity_sha256"],
            "capability_sha256": pc.sha256_json(candidate_cap),
            "inputs_sha256": pc.sha256_json(inputs),
            "baseline_policy_hash": result["baseline_policy_hash"],
            "candidate_policy_hash": result["candidate_policy_hash"],
            "baseline_epoch": result["baseline_epoch"],
            "candidate_epoch": result["candidate_epoch"],
            "baseline_emitted_token": baseline_token,
            "candidate_emitted_token": candidate_token,
            "reference_emitted_token": int(reference["emitted_token"]),
            "correction_mode": "off",
            "baseline": _worker_bundle_view(baseline),
            "candidate": _worker_bundle_view(candidate),
        }
        verdict["verdict_payload_sha256"] = pc.sha256_json(verdict)
        capability_path = f"{root_path}/capability.json"
        inputs_path = f"{root_path}/inputs.json"
        verdict_path = f"{root_path}/verdict.json"
        _write_json(host, capability_path, candidate_cap)
        _write_json(host, inputs_path, inputs)
        _write_json(host, verdict_path, verdict)
        result["evidence_bundle"] = {
            "capability_path": capability_path,
            "capability_sha256": verdict["capability_sha256"],
            "inputs_path": inputs_path,
            "inputs_sha256": verdict["inputs_sha256"],
            "verdict_path": verdict_path,
            "verdict_payload_sha256": verdict["verdict_payload_sha256"],
        }
    return result


def to_planner_evidence(result: dict, *, context_hash: str, expected_epoch: int) -> dict:
    """Translate isolated A/B proof into planner evidence without faking a live epoch."""
    if result.get("status") != "passed":
        raise GpuPreflightError("only passed isolated preflight can become planner evidence")
    if result.get("backend") != BACKEND:
        raise GpuPreflightError("isolated preflight backend is not mlx_metal")
    if result.get("correction_mode") != "off":
        raise GpuPreflightError("isolated preflight correction mode must be OFF")
    binary_sha256 = result.get("binary_sha256")
    if not isinstance(binary_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", binary_sha256):
        raise GpuPreflightError("isolated preflight lacks valid binary_sha256")
    baseline_hash = result.get("baseline_policy_hash")
    candidate_hash = result.get("candidate_policy_hash")
    if not baseline_hash or not candidate_hash:
        raise GpuPreflightError("isolated preflight lacks policy hashes")
    baseline_epoch = int(result.get("baseline_epoch", -1))
    candidate_epoch = int(result.get("candidate_epoch", -1))
    # PRE and POST are separate isolated processes. Their startup epochs count
    # promotions applied inside each process, so comparing candidate_epoch >
    # baseline_epoch is only valid for the historical empty-baseline case.
    # With a resident baseline policy both workers may legitimately report the
    # same epoch count while binding different exact policies. The immutable
    # policy hashes are the cross-process transition proof.
    if baseline_epoch < 0 or candidate_epoch < 0:
        raise GpuPreflightError("isolated preflight lacks valid startup epochs")
    if baseline_hash == candidate_hash:
        raise GpuPreflightError(
            "isolated candidate policy does not differ from baseline policy"
        )
    if candidate_epoch == 0:
        raise GpuPreflightError(
            "isolated candidate worker did not apply a startup precision policy"
        )
    return {
        "evidence_mode": "isolated_restart",
        "status": "passed",
        "pass": True,
        "backend": BACKEND,
        "context_hash": str(context_hash),
        "expected_epoch": int(expected_epoch),
        "baseline_policy_hash": baseline_hash,
        "requested_policy_hash": candidate_hash,
        "applied_policy_hash": candidate_hash,
        "binary_sha256": binary_sha256.lower(),
        "correction_mode": "off",
        "isolated_baseline_epoch": baseline_epoch,
        "isolated_candidate_epoch": candidate_epoch,
        "baseline_emitted_token": result.get("baseline_emitted_token"),
        "candidate_emitted_token": result.get("candidate_emitted_token"),
        "reference_emitted_token": result.get("reference_emitted_token"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host")
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--architecture", choices=sorted(ARCH_GATE), required=True)
    ap.add_argument("--moe-base", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline-policy-json", required=True)
    ap.add_argument("--candidate-policy-json", required=True)
    ap.add_argument("--event-json", required=True)
    ap.add_argument("--reference-json", required=True)
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    result = run_ab_preflight(
        host=args.host,
        cwd=args.cwd,
        binary=args.binary,
        architecture=args.architecture,
        moe_base=args.moe_base,
        manifest=args.manifest,
        safetensors=args.safetensors,
        root=args.root,
        baseline_policy=json.loads(args.baseline_policy_json),
        candidate_policy=json.loads(args.candidate_policy_json),
        event=json.loads(args.event_json),
        reference=json.loads(args.reference_json),
        prompt_len=args.prompt_len,
        timeout=args.timeout,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
