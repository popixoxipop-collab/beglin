#!/usr/bin/env python3
"""P5 pre-apply live-composition canary.

This validates the *actual* QWEN_MOE_PROMOTION_FILE_NQ pointer path before P5
may touch the live promotion file. qng64_real attribution sweeps exercise a
real packed kernel in the correction/hi-mirror path, but P5-L4 showed that this
can still disagree with the actual base-serving promotion composition.

For every ADD/UPGRADE in a prepared P5 plan:
1. find the latest PRE attribution event for that target;
2. derive a one-request manifest from its recorded manifest+req;
3. run the exact plan preimage as a baseline and require the recorded flip;
4. run the preimage plus the proposed n through QWEN_MOE_PROMOTION_FILE_NQ;
5. PASS only when the emitted token is the recorded corrected token and no
   REAL FLIP at that position was needed.

All promotion files are scratch files; the production file is never modified.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import autopilot_guarded as guarded
import autopilot_observer as observer
import quant_search_n as qsn
import promotion_writeback as pwb

DEFAULT_REPORT = "/private/tmp/qng64_ctl/autopilot_p5_preflight.json"
EVIDENCE_SOURCE = "qng64_live_preflight"
EVIDENCE_TABLE = "moe_live_preflight_results"


class EvidenceStoreUnavailable(RuntimeError):
    pass


def _rest_credentials():
    url = os.environ.get("QWEN_SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("QWEN_SUPABASE_KEY", "")
    if not url or not key:
        raise EvidenceStoreUnavailable(
            "QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY are required for P5 evidence"
        )
    return url, key


def _rest_headers(key, content=False):
    out = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    if content:
        out["Content-Type"] = "application/json"
    return out


def persist_evidence(row):
    """Persist one P5 Evidence Contract v2 row and require representation back."""
    url, key = _rest_credentials()
    payload = dict(row)
    payload.setdefault("source", EVIDENCE_SOURCE)
    req = urllib.request.Request(
        f"{url}/rest/v1/{EVIDENCE_TABLE}",
        data=json.dumps(payload).encode(),
        headers={
            **_rest_headers(key, content=True),
            "Prefer": "return=representation",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        if exc.code in (404, 400) and EVIDENCE_TABLE in body:
            raise EvidenceStoreUnavailable(
                f"P5 evidence table unavailable: HTTP {exc.code}: {body[:300]}"
            ) from exc
        raise RuntimeError(
            f"P5 evidence insert failed: HTTP {exc.code}: {body[:500]}"
        ) from exc
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError(f"P5 evidence insert unverified: response={rows!r}")
    return rows[0]


def fetch_latest_evidence(model, role, layer, n, promotion_preimage_sha256):
    """Return latest durable evidence row for this exact target+n+preimage."""
    url, key = _rest_credentials()
    params = {
        "model": f"eq.{model}",
        "role": f"eq.{role}",
        "layer": f"eq.{int(layer)}",
        "n": f"eq.{int(n)}",
        "promotion_preimage_sha256": f"eq.{promotion_preimage_sha256}",
        "source": f"eq.{EVIDENCE_SOURCE}",
        "select": (
            "id,tested_at,source,model,role,layer,n,corpus,req,pos,"
            "promotion_preimage_sha256,promotion_postimage_sha256,"
            "orig_token,corrected_token,emitted_token,correction_required,"
            "pass,status,reason,manifest,baseline_rc,candidate_rc,evidence_path,"
            "engine_commit"
        ),
        "order": "tested_at.desc",
        "limit": "1",
    }
    qs = urllib.parse.urlencode(params, safe=".,")
    req = urllib.request.Request(
        f"{url}/rest/v1/{EVIDENCE_TABLE}?{qs}",
        headers=_rest_headers(key),
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        if exc.code in (404, 400) and EVIDENCE_TABLE in body:
            raise EvidenceStoreUnavailable(
                f"P5 evidence table unavailable: HTTP {exc.code}: {body[:300]}"
            ) from exc
        raise RuntimeError(
            f"P5 evidence lookup failed: HTTP {exc.code}: {body[:500]}"
        ) from exc
    return rows[0] if rows else None


def _controller_commit():
    repo = Path(__file__).resolve().parents[1]
    p = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    return p.stdout.strip() if p.returncode == 0 else None


def _evidence_row(plan, change, attr, verdict, candidate_hash, status,
                  baseline_rc=None, candidate_rc=None, evidence_path=None,
                  worker_identity=None):
    return {
        "source": EVIDENCE_SOURCE,
        "model": plan["model"],
        "role": change["role"],
        "layer": int(change["layer"]),
        "n": int(change["new_n"]),
        "corpus": attr.get("corpus"),
        "req": int(attr["req"]) if attr.get("req") is not None else None,
        "pos": int(attr["pos"]) if attr.get("pos") is not None else None,
        "promotion_preimage_sha256": plan["before_sha256"],
        "promotion_postimage_sha256": candidate_hash,
        "orig_token": int(attr["orig_argmax"]) if attr.get("orig_argmax") is not None else None,
        "corrected_token": int(attr["corrected_argmax"]) if attr.get("corrected_argmax") is not None else None,
        "emitted_token": verdict.get("emitted_token") if verdict else None,
        "correction_required": bool(verdict.get("real_flips")) if verdict else None,
        "pass": bool(verdict and verdict.get("pass")),
        "status": status,
        "reason": verdict.get("reason") if verdict else status,
        "manifest": attr.get("manifest"),
        "baseline_rc": baseline_rc,
        "candidate_rc": candidate_rc,
        "evidence_path": evidence_path,
        # Never substitute the controller checkout SHA for the worker build.
        # Until a signed/verified worker build manifest exists, leave this NULL.
        "engine_commit": (worker_identity or {}).get("engine_commit"),
    }


def persist_no_current_signal(model, role, layer, n, promotion_preimage_sha256,
                              promotion_postimage_sha256, corpus=None,
                              evidence_path=None, reason=None):
    row = {
        "source": EVIDENCE_SOURCE,
        "model": model,
        "role": role,
        "layer": int(layer),
        "n": int(n),
        "corpus": corpus,
        "req": None,
        "pos": None,
        "promotion_preimage_sha256": promotion_preimage_sha256,
        "promotion_postimage_sha256": promotion_postimage_sha256,
        "orig_token": None,
        "corrected_token": None,
        "emitted_token": None,
        "correction_required": None,
        "pass": False,
        "status": "no_current_signal",
        "reason": reason or "production-matched observation found no attribution",
        "manifest": None,
        "baseline_rc": None,
        "candidate_rc": None,
        "evidence_path": evidence_path,
        "engine_commit": None,
    }
    return persist_evidence(row)
SUPPORTED_PREFLIGHT_BACKENDS = {"cpu"}


def _validate_backend(backend):
    if backend not in SUPPORTED_PREFLIGHT_BACKENDS:
        raise RuntimeError(
            f"backend={backend!r} is not supported by this preflight runner; "
            "GPU evidence must come from the dedicated MLX/Metal adapter"
        )


def _worker_identity(host, bin_path, backend):
    _validate_backend(backend)
    qbin = shlex.quote(bin_path)
    out = _ssh(
        host,
        (
            f"set -e; shasum -a 256 {qbin}; "
            f"stat -f '%z' {qbin}; hostname; uname -m"
        ),
    ).splitlines()
    if len(out) < 4:
        raise RuntimeError(
            f"could not collect worker identity for {host}:{bin_path}: {out!r}"
        )
    sha = out[0].split()[0]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
        raise RuntimeError(f"invalid worker SHA256 from {host}: {out[0]!r}")
    return {
        "backend": backend,
        "host": out[2].strip(),
        "arch": out[3].strip(),
        "binary_path": bin_path,
        "binary_sha256": sha.lower(),
        "binary_size": int(out[1].strip()),
        "engine_commit": None,
    }


def _ssh(host, command, timeout=30):
    p = subprocess.run(
        ["ssh", host, command],
        capture_output=True, text=True, timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"ssh command failed on {host}: "
            f"{p.stderr.strip() or p.stdout.strip()}"
        )
    return p.stdout


def _write_text_remote(host, path, text):
    directory = path.rsplit("/", 1)[0]
    tmp = f"{path}.tmp.{os.getpid()}"
    cmd = (
        f"mkdir -p {shlex.quote(directory)} && "
        f"cat > {shlex.quote(tmp)} && mv {shlex.quote(tmp)} {shlex.quote(path)}"
    )
    p = subprocess.run(
        ["ssh", host, cmd], input=text,
        capture_output=True, text=True, timeout=30,
    )
    if p.returncode != 0:
        raise RuntimeError(f"remote write failed: {p.stderr}")


def _latest_attributed_event(events, role, layer):
    for event in reversed(events):
        for attr in event["attrs"]:
            if attr.get("role") == role and int(attr.get("layer")) == int(layer):
                return event, attr
    raise RuntimeError(
        f"PRE log has no attributed event for {role}/L{layer}"
    )


def _prompt_len(host, selected_line):
    token_file = selected_line.split()[0]
    out = _ssh(host, f"stat -f %z {shlex.quote(token_file)}").strip()
    nbytes = int(out)
    if nbytes % 4:
        raise RuntimeError(
            f"token file byte size {nbytes} is not divisible by 4: {token_file}"
        )
    return nbytes // 4
def _parse_emitted_tokens(output):
    matches = re.findall(
        r"\[moe cb4b\] req 0 .*? tokens:\s*([^\n\r]*)", output
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


def _target_flips(output, pos):
    pattern = re.compile(
        rf"^\[moe neartie\] correct req=0 pos={int(pos)}\b"
        rf"[^\n]*REAL FLIP orig=(\d+) corrected=(\d+)[^\n]*$",
        re.MULTILINE,
    )
    return [(int(a), int(b)) for a, b in pattern.findall(output)]


def classify_candidate(output, pos, prompt_len, corrected):
    flips = _target_flips(output, pos)
    correction_required = bool(flips)

    gen_idx = int(pos) - (int(prompt_len) - 1)
    tokens = _parse_emitted_tokens(output)
    emitted = (
        tokens[gen_idx]
        if 0 <= gen_idx < len(tokens)
        else None
    )
    passed = (not correction_required) and emitted == int(corrected)
    return {
        "pass": passed,
        "emitted_token": emitted,
        "gen_idx": gen_idx,
        "real_flips": flips,
        "reason": (
            "base promotion emitted corrected token without correction"
            if passed else
            (
                "correction REAL FLIP was still required"
                if correction_required else
                f"base promotion emitted {emitted}, expected {corrected}"
            )
        ),
    }


def _run_engine(host, cwd, bin_path, moe_base, manifest, combo,
                promotion_file, safetensors_index, model, corpus,
                events_log, max_pos, timeout):
    env = {
        "QWEN_MOE_BASE": moe_base,
        "QWEN_MOE_CBATCH": "1",
        "QWEN_MOE_CB_ONLINE": "1",
        "QWEN_MOE_CB_PROMPT_MANIFEST": manifest,
        "QWEN_MOE_CB_REQS": "1",
        "QWEN_MOE_NEARTIE_CORRECT": "1",
        "QWEN_MOE_NEARTIE_LOG": "1",
        "QWEN_MOE_NEARTIE_MODEL": model,
        "QWEN_MOE_NEARTIE_CORPUS": corpus,
        "QWEN_MOE_NEARTIE_EVENTS_LOG": events_log,
        "QWEN_MOE_NEARTIE_CORRECT_SAFETENSORS": safetensors_index,
        "QWEN_MOE_ATTRIB": "1",
        "QWEN_MOE_NEARTIE_HI_COMBOS": combo,
        "QWEN_MOE_ATTRIB_MAX_POS": str(max_pos),
        "QWEN_MOE_PROMOTION_FILE_NQ": promotion_file,
    }
    envs = " ".join(
        f"{k}={shlex.quote(str(v))}" for k, v in env.items()
    )
    cmd = (
        f"cd {shlex.quote(cwd)} && "
        f"env -u QWEN_MOE_PROMOTION_FILE -u QWEN_MOE_DEMOTION_FILE_NQ "
        f"{envs} {shlex.quote(bin_path)}"
    )
    p = subprocess.run(
        ["ssh", host, cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return p.returncode, p.stdout + p.stderr


def _baseline_ok(output, orig, corrected, pos):
    return (int(orig), int(corrected)) in _target_flips(output, pos)


def run_preflight(plan_path, log_host, events_log, bin_path, cwd, moe_base,
                  safetensors_index, remote_dir, report_path,
                  timeout=180, backend="cpu"):
    plan = guarded._load_plan(plan_path)
    if plan.get("phase") != "P5-full-auto":
        raise RuntimeError("live preflight only accepts P5-full-auto plans")
    if plan.get("status") != "prepared":
        raise RuntimeError(
            f"preflight requires prepared plan, got {plan.get('status')}"
        )
    if not plan.get("changes"):
        return {"status": "no_changes", "targets": []}

    _validate_backend(backend)
    worker_identity = _worker_identity(plan["ssh_host"], bin_path, backend)
    controller_commit = _controller_commit()

    blob = observer._read_bytes(log_host, events_log)
    events = observer.parse_events(blob, model=plan["model"])
    before = guarded._mapping(plan["before"])
    host = plan["ssh_host"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = f"{remote_dir.rstrip('/')}/{stamp}_{os.getpid()}"
    results = []

    for change in plan["changes"]:
        role = change["role"]
        layer = int(change["layer"])
        n = int(change["new_n"])
        event, attr = _latest_attributed_event(events, role, layer)
        manifest = attr.get("manifest")
        if not manifest:
            raise RuntimeError(
                f"{role}/L{layer} PRE attribution lacks manifest provenance"
            )
        req = int(attr["req"])
        pos = int(attr["pos"])
        orig = int(attr["orig_argmax"])
        corrected = int(attr["corrected_argmax"])

        run_id = f"{role}_L{layer}_req{req}_p{pos}"
        iso, _, selected = qsn.derive_isolated_manifest(
            host, manifest, req, root, run_id,
        )
        plen = _prompt_len(host, selected)
        combo = f"{root}/{run_id}.combo.txt"
        _write_text_remote(host, combo, f"{role} {layer}\n")

        before_file = f"{root}/{run_id}.before.promotions.txt"
        candidate_file = f"{root}/{run_id}.candidate.promotions.txt"
        pwb.write_remote_promotion_file_atomic(host, before_file, before)
        candidate = dict(before)
        candidate[(role, layer)] = n
        candidate_hash = guarded._mapping_hash(candidate)
        pwb.write_remote_promotion_file_atomic(
            host, candidate_file, candidate
        )

        baseline_log = f"{root}/{run_id}.baseline.jsonl"
        candidate_log = f"{root}/{run_id}.candidate.jsonl"
        rc0, out0 = _run_engine(
            host, cwd, bin_path, moe_base, iso, combo, before_file,
            safetensors_index, plan["model"], attr.get("corpus") or "p5-preflight",
            baseline_log, max(pos, 19), timeout,
        )
        if rc0 != 0 or not _baseline_ok(out0, orig, corrected, pos):
            baseline_verdict = {
                "pass": False,
                "emitted_token": None,
                "real_flips": [],
                "reason": (
                    f"baseline isolation failed: rc={rc0}; "
                    f"expected exact {orig}->{corrected} at pos={pos}"
                ),
            }
            persist_evidence(_evidence_row(
                plan, change, attr, baseline_verdict, candidate_hash,
                "baseline_failed", baseline_rc=rc0,
                candidate_rc=None, evidence_path=baseline_log,
                worker_identity=worker_identity,
            ))
            raise RuntimeError(
                f"{role}/L{layer} baseline isolation failed: rc={rc0}; "
                f"expected exact {orig}->{corrected} at pos={pos}"
            )

        rc1, out1 = _run_engine(
            host, cwd, bin_path, moe_base, iso, combo, candidate_file,
            safetensors_index, plan["model"], attr.get("corpus") or "p5-preflight",
            candidate_log, max(pos, 19), timeout,
        )
        verdict = classify_candidate(out1, pos, plen, corrected)
        verdict.update({
            "role": role, "layer": layer, "n": n,
            "req": req, "pos": pos, "orig": orig,
            "corrected": corrected, "baseline_rc": rc0,
            "candidate_rc": rc1, "isolated_manifest": iso,
            "baseline_log": baseline_log,
            "candidate_log": candidate_log,
        })
        results.append(verdict)
        evidence_status = "passed" if rc1 == 0 and verdict["pass"] else "failed"
        persist_evidence(_evidence_row(
            plan, change, attr, verdict, candidate_hash,
            evidence_status, baseline_rc=rc0, candidate_rc=rc1,
            evidence_path=candidate_log,
            worker_identity=worker_identity,
        ))
        if rc1 != 0 or not verdict["pass"]:
            report = {
                "status": "failed",
                "backend": backend,
                "worker_identity": worker_identity,
                "controller_commit": controller_commit,
                "plan_after_sha256": plan["after_sha256"],
                "targets": results,
            }
            guarded._atomic_json(report_path, report)
            raise RuntimeError(
                f"P5 live preflight FAILED for {role}/L{layer} n={n}: "
                f"{verdict['reason']}"
            )

    report = {
        "status": "passed",
        "backend": backend,
        "worker_identity": worker_identity,
        "controller_commit": controller_commit,
        "plan_after_sha256": plan["after_sha256"],
        "targets": results,
    }
    guarded._atomic_json(report_path, report)
    print(
        f"[autopilot-p5-preflight] PASS {len(results)} target(s); "
        f"report={report_path}"
    )
    return report
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--log-host", default="bob")
    ap.add_argument("--events-log", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cwd", default="/Users/bob/vdsp_m4_bench")
    ap.add_argument("--moe-base", default="/Users/bob/moe_base_deepseek")
    ap.add_argument(
        "--safetensors-index",
        default="/Volumes/D50/deepseek_v2lite_bf16_safetensors/model.safetensors.index.json",
    )
    ap.add_argument(
        "--remote-dir",
        default="/private/tmp/qng64_ctl/p5_preflight",
    )
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument(
        "--backend", choices=("cpu", "mlx_metal"), default="cpu",
        help="This runner currently accepts cpu only; mlx_metal fails closed.",
    )
    args = ap.parse_args()
    run_preflight(
        args.plan, args.log_host, args.events_log, args.bin,
        args.cwd, args.moe_base, args.safetensors_index,
        args.remote_dir, args.report, timeout=args.timeout,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
