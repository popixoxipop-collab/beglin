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
from datetime import datetime, timezone

import autopilot_guarded as guarded
import autopilot_observer as observer
import quant_search_n as qsn
import promotion_writeback as pwb

DEFAULT_REPORT = "/private/tmp/qng64_ctl/autopilot_p5_preflight.json"
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


def classify_candidate(output, pos, prompt_len, corrected):
    flip_re = re.compile(
        rf"REAL FLIP orig=(\d+) corrected=(\d+)"
    )
    flips = [
        (int(a), int(b)) for a, b in flip_re.findall(output)
    ]
    exact_flip = any(b == int(corrected) for _, b in flips)

    gen_idx = int(pos) - (int(prompt_len) - 1)
    tokens = _parse_emitted_tokens(output)
    emitted = (
        tokens[gen_idx]
        if 0 <= gen_idx < len(tokens)
        else None
    )
    passed = (not exact_flip) and emitted == int(corrected)
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
                if exact_flip else
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
    marker = f"REAL FLIP orig={orig} corrected={corrected}"
    target = f"correct req=0 pos={pos}"
    return marker in output and target in output


def run_preflight(plan_path, log_host, events_log, bin_path, cwd, moe_base,
                  safetensors_index, remote_dir, report_path,
                  timeout=180):
    plan = guarded._load_plan(plan_path)
    if plan.get("phase") != "P5-full-auto":
        raise RuntimeError("live preflight only accepts P5-full-auto plans")
    if plan.get("status") != "prepared":
        raise RuntimeError(
            f"preflight requires prepared plan, got {plan.get('status')}"
        )
    if not plan.get("changes"):
        return {"status": "no_changes", "targets": []}

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
        if rc1 != 0 or not verdict["pass"]:
            report = {
                "status": "failed",
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
    args = ap.parse_args()
    run_preflight(
        args.plan, args.log_host, args.events_log, args.bin,
        args.cwd, args.moe_base, args.safetensors_index,
        args.remote_dir, args.report, timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
