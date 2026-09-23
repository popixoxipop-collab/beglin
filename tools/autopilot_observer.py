#!/usr/bin/env python3
"""L4 P4 post-promotion observer and automatic demotion controller.

P4 compares equal-size windows from the engine's real near-tie JSONL:
- arm captures the shortest PRE-promotion suffix that contains at least one
  attribution for every target changed by a prepared P3 plan;
- check waits until the same number of POST-promotion near-tie events exists;
- a target passes only if its attribution count strictly falls and, when any
  attributed events remain, their median margin strictly rises.

No arbitrary percentage threshold is used. The comparison is paired by equal
near-tie-event sample count and the required direction comes directly from the
P4 design: fewer target-attributed near-ties and larger margins are improvement.

If QWEN_AUTOPILOT_P4=1 and a target fails, P4 removes only that target from the
cold-start qNg64 promotion file and writes a two-column demotion control file.
The CPU online engine polls QWEN_MOE_DEMOTION_FILE_NQ once per request admission
and swaps the already-resident base pointer back in; it never builds weights.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median

import autopilot_guarded as guarded
import autopilot_lowrisk as lowrisk
import promotion_writeback as pwb

P4_ENABLE_ENV = "QWEN_AUTOPILOT_P4"
DEFAULT_STATE = "/private/tmp/qng64_ctl/autopilot_p4_state.json"
DEFAULT_AUDIT = "/private/tmp/qng64_ctl/autopilot_p4_audit.jsonl"
DEFAULT_DEMOTION_FILE = "/private/tmp/qng64_ctl/demotion_nq_live.txt"
_SAFE_HOST = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
def _validate_remote(host, path):
    if host not in ("local", "localhost") and not _SAFE_HOST.fullmatch(host):
        raise ValueError(f"unsafe host: {host!r}")
    if not _SAFE_PATH.fullmatch(path):
        raise ValueError(f"unsafe absolute path: {path!r}")


def _read_bytes(host, path):
    _validate_remote(host, path)
    if host in ("local", "localhost"):
        with open(path, "rb") as f:
            return f.read()
    p = subprocess.run(
        ["ssh", host, "cat", path],
        capture_output=True,
        timeout=30,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"cannot read {host}:{path}: "
            f"{p.stderr.decode(errors='replace').strip()}"
        )
    return p.stdout


def _atomic_json(path, value):
    guarded._atomic_json(path, value)


def _audit(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = dict(value)
    row["ts"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _event_key(row):
    return (
        row.get("model"), row.get("corpus"),
        row.get("req"), row.get("pos"),
    )
def parse_events(blob, model=None, allow_partial_tail=False):
    """Pair attribution lines to the following event line for the same key."""
    if blob and not blob.endswith(b"\n"):
        if not allow_partial_tail:
            raise RuntimeError(
                "events log has an unterminated trailing line; refuse baseline snapshot"
            )
        cut = blob.rfind(b"\n")
        blob = b"" if cut < 0 else blob[:cut + 1]

    pending = defaultdict(list)
    events = []
    for lineno, raw in enumerate(blob.splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid complete JSONL line {lineno}: {exc}")
        if model and row.get("model") != model:
            continue
        kind = row.get("kind")
        if kind == "attribution":
            pending[_event_key(row)].append(row)
        elif kind == "event":
            key = _event_key(row)
            events.append({
                "model": row.get("model"),
                "corpus": row.get("corpus"),
                "req": row.get("req"),
                "pos": row.get("pos"),
                "margin": float(row["margin"]),
                "attrs": pending.pop(key, []),
            })
    return events


def _target_key(role, layer):
    return f"{role}:{int(layer)}"


def _event_has_target(event, role, layer):
    return any(
        a.get("role") == role and int(a.get("layer")) == int(layer)
        for a in event["attrs"]
    )
def target_metrics(events, role, layer):
    matched = [
        e for e in events if _event_has_target(e, role, layer)
    ]
    margins = [e["margin"] for e in matched]
    return {
        "events": len(events),
        "attributed_events": len(matched),
        "attribution_rate": (
            len(matched) / len(events) if events else None
        ),
        "median_margin": median(margins) if margins else None,
        "margins": margins,
    }


def baseline_suffix(events, targets):
    if not events:
        raise RuntimeError("no pre-promotion near-tie events in log")
    needed = {(t["role"], int(t["layer"])) for t in targets}
    seen = set()
    start = None
    for i in range(len(events) - 1, -1, -1):
        e = events[i]
        for role, layer in needed:
            if _event_has_target(e, role, layer):
                seen.add((role, layer))
        if seen == needed:
            start = i
            break
    if start is None:
        missing = sorted(needed - seen)
        raise RuntimeError(
            f"baseline lacks an attributed event for target(s): {missing}"
        )
    return events[start:]


def _plan_targets(plan):
    targets = []
    for c in plan.get("changes", []):
        if c.get("action") not in ("ADD", "UPGRADE"):
            continue
        if c["role"] not in lowrisk.P3_ROLES:
            continue
        if c.get("new_n") is None:
            continue
        targets.append({
            "role": c["role"],
            "layer": int(c["layer"]),
            "new_n": int(c["new_n"]),
        })
    return targets
def arm(plan_path, log_host, log_path, state_path, demotion_file,
        audit_path, apply_after=False):
    plan = guarded._load_plan(plan_path)
    if plan.get("status") != "prepared":
        raise RuntimeError(
            f"P4 arm requires a prepared pre-apply plan, got {plan.get('status')}"
        )
    targets = _plan_targets(plan)
    if not targets:
        raise RuntimeError("plan has no P3 ADD/UPGRADE target to observe")

    blob = _read_bytes(log_host, log_path)
    events = parse_events(blob, model=plan["model"])
    base = baseline_suffix(events, targets)
    metrics = {
        _target_key(t["role"], t["layer"]):
            target_metrics(base, t["role"], t["layer"])
        for t in targets
    }
    state = {
        "version": 1,
        "phase": "P4-observation",
        "status": "armed",
        "armed_at": datetime.now(timezone.utc).isoformat(),
        "plan_path": plan_path,
        "model": plan["model"],
        "ssh_host": plan["ssh_host"],
        "promotion_file": plan["promotion_file"],
        "demotion_file": demotion_file,
        "log_host": log_host,
        "log_path": log_path,
        "log_offset": len(blob),
        "log_prefix_sha256": hashlib.sha256(blob).hexdigest(),
        "baseline_event_count": len(base),
        "baseline_metrics": metrics,
        "targets": targets,
    }
    _atomic_json(state_path, state)
    _audit(audit_path, {
        "phase": state["phase"], "status": "armed",
        "targets": targets, "baseline_event_count": len(base),
    })
    print(
        f"[autopilot-p4] armed {len(targets)} target(s); "
        f"baseline_window={len(base)} equal-size near-tie event(s)"
    )
    if apply_after:
        result = guarded.apply_plan(plan_path)
        state["plan_status"] = result["status"]
        state["applied_at"] = result.get("applied_at")
        _atomic_json(state_path, state)
        return begin_observation(state_path, audit_path, require_applied=False)
    return state


def begin_observation(state_path, audit_path, require_applied=True):
    """Start the POST window after apply, excluding arm->apply race events."""
    with open(state_path) as f:
        state = json.load(f)
    if state.get("status") != "armed":
        raise RuntimeError(f"P4 begin requires armed state, got {state.get('status')}")
    if require_applied:
        plan = guarded._load_plan(state["plan_path"])
        if plan.get("status") != "applied":
            raise RuntimeError(
                f"P4 begin requires applied plan, got {plan.get('status')}"
            )
    blob = _read_bytes(state["log_host"], state["log_path"])
    old_off = int(state["log_offset"])
    if len(blob) < old_off or hashlib.sha256(blob[:old_off]).hexdigest() != state["log_prefix_sha256"]:
        raise RuntimeError("events log changed/truncated between P4 arm and begin")
    state["log_offset"] = len(blob)
    state["log_prefix_sha256"] = hashlib.sha256(blob).hexdigest()
    state["status"] = "observing"
    state["observation_started_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(state_path, state)
    _audit(audit_path, {
        "phase": state["phase"], "status": "observing",
        "targets": state["targets"], "log_offset": state["log_offset"],
    })
    print(
        f"[autopilot-p4] observation begins at byte {state['log_offset']} "
        "(arm->apply race events excluded)"
    )
    return state


def _read_demotion_targets(host, path):
    _validate_remote(host, path)
    if host in ("local", "localhost"):
        if not os.path.exists(path):
            return set()
        with open(path) as f:
            text = f.read()
    else:
        p = subprocess.run(
            ["ssh", host, "sh", "-c", f"test -f {path} && cat {path} || true"],
            capture_output=True, text=True, timeout=30,
        )
        if p.returncode != 0:
            raise RuntimeError(f"cannot read demotion file: {p.stderr}")
        text = p.stdout
    toks = text.split()
    if len(toks) % 2:
        raise RuntimeError("malformed demotion file: token count is not even")
    return {(toks[i], int(toks[i + 1])) for i in range(0, len(toks), 2)}


def _write_demotion_targets_atomic(host, path, targets):
    _validate_remote(host, path)
    content = "".join(
        f"{role} {layer}\n" for role, layer in sorted(targets)
    )
    if host in ("local", "localhost"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(content)
        os.replace(tmp, path)
        return
    remote_dir = path.rsplit("/", 1)[0]
    tmp = f"{path}.tmp.{os.getpid()}"
    cmd = f"mkdir -p {remote_dir} && cat > {tmp} && mv {tmp} {path}"
    p = subprocess.run(
        ["ssh", host, cmd], input=content,
        capture_output=True, text=True, timeout=30,
    )
    if p.returncode != 0:
        raise RuntimeError(f"cannot write demotion file: {p.stderr}")
def _demote_failed(state, failed):
    host = state["ssh_host"]
    current = pwb.read_remote_promotion_file(
        host, state["promotion_file"]
    )
    merged = dict(current)
    demoted = []
    for t in failed:
        key = (t["role"], int(t["layer"]))
        if current.get(key) != int(t["new_n"]):
            continue
        merged.pop(key, None)
        demoted.append(key)
    if not demoted:
        return []

    pwb.write_remote_promotion_file_atomic(
        host, state["promotion_file"], merged
    )
    live_demotions = _read_demotion_targets(
        host, state["demotion_file"]
    )
    live_demotions.update(demoted)
    _write_demotion_targets_atomic(
        host, state["demotion_file"], live_demotions
    )
    return demoted


def check(state_path, audit_path):
    with open(state_path) as f:
        state = json.load(f)
    if state.get("status") not in ("observing", "collecting"):
        raise RuntimeError(
            f"state status {state.get('status')} is not observable; run begin after apply"
        )

    blob = _read_bytes(state["log_host"], state["log_path"])
    off = int(state["log_offset"])
    if len(blob) < off:
        raise RuntimeError("events log was truncated/rotated after P4 arm")
    if hashlib.sha256(blob[:off]).hexdigest() != state["log_prefix_sha256"]:
        raise RuntimeError("events log prefix changed after P4 arm")
    post_events = parse_events(
        blob[off:], model=state["model"], allow_partial_tail=True
    )
    n = int(state["baseline_event_count"])
    if len(post_events) < n:
        state["status"] = "collecting"
        state["post_events"] = len(post_events)
        _atomic_json(state_path, state)
        print(
            f"[autopilot-p4] collecting: {len(post_events)}/{n} "
            "post-promotion near-tie events"
        )
        return state

    window = post_events[:n]
    results, failed = {}, []
    for t in state["targets"]:
        key = _target_key(t["role"], t["layer"])
        before = state["baseline_metrics"][key]
        after = target_metrics(window, t["role"], t["layer"])
        fewer = after["attributed_events"] < before["attributed_events"]
        margin_up = (
            after["attributed_events"] == 0 or
            (
                after["median_margin"] is not None and
                before["median_margin"] is not None and
                after["median_margin"] > before["median_margin"]
            )
        )
        healthy = fewer and margin_up
        results[key] = {
            "healthy": healthy,
            "baseline": before,
            "post": after,
            "criteria": {
                "attribution_count_strictly_lower": fewer,
                "median_margin_strictly_higher_if_remaining": margin_up,
            },
        }
        if not healthy:
            failed.append(t)

    enabled = os.environ.get(P4_ENABLE_ENV) == "1"
    demoted = []
    if failed and enabled:
        demoted = _demote_failed(state, failed)

    state["checked_at"] = datetime.now(timezone.utc).isoformat()
    state["results"] = results
    state["failed_targets"] = failed
    state["demoted_targets"] = [
        {"role": r, "layer": l} for r, l in demoted
    ]
    if failed and not enabled:
        state["status"] = "regression_detected_killswitch_off"
    elif failed and len(demoted) == len(failed):
        state["status"] = "auto_demoted"
    elif failed:
        state["status"] = "regression_detected_state_changed"
    else:
        state["status"] = "healthy"
    _atomic_json(state_path, state)
    _audit(audit_path, {
        "phase": state["phase"],
        "status": state["status"],
        "failed_targets": failed,
        "demoted_targets": state["demoted_targets"],
        "results": results,
    })
    print(
        f"[autopilot-p4] {state['status']}: "
        f"failed={len(failed)} demoted={len(demoted)}"
    )
    return state
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    arm_p = sub.add_parser("arm")
    arm_p.add_argument("--plan", required=True)
    arm_p.add_argument("--log-host", default="bob")
    arm_p.add_argument("--events-log", required=True)
    arm_p.add_argument("--state", default=DEFAULT_STATE)
    arm_p.add_argument("--audit", default=DEFAULT_AUDIT)
    arm_p.add_argument("--demotion-file", default=DEFAULT_DEMOTION_FILE)
    arm_p.add_argument(
        "--apply", action="store_true",
        help="after capturing baseline, apply the prepared plan transactionally",
    )

    begin_p = sub.add_parser("begin")
    begin_p.add_argument("--state", default=DEFAULT_STATE)
    begin_p.add_argument("--audit", default=DEFAULT_AUDIT)

    check_p = sub.add_parser("check")
    check_p.add_argument("--state", default=DEFAULT_STATE)
    check_p.add_argument("--audit", default=DEFAULT_AUDIT)

    args = ap.parse_args()
    if args.command == "arm":
        arm(
            args.plan, args.log_host, args.events_log,
            args.state, args.demotion_file, args.audit,
            apply_after=args.apply,
        )
    elif args.command == "begin":
        begin_observation(args.state, args.audit)
    else:
        check(args.state, args.audit)


if __name__ == "__main__":
    main()
