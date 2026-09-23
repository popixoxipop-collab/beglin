#!/usr/bin/env python3
"""L4 P2 guarded approval gate for qNg64 precision promotions.

P1 (autopilot_shadow.py) proves decisions without touching live state. P2 turns an
already-real-kernel-verified decision into a reviewable plan, then applies exactly that
plan only after an explicit operator approval (apply). It deliberately does not launch
new sweeps and never silently downgrades precision. Existing live promotions are
re-audited every prepare; if their evidence becomes unsupported, they appear as explicit
REMOVE_UNSAFE changes that still require the separate apply approval.

Safety invariants:
- prepare is read-only and records the exact live-file preimage hash.
- apply takes a remote lock, rejects a stale preimage, recomputes every safe_n, snapshots
  the remote file, writes atomically, reads back exactly, then recomputes safe_n again.
- any post-write failure triggers an atomic rollback to the preimage.
- rollback refuses to overwrite a live file that no longer equals this plan's postimage.
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone

import autopilot_shadow as shadow
import promotion_writeback as pwb
PLAN_VERSION = 1
DEFAULT_PLAN = "/private/tmp/qng64_ctl/autopilot_p2_plan.json"


def _rows(mapping):
    return [
        {"role": role, "layer": layer, "n": n}
        for (role, layer), n in sorted(mapping.items())
    ]


def _mapping(rows):
    return {(r["role"], int(r["layer"])): int(r["n"]) for r in rows}


def _canonical(mapping):
    return "".join(
        f"{role} {layer} {n}\n"
        for (role, layer), n in sorted(mapping.items())
    )


def _mapping_hash(mapping):
    return hashlib.sha256(_canonical(mapping).encode()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _atomic_json(path, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _load_plan(path):
    with open(path) as f:
        plan = json.load(f)
    if plan.get("version") != PLAN_VERSION:
        raise RuntimeError(
            f"unsupported plan version: {plan.get('version')} (expected {PLAN_VERSION})"
        )
    return plan
def _ssh(ssh_host, command, *, input_text=None):
    p = subprocess.run(
        ["ssh", ssh_host, command],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"ssh command failed on {ssh_host}: {p.stderr.strip() or p.stdout.strip()}"
        )
    return p.stdout


def _lock_path(promotion_file):
    return f"{promotion_file}.autopilot.lock"


def _acquire_lock(ssh_host, promotion_file):
    lock = _lock_path(promotion_file)
    q = shlex.quote(lock)
    _ssh(ssh_host, f"mkdir {q}")
    return lock


def _release_lock(ssh_host, lock):
    try:
        _ssh(ssh_host, f"rmdir {shlex.quote(lock)}")
    except RuntimeError:
        pass
def _snapshot_remote_file(ssh_host, promotion_file):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = f"{promotion_file}.autopilot.bak.{stamp}"
    q_src, q_dst = shlex.quote(promotion_file), shlex.quote(backup)
    _ssh(
        ssh_host,
        f"if test -f {q_src}; then cp {q_src} {q_dst}; else : > {q_dst}; fi",
    )
    return backup


def _print_changes(changes):
    if not changes:
        print("  no live changes proposed")
        return
    for c in changes:
        old = "<none>" if c["old_n"] is None else f"n={c['old_n']}"
        new = "<removed>" if c["new_n"] is None else f"n={c['new_n']}"
        print(
            f"  {c['action']:<13} {c['role']}/L{c['layer']}: "
            f"{old} -> {new} (events={c['event_count']})"
        )


def prepare_plan(model, limit, ssh_host, promotion_file, plan_path):
    existing = pwb.read_remote_promotion_file(ssh_host, promotion_file)
    # Defense in depth: enforce the operator cap locally even if PostgREST/proxy
    # ignores or rewrites its `limit` query parameter.
    candidates = shadow.fetch_candidates(model, limit)[:limit]
    ranked_candidate_count = len(candidates)
    by_target = {
        (c["role"], int(c["layer"])): dict(c)
        for c in candidates
    }
    for (role, layer), old_n in existing.items():
        if (role, layer) not in by_target:
            by_target[(role, layer)] = {
                "role": role,
                "layer": layer,
                "event_count": None,
                "current_bits": None,
                "source": "live-audit",
            }
    candidates = list(by_target.values())
    after = dict(existing)
    decisions, changes = [], []
    for c in candidates:
        role, layer = c["role"], int(c["layer"])
        safe_n, detail = pwb.target_safe_n(model, role, layer)
        old_n = existing.get((role, layer))
        decision = {
            "role": role,
            "layer": layer,
            "old_n": old_n,
            "safe_n": safe_n,
            "event_count": c.get("event_count"),
            "current_bits": c.get("current_bits"),
            "detail": _json_safe(detail),
        }
        if safe_n is None:
            if old_n is None:
                decision["action"] = "SKIP_UNSAFE"
            else:
                decision["action"] = "REMOVE_UNSAFE"
                after.pop((role, layer), None)
        elif old_n is None:
            decision["action"] = "ADD"
            after[(role, layer)] = safe_n
        elif safe_n > old_n:
            decision["action"] = "UPGRADE"
            after[(role, layer)] = safe_n
        elif safe_n == old_n:
            decision["action"] = "NOOP"
        else:
            decision["action"] = "HOLD_HIGHER"
        decisions.append(decision)
        if decision["action"] in ("ADD", "UPGRADE", "REMOVE_UNSAFE"):
            changes.append({
                "action": decision["action"],
                "role": role,
                "layer": layer,
                "old_n": old_n,
                "new_n": safe_n,
                "event_count": c.get("event_count"),
            })

    now = datetime.now(timezone.utc).isoformat()
    plan = {
        "version": PLAN_VERSION,
        "phase": "P2-approval-gate",
        "status": "prepared",
        "created_at": now,
        "model": model,
        "limit": limit,
        "ssh_host": ssh_host,
        "promotion_file": promotion_file,
        "before": _rows(existing),
        "after": _rows(after),
        "before_sha256": _mapping_hash(existing),
        "after_sha256": _mapping_hash(after),
        "decisions": decisions,
        "changes": changes,
    }
    _atomic_json(plan_path, plan)
    print(f"[autopilot-p2] prepared {plan_path}")
    print(
        f"  ranked_candidates={ranked_candidate_count} audited_targets={len(candidates)} "
        f"changes={len(changes)} live_entries_before={len(existing)}"
    )
    _print_changes(changes)
    print("  no live state changed; review this diff, then run the apply subcommand")
    return plan


def _revalidate_changes(plan):
    for c in plan["changes"]:
        safe_n, detail = pwb.target_safe_n(
            plan["model"], c["role"], int(c["layer"])
        )
        expected_n = c["new_n"]
        if expected_n is not None:
            expected_n = int(expected_n)
        if safe_n != expected_n:
            raise RuntimeError(
                f"evidence changed for {c['role']}/L{c['layer']}: "
                f"plan n={c['new_n']}, now n={safe_n}; "
                f"reason={detail.get('reason', detail)}"
            )


def apply_plan(plan_path):
    plan = _load_plan(plan_path)
    if plan.get("status") not in ("prepared", "apply_failed_rolled_back"):
        raise RuntimeError(f"plan status is {plan.get('status')}, not applicable")
    if not plan["changes"]:
        plan["status"] = "no_changes"
        plan["applied_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(plan_path, plan)
        print("[autopilot-p2] no changes in plan; live state untouched")
        return plan

    before, after = _mapping(plan["before"]), _mapping(plan["after"])
    ssh_host, promotion_file = plan["ssh_host"], plan["promotion_file"]
    lock = _acquire_lock(ssh_host, promotion_file)
    wrote = False
    backup = None
    try:
        current = pwb.read_remote_promotion_file(ssh_host, promotion_file)
        if _mapping_hash(current) != plan["before_sha256"]:
            raise RuntimeError(
                "stale approval plan: live promotion file changed since prepare"
            )

        _revalidate_changes(plan)
        backup = _snapshot_remote_file(ssh_host, promotion_file)
        pwb.write_remote_promotion_file_atomic(ssh_host, promotion_file, after)
        wrote = True

        readback = pwb.read_remote_promotion_file(ssh_host, promotion_file)
        if readback != after:
            raise RuntimeError("post-write exact readback mismatch")

        _revalidate_changes(plan)
        plan["status"] = "applied"
        plan["applied_at"] = datetime.now(timezone.utc).isoformat()
        plan["remote_backup"] = backup
        _atomic_json(plan_path, plan)
        print(
            f"[autopilot-p2] APPLIED {len(plan['changes'])} guarded change(s); "
            f"backup={backup}"
        )
        _print_changes(plan["changes"])
        return plan
    except Exception as exc:
        if wrote:
            try:
                pwb.write_remote_promotion_file_atomic(
                    ssh_host, promotion_file, before
                )
                rolled = pwb.read_remote_promotion_file(
                    ssh_host, promotion_file
                )
                if rolled != before:
                    raise RuntimeError("rollback readback mismatch")
                plan["status"] = "apply_failed_rolled_back"
                plan["failure"] = str(exc)
                plan["remote_backup"] = backup
                _atomic_json(plan_path, plan)
            except Exception as rollback_exc:
                raise RuntimeError(
                    f"apply failed ({exc}); ROLLBACK ALSO FAILED: {rollback_exc}"
                ) from rollback_exc
        raise
    finally:
        _release_lock(ssh_host, lock)


def rollback_plan(plan_path):
    plan = _load_plan(plan_path)
    if plan.get("status") != "applied":
        raise RuntimeError(f"plan status is {plan.get('status')}, not applied")
    before, after = _mapping(plan["before"]), _mapping(plan["after"])
    ssh_host, promotion_file = plan["ssh_host"], plan["promotion_file"]
    lock = _acquire_lock(ssh_host, promotion_file)
    try:
        current = pwb.read_remote_promotion_file(ssh_host, promotion_file)
        if current != after:
            raise RuntimeError(
                "refusing rollback: live file no longer equals this plan's postimage"
            )
        pwb.write_remote_promotion_file_atomic(ssh_host, promotion_file, before)
        readback = pwb.read_remote_promotion_file(ssh_host, promotion_file)
        if readback != before:
            raise RuntimeError("rollback exact readback mismatch")
        plan["status"] = "rolled_back"
        plan["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(plan_path, plan)
        print(
            f"[autopilot-p2] ROLLED BACK {len(plan['changes'])} change(s)"
        )
        return plan
    finally:
        _release_lock(ssh_host, lock)
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="read-only: build a reviewable P2 diff")
    prep.add_argument("--model", default="deepseek-v2-lite")
    prep.add_argument("--limit", type=int, default=10)
    prep.add_argument("--ssh-host", default="bob")
    prep.add_argument("--promotion-file", default=pwb.DEFAULT_PROMOTION_FILE)
    prep.add_argument("--plan", default=DEFAULT_PLAN)

    apply_p = sub.add_parser(
        "apply", help="explicit approval: apply a prepared plan transactionally"
    )
    apply_p.add_argument("--plan", default=DEFAULT_PLAN)

    rb = sub.add_parser(
        "rollback", help="restore the exact preimage of an applied plan"
    )
    rb.add_argument("--plan", default=DEFAULT_PLAN)

    args = ap.parse_args()
    if args.command == "prepare":
        prepare_plan(
            args.model, args.limit, args.ssh_host,
            args.promotion_file, args.plan,
        )
    elif args.command == "apply":
        apply_plan(args.plan)
    else:
        rollback_plan(args.plan)


if __name__ == "__main__":
    main()
