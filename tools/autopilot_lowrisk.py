#!/usr/bin/env python3
"""L4 P3 low-risk autonomous promotion controller.

P3 is deliberately narrower than P2:
- only dense/shared FFN roles are eligible for approval-free changes;
- only already-collected qng64_real evidence is used via target_safe_n();
- no new sweep is launched here;
- ADD/UPGRADE are automatic, but REMOVE/downgrade stays in P2 review;
- attention/expert/embed/lm-head roles are never changed by this script.

Mutation reuses P2's guarded transaction: remote lock, stale-preimage rejection,
evidence revalidation, remote snapshot, atomic write, exact readback, second
evidence revalidation, and automatic rollback on any post-write failure.

The controller is opt-in. Unless QWEN_AUTOPILOT_P3=1, run mode is read-only.
"""
import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import autopilot_guarded as guarded
import autopilot_shadow as shadow
import promotion_writeback as pwb

P3_ENABLE_ENV = "QWEN_AUTOPILOT_P3"
DEFAULT_PLAN = "/private/tmp/qng64_ctl/autopilot_p3_plan.json"
DEFAULT_AUDIT = "/private/tmp/qng64_ctl/autopilot_p3_audit.jsonl"

P3_ROLES = frozenset({
    "dense_gate_proj",
    "dense_up_proj",
    "dense_down_proj",
    "shared_gate_proj",
    "shared_up_proj",
    "shared_down_proj",
})
DEFAULT_QUARANTINE_FILE = "/private/tmp/qng64_ctl/demotion_nq_live.txt"
_SAFE_HOST = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")


def _read_quarantine(ssh_host, path):
    if not path:
        return set()
    if not _SAFE_HOST.fullmatch(ssh_host) or not _SAFE_PATH.fullmatch(path):
        raise ValueError("unsafe quarantine host/path")
    p = subprocess.run(
        ["ssh", ssh_host, "sh", "-c", f"test -f {path} && cat {path} || true"],
        capture_output=True, text=True, timeout=30,
    )
    if p.returncode != 0:
        raise RuntimeError(f"cannot read P4 quarantine file: {p.stderr}")
    toks = p.stdout.split()
    if len(toks) % 2:
        raise RuntimeError("malformed P4 quarantine file")
    return {(toks[i], int(toks[i + 1])) for i in range(0, len(toks), 2)}


def _audit(path, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    row = dict(payload)
    row["ts"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _decision(role, layer, old_n, safe_n, event_count, detail):
    base = {
        "role": role,
        "layer": int(layer),
        "old_n": old_n,
        "safe_n": safe_n,
        "event_count": event_count,
        "detail": guarded._json_safe(detail),
    }
    if safe_n is None:
        base["action"] = "P2_REVIEW_UNSAFE" if old_n is not None else "SKIP_UNSAFE"
    elif old_n is None:
        base["action"] = "ADD"
    elif safe_n > old_n:
        base["action"] = "UPGRADE"
    elif safe_n == old_n:
        base["action"] = "NOOP"
    else:
        base["action"] = "P2_REVIEW_DOWNGRADE"
    return base


def build_plan(model, limit, ssh_host, promotion_file, plan_path,
               quarantine_file=None):
    existing = pwb.read_remote_promotion_file(ssh_host, promotion_file)
    quarantine = _read_quarantine(ssh_host, quarantine_file) if quarantine_file else set()
    ranked = shadow.fetch_candidates(model, limit)[:limit]
    by_target = {}
    for c in ranked:
        role = c["role"]
        if role not in P3_ROLES:
            continue
        by_target[(role, int(c["layer"]))] = dict(c)

    # Existing P3-owned live entries are always re-audited, even if the aggregate
    # candidate query no longer returns them. P3 does not auto-remove them; an
    # evidence regression is surfaced to P2.
    for (role, layer), old_n in existing.items():
        if role not in P3_ROLES or (role, layer) in by_target:
            continue
        by_target[(role, layer)] = {
            "role": role,
            "layer": layer,
            "event_count": None,
            "current_bits": None,
            "source": "live-audit",
        }

    after = dict(existing)
    decisions, changes = [], []
    for (role, layer), c in sorted(by_target.items()):
        old_n = existing.get((role, layer))
        if (role, layer) in quarantine:
            decisions.append({
                "role": role, "layer": layer, "old_n": old_n,
                "safe_n": None, "event_count": c.get("event_count"),
                "action": "P4_QUARANTINED",
                "detail": {"reason": "P4 demotion quarantine blocks automatic re-promotion"},
            })
            continue
        safe_n, detail = pwb.target_safe_n(model, role, layer)
        d = _decision(
            role, layer, old_n, safe_n, c.get("event_count"), detail
        )
        decisions.append(d)
        if d["action"] in ("ADD", "UPGRADE"):
            after[(role, layer)] = safe_n
            changes.append({
                "action": d["action"],
                "role": role,
                "layer": layer,
                "old_n": old_n,
                "new_n": safe_n,
                "event_count": c.get("event_count"),
            })

    plan = {
        "version": guarded.PLAN_VERSION,
        "phase": "P3-lowrisk-auto",
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "limit": limit,
        "ssh_host": ssh_host,
        "promotion_file": promotion_file,
        "quarantine_file": quarantine_file,
        "before": guarded._rows(existing),
        "after": guarded._rows(after),
        "before_sha256": guarded._mapping_hash(existing),
        "after_sha256": guarded._mapping_hash(after),
        "decisions": decisions,
        "changes": changes,
        "p3_roles": sorted(P3_ROLES),
    }
    guarded._atomic_json(plan_path, plan)
    return plan


def print_plan(plan):
    print(
        f"[autopilot-p3] eligible_targets={len(plan['decisions'])} "
        f"changes={len(plan['changes'])}"
    )
    for d in plan["decisions"]:
        old = "<none>" if d["old_n"] is None else f"n={d['old_n']}"
        new = "<unsafe>" if d["safe_n"] is None else f"n={d['safe_n']}"
        print(
            f"  {d['action']:<20} {d['role']}/L{d['layer']}: "
            f"{old} -> {new} (events={d['event_count']})"
        )


def run_once(model, limit, ssh_host, promotion_file, plan_path, audit_path,
             quarantine_file=None, dry_run=False, prepare_only=False):
    plan = build_plan(
        model, limit, ssh_host, promotion_file, plan_path,
        quarantine_file=quarantine_file,
    )
    print_plan(plan)

    if prepare_only:
        _audit(audit_path, {
            "phase": plan["phase"],
            "status": "prepared",
            "model": model,
            "changes": plan["changes"],
        })
        print("[autopilot-p3] prepared only; hand this plan to P4 arm --apply")
        return plan

    enabled = os.environ.get(P3_ENABLE_ENV) == "1"
    if dry_run or not enabled:
        mode = "dry_run" if dry_run else "disabled"
        plan["status"] = mode
        guarded._atomic_json(plan_path, plan)
        _audit(audit_path, {
            "phase": plan["phase"],
            "status": mode,
            "model": model,
            "changes": plan["changes"],
        })
        if not enabled and not dry_run:
            print(
                f"[autopilot-p3] kill switch is OFF: set {P3_ENABLE_ENV}=1 "
                "for autonomous apply"
            )
        else:
            print("[autopilot-p3] dry run; live state untouched")
        return plan

    if not plan["changes"]:
        plan["status"] = "no_changes"
        guarded._atomic_json(plan_path, plan)
        _audit(audit_path, {
            "phase": plan["phase"],
            "status": "no_changes",
            "model": model,
            "changes": [],
        })
        print("[autopilot-p3] no eligible ADD/UPGRADE; live state untouched")
        return plan

    try:
        result = guarded.apply_plan(plan_path)
    except Exception as exc:
        _audit(audit_path, {
            "phase": plan["phase"],
            "status": "failed",
            "model": model,
            "changes": plan["changes"],
            "error": str(exc),
        })
        raise

    _audit(audit_path, {
        "phase": plan["phase"],
        "status": result["status"],
        "model": model,
        "changes": result["changes"],
        "remote_backup": result.get("remote_backup"),
    })
    print("[autopilot-p3] autonomous low-risk transaction complete")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-v2-lite")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ssh-host", default="bob")
    ap.add_argument("--promotion-file", default=pwb.DEFAULT_PROMOTION_FILE)
    ap.add_argument("--quarantine-file", default=DEFAULT_QUARANTINE_FILE,
                    help="P4 two-column demotion/quarantine file; listed targets are never auto-repromoted")
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--audit", default=DEFAULT_AUDIT)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--prepare-only", action="store_true",
        help="build a prepared P3 plan without applying it (for P4 arm --apply)",
    )
    args = ap.parse_args()
    run_once(
        args.model,
        args.limit,
        args.ssh_host,
        args.promotion_file,
        args.plan,
        args.audit,
        quarantine_file=args.quarantine_file,
        dry_run=args.dry_run,
        prepare_only=args.prepare_only,
    )


if __name__ == "__main__":
    main()
