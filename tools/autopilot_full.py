#!/usr/bin/env python3
"""L4 P5 full autonomous precision controller.

P5 extends the P3/P4 machinery from dense/shared FFN to attention roles.
It never promotes from simulated data: target_safe_n() still requires qng64_real
coverage for every known event/corpus. Routed experts and global embed/lm-head
remain outside this rollout because the P5 design explicitly expands attention.

A P5 plan is not applied directly. --arm-apply requires QWEN_AUTOPILOT_P5=1
and hands the prepared plan to the P4 observer, which must capture a PRE window
before guarded apply and later owns regression detection + auto-demotion.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import autopilot_guarded as guarded
import autopilot_lowrisk as lowrisk
import autopilot_observer as observer
import autopilot_shadow as shadow
import promotion_writeback as pwb

DEFAULT_PLAN = "/private/tmp/qng64_ctl/autopilot_p5_plan.json"
DEFAULT_AUDIT = "/private/tmp/qng64_ctl/autopilot_p5_audit.jsonl"
DEFAULT_STATE = "/private/tmp/qng64_ctl/autopilot_p5_state.json"
def _audit(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = dict(payload)
    row["ts"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _merge_candidate(by_target, row):
    key = (row["role"], int(row["layer"]))
    if key not in by_target:
        by_target[key] = {
            "role": key[0],
            "layer": key[1],
            "event_count": 0,
            "current_bits": row.get("current_bits"),
            "source_rows": 0,
        }
    dst = by_target[key]
    dst["event_count"] += int(row.get("event_count") or 0)
    dst["source_rows"] += 1
    bits = row.get("current_bits")
    if dst["current_bits"] is None:
        dst["current_bits"] = bits
    elif bits is not None and bits != dst["current_bits"]:
        raise RuntimeError(
            f"inconsistent current_bits for {key}: "
            f"{dst['current_bits']} vs {bits}"
        )
def build_plan(model, limit, ssh_host, promotion_file, quarantine_file,
               plan_path):
    existing = pwb.read_remote_promotion_file(ssh_host, promotion_file)
    quarantine = lowrisk._read_quarantine(
        ssh_host, quarantine_file
    ) if quarantine_file else set()

    ranked = shadow.fetch_candidates(model, limit)[:limit]
    by_target = {}
    for c in ranked:
        if c["role"] not in lowrisk.P5_ROLES:
            continue
        _merge_candidate(by_target, c)

    # Existing P5-owned live entries are always re-audited even when the
    # aggregate candidate query no longer returns them.
    for (role, layer), old_n in existing.items():
        if role not in lowrisk.P5_ROLES or (role, layer) in by_target:
            continue
        by_target[(role, layer)] = {
            "role": role,
            "layer": layer,
            "event_count": None,
            "current_bits": None,
            "source_rows": 0,
            "source": "live-audit",
        }

    after = dict(existing)
    decisions, changes = [], []
    for (role, layer), c in sorted(by_target.items()):
        old_n = existing.get((role, layer))
        if (role, layer) in quarantine:
            decisions.append({
                "role": role,
                "layer": layer,
                "old_n": old_n,
                "safe_n": None,
                "event_count": c.get("event_count"),
                "action": "P4_QUARANTINED",
                "detail": {
                    "reason": "P4 demotion quarantine blocks automatic re-promotion"
                },
            })
            continue

        safe_n, detail = pwb.target_safe_n(model, role, layer)
        d = lowrisk._decision(
            role, layer, old_n, safe_n,
            c.get("event_count"), detail,
        )
        d["source_rows"] = c.get("source_rows", 0)
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
        "phase": "P5-full-auto",
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
        "p5_roles": sorted(lowrisk.P5_ROLES),
    }
    guarded._atomic_json(plan_path, plan)
    return plan


def print_plan(plan):
    print(
        f"[autopilot-p5] targets={len(plan['decisions'])} "
        f"changes={len(plan['changes'])}"
    )
    for d in plan["decisions"]:
        old = "<none>" if d["old_n"] is None else f"n={d['old_n']}"
        new = "<unsafe>" if d["safe_n"] is None else f"n={d['safe_n']}"
        print(
            f"  {d['action']:<20} {d['role']}/L{d['layer']}: "
            f"{old} -> {new} (events={d['event_count']})"
        )
def prepare(model, limit, ssh_host, promotion_file, quarantine_file,
            plan_path, audit_path):
    plan = build_plan(
        model, limit, ssh_host, promotion_file,
        quarantine_file, plan_path,
    )
    print_plan(plan)
    _audit(audit_path, {
        "phase": plan["phase"],
        "status": "prepared",
        "model": model,
        "changes": plan["changes"],
    })
    return plan


def arm_apply(model, limit, ssh_host, promotion_file, quarantine_file,
              plan_path, audit_path, log_host, events_log, state_path,
              observer_audit):
    plan = prepare(
        model, limit, ssh_host, promotion_file,
        quarantine_file, plan_path, audit_path,
    )
    if not plan["changes"]:
        plan["status"] = "no_changes"
        guarded._atomic_json(plan_path, plan)
        print("[autopilot-p5] no safe ADD/UPGRADE target; live state untouched")
        return plan

    if os.environ.get(lowrisk.P5_ENABLE_ENV) != "1":
        plan["status"] = "disabled"
        guarded._atomic_json(plan_path, plan)
        print(
            f"[autopilot-p5] kill switch is OFF: "
            f"set {lowrisk.P5_ENABLE_ENV}=1"
        )
        return plan

    return observer.arm(
        plan_path,
        log_host,
        events_log,
        state_path,
        quarantine_file,
        observer_audit,
        apply_after=True,
    )
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--arm-apply", action="store_true")

    ap.add_argument("--model", default="deepseek-v2-lite")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--ssh-host", default="bob")
    ap.add_argument("--promotion-file", default=pwb.DEFAULT_PROMOTION_FILE)
    ap.add_argument(
        "--quarantine-file",
        default=lowrisk.DEFAULT_QUARANTINE_FILE,
    )
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--audit", default=DEFAULT_AUDIT)

    ap.add_argument("--log-host", default="bob")
    ap.add_argument("--events-log")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument(
        "--observer-audit",
        default=observer.DEFAULT_AUDIT,
    )
    args = ap.parse_args()

    if args.prepare_only:
        prepare(
            args.model, args.limit, args.ssh_host,
            args.promotion_file, args.quarantine_file,
            args.plan, args.audit,
        )
        return
    if not args.events_log:
        ap.error("--arm-apply requires --events-log")
    arm_apply(
        args.model, args.limit, args.ssh_host,
        args.promotion_file, args.quarantine_file,
        args.plan, args.audit, args.log_host,
        args.events_log, args.state, args.observer_audit,
    )


if __name__ == "__main__":
    main()
