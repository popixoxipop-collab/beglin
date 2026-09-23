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
import autopilot_live_preflight as live_preflight
import attribution_provenance as provenance
import autopilot_shadow as shadow
import promotion_writeback as pwb
import quant_search_n as qsn

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
def _live_evidence_gate(model, role, layer, safe_n, preimage_sha256):
    ladder = [n for n in qsn.REAL_LADDER if int(n) >= int(safe_n)]
    if not ladder:
        return "LIVE_LADDER_UNSAFE", {
            "reason": f"no live ladder value >= qng64 safe_n={safe_n}",
            "qng64_safe_n": safe_n,
            "evidence_by_n": {},
        }

    evidence_by_n = {}
    for candidate_n in ladder:
        try:
            evidence = live_preflight.fetch_latest_evidence(
                model, role, layer, candidate_n, preimage_sha256
            )
        except live_preflight.EvidenceStoreUnavailable as exc:
            return "EVIDENCE_STORE_UNAVAILABLE", {
                "reason": str(exc),
                "qng64_safe_n": safe_n,
                "evidence_by_n": evidence_by_n,
            }
        evidence_by_n[str(candidate_n)] = evidence
        if evidence is None:
            return "NEEDS_LIVE_PREFLIGHT", {
                "reason": (
                    "no durable live-preflight evidence for current production "
                    f"preimage at n={candidate_n}"
                ),
                "qng64_safe_n": safe_n,
                "next_n": candidate_n,
                "evidence_by_n": evidence_by_n,
            }

        status = evidence.get("status")
        if status == "passed" and evidence.get("pass") is True:
            return None, {
                "reason": (
                    "live-preflight passed for current preimage at "
                    f"n={candidate_n}"
                ),
                "qng64_safe_n": safe_n,
                "selected_n": candidate_n,
                "evidence": evidence,
                "evidence_by_n": evidence_by_n,
            }
        if status == "no_current_signal":
            return "NO_CURRENT_SIGNAL", {
                "reason": evidence.get("reason") or (
                    "no current production-matched attribution signal"
                ),
                "qng64_safe_n": safe_n,
                "evidence": evidence,
                "evidence_by_n": evidence_by_n,
            }
        if status == "baseline_failed":
            return "LIVE_PREFLIGHT_FAILED", {
                "reason": evidence.get("reason") or "baseline isolation failed",
                "qng64_safe_n": safe_n,
                "evidence": evidence,
                "evidence_by_n": evidence_by_n,
            }
        # A candidate-path failure does not prove that a higher n fails.
        # Continue up the deployable ladder before declaring the target unsafe.

    return "LIVE_LADDER_UNSAFE", {
        "reason": (
            f"actual serving-path preflight failed for every n in {ladder} "
            f"at production preimage {preimage_sha256}"
        ),
        "qng64_safe_n": safe_n,
        "tested_ns": ladder,
        "evidence_by_n": evidence_by_n,
    }


def _bootstrap_preflight_plan(plan, path):
    preflight = dict(plan)
    preflight["status"] = "prepared"
    preflight["changes"] = [dict(c) for c in plan.get("preflight_candidates", [])]
    after = guarded._mapping(plan["before"])
    for c in preflight["changes"]:
        after[(c["role"], int(c["layer"]))] = int(c["new_n"])
    preflight["after"] = guarded._rows(after)
    preflight["after_sha256"] = guarded._mapping_hash(after)
    guarded._atomic_json(path, preflight)
    return preflight


def _work_priority(c):
    count = c.get("event_count")
    return (-(count if count is not None else -1), c["role"], int(c["layer"]))


def _mark_deferred(decisions, candidate):
    for d in decisions:
        if d.get("role") == candidate["role"] and int(d.get("layer")) == int(candidate["layer"]):
            d["action"] = "DEFERRED_SERIAL_PREIMAGE"
            d["deferred_reason"] = (
                "P5 v2 admits one new target per production preimage; "
                "re-evaluate after the selected target changes live state"
            )
            return


def _serialize_new_work(existing, decisions, changes, preflight_candidates):
    """Admit at most one new target for a given production preimage."""
    selected_changes = []
    selected_preflight = []

    if changes:
        chosen = sorted(changes, key=_work_priority)[0]
        selected_changes = [chosen]
        for c in changes:
            if c is not chosen:
                _mark_deferred(decisions, c)
        for c in preflight_candidates:
            _mark_deferred(decisions, c)
    elif preflight_candidates:
        chosen = sorted(preflight_candidates, key=_work_priority)[0]
        selected_preflight = [chosen]
        for c in preflight_candidates:
            if c is not chosen:
                _mark_deferred(decisions, c)

    after = dict(existing)
    for c in selected_changes:
        after[(c["role"], int(c["layer"]))] = int(c["new_n"])
    return after, selected_changes, selected_preflight


def _needs_real_sweep(detail):
    per = detail.get("per_corpus", {}) if isinstance(detail, dict) else {}
    if any(v.get("unsafe_events") for v in per.values() if isinstance(v, dict)):
        return False
    reason = (detail or {}).get("reason", "")
    markers = (
        "no prior sweep data",
        "no source='qng64_real' rows exist yet",
        "have no source='qng64_real' data yet",
        "not fully real-kernel-verified",
    )
    return any(m in reason for m in markers)


def build_plan(model, limit, ssh_host, promotion_file, quarantine_file,
               plan_path):
    existing = pwb.read_remote_promotion_file(ssh_host, promotion_file)
    quarantine = lowrisk._read_quarantine(
        ssh_host, quarantine_file
    ) if quarantine_file else set()

    preimage_sha256 = guarded._mapping_hash(existing)
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
    decisions, changes, preflight_candidates, real_sweep_candidates = [], [], [], []
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

        proposed_action = d["action"]
        if safe_n is None and old_n is None and _needs_real_sweep(detail):
            try:
                prov = provenance.fetch_best(model, role, layer)
            except provenance.ProvenanceStoreUnavailable as exc:
                prov = None
                d["action"] = "PROVENANCE_STORE_UNAVAILABLE"
                d["provenance_error"] = str(exc)
            if prov is not None:
                d["action"] = "NEEDS_REAL_SWEEP"
                d["provenance"] = prov
                real_sweep_candidates.append({
                    "model": model, "role": role, "layer": layer,
                    "event_count": c.get("event_count"),
                    "provenance": prov,
                })
            elif d.get("action") == "SKIP_UNSAFE":
                d["action"] = "NEEDS_ATTRIBUTION_PROVENANCE"

        proposed_action = d["action"]
        if proposed_action in ("ADD", "UPGRADE"):
            gate_action, live_detail = _live_evidence_gate(
                model, role, layer, safe_n, preimage_sha256
            )
            d["live_evidence"] = live_detail
            candidate_n = int(
                live_detail.get("selected_n",
                                live_detail.get("next_n", safe_n))
            )
            d["qng64_safe_n"] = safe_n
            d["live_selected_n"] = live_detail.get("selected_n")
            candidate = {
                "action": proposed_action,
                "role": role,
                "layer": layer,
                "old_n": old_n,
                "new_n": candidate_n,
                "event_count": c.get("event_count"),
            }
            if gate_action is None:
                changes.append(candidate)
            else:
                d["action"] = gate_action
                if gate_action == "NEEDS_LIVE_PREFLIGHT":
                    preflight_candidates.append(candidate)
        decisions.append(d)

    after, changes, preflight_candidates = _serialize_new_work(
        existing, decisions, changes, preflight_candidates
    )
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
        "promotion_preimage_sha256": preimage_sha256,
        "before": guarded._rows(existing),
        "after": guarded._rows(after),
        "before_sha256": guarded._mapping_hash(existing),
        "after_sha256": guarded._mapping_hash(after),
        "decisions": decisions,
        "changes": changes,
        "preflight_candidates": preflight_candidates,
        "real_sweep_candidates": sorted(real_sweep_candidates, key=_work_priority),
        "evidence_contract": "p5-v2",
        "serialization": "one-target-per-preimage",
        "p5_roles": sorted(lowrisk.P5_ROLES),
    }
    guarded._atomic_json(plan_path, plan)
    return plan


def print_plan(plan):
    print(
        f"[autopilot-p5] targets={len(plan['decisions'])} "
        f"changes={len(plan['changes'])} "
        f"preflight_candidates={len(plan.get('preflight_candidates', []))} "
        f"real_sweep_candidates={len(plan.get('real_sweep_candidates', []))}"
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
              observer_audit, preflight_bin=None,
              preflight_cwd="/Users/bob/vdsp_m4_bench",
              preflight_moe_base="/Users/bob/moe_base_deepseek",
              preflight_safetensors_index="/Volumes/D50/deepseek_v2lite_bf16_safetensors/model.safetensors.index.json",
              preflight_remote_dir="/private/tmp/qng64_ctl/p5_preflight",
              preflight_report=live_preflight.DEFAULT_REPORT,
              preflight_timeout=180):
    plan = prepare(
        model, limit, ssh_host, promotion_file,
        quarantine_file, plan_path, audit_path,
    )

    unavailable = [
        d for d in plan["decisions"]
        if d.get("action") == "EVIDENCE_STORE_UNAVAILABLE"
    ]
    if unavailable:
        raise RuntimeError(
            "P5 evidence store unavailable; fail-closed before live apply"
        )

    has_work = bool(plan["changes"] or plan.get("preflight_candidates"))
    if not has_work:
        plan["status"] = "no_changes"
        guarded._atomic_json(plan_path, plan)
        print("[autopilot-p5] no evidence-admissible work; live state untouched")
        return plan

    if os.environ.get(lowrisk.P5_ENABLE_ENV) != "1":
        plan["status"] = "disabled"
        guarded._atomic_json(plan_path, plan)
        print(
            f"[autopilot-p5] kill switch is OFF: "
            f"set {lowrisk.P5_ENABLE_ENV}=1"
        )
        return plan

    if plan.get("preflight_candidates"):
        if not preflight_bin:
            raise RuntimeError(
                "P5 arm-apply requires --preflight-bin: "
                "live-composition canary is mandatory"
            )
        bootstrap_path = plan_path + ".preflight"
        _bootstrap_preflight_plan(plan, bootstrap_path)
        live_preflight.run_preflight(
            bootstrap_path, log_host, events_log, preflight_bin,
            preflight_cwd, preflight_moe_base, preflight_safetensors_index,
            preflight_remote_dir, preflight_report, timeout=preflight_timeout,
        )
        # Durable evidence was written by preflight. Rebuild the planner view;
        # only same-preimage PASS rows can now become ADD/UPGRADE changes.
        plan = prepare(
            model, limit, ssh_host, promotion_file,
            quarantine_file, plan_path, audit_path,
        )

    if not plan["changes"]:
        plan["status"] = "no_changes"
        guarded._atomic_json(plan_path, plan)
        print("[autopilot-p5] preflight produced no admissible change")
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
    ap.add_argument("--preflight-bin")
    ap.add_argument("--preflight-cwd", default="/Users/bob/vdsp_m4_bench")
    ap.add_argument("--preflight-moe-base", default="/Users/bob/moe_base_deepseek")
    ap.add_argument(
        "--preflight-safetensors-index",
        default="/Volumes/D50/deepseek_v2lite_bf16_safetensors/model.safetensors.index.json",
    )
    ap.add_argument(
        "--preflight-remote-dir",
        default="/private/tmp/qng64_ctl/p5_preflight",
    )
    ap.add_argument("--preflight-report", default=live_preflight.DEFAULT_REPORT)
    ap.add_argument("--preflight-timeout", type=int, default=180)
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
        preflight_bin=args.preflight_bin,
        preflight_cwd=args.preflight_cwd,
        preflight_moe_base=args.preflight_moe_base,
        preflight_safetensors_index=args.preflight_safetensors_index,
        preflight_remote_dir=args.preflight_remote_dir,
        preflight_report=args.preflight_report,
        preflight_timeout=args.preflight_timeout,
    )


if __name__ == "__main__":
    main()
