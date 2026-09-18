#!/usr/bin/env python3
"""L4 Phase 1 (shadow mode) -- the first real piece of the online-learning autopilot design
(.claude/history/2026-09-18_online-learning-autopilot-design.md).

D-l4-1: this is the smallest REAL, RUNNABLE slice of Phase 1 -- not a stub, not a plan, an
actual decision pipeline that runs today.
  WHY this scope and not more: target_safe_n() only reads Supabase (moe_quant_sweep_results),
  it never drives a new remote engine invocation -- so this script can make and log real
  decisions against real, already-collected data with ZERO new risk to bob (no SSH, no new
  compute, no chance of resource contention with brain.selfplay or anything else). Actually
  triggering NEW real-kernel sweeps for candidates with no prior sweep data at all is the part
  that needs the resource-gate + rate-limit machinery the design doc calls for -- deliberately
  not built here yet, see EXIT.
  COST: this run's decisions are only as good as whatever moe_quant_sweep_results already has.
  A candidate with zero real-kernel history gets "no data yet" and is skipped, not swept live.
  EXIT: to close the gap to a full Phase 1 (auto-triggers NEW sweeps for undercovered
  candidates), add: (a) a resource-gate check before any new sweep (reuse
  bob-macstudio-concurrent-load-guard's own live-load check as a library call), (b) a rate
  limiter (sweep_one_n costs up to ~9min/target for the 3-point REAL_LADDER), (c) call
  quant_search_n.sweep_one_n() directly instead of only reading existing rows.

Never writes to moe_role_precision_state.promoted or any live promotion file -- shadow
decisions land ONLY in moe_autopilot_shadow_decisions, which nothing else reads yet.

Usage:
    python3 tools/autopilot_shadow.py --model deepseek-v2-lite --limit 10
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from promotion_writeback import target_safe_n  # noqa: E402  (reuses the exact, already-reviewed decision logic)


def _supabase_env():
    url = os.environ.get("QWEN_SUPABASE_URL")
    key = os.environ.get("QWEN_SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("QWEN_SUPABASE_URL / QWEN_SUPABASE_KEY must be set")
    return url.rstrip("/"), key


def fetch_candidates(model, limit):
    """Real, undecided candidates: event_count>0 (flagged by live near-tie attribution) and not
    already promoted. This is exactly what a human reviewing promotion_controller.py's report
    would look at first -- same data, just queried directly here instead of via that script's
    JSONL-file mode (which reads local attribution logs; this reads the Supabase aggregate,
    a v2 later)."""
    url, key = _supabase_env()
    qs = (f"model=eq.{model}&event_count=gt.0&promoted=eq.false"
          f"&select=model,role,layer,event_count,current_bits,min_margin_observed"
          f"&order=event_count.desc&limit={limit}")
    p = subprocess.run(
        ["curl", "-s", "-H", f"apikey: {key}", "-H", f"Authorization: Bearer {key}",
         f"{url}/rest/v1/moe_role_precision_state?{qs}"],
        capture_output=True, text=True, timeout=30,
    )
    return json.loads(p.stdout)


def log_shadow_decision(model, role, layer, decided_n, reason, event_count, current_bits):
    url, key = _supabase_env()
    row = {
        "model": model, "role": role, "layer": layer,
        "decided_n": decided_n, "reason": reason[:2000],
        "event_count_at_decision": event_count, "current_bits_at_decision": current_bits,
        "phase": "P1-shadow",
    }
    p = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{url}/rest/v1/moe_autopilot_shadow_decisions",
         "-H", f"apikey: {key}", "-H", f"Authorization: Bearer {key}",
         "-H", "Content-Type: application/json", "-H", "Prefer: return=minimal",
         "-d", json.dumps([row])],
        capture_output=True, text=True, timeout=30,
    )
    if p.returncode != 0 or (p.stdout and "message" in p.stdout):
        print(f"  WARN: shadow-log insert may have failed: {p.stdout or p.stderr}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v2-lite")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    candidates = fetch_candidates(args.model, args.limit)
    print(f"[autopilot-shadow] {len(candidates)} candidate(s) for model={args.model}\n")

    for c in candidates:
        role, layer = c["role"], c["layer"]
        n, detail = target_safe_n(args.model, role, layer)
        if n is None:
            reason = detail.get("reason", "unsafe/no data")
            print(f"  {role}/L{layer}  (event_count={c['event_count']}, bits={c['current_bits']})  -> NO DECISION: {reason}")
        else:
            reason = json.dumps(detail.get("per_corpus", {}), default=str)
            print(f"  {role}/L{layer}  (event_count={c['event_count']}, bits={c['current_bits']})  -> shadow decision: n={n}")
        log_shadow_decision(args.model, role, layer, n, reason, c["event_count"], c["current_bits"])

    print("\n[autopilot-shadow] all decisions logged to moe_autopilot_shadow_decisions (phase=P1-shadow) -- no live promotion file touched")


if __name__ == "__main__":
    main()
