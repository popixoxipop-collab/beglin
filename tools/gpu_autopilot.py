#!/usr/bin/env python3
"""GPU precision autopilot -- real end-to-end automation for ONE candidate,
wired entirely from the project's own committed library code:

    gpu_isolated_preflight.run_ab_preflight / to_planner_evidence   (G4)
    precision_planner_v3.evaluate_candidate                          (admission)
    gpu_restart_canary.build_restart_canary / evaluate_restart_canary (G6)
    gpu_observer_control.request_regression_rollback /
        complete_regression_rollback                                 (G5 safety net)
    backend_adapters.MlxMetalBackendAdapter
    precision_control_state.ControlStore

No step re-derives decision logic by hand (that was the bug class this
replaces -- a human reconstructing "ADMIT_ONE_TARGET_RESTART_CANARY" or the
attribution-rate direction from memory, and occasionally getting it backwards).
Every decision here is made by calling the real function.

SCOPE, DELIBERATELY: this tool only ever reads/writes its own --control-root
(a scratch directory, defaults under the GPU worktree). It never touches any
real production promotion file or control directory (the CPU-side
autopilot_full.py's ssh_host="bob" / DEFAULT_PROMOTION_FILE equivalents are
NOT wired here). Pointing this at real production is a separate, explicit,
human decision this tool does not make on its own.

One run = one candidate = the full G4 -> admit -> G6 -> (G5 rollback if
needed) pipeline, fully automatic once launched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precision_context as pc
import precision_evidence_v3 as ev3
import precision_control_state as pcs
import precision_planner_v3 as planner
import backend_adapters as ba
import autopilot_observer_v3 as observer
import gpu_observer_control as goc
import gpu_runtime_control as grc
import gpu_restart_canary as canary
import gpu_isolated_preflight as preflight


class AutopilotError(RuntimeError):
    pass


def log(msg):
    print(f"[gpu-autopilot] {msg}", flush=True)


def build_context(*, binary_sha256, checkpoint_sha256, model_id="deepseek-v2-lite",
                   architecture="mla", kernel_revision="unspecified"):
    return pc.ExecutionContext(
        schema="precision-context-v3",
        model_id=model_id,
        architecture=architecture,
        checkpoint_sha256=checkpoint_sha256,
        tokenizer_sha256=checkpoint_sha256,
        base_artifact_sha256=checkpoint_sha256,
        backend="mlx_metal",
        device_fingerprint="gpu-autopilot",
        binary_sha256=binary_sha256,
        build_manifest_sha256=binary_sha256,
        kernel_revision=kernel_revision,
        execution_mode="online_cbatch",
        runtime_config_sha256=checkpoint_sha256,
        quant_format="qng64",
        group_size=64,
        correction_mode="off",
    )


def _persist_context_v3(context):
    if not ev3.configured():
        log("Supabase v3 evidence disabled: credentials absent")
        return False
    row = ev3.upsert_execution_context(context)
    if row.get("context_hash") != context.context_hash:
        raise AutopilotError("Supabase v3 context upsert verification mismatch")
    log(f"Supabase v3 context persisted: {context.context_hash}")
    return True


def _new_v3_run_id(stage, context_hash):
    return (
        f"gpu-autopilot-{stage}-{context_hash[:12]}-"
        f"{time.time_ns()}"
    )


def _persist_g4_v3(*, args, context, g4_result, event, reference):
    if not ev3.configured():
        return None

    bundle = g4_result.get("evidence_bundle") or {}
    baseline = g4_result["baseline"]
    candidate = g4_result["candidate"]
    baseline_run_id = _new_v3_run_id("g4-baseline", context.context_hash)
    candidate_run_id = _new_v3_run_id("g4-candidate", context.context_hash)

    def validation_row(run_id, run_kind, run, n, emitted_token, policy_hash):
        row = {
            "run_id": run_id,
            "context_hash": context.context_hash,
            "backend": context.backend,
            "run_kind": run_kind,
            "status": "PASS",
            "model_id": args.model,
            "role": args.role,
            "layer": int(args.layer),
            "n": n,
            "policy_postimage_sha256": policy_hash,
            "weight_epoch": int(run["weight_epoch"]),
            "manifest_sha256": run.get("manifest_sha256"),
            "req": event.get("req"),
            "pos": int(event["pos"]),
            "pass": True,
            "reason": "gpu_autopilot isolated G4 replay passed",
            "evidence_path": run.get("worker_log_path"),
            "evidence_sha256": run.get("worker_log_sha256"),
            "metrics": {
                "emitted_token": int(emitted_token),
                "reference_token": int(reference["emitted_token"]),
                "returncode": int(run.get("returncode", 0)),
                "ack_sha256": run.get("ack_sha256"),
            },
        }
        return {k: v for k, v in row.items() if v is not None}

    ev3.insert_validation_run(validation_row(
        baseline_run_id, "baseline", baseline, None,
        g4_result["baseline_emitted_token"],
        g4_result["baseline_policy_hash"],
    ))
    ev3.insert_validation_run(validation_row(
        candidate_run_id, "candidate", candidate, int(args.n),
        g4_result["candidate_emitted_token"],
        g4_result["candidate_policy_hash"],
    ))

    preflight_row = {
        "context_hash": context.context_hash,
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "model_id": args.model,
        "role": args.role,
        "layer": int(args.layer),
        "n": int(args.n),
        "baseline_policy_hash": g4_result["baseline_policy_hash"],
        "requested_policy_hash": g4_result["candidate_policy_hash"],
        "applied_policy_hash": g4_result["candidate_policy_hash"],
        "expected_epoch": 0,
        "observed_epoch": int(g4_result["candidate_epoch"]),
        "pass": True,
        "status": "PASS",
        "reason": "isolated GPU A/B preflight matched immutable reference",
        "emitted_token": int(g4_result["candidate_emitted_token"]),
        "reference_token": int(g4_result["reference_emitted_token"]),
        "evidence_sha256": bundle.get("verdict_payload_sha256"),
    }
    persisted = ev3.insert_live_preflight({
        k: v for k, v in preflight_row.items() if v is not None
    })
    log(
        "Supabase v3 G4 evidence persisted: "
        f"baseline={baseline_run_id} candidate={candidate_run_id}"
    )
    return {
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "preflight_id": persisted.get("id"),
    }


def run_worker(binary, cwd, moe_base, safetensors, manifest, run_dir,
               promotion_rows, slots=4, timeout=180):
    """Launch one real qwen_infer_gpu process to completion; returns
    (returncode, stdout, ack_dict)."""
    os.makedirs(run_dir, exist_ok=True)
    ack_path = os.path.join(run_dir, "applied_ack.json")
    if os.path.exists(ack_path):
        os.remove(ack_path)
    promo_path = os.path.join(run_dir, "promotion_nq.txt")
    with open(promo_path, "w") as f:
        if promotion_rows:
            for role, layer, n in promotion_rows:
                f.write(f"{role} {layer} {n}\n")
    env = dict(os.environ)
    env.update({
        "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
        "QWEN_MOE_BASE": moe_base,
        "QWEN_MOE_NEARTIE_CORRECT": "0",
        "QWEN_MOE_CB_PROMPT_MANIFEST": manifest,
        "QWEN_MOE_CB_SLOTS": str(slots),
        "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
        "QWEN_MOE_GPU_APPLIED_ACK": ack_path,
        "QWEN_MOE_PROMOTION_FILE_NQ": promo_path,
        "QWEN_MOE_PROMOTION_SAFETENSORS": safetensors,
    })
    proc = subprocess.Popen(
        [binary], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    out, _ = proc.communicate(timeout=timeout)
    with open(os.path.join(run_dir, "worker.log"), "w") as f:
        f.write(out)
    ack = grc.read_runtime_ack(ack_path) if os.path.exists(ack_path) else None
    return proc.returncode, proc.pid, out, ack


def run_worker_live(binary, cwd, moe_base, safetensors, manifest, run_dir,
                     promotion_rows, slots=4):
    """Launch one real qwen_infer_gpu process WITHOUT waiting for it to
    finish -- caller must wait_for_ack() then eventually proc.communicate()."""
    os.makedirs(run_dir, exist_ok=True)
    ack_path = os.path.join(run_dir, "applied_ack.json")
    txn_path = os.path.join(run_dir, "txn.cmd")
    for p in (ack_path, txn_path):
        if os.path.exists(p):
            os.remove(p)
    promo_path = os.path.join(run_dir, "promotion_nq.txt")
    with open(promo_path, "w") as f:
        if promotion_rows:
            for role, layer, n in promotion_rows:
                f.write(f"{role} {layer} {n}\n")
    env = dict(os.environ)
    env.update({
        "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
        "QWEN_MOE_BASE": moe_base,
        "QWEN_MOE_NEARTIE_CORRECT": "0",
        "QWEN_MOE_CB_PROMPT_MANIFEST": manifest,
        "QWEN_MOE_CB_SLOTS": str(slots),
        "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
        "QWEN_MOE_GPU_APPLIED_ACK": ack_path,
        "QWEN_MOE_GPU_TXN_FILE": txn_path,
        "QWEN_MOE_PROMOTION_FILE_NQ": promo_path,
        "QWEN_MOE_PROMOTION_SAFETENSORS": safetensors,
    })
    proc = subprocess.Popen(
        [binary], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return proc, ack_path, txn_path


def wait_for_ack(ack_path, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(ack_path) and os.path.getsize(ack_path) > 0:
            try:
                with open(ack_path) as f:
                    data = json.load(f)
                if data.get("status") in ("PROMOTION_APPLIED", "STARTUP_STATE"):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.1)
    raise AutopilotError(f"timed out waiting for ack at {ack_path}")


def parse_requests(output):
    reqs = {}
    for m in re.finditer(r"\[moe gpu cb online\] req (\d+) .*? tokens:\s*([^\n\r]*)", output):
        req = int(m.group(1))
        toks = [int(x) for x in m.group(2).split()]
        reqs[req] = toks
    return reqs


def observation_from_replay(*, context_hash, ack, output, pos, prompt_len,
                             correct_token, worker_instance_id=None,
                             target_replay_pass=None):
    """Build real ObservationEvidence from one worker's actual token output.

    attribution_hits convention (verified against RESULTS.md's real P4
    language, and gotten backwards TWICE now by hand-reasoning about it --
    once in the manual g6_drill.py session work, and again on this
    function's own first draft, which wrongly took a per-call `expected_token`
    and got passed the PRE observation's `orig_token` instead of the
    reference. Fixed here for good: attribution_hits ALWAYS counts replays
    that fail to reproduce the one true `correct_token` (the event's
    `reference.emitted_token`), for BOTH baseline and candidate observations
    -- there is no other correct value to compare against. Lower is
    healthier (fewer replays where this target is still attributably wrong).
    `target_replay_pass` is a separate, POST-only concept (computed the same
    way here, since it's the same comparison) -- callers pass an explicit
    override only when they want to force it (e.g. None for PRE, where it
    doesn't apply).
    """
    gen_idx = pos - (prompt_len - 1)
    reqs = parse_requests(output)
    n = len(reqs)
    matches = sum(1 for toks in reqs.values() if 0 <= gen_idx < len(toks) and toks[gen_idx] == correct_token)
    attribution_misses = n - matches
    if target_replay_pass is None:
        target_replay_pass = (matches == n and n > 0)
    return observer.ObservationEvidence(
        context_hash=context_hash,
        backend="mlx_metal",
        policy_hash=ack["active_policy_hash"],
        weight_epoch=ack["weight_epoch"],
        requests_completed=n,
        tokens_evaluated=sum(len(t) for t in reqs.values()),
        near_tie_events=n,
        reference_checks_attempted=n,
        effective_attribution_checks=n,
        attribution_hits=attribution_misses,
        run_errors=0,
        median_margin=None,
        target_replay_pass=target_replay_pass,
        worker_instance_id=worker_instance_id,
    ), reqs, matches


def run_candidate(args):
    binary_sha256 = args.binary_sha256
    checkpoint_sha256 = args.checkpoint_sha256
    context = build_context(
        binary_sha256=binary_sha256, checkpoint_sha256=checkpoint_sha256,
        architecture=args.architecture,
    )
    context_hash = context.context_hash
    log(f"context_hash={context_hash}")
    _persist_context_v3(context)

    root = args.control_root
    os.makedirs(root, exist_ok=True)
    event = json.loads(args.event_json)
    reference = json.loads(args.reference_json)

    # ---- G4: real isolated A/B preflight (baseline=[], candidate=this one target) ----
    # This is the FIRST real gate: a candidate that doesn't actually reproduce the
    # known corrected token is expected to be rejected right here, before ever
    # reaching a live restart canary. run_ab_preflight raises rather than
    # returning a failure dict, so a bad candidate is a normal, clean outcome
    # here -- not a crash.
    log(f"G4: isolated preflight for {args.role}/L{args.layer} n={args.n}")
    g4_root = os.path.join(root, "g4")
    try:
        g4_result = preflight.run_ab_preflight(
            host=None,
            cwd=args.cwd,
            binary=args.binary,
            architecture=args.architecture,
            moe_base=args.moe_base,
            manifest=args.g4_manifest,
            safetensors=args.safetensors,
            root=g4_root,
            baseline_policy=[],
            candidate_policy=[{"role": args.role, "layer": args.layer, "n": args.n}],
            event=event,
            reference=reference,
            prompt_len=args.prompt_len,
            timeout=args.timeout,
        )
    except preflight.GpuPreflightError as exc:
        log(f"G4 REJECTED: {exc}")
        return {"final_status": "REJECTED_AT_G4", "reason": str(exc)}
    log(f"G4 result: status={g4_result['status']} "
        f"baseline={g4_result['baseline_emitted_token']} "
        f"candidate={g4_result['candidate_emitted_token']} "
        f"reference={g4_result['reference_emitted_token']}")

    planner_evidence = preflight.to_planner_evidence(
        g4_result, context_hash=context_hash, expected_epoch=0,
    )
    _persist_g4_v3(
        args=args,
        context=context,
        g4_result=g4_result,
        event=event,
        reference=reference,
    )

    # ---- admission: the REAL planner decides, not a hand-built dict ----
    admission = planner.evaluate_candidate(
        context=context,
        current_policy=[],
        current_epoch=0,
        role=args.role,
        layer=args.layer,
        n=args.n,
        preflight_evidence=planner_evidence,
    )
    log(f"planner admission: {admission['action']}")
    if admission["action"] != "ADMIT_ONE_TARGET_RESTART_CANARY":
        log(f"STOP: planner did not admit this candidate ({admission.get('reason', '')})")
        return {"final_status": "NOT_ADMITTED", "admission": admission}

    # ---- G6: real restart canary, built by the real module ----
    plan = canary.build_restart_canary(
        admission, baseline_policy=[],
        min_requests=args.min_requests, max_requests=args.max_requests,
        min_effective_checks=1,
    )
    log(f"G6 plan built: target={plan['target']}")

    log("G6: PRE restart (baseline policy)")
    pre_rc, pre_pid, pre_out, pre_ack = run_worker(
        args.binary, args.cwd, args.moe_base, args.safetensors, args.g6_manifest,
        os.path.join(root, "g6_pre"), promotion_rows=None, slots=args.slots,
        timeout=args.timeout,
    )
    pre_evidence, pre_reqs, pre_matches_correct = observation_from_replay(
        context_hash=context_hash, ack=pre_ack, output=pre_out,
        pos=event["pos"], prompt_len=args.prompt_len,
        correct_token=reference["emitted_token"],
        worker_instance_id=f"pid-{pre_pid}",
        target_replay_pass=None,
    )
    log(f"G6 PRE: pid={pre_pid} {len(pre_reqs)} requests, "
        f"attribution_hits={pre_evidence.attribution_hits}")
    # Sanity check (not part of ObservationEvidence): baseline should still
    # reproduce the ORIGINAL known-wrong token, same as G4's own baseline arm.
    gen_idx = event["pos"] - (args.prompt_len - 1)
    pre_orig_matches = sum(
        1 for toks in pre_reqs.values()
        if 0 <= gen_idx < len(toks) and toks[gen_idx] == event["orig_token"]
    )
    log(f"G6 PRE sanity: {pre_orig_matches}/{len(pre_reqs)} reproduced orig_token={event['orig_token']}")

    log("G6: POST restart (candidate policy, fresh process)")
    post_rc, post_pid, post_out, post_ack = run_worker(
        args.binary, args.cwd, args.moe_base, args.safetensors, args.g6_manifest,
        os.path.join(root, "g6_post"),
        promotion_rows=[(args.role, args.layer, args.n)], slots=args.slots,
        timeout=args.timeout,
    )
    post_evidence, post_reqs, post_matches = observation_from_replay(
        context_hash=context_hash, ack=post_ack, output=post_out,
        pos=event["pos"], prompt_len=args.prompt_len,
        correct_token=reference["emitted_token"],
        worker_instance_id=f"pid-{post_pid}",
    )
    log(f"G6 POST: pid={post_pid} {len(post_reqs)} requests, "
        f"{post_matches} matched reference, "
        f"attribution_hits={post_evidence.attribution_hits}, "
        f"target_replay_pass={post_evidence.target_replay_pass}")
    if post_pid == pre_pid:
        raise AutopilotError("PRE and POST were not genuinely separate processes")

    canary_result = canary.evaluate_restart_canary(plan, pre_evidence, post_evidence)
    log(f"G6 verdict: status={canary_result['status']} decision={canary_result['decision']} "
        f"rollback_required={canary_result['rollback_required']}")

    store = pcs.ControlStore(root=os.path.join(root, "state"),
                              model_revision_id=args.model, backend="mlx_metal")

    if not canary_result["rollback_required"]:
        store.set_desired(context_hash=context_hash,
                           policy=[{"role": args.role, "layer": args.layer, "n": args.n}],
                           reason="gpu_autopilot: restart canary passed", source="gpu_autopilot")
        store.set_applied(context_hash=context_hash, policy=post_ack["active_policy"],
                           epoch=post_ack["weight_epoch"], txn_id="autopilot-canary-pass",
                           ack_sha256=post_ack["ack_sha256"])
        log(f"reconcile: {store.reconcile()['status']}")
        return {
            "final_status": "ADMITTED",
            "canary": canary_result,
            "note": ("Approved in this tool's own --control-root only. "
                     "No production promotion file was read or written."),
        }

    # ---- G5 safety net: the canary itself says rollback is required. The POST
    # worker has already exited (isolated restart canary, not a live demote),
    # so "rollback" here means: never adopt this candidate, and relaunch a
    # fresh live worker with it applied ONLY to prove the real live-DEMOTE
    # mechanism still works end to end, matching the G5 invariant that a
    # demote command is not success until a terminal ACK says so.
    log("G6 says rollback required -- exercising the real G5 rollback path")
    store.set_desired(context_hash=context_hash, policy=[],
                       reason="gpu_autopilot: candidate never admitted", source="gpu_autopilot")
    store.set_applied(context_hash=context_hash, policy=[{"role": args.role, "layer": args.layer, "n": args.n}],
                       epoch=post_ack["weight_epoch"], txn_id="autopilot-pre-rollback",
                       ack_sha256=post_ack["ack_sha256"])

    live_proc, live_ack_path, live_txn_path = run_worker_live(
        args.binary, args.cwd, args.moe_base, args.safetensors, args.g6_manifest,
        os.path.join(root, "g5_live"), promotion_rows=[(args.role, args.layer, args.n)],
        slots=args.slots,
    )
    live_ack = wait_for_ack(live_ack_path)
    adapter = ba.MlxMetalBackendAdapter(verified=True, context=context,
                                         ack_path=live_ack_path, txn_path=live_txn_path)
    live_evidence = goc.evidence_from_runtime_ack(
        context_hash=context_hash, runtime_ack=live_ack,
        metrics={
            "requests_completed": post_evidence.requests_completed,
            "tokens_evaluated": post_evidence.tokens_evaluated,
            "near_tie_events": post_evidence.near_tie_events,
            "reference_checks_attempted": post_evidence.reference_checks_attempted,
            "effective_attribution_checks": post_evidence.effective_attribution_checks,
            "attribution_hits": post_evidence.attribution_hits,
            "run_errors": 0,
        },
        target_replay_pass=post_evidence.target_replay_pass,
    )
    rollback_request = goc.request_regression_rollback(
        adapter=adapter, store=store,
        baseline=pre_evidence, post=live_evidence,
        baseline_policy=[], failed_policy=[{"role": args.role, "layer": args.layer, "n": args.n}],
        role=args.role, layer=args.layer, n=args.n,
        txn_id=f"gpu-autopilot-rollback-{int(time.time())}",
        min_requests=args.min_requests, min_effective_checks=1,
    )
    log(f"rollback request: action={rollback_request['action']}")
    _, _ = live_proc.communicate(timeout=args.timeout)
    rollback_complete = goc.complete_regression_rollback(
        adapter=adapter, store=store,
        txn_id=rollback_request["txn_id"], context_hash=context_hash,
        baseline_policy=[], failed_epoch=live_evidence.weight_epoch,
    )
    log(f"rollback complete: status={rollback_complete['status']}")
    return {
        "final_status": "REJECTED_AND_ROLLED_BACK",
        "canary": canary_result,
        "rollback": rollback_complete,
        "note": ("Nothing was ever admitted. The live-DEMOTE path was exercised "
                 "against this tool's own scratch worker only."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--event-json", required=True)
    ap.add_argument("--reference-json", required=True)
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--g4-manifest", required=True, help="single-entry manifest for the isolated G4 replay")
    ap.add_argument("--g6-manifest", required=True, help="repeated-entry manifest for real G6 PRE/POST traffic")
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--binary-sha256", required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--moe-base", required=True)
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--architecture", default="mla")
    ap.add_argument("--model", default="deepseek-v2-lite")
    ap.add_argument("--control-root", required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--min-requests", type=int, default=10)
    ap.add_argument("--max-requests", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    result = run_candidate(args)
    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2, default=str))
    clean_outcomes = ("ADMITTED", "REJECTED_AND_ROLLED_BACK", "REJECTED_AT_G4", "NOT_ADMITTED")
    sys.exit(0 if result["final_status"] in clean_outcomes else 1)


if __name__ == "__main__":
    main()
