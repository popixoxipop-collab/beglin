#!/usr/bin/env python3
"""Isolated P9 one-shot abort after first GPU binding: true rollback and recovery.

Requires an extra *scratch-only* binary built from exact P5 CB128 plus injected
test failure; no production ACK override, worker state monkeypatch or live write.
"""
import hashlib,json,os,pathlib,subprocess,sys,shutil
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import hcp_p9_native_trial as t
import gpu_runtime_control as grc
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,sha

ROOT=t.ROOT
FAULT_SHA="6611036998baa344a83809a526d241f7752530d89b9a9b9cbb0102f6a2466d2a"
def main():
    prompts,expected,before,_=t.preflight()
    binary=ROOT/"build-fault/qwen_infer_gpu"
    if not binary.is_file() or binary.is_symlink():raise ValueError("fault-only binary absent")
    if sha(ROOT/"src-fault/qwen_infer.c")!=FAULT_SHA:raise ValueError("fault source not SHA-sealed")
    p=ROOT/"workers/p9-fault-native"
    if p.exists():raise ValueError("fault worker root already exists")
    oldbin,oldsha,oldtimeout,oldrepo=ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO
    ps.PERSISTENT_BINARY=binary
    ps.EXPECTED_PERSISTENT_BINARY_SHA=sha(binary)
    ps.REQUEST_TIMEOUT_SECONDS=max(int(ps.REQUEST_TIMEOUT_SECONDS),180)
    base.REPO=ROOT/"src"
    w=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=p)
    old_env=w._env
    def env():
        v=old_env()
        v["QWEN_MOE_CB_SLOTS"]="1";v["QWEN_MOE_CB_REQS"]="1"
        v["BEGLIN_HCP_P9_PREPARED_CACHE"]="1"
        v["BEGLIN_HCP_P9_FAULT_AFTER_FIRST_BIND_ONCE"]="1"
        return v
    w._env=env
    outcomes=[]
    try:
        health=w.start()
        pid=int(health["pid"])
        if health.get("runtime_policy_hash")!=pc.policy_hash(A) or not health.get("alive"):raise ValueError("incorrect initial A")
        row=prompts[0]
        caught=None
        try:
            w.submit_with_precision_policy([(row["token_ids"],4)],target_policy=B,
                admission_id="p9-rollback-first-B-deliberately-fail")
        except Exception as e:
            caught=type(e).__name__
        ack=grc.read_runtime_ack(w.ack_path)
        alive=w.health()
        first={"caught_error_type":caught,"terminal_status":ack["status"],"policy_hash":ack["active_policy_hash"],
               "epoch":int(ack["weight_epoch"]),"worker_pid_preserved":int(alive["pid"])==pid}
        if (caught is None or first["terminal_status"]!="REBIND_SET_FAILED" or
            first["policy_hash"]!=pc.policy_hash(A) or not first["worker_pid_preserved"]):
            raise ValueError("P9 intentionally failed rebind did not preserve entire A preimage")
        outcomes.append(first)
        for i,pol in enumerate(("B","A","B")):
            res=w.submit_with_precision_policy([(row["token_ids"],4)],target_policy=t.POLICIES[pol],
                   admission_id=f"p9-rollback-recover-{i}-{pol}")
            response=res.get("responses")
            if not response or len(response)!=1 or not response[0] or t.sseq(response[0])!=expected[row["id"]][pol]:
                raise ValueError("P9 repaired/rebound output disagrees with P6 reference")
            ack=grc.read_runtime_ack(w.ack_path)
            alive=w.health()
            if (ack["status"]!="REBIND_SET_APPLIED" or
                ack["active_policy_hash"]!=pc.policy_hash(t.POLICIES[pol]) or
                not alive.get("alive") or int(alive["pid"])!=pid):
                raise ValueError("P9 rollback recovery terminal ACK or worker PID changed")
            outcomes.append({"policy":pol,"terminal_status":ack["status"],
                 "output_SHA_matched":True,"epoch":int(ack["weight_epoch"]),
                 "worker_pid_preserved":True})
    finally:
        try:w.stop(force=True)
        finally:
            ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO=oldbin,oldsha,oldtimeout,oldrepo
    log=p/"worker.log"
    msg=log.read_text(errors="replace") if log.is_file() else ""
    if msg.count("P9_FAULT_AFTER_FIRST_BIND_ONCE")!=1:
        raise ValueError("P9 fault injection did not fire exactly once")
    if msg.count("P9_PREPARED_CACHE_HIT")<1:
        raise ValueError("P9 failed to reuse packed cache after restored transaction")
    after=t.liveproof()
    if after!=before:raise ValueError("P9 fault study touched live stage or route")
    record={"schema":"beglin-hcp-p9-real-native-one-shot-rollback-v1",
       "status":"EXACT_REBIND_SET_FAILED_ACK_AND_RECOVERY_CONFIRMED",
       "fault_source_sha256":FAULT_SHA,"fault_binary_sha256":sha(binary),
       "cached_mlx_source_sha256":t.MLX_SHA,
       "fault_injected_once":True,"error_expected_not_hidden":True,
       "failed_B_preimage_A_retained":True,
       "native_worker_same_PID":True,"recovery_exact_ABB_policy_sequence":["B","A","B"],
       "recovery_tokens_match_P6":True,
       "cache_hit_after_failed_transaction":True,
       "terminal_evidence":outcomes,
       "live_preimage":before,"live_postimage":after,
       "production_touched":False,"automatic_live_promotion":False,
       "full_concurrency_or_restart_certified":False}
    t.save(ROOT/"results/p9_rollback.json",record)
    shutil.rmtree(p)
    print("P9_REAL_ROLLBACK_RECOVERY",json.dumps({"status":record["status"],
       "worker_PID_stable":True,"after_failed_txn":outcomes[0],
       "cache_hit_after_restore":True,"production_modified":False},sort_keys=True),flush=True)

if __name__=="__main__":main()
