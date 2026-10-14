#!/usr/bin/env python3
"""Real, precisely witnessed mid-REBIND_SET crash and run-owned quarantine.

Two crash windows after first actual bind or after native epoch mutation/before
durable ACK. Fail closed until prior txn is atomically quarantined with fsync.
Restart is fresh; no automatic replay, no exactly-once outcome claim.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys,threading,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import gpu_runtime_control as grc
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_native_policies_ab import A,B,sha
from hcp_p4_long_prompt_registry import prepare
from hcp_p11_quarantine import quarantine,verify_quarantine

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p11-inflight-20261014-1816")
P6=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104")
TOK=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
SOURCE_SHA="6a3790590321b35e353f0b06978517e5407dcce9"
C128="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
P9_MLX="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
P6_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
MAX_RSS=14*1024**3
WINDOWS=("first_bind","before_ack")

def seqhash(a):return hashlib.sha256(",".join(map(str,a)).encode()).hexdigest()
def state():
    text=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True)
    pids=sorted(int(row.split()[0]) for row in text.splitlines() if str(STAGE) in row)
    return {"stage_sha":sha(STAGE),"route_sha":sha(ROUTE),"original_worker_pids":pids}
def assert_state(pre):
    got=state()
    if got!=pre:raise RuntimeError("P11 original live stage/route/PIDs changed")
    return got
def prescan():
    if (not ROOT.is_dir() or ROOT.resolve(strict=True)!=ROOT or ROOT.is_symlink()
        or ROOT.parent.name!=".tailnet-tmp"):
        raise ValueError("P11 test confined to pinned XOX scratch")
    commit=subprocess.check_output(["/usr/bin/git","-C",str(ROOT/"src"),"rev-parse","HEAD"],text=True).strip()
    if commit!=SOURCE_SHA:raise RuntimeError("P11 source commit mismatch")
    report=json.loads((ROOT/"preflight.json").read_text())
    if (report["source_commit"]!=SOURCE_SHA or
        sha(ROOT/"src/qwen_infer.c")!=report["fault_C_sha"] or
        sha(ROOT/"src/mlx_moe.cpp")!=P9_MLX or
        sha(ROOT/"build/qwen_infer_gpu")!=report["fault_binary_sha"]):
        raise RuntimeError("P11 signed scratch patch or build not pinned")
    if (sha(TOK/"model.safetensors.index.json")!=MODEL_SHA or
        sha(P6/"repository/experiments/hcp_p6/evidence/quality.json")!=P6_SHA):
        raise RuntimeError("P11 model/checkpoint/P6 quality evidence mismatch")
    free=shutil.disk_usage(ROOT).free
    if free<20*1000**3+300*1000**2:raise RuntimeError("XOX 20GB storage reserve breached")
    pre=state()
    if pre!={"stage_sha":STAGE_SHA,"route_sha":ROUTE_SHA,"original_worker_pids":[35481,35493]}:
        raise RuntimeError("P11 live serving identity unknown")
    corpus=prepare(P6/"inputs/bank",TOK)["prompts"]
    entry=corpus[0]
    quality=json.loads((P6/"repository/experiments/hcp_p6/evidence/quality.json").read_text())
    golden={r["prompt_id"]:r for r in quality["records"]}
    ref={p:golden[entry["id"]][p]["sequence_sha256"] for p in ("A","B")}
    return entry,ref,pre,report

def real_submit(w,tokens,policy,admission,expected):
    out=w.submit_with_precision_policy([(tokens,4)],target_policy={"A":A,"B":B}[policy],admission_id=admission)
    got=out["responses"]
    if (not isinstance(got,list) or len(got)!=1 or not got[0] or
        seqhash(got[0])!=expected[policy] or out.get("finite_logits") is not True):
        raise RuntimeError("P11 native recovery output differs from frozen P6")
    epoch=out.get("precision_epoch") or {}
    ack=grc.read_runtime_ack(w.ack_path)
    if (ack.get("status")!="REBIND_SET_APPLIED" or
        ack["active_policy_hash"]!=pc.policy_hash({"A":A,"B":B}[policy]) or
        ack.get("txn_id")!=epoch.get("txn_id") or
        len(epoch.get("changed_targets",[]))!=2 or
        int(epoch["after_epoch"])!=int(epoch["before_epoch"])+1):
        raise RuntimeError("P11 recovery native terminal ACK not exact")
    if not w.health().get("alive") or int(w.health()["rss_bytes"])>=MAX_RSS:
        raise RuntimeError("P11 recovery worker unstable")
    return {"policy":policy,"after_epoch":int(ack["weight_epoch"]),
      "txn_id_sha256":hashlib.sha256(ack["txn_id"].encode()).hexdigest(),
      "sequence_sha256":seqhash(got[0]),
      "worker_pid_stable":True}

def one_window(window,entry,ref,pre,build):
    label="p11-midtxn-"+window
    marker=ROOT/(window+".marker")
    owned=ROOT/"workers"/label
    if owned.exists() or marker.exists():raise RuntimeError("P11 crash trial already used")
    old=(ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,
         ps.REQUEST_TIMEOUT_SECONDS,base.REPO)
    ps.PERSISTENT_BINARY=ROOT/"build/qwen_infer_gpu"
    ps.EXPECTED_PERSISTENT_BINARY_SHA=build["fault_binary_sha"]
    ps.REQUEST_TIMEOUT_SECONDS=120
    base.REPO=ROOT/"src"
    w=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=owned)
    orig_env=w._env
    armed={"yes":True}
    def env():
        obj=orig_env()
        obj.update({"QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"1",
         "BEGLIN_HCP_P9_PREPARED_CACHE":"1","OMP_NUM_THREADS":"2",
         "OPENBLAS_NUM_THREADS":"2"})
        if armed["yes"]:
            obj["BEGLIN_P11_CRASH_WINDOW"]=window
            obj["BEGLIN_P11_CRASH_MARKER"]=str(marker)
        else:
            obj.pop("BEGLIN_P11_CRASH_WINDOW",None)
            obj.pop("BEGLIN_P11_CRASH_MARKER",None)
        return obj
    w._env=env
    exceptions=[];late=[]
    try:
        start=w.start()
        pid=int(start["pid"])
        baseline=grc.read_runtime_ack(w.ack_path)
        if baseline["active_policy_hash"]!=pc.policy_hash(A):
            raise RuntimeError("P11 research worker did not start with policy A")
        def attempt():
            try:
                result=w.submit_with_precision_policy([(entry["token_ids"],4)],
                  target_policy=B,admission_id="p11-attempt-"+window)
                late.append(result)
            except BaseException as e:
                exceptions.append(type(e).__name__)
        thread=threading.Thread(target=attempt,daemon=True)
        thread.start()
        limit=time.monotonic()+45.0
        while time.monotonic()<limit and not marker.is_file():
            if not w.is_alive():break
            time.sleep(.012)
        if not marker.is_file() or not w.is_alive():
            raise RuntimeError("P11 no durable in-flight marker or native worker exited before kill")
        markertext=marker.read_text().split()
        if len(markertext)!=2 or markertext[0]!=window:
            raise RuntimeError("P11 wrong crash phase marker")
        observed_ack=grc.read_runtime_ack(w.ack_path)
        if (observed_ack.get("txn_id")==markertext[1] or
            observed_ack["weight_epoch"]!=baseline["weight_epoch"] or
            observed_ack["active_policy_hash"]!=baseline["active_policy_hash"]):
            raise RuntimeError("P11 ACK was applied already; collision is not truly in-flight")
        command_before=owned/"txn.cmd"
        cmd_sha=sha(command_before)
        w.proc.kill()  # ONLY w.proc: exact PID returned by this run-owned worker
        w.proc.wait(timeout=10)
        thread.join(timeout=10)
        if thread.is_alive() or late or not exceptions or w.is_alive():
            raise RuntimeError("P11 failed to prove mid-transaction worker death and aborted admission")
        receipt=quarantine(ROOT,owned,marker,pid_is_dead=True)
        verified=verify_quarantine(owned)
        if (verified["source_command_sha256"]!=cmd_sha or
           verified["preimage_epoch"]!=baseline["weight_epoch"]):
            raise RuntimeError("P11 quarantine receipt does not match crashed command")
        armed["yes"]=False
        w.stop(force=True)
        if (owned/"txn.cmd").exists():raise RuntimeError("unquarantined command would be replayed")
        ready=w.start()
        if not ready.get("alive") or int(ready["pid"])==pid:
            raise RuntimeError("P11 did not fresh-restart after crash")
        ack=grc.read_runtime_ack(w.ack_path)
        if (ack["active_policy_hash"]!=pc.policy_hash(A) or
            ack["status"]!="PROMOTION_APPLIED" or ack.get("txn_id")==markertext[1]):
            raise RuntimeError("P11 stale ambiguous command may have replayed on startup")
        if not verify_quarantine(owned)["no_automatic_retry"]:
            raise RuntimeError("P11 quarantine no-retry invariant failed")
        recovery=[]
        for i,policy in enumerate(("B","A","B")):
            rec=real_submit(w,entry["token_ids"],policy,
               "p11-fresh-after-"+window+"-"+str(i),ref)
            recovery.append(rec)
        if not w.health().get("alive") or int(w.health()["pid"])!=int(ready["pid"]):
            raise RuntimeError("P11 native worker restarted during fresh recovery")
        log=(owned/"worker.log").read_text(errors="replace")
        if "P11_INFLIGHT_TEST_MARKER_PERSISTED" in log:
            raise RuntimeError("P11 crashed gate unexpectedly activated in recovery worker")
        assert_state(pre)
        report={"schema":"beglin-hcp-p11-real-midtxn-crash-reconcile-v1",
            "status":"TRUE_INFLIGHT_WINDOW_WITNESSED_AND_QUARANTINED",
            "phase":window,"source_commit":SOURCE_SHA,
            "isolated_binary_sha256":build["fault_binary_sha"],
            "controlled_PID_kill_after_fsynced_marker":True,
            "native_crash_marker_sha256":sha(marker),
            "native_cmd_SHA256":cmd_sha,
            "native_old_terminal_ack_for_txn":False,
            "old_policy_A_preimage_retained":True,
            "crashed_admission_thread_error_types":exceptions,
            "old_transaction_outcome":"UNKNOWN_CRASHED_BEFORE_DURABLE_ACK",
            "old_transaction_never_replayed":True,
            "old_transaction_retained_in_fsynced_quarantine":True,
            "quarantine_receipt":verified,
            "startup_fresh_pid":True,
            "startup_fresh_policy_A":True,
            "fresh_B_A_B_accepted_with_P6_output_SHA":True,
            "fresh_recovery":recovery,
            "original_stage_route_pids_unchanged":True,
            "production_touched":False,"automatic_live_promotion":False,
            "production_automatic_reconciler_installed":False}
        dst=ROOT/"evidence"/(window+".json")
        with dst.open("x") as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
        # Preserve the command as an exact, auditable retained research receipt;
        # no generated output ids or checkpoint bytes are included.
        escrow=ROOT/"evidence"/(window+"_quarantined_transaction.cmd")
        escrow.write_bytes(pathlib.Path(verified["quarantine_cmd_path"]).read_bytes())
        if sha(escrow)!=cmd_sha:raise RuntimeError("P11 escrow command SHA mismatch")
        print("P11_NATIVE_WINDOW_PASS",json.dumps({"phase":window,
          "old_txn_outcome":"UNKNOWN",
          "worker_killed_in_transaction":True,
          "fresh_BAB_count":len(recovery),
          "quarantined_cmd_sha256":cmd_sha,
          "receipt_sha256":sha(dst),"production_changed":False}),flush=True)
        return report
    finally:
        try:w.stop(force=True)
        finally:
            ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO=old
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--window",choices=WINDOWS,required=True)
    args=ap.parse_args()
    entry,ref,pre,build=prescan()
    try:
        one_window(args.window,entry,ref,pre,build)
    finally:
        # Do not erase raw quarantine in case forensic work is necessary.
        # Entire run-owned root is archived and later copied with SHA to EOE.
        pass
if __name__=="__main__":main()
