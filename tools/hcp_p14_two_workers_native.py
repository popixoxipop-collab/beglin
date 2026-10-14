#!/usr/bin/env python3
"""P14 two real *simultaneous* P9-native DeepSeek workers + crashed peer survivor.

Two independent Python controller processes each own exactly one native MLX GPU
worker and one private REBIND_SET/ACK directory. Fresh GPU workers and a separate
P11 exact pre-ACK crash-injection binary are used. The surviving original P9 GPU
worker is never restarted while the sibling crashes and is quarantined.
No production route, binary, worker or release writes; evidence records only
hashes and bounded numerical telemetry, not original token values.
"""
from __future__ import annotations

import argparse,hashlib,json,multiprocessing as mp,os,pathlib,shutil,subprocess,sys,time,traceback
from collections import Counter

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p11-inflight-p14-multiworker-20261014")
RELEASE="g6-0.9.0-operator.900343"
COMMIT="93ac73bb6feefe4bf020b7882fbc0bf3d8d977e7"
P6=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104")
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
BASE_SHA="103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd"
FAULT_SHA="a5fd3f590ad6bbddcb69e690ccf738b711b8f7ceb3319f99ee933123f5ac7e78"
MLX_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
C128_SHA="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
FAULT_C_SHA="3d5a6d9f2f63d53e960c14a88dc4892ffa27cfd67b6d15258b6c930a7c17f2bd"
MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
QUALITY_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
RSS_PER_WORKER_LIMIT=7*(1024**3)
RSS_SUM_LIMIT=12*(1024**3)

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def seqsha(seq):return hashlib.sha256(",".join(map(str,seq)).encode()).hexdigest()
def stage_state():
    txt=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=5)
    return {"stage_sha256":sha(STAGE),"route_sha256":sha(ROUTE),
         "source_worker_pids":sorted(int(l.split()[0]) for l in txt.splitlines() if str(STAGE) in l)}
def preflight():
    if (not ROOT.is_dir() or ROOT.is_symlink() or
        ROOT.resolve(strict=True)!=ROOT or ROOT.parent.name!=".tailnet-tmp" or
        ROOT.name!="hcp-p11-inflight-p14-multiworker-20261014"):
        raise RuntimeError("P14 not in exact owned XOX research root")
    if shutil.disk_usage(ROOT).free < 20_300_000_000:
        raise RuntimeError("P14 XOX 20GB SSD reserve + safety margin not available")
    if subprocess.check_output(["/usr/bin/git","-C",str(ROOT/"src"),"rev-parse","HEAD"],text=True).strip()!=COMMIT:
        raise RuntimeError("P14 pinned preregistration source commit differs")
    checks={
        ROOT/"src/qwen_infer.c":C128_SHA,
        ROOT/"src-fault/qwen_infer.c":FAULT_C_SHA,
        ROOT/"src/mlx_moe.cpp":MLX_SHA,
        ROOT/"build/qwen_infer_gpu":BASE_SHA,
        ROOT/"build-fault/qwen_infer_gpu":FAULT_SHA,
        MODEL/"model.safetensors.index.json":MODEL_SHA,
        P6/"repository/experiments/hcp_p6/evidence/quality.json":QUALITY_SHA
    }
    for p,h in checks.items():
        if not p.is_file() or p.is_symlink() or sha(p)!=h:
            raise RuntimeError(f"P14 source/checkpoint SHA mismatch {p}")
    original=stage_state()
    expected={"stage_sha256":STAGE_SHA,"route_sha256":ROUTE_SHA,"source_worker_pids":[35481,35493]}
    if original!=expected:raise RuntimeError("P14 immutable original live stage or workers changed")
    # The whole 29GiB BF16 model is deliberately NOT loaded for these native
    # quantized P9 GPU worker trials.
    return original

def controller(conn,role):
    """Separate OS process; globals/process control never shared with the other."""
    native=None
    try:
        import precision_context as pc
        import gpu_runtime_control as grc
        import production_serving_supervisor as base
        import production_serving_supervisor_persistent as ps
        from hcp_p4_long_prompt_registry import prepare
        from hcp_p4_native_policies_ab import A,B
        from hcp_p11_quarantine import quarantine,verify_quarantine,inspect
        import threading
        from time import monotonic_ns

        policies={"A":A,"B":B}
        binary=ROOT/("build-fault" if role=="fault" else "build")/"qwen_infer_gpu"
        binary_expected=FAULT_SHA if role=="fault" else BASE_SHA
        if sha(binary)!=binary_expected:raise RuntimeError("controller binary not sealed")
        ps.PERSISTENT_BINARY=binary
        ps.EXPECTED_PERSISTENT_BINARY_SHA=binary_expected
        ps.REQUEST_TIMEOUT_SECONDS=max(180,int(ps.REQUEST_TIMEOUT_SECONDS))
        base.REPO=ROOT/"src"
        wr=ROOT/"workers"/("worker-"+role)
        if wr.exists():raise RuntimeError("P14 worker owns a foreign/previous directory")
        native=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=wr)
        original_env=native._env
        arm={"fault":role=="fault"}
        marker=ROOT/"before_ack.fault.marker"
        def env():
            e=original_env()
            e.update({"QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"1",
                "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2",
                "BEGLIN_HCP_P9_PREPARED_CACHE":"1"})
            if arm["fault"]:
                e["BEGLIN_P11_CRASH_WINDOW"]="before_ack"
                e["BEGLIN_P11_CRASH_MARKER"]=str(marker)
            else:
                e.pop("BEGLIN_P11_CRASH_WINDOW",None)
                e.pop("BEGLIN_P11_CRASH_MARKER",None)
            return e
        native._env=env
        prompts=prepare(P6/"inputs/bank",MODEL)["prompts"]
        quality=json.loads((P6/"repository/experiments/hcp_p6/evidence/quality.json").read_text())["records"]
        truth={r["prompt_id"]:{p:r[p]["sequence_sha256"] for p in ("A","B")} for r in quality}
        chosen=[prompts[3*i+i%3] for i in range(6)]
        if len(chosen)!=6 or len({z["domain"] for z in chosen})!=6:
            raise RuntimeError("P14 corpus not original six distinct P6 domains")
        def act(idx,policy,admission):
            row=chosen[idx]
            t0=monotonic_ns()
            out=native.submit_with_precision_policy([(row["token_ids"],4)],
                 target_policy=policies[policy],admission_id=admission)
            t1=monotonic_ns()
            seq=out.get("responses")
            if (not isinstance(seq,list) or len(seq)!=1 or not seq[0] or
                out.get("finite_logits") is not True or
                seqsha(seq[0])!=truth[row["id"]][policy]):
                raise RuntimeError("P14 native model generated output differs from original P6")
            ack=grc.read_runtime_ack(native.ack_path)
            health=native.health()
            if (ack["active_policy_hash"]!=pc.policy_hash(policies[policy]) or
                not health["alive"] or
                int(health["rss_bytes"])>=RSS_PER_WORKER_LIMIT or
                int(health["pid"])!=original_pid["value"]):
                raise RuntimeError("P14 native worker moved PID/policy or exhausted memory")
            ep=out.get("precision_epoch") or {}
            switched=bool(ep.get("transitioned"))
            if switched and (ack.get("status")!="REBIND_SET_APPLIED" or
                ack.get("txn_id")!=ep.get("txn_id") or
                len(ep.get("changed_targets",[]))!=2 or
                int(ep["after_epoch"])!=int(ep["before_epoch"])+1):
                raise RuntimeError("P14 genuine native txn/epoch ACK invalid")
            return {"status":"inference_ok","controller":role,
                "prompt_id":row["id"],"domain":row["domain"],
                "policy":policy,"seq_sha256":seqsha(seq[0]),
                "switched":switched,"terminal_ACK":ack["status"],
                "epoch":int(ack["weight_epoch"]),"PID":int(health["pid"]),
                "rss_bytes":int(health["rss_bytes"]),
                "start_ns":t0,"end_ns":t1}

        original_pid={"value":None}
        def start_worker():
            state=native.start()
            ack=grc.read_runtime_ack(native.ack_path)
            if (not state.get("alive") or
                ack["active_policy_hash"]!=pc.policy_hash(A) or
                int(state["rss_bytes"])>=RSS_PER_WORKER_LIMIT):
                raise RuntimeError("P14 source worker initial native ACK or RSS error")
            original_pid["value"]=int(state["pid"])
            return {"status":"started","controller":role,
                "PID":original_pid["value"],"rss_bytes":int(state["rss_bytes"]),
                "active_policy_hash":ack["active_policy_hash"]}

        def crash_after_before_ack():
            if role!="fault" or not arm["fault"] or marker.exists():
                raise RuntimeError("P14 fault-armed worker or marker identity missing")
            start_ack=grc.read_runtime_ack(native.ack_path)
            if start_ack["active_policy_hash"]!=pc.policy_hash(A):
                raise RuntimeError("P14 fault preimage is not A")
            errors=[];unexpected=[]
            def make_request():
                try:
                    result=native.submit_with_precision_policy(
                        [(chosen[0]["token_ids"],4)],
                        target_policy=B,admission_id="p14-beforeACK-owned-crash")
                    unexpected.append(result)
                except BaseException as exc:
                    errors.append(type(exc).__name__)
            workerthread=threading.Thread(target=make_request,daemon=True)
            workerthread.start()
            deadline=time.monotonic()+45
            while not marker.exists() and time.monotonic()<deadline:
                if not native.is_alive():break
                time.sleep(.01)
            if not marker.is_file() or not native.is_alive():
                raise RuntimeError("P14 did not witness a running native worker at fsynced marker")
            marks=marker.read_text().split()
            if len(marks)!=2 or marks[0]!="before_ack":
                raise RuntimeError("P14 crash marker not the exact before_ACK native phase")
            oldcmd=sha(native.txn_path)
            midack=grc.read_runtime_ack(native.ack_path)
            if (midack["weight_epoch"]!=start_ack["weight_epoch"] or
                midack["active_policy_hash"]!=start_ack["active_policy_hash"] or
                midack.get("txn_id")==marks[1]):
                raise RuntimeError("P14 native interrupted transaction already terminal; no real crash")
            crashed_pid=original_pid["value"]
            if native.proc.pid!=crashed_pid:raise RuntimeError("P14 wrong crash PID")
            native.proc.kill()  # this child owns exactly this research native PID
            native.proc.wait(timeout=10)
            workerthread.join(timeout=12)
            if native.is_alive() or workerthread.is_alive() or unexpected or not errors:
                raise RuntimeError("P14 fail-inflight worker didn't become dead and reject admission")
            pre=inspect(ROOT,wr,marker,pid_is_dead=True)
            if (pre["source_command_sha256"]!=oldcmd or
                pre["txn_id"]!=marks[1]):
                raise RuntimeError("P14 crash source command did not match durable marker")
            sealed=quarantine(ROOT,wr,marker,pid_is_dead=True)
            verified=verify_quarantine(wr)
            if sealed!=verified or (wr/"txn.cmd").exists():
                raise RuntimeError("P14 uncommitted original command not safely quarantined")
            arm["fault"]=False
            native.stop(force=True)
            fresh=native.start()
            if (int(fresh["pid"])==crashed_pid or
                grc.read_runtime_ack(native.ack_path)["active_policy_hash"]!=pc.policy_hash(A)):
                raise RuntimeError("P14 new worker replayed original ambiguous policy or PID")
            original_pid["value"]=int(fresh["pid"])
            recovery=[]
            for i,p in enumerate(("B","A","B")):
                recovery.append(act(0,p,f"p14-new-post-crash-{i}"))
            return {"status":"actual_midtxn_fault_quarantined_and_recovered",
                "crashed_worker_pid":crashed_pid,"fresh_worker_pid":original_pid["value"],
                "original_outcome":"UNKNOWN_BEFORE_DURABLE_ACK",
                "original_command_sha256":oldcmd,
                "marker_sha256":sha(marker),
                "old_terminal_ACK_for_crashed_txn":False,
                "quarantine_VERIFIED_sha256":sha(wr/"quarantine/VERIFIED.json"),
                "recovered_P6_sha_requests":len(recovery),
                "recovery":recovery,"source_worker_unchanged":True,
                "production_touched":False}

        while True:
            msg=conn.recv()
            try:
                op=msg["op"]
                if op=="start":response=start_worker()
                elif op=="run":
                    response=act(int(msg["index"]),str(msg["policy"]),str(msg["admission"]))
                elif op=="crash":response=crash_after_before_ack()
                elif op=="stop":
                    native.stop(force=True)
                    response={"status":"stopped","controller":role}
                elif op=="exit":
                    native.stop(force=True)
                    conn.send({"status":"exited","controller":role})
                    break
                else:raise RuntimeError("P14 invalid controller operation")
                conn.send(response)
            except Exception as exc:
                conn.send({"status":"ERROR","controller":role,
                    "error_type":type(exc).__name__,"error_message":str(exc)[:600]})
                break
    except BaseException as exc:
        try:conn.send({"status":"ERROR","controller":role,"fatal_type":type(exc).__name__,
                       "fatal_message":str(exc)[:950]})
        except Exception:pass
    finally:
        if native is not None:
            try:native.stop(force=True)
            except Exception:pass
        try:conn.close()
        except Exception:pass

def request(conn,operation,timeout=120):
    conn.send(operation)
    if not conn.poll(timeout):
        raise RuntimeError(f"P14 controller response timeout: {operation['op']}")
    reply=conn.recv()
    if reply.get("status")=="ERROR":
        raise RuntimeError(f"P14 {reply.get('controller')} error: {reply}")
    return reply
def assert_safestage(original):
    if stage_state()!=original:
        raise RuntimeError("P14 test mutated original live stage/route/PID")

def main():
    original=preflight()
    ctx=mp.get_context("spawn")
    workers={}
    def spawn(role):
        a,b=ctx.Pipe()
        p=ctx.Process(target=controller,args=(b,role),name="p14-controller-"+role)
        p.start();b.close()
        workers[role]={"process":p,"pipe":a,"state":"new"}
        return a
    records=[];crashproof=None;overlap=0;rss_peak=0
    try:
        alpha=spawn("alpha")
        alpha_start=request(alpha,{"op":"start"},timeout=160)
        workers["alpha"]["state"]="started"
        beta=spawn("beta")
        beta_start=request(beta,{"op":"start"},timeout=160)
        workers["beta"]["state"]="started"
        if alpha_start["PID"]==beta_start["PID"]:
            raise RuntimeError("P14 both controllers reused same native PID")
        rss_sum=alpha_start["rss_bytes"]+beta_start["rss_bytes"]
        if rss_sum>=RSS_SUM_LIMIT:
            raise RuntimeError("P14 sum RSS of simultaneous native workers exceeds 12GiB")
        rss_peak=rss_sum
        print("P14_TWO_REAL_MLX_NATIVE_WORKERS_STARTED",json.dumps({
            "distinct_native_PIDs":[alpha_start["PID"],beta_start["PID"]],
            "sum_rss_GiB":round(rss_sum/1024**3,4),
            "cache_opt_in":True,"original_live_stage_PID_set":original["source_worker_pids"]}),flush=True)
        for ix in range(6):
            for pos in range(4):
                policy_a="B" if pos%2==0 else "A"
                policy_b="A" if pos%2==0 else "B"
                alpha.send({"op":"run","index":ix,"policy":policy_a,
                            "admission":f"p14-a-{ix}-{pos}"})
                beta.send({"op":"run","index":ix,"policy":policy_b,
                           "admission":f"p14-b-{ix}-{pos}"})
                if not alpha.poll(140) or not beta.poll(140):
                    raise RuntimeError("P14 native worker concurrent request timeout")
                ra,rb=alpha.recv(),beta.recv()
                if ra.get("status")!="inference_ok" or rb.get("status")!="inference_ok":
                    raise RuntimeError(f"P14 dual native admission failed {ra} {rb}")
                if ra["PID"]!=alpha_start["PID"] or rb["PID"]!=beta_start["PID"]:
                    raise RuntimeError("P14 native worker restarted or shared PID unexpectedly")
                if (ra["prompt_id"]!=rb["prompt_id"] or
                    ra["policy"]==rb["policy"] or
                    ra["seq_sha256"]=="" or rb["seq_sha256"]==""):
                    raise RuntimeError("P14 cross-worker corpus or role ambiguous")
                pair_rss=ra["rss_bytes"]+rb["rss_bytes"]
                if pair_rss>=RSS_SUM_LIMIT:raise RuntimeError("P14 two resident native workers used >=12GiB")
                rss_peak=max(rss_peak,pair_rss)
                true_overlap=max(ra["start_ns"],rb["start_ns"])<min(ra["end_ns"],rb["end_ns"])
                overlap+=int(true_overlap)
                records.append({"domain":ra["domain"],"position":pos,
                 "simultaneous_calls_interval_overlapped":true_overlap,
                 "left":ra,"right":rb})
            print("P14_DUAL_WORKER_ACTUAL_DOMAIN",
                  json.dumps({"domain_index":ix+1,"domains_total":6,
                              "real_responses":len(records)*2,
                              "overlap_count":overlap,
                              "combined_max_rss_GiB":round(rss_peak/1024**3,3)}),flush=True)
        if overlap<1:raise RuntimeError("P14 no actual independent admission call intervals overlapped")
        stopped=request(beta,{"op":"stop"},timeout=30)
        workers["beta"]["state"]="stopped"
        exited=request(beta,{"op":"exit"},timeout=10)
        workers["beta"]["process"].join(timeout=10)
        if workers["beta"]["process"].is_alive():
            raise RuntimeError("P14 baseline beta process did not cleanly exit")
        beta_fault=spawn("fault")
        fault_state=request(beta_fault,{"op":"start"},timeout=160)
        workers["fault"]["state"]="started"
        if fault_state["PID"]==alpha_start["PID"]:
            raise RuntimeError("P14 fault worker PID equals healthy survivor")
        if fault_state["rss_bytes"]+rss_peak//2>=RSS_SUM_LIMIT:
            raise RuntimeError("P14 fault plus survivor preliminary RSS bound failed")
        beta_fault.send({"op":"crash"})
        survivor=[]
        for j in range(6):
            policy="B" if j%2==0 else "A"
            survivor.append(request(alpha,{"op":"run","index":j%6,"policy":policy,
                "admission":f"p14-alpha-while-peer-crashes-{j}"},timeout=130))
        if not beta_fault.poll(145):
            raise RuntimeError("P14 midtxn fault worker did not return quarantine proof")
        crashproof=beta_fault.recv()
        if crashproof.get("status")!="actual_midtxn_fault_quarantined_and_recovered":
            raise RuntimeError("P14 peer fault/repair failed: "+str(crashproof))
        if (any(z["PID"]!=alpha_start["PID"] for z in survivor) or
            crashproof["crashed_worker_pid"]==crashproof["fresh_worker_pid"]):
            raise RuntimeError("P14 healthy survivor changed PID or fault worker did not restart")
        for name in ("alpha","fault"):
            request(workers[name]["pipe"],{"op":"stop"},timeout=45)
            workers[name]["state"]="stopped"
            request(workers[name]["pipe"],{"op":"exit"},timeout=10)
            workers[name]["process"].join(timeout=12)
            if workers[name]["process"].is_alive():
                raise RuntimeError("P14 controller did not exit normally")
        assert_safestage(original)
        summary={"schema":"beglin-hcp-p14-two-real-worker-native-overlap-and-peer-inflight-crash-v1",
          "status":"TWO_CONCURRENT_NATIVE_WORKERS_AND_PEER_SURVIVOR_CRASH_RECOVERY_PASS",
          "source_preregistered_commit":COMMIT,
          "real_separate_controller_processes":2,
          "real_simultaneously_alive_P9_native_workers":2,
          "real_gpu_slots_total":2,
          "native_P9_SHA256":BASE_SHA,
          "native_fault_only_P11_SHA256":FAULT_SHA,
          "P9_MLX_KV128_source_SHA256":MLX_SHA,
          "real_concurrent_paired_calls":len(records),
          "real_concurrent_native_generated_responses":len(records)*2,
          "actual_matched_same_prompt_opposite_policy_pairs":len(records),
          "overlapped_admission_time_windows":overlap,
          "all_unchanged_P6_sequence_SHA_and_real_policy_ACKs":True,
          "simultaneous_native_PIDs":[alpha_start["PID"],beta_start["PID"]],
          "peak_combined_worker_RSS_bytes":rss_peak,
          "RSS_limits_bytes":{"per_native":RSS_PER_WORKER_LIMIT,"both":RSS_SUM_LIMIT},
          "crash_at_actual_native_window":"before_ack",
          "fault_original_txn_outcome":"UNKNOWN_NOT_RETRIED",
          "fault_original_txn_escrow_SHA256":crashproof["original_command_sha256"],
          "real_peer_crash_and_quarantine":crashproof,
          "survivor_native_PID_never_restarted":True,
          "survivor_requests_during_peer_crash":len(survivor),
          "survivor_after_before_original_production_unchanged":True,
          "original_service_before":original,"original_service_after":stage_state(),
          "full_BF16_teacher_logits_measured":False,
          "production_touched":False,"automatic_live_promotion":False,
          "real_concurrent_pair_records":records,
          "survivor_records":survivor}
        path=ROOT/"evidence/real_dual_worker_and_crash.json"
        with path.open("x") as f:json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
        print("P14_REAL_PEER_NATIVE_RECOVERY_SUCCESS",json.dumps({
            "paired":len(records),"total_admissions":len(records)*2,
            "overlap":overlap,"fault_stage":"before_ack",
            "survivor_pid_stable":True,"new_recoveries":3,
            "result_SHA256":sha(path),"stage_untouched":True},sort_keys=True),flush=True)
    finally:
        for name,w in workers.items():
            pipe=w["pipe"];proc=w["process"]
            if proc.is_alive():
                try:
                    if pipe.poll(0.1):pipe.recv()
                    pipe.send({"op":"exit"})
                    if pipe.poll(8):pipe.recv()
                    proc.join(timeout=8)
                except Exception:pass
                if proc.is_alive():
                    # Do not use broad kill commands in a shared host. Preserve
                    # run-owned forensic files and fail manual stop needed.
                    print("P14_CONTROLLER_NOT_STOPPED",name,proc.pid,flush=True)
            try:pipe.close()
            except Exception:pass

if __name__=="__main__":
    main()
