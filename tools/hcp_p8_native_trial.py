#!/usr/bin/env python3
"""P8 XOX isolated MLX CB128 switch-cost phases and same-input research comparison.

Two scratch-only binaries: unmodified P5 CB128 (timing), P8 observer (9 phases).
No production service, release, policy, source checkpoint or active route changes.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, pathlib, random
import shutil, statistics, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import precision_context as pc
import gpu_runtime_control as grc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A, B, sha, stats
from hcp_p8_profile_patch import validate_phases, strip_observer

ROOT = pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p8-native-20261014-1245")
P6 = pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104")
STAGE = pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
LIVE = pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
TOK = pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
MODEL_INDEX_SHA = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
P4_FEATURE_SHA = "c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53"
P6_QUALITY_SHA = "2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
P5_STAGE_SHA = "8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
C128_SHA = "bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
PROFILE_SHA = "84ff47596d6912741b8dcd14c8bdb2028f09031c4c48b78272314a4374d88edf"
MLX_SHA = "daded3c0e4bf7cc2a86e0fab8c1f73322b5b0ffea0002d9b2d38857d2e9591ae"
PHASE_FIELDS = ("precheck_ms","synchronize_ms","snapshot_ms","qng64_cache_lookup_or_build_ms",
                "mlx_bind_ms","bind_verify_and_loop_ms","epoch_reset_ms","metadata_commit_ms","durable_ack_ms")
POLICIES = {"AF": [], "A": A, "B": B}
RSS_MAX = 14 * 1024**3
SEED = 20261014

def seqhash(seq):
    return hashlib.sha256(",".join(map(str,seq)).encode()).hexdigest()

def write_json(path,doc):
    path=pathlib.Path(path)
    if path.exists(): raise ValueError("immutable result already exists")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as fp:json.dump(doc,fp,sort_keys=True,indent=2,allow_nan=False);fp.write("\n")

def pids():
    txt=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True)
    return sorted(int(line.split()[0]) for line in txt.splitlines()
                  if str(STAGE) in line)

def preflight():
    if ROOT.resolve(strict=True)!=ROOT or not ROOT.is_dir() or ROOT.is_symlink():
        raise ValueError("P8 root must be fixed owned scratch")
    h=lambda p:sha(pathlib.Path(p))
    orig=json.loads((ROOT/"preflight.json").read_text())
    if (orig.get("source_head")!="9d55fff527aae34bd2568b149661e52cc75a8504"
       or orig.get("route_sha_before")!=h(LIVE)):
        raise ValueError("P8 source commit or live route changed")
    if h(STAGE)!=P5_STAGE_SHA or h(TOK/"model.safetensors.index.json")!=MODEL_INDEX_SHA:
        raise ValueError("stage/checkpoint mismatch")
    if h(ROOT/"plain_cb128_qwen_infer.c")!=C128_SHA:
        raise ValueError("P8 exact C128 source missing")
    if h(ROOT/"src/qwen_infer.c")!=PROFILE_SHA or h(ROOT/"src/mlx_moe.cpp")!=MLX_SHA:
        raise ValueError("P8 native observer source mismatch")
    if strip_observer((ROOT/"src/qwen_infer.c").read_text())!=(ROOT/"plain_cb128_qwen_infer.c").read_text():
        raise ValueError("P8 observer changed mathematics")
    features=P6/"repository/experiments/hcp_p6/evidence/quality.json"
    f=P6/"inputs/p4_derived_long_features.json"
    if h(features)!=P6_QUALITY_SHA or h(f)!=P4_FEATURE_SHA:
        raise ValueError("P6 quality or original P4 prompt identity mismatch")
    if shutil.disk_usage(ROOT).free < 20*1000**3 + 300*1000**2:
        raise ValueError("20GB XOX storage reserve + 300MB margin unavailable")
    if int(subprocess.check_output(["/usr/sbin/sysctl","-n","hw.memsize"]))<40*1024**3:
        raise ValueError("insufficient physical memory")
    src=ROOT/"src"
    if subprocess.check_output(["/usr/bin/git","-C",str(src),"rev-parse","HEAD"],text=True).strip()!=orig["source_head"]:
        raise ValueError("incorrect source commit")
    plain=ROOT/"build-plain/qwen_infer_gpu"; prof=ROOT/"build-profile/qwen_infer_gpu"
    if any(not z.is_file() or z.is_symlink() for z in (plain,prof)):
        raise ValueError("missing exact scratch native binaries")
    all_prompts=prepare(P6/"inputs/bank",TOK)
    selected=[all_prompts["prompts"][3*i+i%3] for i in range(12)]
    frozen={x["prompt_id"]:x for x in json.loads(f.read_text())["runs"]}
    expected={}
    for x in json.loads(features.read_text())["records"]:
        expected[x["prompt_id"]]={p:x[p]["sequence_sha256"] for p in ("A","B")}
    if len(expected)!=12 or len({x["domain"] for x in selected})!=12:
        raise ValueError("P8 missing 12 original research domains")
    for row in selected:
        old=frozen[row["id"]]
        if old["token_sha256"]!=row["token_sha256"] or old["prompt_length"]!=row["tokens"] or not 104<=row["tokens"]<=112:
            raise ValueError("P8 unmatched prompt tokens")
    snapshot={"stage":h(STAGE),"route":h(LIVE),"pids":pids()}
    return selected,expected,{"plain_binary_sha256":h(plain),"observer_binary_sha256":h(prof),
                             "source_git_head":orig["source_head"],"before":snapshot}

@contextlib.contextmanager
def launch(name,kind,*,profile=False):
    binary=ROOT/("build-profile" if profile else "build-plain")/"qwen_infer_gpu"
    source=ROOT/"src"
    worker_path=ROOT/"workers"/name
    if worker_path.exists():raise ValueError("duplicate P8 worker directory")
    trace=ROOT/("trace-"+name)
    if profile:trace.mkdir(exist_ok=False)
    old_bin,old_expected,old_repo,old_timeout=ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,base.REPO,ps.REQUEST_TIMEOUT_SECONDS
    ps.PERSISTENT_BINARY=binary.resolve(strict=True)
    ps.EXPECTED_PERSISTENT_BINARY_SHA=sha(binary)
    ps.REQUEST_TIMEOUT_SECONDS=max(int(ps.REQUEST_TIMEOUT_SECONDS),180)
    base.REPO=source
    worker=(ps.PersistentRouteWorker(route=base.baseline_route(),root=worker_path)
            if kind=="AF" else ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=worker_path))
    native_env=worker._env
    def env():
        e=native_env()
        e.update({"QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"1",
                  "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"})
        if profile:e["BEGLIN_HCP_P8_PROFILE_DIR"]=str(trace)
        return e
    worker._env=env
    try:
        health=worker.start()
        ack=grc.read_runtime_ack(worker.ack_path)
        assert health.get("alive") and ack["active_policy_hash"]==pc.policy_hash(POLICIES["AF" if kind=="AF" else "A"])
        assert int(health["rss_bytes"])<RSS_MAX
        worker.p8_pid=int(health["pid"])
        yield worker,trace
    finally:
        try:worker.stop(force=True)
        finally:
            ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,base.REPO,ps.REQUEST_TIMEOUT_SECONDS=old_bin,old_expected,old_repo,old_timeout

def infer(worker,row,policy,label,*,trace=None):
    start=time.perf_counter_ns()
    if policy=="AF":
        result=worker.submit([(row["token_ids"],4)])
    else:
        result=worker.submit_with_precision_policy([(row["token_ids"],4)],
                target_policy=POLICIES[policy],admission_id=label)
    elapsed=(time.perf_counter_ns()-start)/1e6
    seq=result.get("responses")
    if (result.get("finite_logits") is not True or result.get("inference_passes",1)!=1 or
        not isinstance(seq,list) or len(seq)!=1 or not (1<=len(seq[0])<=4)):
        raise ValueError("nonfinite/missing native inference result")
    ack=grc.read_runtime_ack(worker.ack_path)
    health=worker.health()
    if (not health.get("alive") or int(health["pid"])!=worker.p8_pid
        or int(health["rss_bytes"])>=RSS_MAX
        or ack["active_policy_hash"]!=pc.policy_hash(POLICIES[policy])):
        raise ValueError("P8 native policy ACK/PID/RSS mismatch")
    epoch=result.get("precision_epoch") or {}
    switched=bool(epoch.get("transitioned"))
    if switched and (ack.get("status")!="REBIND_SET_APPLIED" or
                     ack.get("txn_id")!=epoch.get("txn_id") or
                     len(epoch.get("changed_targets",[]))!=2 or
                     int(epoch["after_epoch"])!=int(epoch["before_epoch"])+1):
        raise ValueError("P8 two-target exact native terminal ACK mismatch")
    phase=None
    if trace:
        path=trace/"native_phases.jsonl"
        nlines=path.read_text().splitlines() if path.is_file() else []
        if switched:
            if not nlines:raise ValueError("P8 switched with no native phase trace")
            phase=json.loads(nlines[-1])
            validate_phases(phase)
            if int(phase["epoch"])!=int(ack["weight_epoch"]):
                raise ValueError("P8 phase/ACK epoch mismatch")
        worker.p8_previous_phase_count=getattr(worker,"p8_previous_phase_count",0)+(1 if switched else 0)
        if len(nlines)!=worker.p8_previous_phase_count:
            raise ValueError("P8 unexpected phase count (stale/duplicate record)")
    vals={k:float(result[k]) for k in ("engine_wall_ms","roundtrip_ms")}
    vals["admission_wall_ms"]=elapsed
    vals["transition_wall_ms"]=float((epoch.get("transition_cost") or {}).get("transition_wall_ms",0))
    if (any(not math.isfinite(v) or v<0 for v in vals.values()) or
        min(vals["engine_wall_ms"],vals["roundtrip_ms"],elapsed)<=0):
        raise ValueError("P8 invalid timing")
    return {"policy":policy,"seq_sha256":seqhash(seq[0]),"generation_length":len(seq[0]),
            "switched":switched,"ack":ack.get("status"),
            "epoch":int(ack["weight_epoch"]),"rss_bytes":int(health["rss_bytes"]),
            "qng_cache_hits":int(ack.get("transition_cache_hits",0)),
            "qng_cache_misses":int(ack.get("transition_cache_misses",0)),
            **vals,"phases":phase}

def make_phase_trace(selected,expected):
    allruns={}
    for profiling in (False,True):
        label="observed" if profiling else "control"
        records=[]
        with launch("parity-"+label,"A",profile=profiling) as (w,trace):
            for i,row in enumerate(selected[:2]):
                order=("A","B","B","A") if i%2==0 else ("B","A","A","B")
                for j,p in enumerate(order):
                    data=infer(w,row,p,f"p8-parity-{label}-{i}-{j}",trace=trace if profiling else None)
                    if data["seq_sha256"]!=expected[row["id"]][p]:
                        raise ValueError("P8 logit-observer or control generated different tokens from P6")
                    records.append({"prompt_id":row["id"],"domain":row["domain"],"position":j,**data})
        allruns[label]=records
    left,right=allruns["control"],allruns["observed"]
    if [(r["prompt_id"],r["policy"],r["seq_sha256"]) for r in left]!=[
         (r["prompt_id"],r["policy"],r["seq_sha256"]) for r in right]:
        raise ValueError("P8 instrumented/control identical input mismatch")
    write_json(ROOT/"results/parity.json",{"schema":"beglin-hcp-p8-native-observer-parity-v1",
        "pairs":len(left),"all_native_sequences_exact":True,"ACKs_verified":True,
        "control":left,"observed":right,"production_touched":False})
    phase_rows=[]
    with launch("phase-measure","A",profile=True) as (w,trace):
        for i,row in enumerate(selected):
            order=("A","B","B","A") if i%2==0 else ("B","A","A","B")
            for j,p in enumerate(order):
                data=infer(w,row,p,f"p8-phase-{i}-{j}",trace=trace)
                if data["seq_sha256"]!=expected[row["id"]][p]:
                    raise ValueError("P8 measurement tokens disagree with sealed P6 source")
                if data["switched"]:
                    phase_rows.append({"domain":row["domain"],"prompt_id":row["id"],
                        "input_tokens":row["tokens"],"policy":p,"cache_hits":data["qng_cache_hits"],
                        "cache_misses":data["qng_cache_misses"],**data["phases"]})
            print("P8_REAL_PHASE_DOMAIN",json.dumps({"completed":i+1,"out_of":12,
                 "phase_records":len(phase_rows),"source_verified":True}),flush=True)
    if len(phase_rows)<12:raise ValueError("P8 missing native transition phase coverage")
    ag={}
    for col in PHASE_FIELDS+("entry_to_ack_ms",):
        values=[float(r[col]) for r in phase_rows]
        ag[col]=dict(count=len(values),mean=statistics.mean(values),
                     p50=statistics.median(values),max=max(values))
    doc={"schema":"beglin-hcp-p8-xox-real-9phase-v1",
         "status":"REAL_NATIVE_PROFILE_RESEARCH_ONLY","records":phase_rows,
         "transition_count":len(phase_rows),"summary":ag,
         "control_binary_sha256":sha(ROOT/"build-plain/qwen_infer_gpu"),
         "profiled_binary_sha256":sha(ROOT/"build-profile/qwen_infer_gpu"),
         "seq_ack_parity_pass":True,"observer_overhead_excluded_from_speed":True,
         "production_touched":False,"automatic_live_promotion":False}
    write_json(ROOT/"results/native-phases.json",doc)
    return doc

def timing_comparison(selected,expected):
    seed=random.Random(SEED)
    cases=["fixed_AF","fixed_A","fixed_B","alternating","minimum_hold_8"]
    runorder=cases[:]
    seed.shuffle(runorder)
    records=[]
    for ix,name in enumerate(runorder):
        kind="AF" if name=="fixed_AF" else "A"
        row_out=[]
        seen_af_sequences={}
        with launch("performance-"+name,kind,profile=False) as (w,trace):
            for j in range(24):
                row=selected[j%12]
                if name=="fixed_AF":pol="AF"
                elif name=="fixed_A":pol="A"
                elif name=="fixed_B":pol="B"
                elif name=="alternating":pol="A" if j%2==0 else "B"
                else:pol="A" if (j//8)%2==0 else "B"
                data=infer(w,row,pol,f"p8-timing-{name}-{j}")
                if pol != "AF" and data["seq_sha256"]!=expected[row["id"]][pol]:
                    raise ValueError("P8 real speed trial model outputs not equal P6 sealed results")
                if pol == "AF":
                    prev=seen_af_sequences.setdefault(row["id"],data["seq_sha256"])
                    if prev != data["seq_sha256"]:
                        raise ValueError("P8 fixed AF repeated prompt generated inconsistent tokens")
                row_out.append({"prompt_id":row["id"],"domain":row["domain"],"ordinal":j,
                                "block":j//12,**data})
            print("P8_REAL_TIMING_ARM",json.dumps({"arm":name,"measured":len(row_out),
                 "switches":sum(x["switched"] for x in row_out),"no_errors":True}),flush=True)
        records.extend([{"arm":name,**x} for x in row_out])
    summaries={}
    for name in cases:
        rows=[r for r in records if r["arm"]==name]
        times=[r["admission_wall_ms"] for r in rows]
        engine=[r["engine_wall_ms"] for r in rows]
        times.sort()
        summaries[name]={"n":len(rows),"switches":sum(r["switched"] for r in rows),
            "admission_ms_mean":statistics.mean(times),"admission_ms_p50":statistics.median(times),
            "admission_ms_p95":times[int(round((len(times)-1)*.95))],
            "engine_ms_mean":statistics.mean(engine)}
    doc={"schema":"beglin-hcp-p8-xox-native-5arm-pilot-v1",
        "status":"REAL_NATIVE_PILOT_DESCRIPTIVE_ONLY","seed":SEED,"arm_order":runorder,
        "fixed_AF_native_measured":True,"records":records,"summary":summaries,
        "no_new_holdout":True,"insufficient_temporal_blocks_for_speedup_claim":True,
        "quality_authority_for_B_not_granted":True,
        "no_proven_speedup":True,"production_touched":False,"automatic_live_promotion":False}
    write_json(ROOT/"results/performance.json",doc)
    return doc

def main():
    args=argparse.ArgumentParser()
    args.add_argument("--phase",choices=("profile","performance","all"),default="profile")
    a=args.parse_args()
    selected,expected,env=preflight()
    ps.REQUEST_TIMEOUT_SECONDS=max(ps.REQUEST_TIMEOUT_SECONDS,180)
    # Route and stage are cryptographically verified again after each phase.
    try:
        if a.phase in ("profile","all"):make_phase_trace(selected,expected)
        if a.phase in ("performance","all"):timing_comparison(selected,expected)
    finally:
        # Persistent run-owned worker directories may contain prompt IDs; remove
        # after all trials (including failures), retaining only sanitized JSON.
        for folder in (ROOT/"workers").iterdir():
            if folder.is_symlink():raise ValueError("unsafe symlink worker cleanup")
            shutil.rmtree(folder)
        for folder in ROOT.glob("trace-*"):
            if folder.is_symlink():raise ValueError("unsafe symlink trace cleanup")
            shutil.rmtree(folder)
    after={"stage":sha(STAGE),"route":sha(LIVE),"pids":pids()}
    if after != env["before"]:raise ValueError("P8 operation changed original live route/stage/PIDs")
    report={"schema":"beglin-hcp-p8-native-run-receipt-v1","phase":a.phase,
      "control_binary_sha256":env["plain_binary_sha256"],
      "instrumented_binary_sha256":env["observer_binary_sha256"],
      "git_commit":env["source_git_head"],"model_index_sha256":MODEL_INDEX_SHA,
      "before":env["before"],"after":after,
      "live_preserved":True,"production_touched":False,"automatic_live_promotion":False,
      "speedup_proven":False}
    write_json(ROOT/"results"/("research-receipt-"+a.phase+".json"),report)
    print("P8_REAL_NATIVE_COMPLETE",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":main()
