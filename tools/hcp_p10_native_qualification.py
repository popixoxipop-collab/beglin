#!/usr/bin/env python3
"""Beglin P10 bounded XOX runtime qualification (research-only).

Exact P9 cached-MLX binary on a run-owned scratch root; true unmodified
PrecisionEpochScheduler / durable ACK, no production writes or promotion.
Phases: freshly frozen token-disjoint 12-domain new prompts +12 tokens;
idle SIGKILL/graceful worker restart; two-slot native batch; fresh-block
steady-state tail crossover. BF16 teacher is NOT performed here.
"""
from __future__ import annotations
import argparse,contextlib,hashlib,json,math,os,pathlib,random,shutil
import statistics,subprocess,sys,time
from transformers import AutoTokenizer

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import precision_context as pc
import gpu_runtime_control as grc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,sha

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p10-qualification-20261014-1410")
P6=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104")
TOK=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
SRC_HEAD="b551237fb976e65447e6eb0e82b880cc57626caa"
BANK_SHA="7523f8e8ef5d2dea21391ad18ab575606b5804299386bf715a9647c815758083"
FROZEN_SHA="93f6f9f21983cb0c93e78b813b2972724aff45501b773943be2020574ada3b9f"
P6_QUALITY_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
P4_FEATURE_SHA="c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53"
C128_SHA="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
P9_MLX_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
P9_BINARY_SHA="103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
POLICIES={"A":A,"B":B}
MAX_RSS=14*1024**3
SEED=20261014

def d_ids(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def sequence(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def livepids():
    p=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True)
    return sorted(int(l.split()[0]) for l in p.splitlines() if str(STAGE) in l)
def liveproof():return {"stage_sha256":sha(STAGE),"route_sha256":sha(ROUTE),"pids":livepids()}
def assert_live(before):
    after=liveproof()
    if before!=after:raise RuntimeError("P10 live stage/route/PIDs changed")
    return after

def preflight():
    assert ROOT.is_dir() and ROOT.resolve(strict=True)==ROOT and not ROOT.is_symlink()
    assert sha(ROOT/"src/qwen_infer.c")==C128_SHA
    assert sha(ROOT/"src/mlx_moe.cpp")==P9_MLX_SHA
    assert sha(ROOT/"build/qwen_infer_gpu")==P9_BINARY_SHA
    assert sha(ROOT/"holdout_token_manifest.json")==FROZEN_SHA
    assert sha(ROOT/"src/experiments/hcp_p10/new_12domain_holdout.json")==BANK_SHA
    assert sha(TOK/"model.safetensors.index.json")==MODEL_SHA
    assert sha(P6/"inputs/p4_derived_long_features.json")==P4_FEATURE_SHA
    assert sha(P6/"repository/experiments/hcp_p6/evidence/quality.json")==P6_QUALITY_SHA
    got=subprocess.check_output(["/usr/bin/git","-C",str(ROOT/"src"),"rev-parse","HEAD"],text=True).strip()
    assert got==SRC_HEAD
    assert shutil.disk_usage(ROOT).free>20*1000**3+300*1000**2
    assert int(subprocess.check_output(["/usr/sbin/sysctl","-n","hw.memsize"]))>=40*1024**3
    before=liveproof()
    assert before=={"stage_sha256":STAGE_SHA,"route_sha256":ROUTE_SHA,"pids":[35481,35493]}
    p4=prepare(P6/"inputs/bank",TOK)
    selected=[p4["prompts"][3*i+i%3] for i in range(12)]
    original=json.loads((P6/"repository/experiments/hcp_p6/evidence/quality.json").read_text())["records"]
    expected={r["prompt_id"]:{pol:r[pol]["sequence_sha256"] for pol in ("A","B")} for r in original}
    assert len(expected)==12
    for row in selected:assert row["id"] in expected
    return selected,expected,before

@contextlib.contextmanager
def open_worker(label:str,cache:bool,*,slots:int=1):
    root=ROOT/"workers"/label
    if root.exists() or slots not in (1,2):
        raise RuntimeError("invalid fresh run-owned native worker or slot")
    old=(ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,
         ps.REQUEST_TIMEOUT_SECONDS,base.REPO)
    ps.PERSISTENT_BINARY=ROOT/"build/qwen_infer_gpu"
    ps.EXPECTED_PERSISTENT_BINARY_SHA=P9_BINARY_SHA
    ps.REQUEST_TIMEOUT_SECONDS=max(180,int(ps.REQUEST_TIMEOUT_SECONDS))
    base.REPO=ROOT/"src"
    w=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root)
    original_env=w._env
    def env():
        r=original_env()
        r["QWEN_MOE_CB_SLOTS"]=str(slots)
        r["QWEN_MOE_CB_REQS"]=str(slots)
        r["OMP_NUM_THREADS"]="2";r["OPENBLAS_NUM_THREADS"]="2"
        if cache:r["BEGLIN_HCP_P9_PREPARED_CACHE"]="1"
        else:r.pop("BEGLIN_HCP_P9_PREPARED_CACHE",None)
        return r
    w._env=env
    try:
        health=w.start()
        ack=grc.read_runtime_ack(w.ack_path)
        if (not health.get("alive") or ack["active_policy_hash"]!=pc.policy_hash(A)
            or int(health["rss_bytes"])>=MAX_RSS):
            raise RuntimeError("P10 wrong startup ACK/size")
        yield w
    finally:
        try:w.stop(force=True)
        finally:
            (ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,
            ps.REQUEST_TIMEOUT_SECONDS,base.REPO)=old

def submit(w,items:list[tuple[list[int],int]],policy:str,uid:str):
    start=time.perf_counter_ns()
    out=w.submit_with_precision_policy(items,target_policy=POLICIES[policy],admission_id=uid)
    wall=(time.perf_counter_ns()-start)/1e6
    ack=grc.read_runtime_ack(w.ack_path)
    state=w.health()
    rows=out.get("responses")
    if (out.get("finite_logits") is not True or out.get("inference_passes")!=1
        or not isinstance(rows,list) or len(rows)!=len(items)
        or any(not x or not(1<=len(x)<=cap) for x,(_,cap) in zip(rows,items))):
        raise RuntimeError("P10 invalid native generation count or finite logits")
    if (not state.get("alive") or int(state["rss_bytes"])>=MAX_RSS
        or ack["active_policy_hash"]!=pc.policy_hash(POLICIES[policy])):
        raise RuntimeError("P10 ACK/PID/RSS failed")
    epoch=out.get("precision_epoch") or {}
    sw=bool(epoch.get("transitioned"))
    if sw:
        if (ack.get("status")!="REBIND_SET_APPLIED" or
            ack.get("txn_id")!=epoch.get("txn_id") or
            len(epoch.get("changed_targets",[]))!=2 or
            int(epoch["after_epoch"])!=int(epoch["before_epoch"])+1):
            raise RuntimeError("P10 real native REBIND_SET terminal ACK failed")
    timing={"admission_ms":wall,"roundtrip_ms":float(out["roundtrip_ms"]),
            "engine_ms":float(out["engine_wall_ms"]),
            "transition_ms":float((epoch.get("transition_cost") or {}).get("transition_wall_ms",0))}
    if any(not math.isfinite(v) or v<0 for v in timing.values()) or min(wall,timing["roundtrip_ms"],timing["engine_ms"])<=0:
        raise RuntimeError("P10 timing invalid")
    return {"sequences":[sequence(x) for x in rows],"tokens":rows,
            "generated":[len(x) for x in rows],"policy":policy,
            "switched":sw,"rss_bytes":int(state["rss_bytes"]),
            "worker_pid":int(state["pid"]),
            "native_ack_status":ack["status"],"epoch":int(ack["weight_epoch"]),
            **timing}

def write_report(name,document):
    f=ROOT/"evidence"/(name+".json")
    if f.exists():raise RuntimeError("P10 evidence already sealed")
    with f.open("x") as out:json.dump(document,out,sort_keys=True,indent=2,allow_nan=False);out.write("\n")
    print("P10_EVIDENCE_SEALED",json.dumps({"name":name,"sha256":sha(f),"bytes":f.stat().st_size},sort_keys=True),flush=True)

def events(label):
    path=ROOT/"workers"/label/"worker.log"
    txt=path.read_text(errors="replace") if path.exists() else ""
    return {part:txt.count("P9_PREPARED_CACHE_"+part) for part in
       ("STORE name=","HIT name=","SOURCE_MISMATCH","BUDGET_SKIP","STORE_SKIPPED")}

def prime(w,old,expected,uid,*,ending="A"):
    order=("A","B","A","B","A") if ending=="A" else ("A","B","A","B")
    for j,p in enumerate(order):
        v=submit(w,[(old["token_ids"],4)],p,f"{uid}-warm-{j}-{p}")
        if v["sequences"][0]!=expected[old["id"]][p]:
            raise RuntimeError("P10 prime changed original P6 output")
    return len(order)

def run_holdout(old,expected,before):
    source=json.loads((ROOT/"src/experiments/hcp_p10/new_12domain_holdout.json").read_text())
    frozen=json.loads((ROOT/"holdout_token_manifest.json").read_text())
    assert len(source["items"])==len(frozen["records"])==12
    tok=AutoTokenizer.from_pretrained(str(TOK),local_files_only=True,
                                      trust_remote_code=False,use_fast=True)
    ids=[];ref={r["id"]:r for r in frozen["records"]}
    for item in source["items"]:
        txt=(source["instruction"]+" Task domain: "+item["domain"]+
             ". Question: "+item["question"]+" Answer:")
        row=ref[item["id"]]
        generated=tok.encode(txt,add_special_tokens=True)
        if d_ids(generated)!=row["token_sha256"] or len(generated)!=row["tokens"]:
            raise RuntimeError("frozen new prompt bytes changed")
        if len(generated)+12>=128:raise RuntimeError("unsafe P10 token context")
        ids.append((item,generated))
    data=[];stats={}
    for enabled in (False,True):
        label="p10-holdout-"+("on" if enabled else "off")
        with open_worker(label,enabled) as w:
            prime(w,old[0],expected,label)
            for ix,(item,tokens) in enumerate(ids):
                for p in (("A","B") if ix%2==0 else ("B","A")):
                    s=submit(w,[(tokens,12)],p,f"{label}-{ix}-{p}")
                    decoded=tok.decode(s["tokens"][0],skip_special_tokens=True).strip()
                    actual=" ".join(decoded.casefold().split())
                    target=" ".join(item["expected_short_answer"].casefold().split())
                    # Deliberately descriptive string match only; NOT external validity.
                    elementary_match=target in actual
                    data.append({"mode":"on" if enabled else "off","policy":p,
                         "item_id":item["id"],"domain":item["domain"],
                         "tokens_in":len(tokens),"budget_out":12,
                         "output_sequence_sha256":s["sequences"][0],
                         "generated_length":s["generated"][0],
                         "elementary_reference_substring_match":elementary_match,
                         "switched":s["switched"],"rss_bytes":s["rss_bytes"],
                         "admission_ms":s["admission_ms"]})
        stats[label]=events(label)
        ec=stats[label]
        if any(ec[k] for k in ("SOURCE_MISMATCH","BUDGET_SKIP","STORE_SKIPPED")):
            raise RuntimeError("P10 heldout cache source/budget error")
        if enabled and (ec["STORE name="]<4 or ec["HIT name="]<4):
            raise RuntimeError("P10 heldout cache not filled and reused")
        if not enabled and (ec["STORE name="]!=0 or ec["HIT name="]!=0):
            raise RuntimeError("P10 heldout OFF used cache")
    paired={}
    for row in data:
        key=(row["item_id"],row["policy"])
        mode=row["mode"]
        slot=paired.setdefault(key,{})
        if mode in slot:raise RuntimeError("duplicate P10 heldout mode")
        slot[mode]=row
    if len(paired)!=24 or any(set(v)!={"on","off"} for v in paired.values()):
        raise RuntimeError("P10 heldout A/B mode coverage incomplete")
    mismatches=0
    for values in paired.values():
        left,right=values["off"],values["on"]
        if (left["output_sequence_sha256"]!=right["output_sequence_sha256"] or
            left["generated_length"]!=right["generated_length"] or
            left["elementary_reference_substring_match"]!=right["elementary_reference_substring_match"]):
            mismatches+=1
    if mismatches:raise RuntimeError(f"P10 new holdout cache output divergence: {mismatches}")
    assert_live(before)
    d={"schema":"beglin-hcp-p10-new-short-answer-native-parity-v1",
       "status":"REAL_FIRST_USE_OUTPUT_PARITY_COMPLETED","source_bank_sha256":BANK_SHA,
       "token_manifest_sha256":FROZEN_SHA,"model_index_sha256":MODEL_SHA,
       "new_domains":12,"generation_budget":12,"total_native_requests":len(data),
       "paired_on_off_generations":24,"mismatched_generated_sequences":mismatches,
       "small_manual_rubric_not_external_benchmark":True,
       "BF16_reference_measured":False,"cache_events":stats,
       "records":data,"live_unchanged":True,"production_touched":False,"automatic_promotion":False}
    write_report("new_holdout",d)
    return d

def run_restart(old,expected,before):
    label="p10-crash-recovery-on"
    outputs=[]
    with open_worker(label,True) as w:
        first_pid=int(w.health()["pid"])
        for j,p in enumerate(("B","A","B")):
            r=submit(w,[(old[0]["token_ids"],4)],p,f"{label}-pre-{j}")
            if r["sequences"][0]!=expected[old[0]["id"]][p]:raise RuntimeError("restart preimage output mismatch")
        old_ack=grc.read_runtime_ack(w.ack_path)
        prev_epoch=int(old_ack["weight_epoch"])
        w.proc.kill() # only exact subprocess created above in our run-owned worker
        w.proc.wait(timeout=10)
        if w.is_alive():raise RuntimeError("P10 crash injection did not stop test worker")
        refused=False
        try:w.submit([(old[0]["token_ids"],4)])
        except Exception:refused=True
        if not refused:raise RuntimeError("dead worker unexpectedly accepted request")
        w.stop(force=True)
        restarted=w.start()
        if not restarted["alive"] or restarted["pid"]==first_pid:
            raise RuntimeError("P10 crash recovery did not allocate a new native process")
        ack=grc.read_runtime_ack(w.ack_path)
        if ack["active_policy_hash"]!=pc.policy_hash(A) or ack.get("status")!="PROMOTION_APPLIED":
            raise RuntimeError("P10 restart erroneously reused old policy or ACK")
        outputs.append({"event":"crash_and_restart","stale_request_refused":refused,
               "old_policy_epoch":prev_epoch,"new_runtime_epoch":int(ack["weight_epoch"]),
               "new_process":int(restarted["pid"])!=first_pid,"policy_reset_to_A":True})
        for j,p in enumerate(("B","A","B")):
            result=submit(w,[(old[0]["token_ids"],4)],p,f"{label}-aftercrash-{j}")
            if result["sequences"][0]!=expected[old[0]["id"]][p]:
                raise RuntimeError("crash recovery did not preserve original P6 output")
        pid_second=int(w.health()["pid"])
        w.stop(force=False)  # graceful journal/shutdown
        ready=w.start()
        if not ready["alive"] or ready["pid"]==pid_second:
            raise RuntimeError("graceful restart failed")
        ack=grc.read_runtime_ack(w.ack_path)
        if ack.get("status")!="PROMOTION_APPLIED" or ack["active_policy_hash"]!=pc.policy_hash(A):
            raise RuntimeError("graceful restart retained stale policy")
        for j,p in enumerate(("B","A","B")):
            result=submit(w,[(old[0]["token_ids"],4)],p,f"{label}-aftergraceful-{j}")
            if result["sequences"][0]!=expected[old[0]["id"]][p]:
                raise RuntimeError("graceful restart output mismatch")
        outputs.append({"event":"graceful_restart","new_process":int(ready["pid"])!=pid_second,
                    "policy_reset_to_A":True,"recovered_BAB_requests":3})
    ev=events(label) # final restarted worker log only; old log rotations explicitly excluded
    if ev["STORE name="]<1 or ev["HIT name="]<1 or ev["SOURCE_MISMATCH"]:
        raise RuntimeError("postrestart cache did not safely initialize")
    assert_live(before)
    report={"schema":"beglin-hcp-p10-worker-restart-recovery-v1",
       "status":"IDLE_CRASH_AND_GRACEFUL_NATIVE_RESTART_COMPLETED",
       "single_owned_pid_killed_after_complete_request":True,
       "inflight_txn_crash_not_tested":True,
       "successfully_recovered_requests":9,
       "P6_original_sequences_match":True,"real_ACKs_verified":True,
       "restart_records":outputs,"postrestart_cache_events":ev,
       "live_unchanged":True,"production_touched":False,"automatic_promotion":False}
    write_report("restart",report);return report

def run_two_slots(old,expected,before):
    data=[]
    for enabled in (False,True):
        label="p10-two-slot-"+("on" if enabled else "off")
        with open_worker(label,enabled,slots=2) as w:
            state=w.health()
            for j,p in enumerate(("A","B","A","B")):
                pair=[(old[0]["token_ids"],4),(old[1]["token_ids"],4)]
                result=submit(w,pair,p,f"{label}-{j}-{p}")
                if len(result["sequences"])!=2:
                    raise RuntimeError("two-slot response count mismatch")
                match=[result["sequences"][k]==expected[old[k]["id"]][p] for k in (0,1)]
                if not all(match):raise RuntimeError("two-slot native output differs from original P6 reference")
                data.append({"cache_enabled":enabled,"policy":p,"batch_reqs":2,
                   "input_tokens":[old[0]["tokens"],old[1]["tokens"]],
                   "both_outputs_match_P6":True,
                   "admission_ms":result["admission_ms"],
                   "rss_bytes":result["rss_bytes"],
                   "switched":result["switched"]})
        e=events(label)
        if enabled and e["SOURCE_MISMATCH"]:raise RuntimeError("two-slot cache source integrity failure")
    assert_live(before)
    record={"schema":"beglin-hcp-p10-two-slot-real-native-mla-v1",
       "status":"TWO_SLOT_BATCH_PASS",
       "slots":2,"requests_per_batch":2,
       "batch_admissions":len(data),"inferred_sequences":len(data)*2,
       "all_exact_P6_one_slot_reference_sequences":True,
       "cache_on_off_both_tested":True,
       "max_worker_rss_bytes":max(x["rss_bytes"] for x in data),
       "records":data,"live_unchanged":True,"production_touched":False,
       "multi_worker_or_arbitrary_concurrency_not_certified":True,
       "automatic_promotion":False}
    write_report("two_slots",record);return record

def linear_percentile(a,p):
    a=sorted(a);v=p*(len(a)-1);lo=math.floor(v);hi=math.ceil(v)
    return a[lo]+(a[hi]-a[lo])*(v-lo)

def run_tail(old,expected,before):
    rows=[];runs=[]
    for block in range(4):
        mode_order=(False,True) if block%2==0 else (True,False)
        for enabled in mode_order:
            label=f"p10-steady-b{block}-"+("on" if enabled else "off")
            with open_worker(label,enabled) as w:
                nw=prime(w,old[0],expected,label,ending="A")
                for idx,entry in enumerate(old):
                    result=submit(w,[(entry["token_ids"],4)],"A",f"{label}-steady-{idx}")
                    if result["switched"]:raise RuntimeError("steady cell unexpectedly changed policy")
                    if result["sequences"][0]!=expected[entry["id"]]["A"]:
                        raise RuntimeError("P10 steady cache output differs from original P6")
                    rows.append({"block":block,"enabled":enabled,
                         "domain":entry["domain"],"prompt_id":entry["id"],
                         "input_tokens":entry["tokens"],"cache_enabled":enabled,
                         "policy":"A","admission_ms":result["admission_ms"],
                         "engine_ms":result["engine_ms"],"rss_bytes":result["rss_bytes"],
                         "seq_sha_match_P6":True})
            e=events(label)
            if enabled and (e["STORE name="]<4 or e["HIT name="]<4 or e["SOURCE_MISMATCH"]):
                raise RuntimeError("P10 steady-state ON cache missing exact prepared entries")
            if not enabled and (e["STORE name="] or e["HIT name="]):
                raise RuntimeError("P10 steady OFF mode used cache")
            runs.append({"block":block,"cache_enabled":enabled,"warmups_excluded":nw,"cache_events":e})
            print("P10_STEADY_BLOCK_DONE",json.dumps({
                "block":block,"cache_enabled":enabled,"measurements":12,
                "p95_ms":round(linear_percentile([r["admission_ms"] for r in rows if r["block"]==block and r["enabled"]==enabled],.95),3),
                "elapsed_blocks":len(runs)}),flush=True)
    by={}
    for r in rows:by[(r["block"],r["prompt_id"],r["enabled"])]=r
    if len(by)!=4*12*2:raise RuntimeError("incomplete P10 paired steady trials")
    diffs=[];blockmeans=[]
    for block in range(4):
        per=[]
        for entry in old:
            on=by[(block,entry["id"],True)]["admission_ms"]
            off=by[(block,entry["id"],False)]["admission_ms"]
            per.append(on-off);diffs.append(on-off)
        blockmeans.append(statistics.mean(per))
    rng=random.Random(SEED)
    draws=10000
    bs=sorted(statistics.mean([blockmeans[rng.randrange(4)] for i in range(4)]) for j in range(draws))
    summary={}
    for enabled in (False,True):
        a=[r["admission_ms"] for r in rows if r["enabled"]==enabled]
        summary["on" if enabled else "off"]={
          "n":len(a),"mean_ms":statistics.mean(a),"p50_ms":linear_percentile(a,.5),
          "p95_ms":linear_percentile(a,.95),
          "block_mean_ms":[statistics.mean(r["admission_ms"] for r in rows if r["block"]==b and r["enabled"]==enabled) for b in range(4)],
          "max_rss_gib":max(r["rss_bytes"] for r in rows if r["enabled"]==enabled)/1024**3}
    paired={"mean_on_minus_off_ms":statistics.mean(diffs),
         "block_bootstrap95_ci_ms":[bs[250],bs[9750]],"blocks":4,"reps_per_block":12}
    assert_live(before)
    report={"schema":"beglin-hcp-p10-repeated-steady-block-tail-v1",
        "status":"REAL_FRESH_SESSION_STEADY_TAIL_CROSSOVER_COMPLETED",
        "single_gpu_slot":True,"temp_p9_cached_binary_sha256":P9_BINARY_SHA,
        "P9_original_corpus_reused_for_latency_only":True,
        "independent_fresh_holdout_not_this_timing_suite":True,
        "all_output_sha_match_P6":True,
        "per_policy_rest_state_A":True,"warmups_excluded":True,
        "summary":summary,"paired":paired,"rows":rows,
        "run_receipts":runs,"production_touched":False,
        "automatic_promotion":False}
    write_report("steady_tail",report);return report

def run_phase(mode:str):
    old,expected,before=preflight()
    try:
        if mode=="holdout":run_holdout(old,expected,before)
        elif mode=="restart":run_restart(old,expected,before)
        elif mode=="two_slots":run_two_slots(old,expected,before)
        elif mode=="steady_tail":run_tail(old,expected,before)
        else:raise ValueError("invalid P10 bounded phase")
        assert_live(before)
    finally:
        # No request manifests, raw token values or worker logs are exported.
        prefix={"holdout":"p10-holdout-","restart":"p10-crash-recovery-on",
          "two_slots":"p10-two-slot-","steady_tail":"p10-steady-b"}[mode]
        for path in (ROOT/"workers").iterdir():
            if not path.name.startswith(prefix):continue
            if path.is_symlink() or path.parent!=(ROOT/"workers"):raise RuntimeError("unsafe P10 cleanup path")
            shutil.rmtree(path)
    print("P10_PHASE_DONE",json.dumps({"phase":mode,"native_process_exit":0,
        "stage_unchanged":True,"route_unchanged":True,"production_touched":False}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=("holdout","restart","two_slots","steady_tail"),required=True)
    a=ap.parse_args();run_phase(a.phase)
