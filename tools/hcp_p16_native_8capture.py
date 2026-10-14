#!/usr/bin/env python3
"""P16 bounded REAL native MLX A/B first-token full-logits extraction.

No mock output. Every 409600-byte native C float32 dump is generated at exact
last input prefill, pass=1, before argmax, one request / one new token.
Identical P15 frozen original token ids, opposite 5/6-bit P6 policies A/B.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,re,shutil,subprocess,sys,time
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import gpu_runtime_control as grc
import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_native_policies_ab import A,B

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p16-native-logits-20261015")
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
PREREG_HEAD="f2ed4b49541db2cdedc2103fcf1aade09a9997c7"
P16_CAPTURE_C_SHA="aedb3f915c532b7e454515598b6d9836a4962e23391e2e304980cde64bc7c1af"
P9_MLX_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
REFERENCE_SHA="0ea208ca6b6bf9fa404fd291679e164f63217c91b1d6e15756d331261998c36e"
FROZEN_INPUTS_SHA="c85d16c113780d73692fd0bdb9b1afaacb40bda9a998876420f361899d3cbb8b"
MODEL_INDEX_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
LIVE_BINARY_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
LIVE_ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
EXPECTED_LIVE_PIDS=[35481,35493]
MAX_RSS=14*1024**3
ROWS=re.compile(r"^P16_NATIVE_FULL_LOGITS rid=([0-9a-f]{32}) inputsha=([0-9a-f]{64}) plen=(7|9) pos=(6|8) next=([0-9]+) vocab=(102400) bytes=(409600) floatsha=([0-9a-f]{64})$")
POLICY={"A":A,"B":B}

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open("rb") as f:
        for v in iter(lambda:f.read(1024*1024),b""):h.update(v)
    return h.hexdigest()

def original_live():
    output=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=5)
    return {"binary_sha256":sha(STAGE),"route_sha256":sha(ROUTE),
      "PIDs":sorted(int(z.split()[0]) for z in output.splitlines() if str(STAGE) in z)}

def before():
    if (ROOT.resolve(strict=True)!=ROOT or ROOT.is_symlink() or
        ROOT.parent.name!=".tailnet-tmp" or ROOT.name!="hcp-p16-native-logits-20261015"):
        raise RuntimeError("P16 sandbox source not exact owned XOX root")
    if shutil.disk_usage(ROOT).free<20_000_000_000:
        raise RuntimeError("P16 20GB physical SSD protection gate failed")
    head=subprocess.check_output(["/usr/bin/git","-C",str(ROOT/"src"),"rev-parse","HEAD"],text=True).strip()
    if head!=PREREG_HEAD:raise RuntimeError("P16 native run not from preregistered Git commit")
    src=ROOT/"src"
    if (sha(src/"qwen_infer.c")!=P16_CAPTURE_C_SHA or sha(src/"mlx_moe.cpp")!=P9_MLX_SHA
        or sha(src/"experiments/hcp_p16/FROZEN_TEACHER_VS_NATIVE_V1.json")!=REFERENCE_SHA
        or sha(src/"experiments/hcp_p15/FROZEN_INPUTS.json")!=FROZEN_INPUTS_SHA
        or sha(MODEL/"model.safetensors.index.json")!=MODEL_INDEX_SHA):
        raise RuntimeError("P16 C / MLX / frozen teacher input or original model changed")
    live=original_live()
    if live!={"binary_sha256":LIVE_BINARY_SHA,"route_sha256":LIVE_ROUTE_SHA,
             "PIDs":EXPECTED_LIVE_PIDS}:
        raise RuntimeError("P16 immutable production stage or serving PID differs")
    if not (ROOT/"build/qwen_infer_gpu").is_file():raise RuntimeError("P16 test-only GPU executable absent")
    frozen=json.loads((src/"experiments/hcp_p15/FROZEN_INPUTS.json").read_text())["entries"]
    ref=json.loads((src/"experiments/hcp_p16/FROZEN_TEACHER_VS_NATIVE_V1.json").read_text())
    assert len(frozen)==len(ref["full_teacher_refs"])==2
    for item,expect in zip(frozen,ref["full_teacher_refs"]):
        if item["id"]!=expect["prompt_id"] or len(item["token_ids"])!=expect["length"]:
            raise RuntimeError("P16 teacher token input geometry changed")
        text=",".join(map(str,item["token_ids"]))
        if hashlib.sha256(text.encode()).hexdigest()!=expect["token_sha256"]:
            raise RuntimeError("P16 teacher token identities no longer identical")
    return live,frozen,ref,sha(ROOT/"build/qwen_infer_gpu")

def capture_one(w,item,policy,order,old_log,old_files):
    ids=item["token_ids"];token_sha=item["token_sha256"]
    started=time.perf_counter_ns()
    out=w.submit_with_precision_policy([(ids,1)],target_policy=POLICY[policy],
         admission_id=f"p16-original-p15-token-{order:02d}-{policy}")
    wall=(time.perf_counter_ns()-started)/1e6
    seq=out.get("responses")
    ack=grc.read_runtime_ack(w.ack_path)
    health=w.health()
    if (out.get("finite_logits") is not True or
        not isinstance(seq,list) or len(seq)!=1 or len(seq[0])!=1 or
        int(health["rss_bytes"])>=MAX_RSS or not health["alive"] or
        ack["active_policy_hash"]!=pc.policy_hash(POLICY[policy]) or
        int(health["pid"])!=int(w.proc.pid)):
        raise RuntimeError("P16 native generation/ACK/worker state invalid")
    ep=out.get("precision_epoch") or {}
    if ep.get("transitioned"):
        if (ack.get("status")!="REBIND_SET_APPLIED" or
            ack.get("txn_id")!=ep.get("txn_id") or
            len(ep.get("changed_targets",[]))!=2 or
            int(ep["after_epoch"])!=int(ep["before_epoch"])+1):
            raise RuntimeError("P16 native weight transition terminal ACK not verified")
    logdata=w.log_path.read_text(errors="replace")
    newlog=logdata[old_log:]
    lines=[s.strip() for s in newlog.splitlines() if "P16_NATIVE_FULL_LOGITS" in s]
    if len(lines)!=1:raise RuntimeError(f"P16 expected exactly one native first-prefill capture; found {len(lines)}")
    match=ROWS.fullmatch(lines[0])
    if not match:raise RuntimeError("P16 native C capture metadata contract malformed")
    rid,tok,plen,pos,first,vocab,bytes_,digest=match.groups()
    if (tok!=token_sha or int(plen)!=len(ids) or int(pos)!=len(ids)-1 or
        int(first)!=seq[0][0] or int(vocab)!=102400 or int(bytes_)!=409600):
        raise RuntimeError("P16 actual native input / generated token / capture position mismatch")
    captures=ROOT/"capture"
    files={p.name for p in captures.iterdir()}
    new=files-old_files
    if new!={rid+".f32"}:raise RuntimeError("P16 captured data absent / extra/stale unexpected file")
    file=captures/(rid+".f32")
    if file.is_symlink() or not file.is_file() or file.stat().st_size!=409600:
        raise RuntimeError("P16 real C GPU logits file invalid")
    if sha(file)!=digest:
        raise RuntimeError("P16 whole native logits digest differs from original C witness")
    values=np.fromfile(file,dtype="<f4")
    if len(values)!=102400 or not np.isfinite(values).all():
        raise RuntimeError("P16 native full logits not finite or correct vocabulary")
    if int(np.argmax(values))!=int(seq[0][0]):
        raise RuntimeError("P16 actual raw native argmax differs from generated first token")
    if not (0<wall<180000):
        raise RuntimeError("P16 native admission timing invalid")
    return {"schema":"beglin-hcp-p16-real-native-one-request-full-logits-v1",
      "request_index":order,"input_name":item["id"],"prompt_tokens_sha256":tok,
      "prompt_token_count":len(ids),"policy":policy,
      "actual_native_precision_epoch":int(ack["weight_epoch"]),
      "actual_native_policy_sha256":ack["active_policy_hash"],
      "native_transitioned":bool(ep.get("transitioned")),
      "terminal_ACK_status":ack["status"],"terminal_ACK_SHA256":ack["ack_sha256"],
      "actual_worker_pid":int(health["pid"]),"max_worker_rss_observed_bytes":int(health["rss_bytes"]),
      "native_capture_request_id":rid,"native_real_first_generated_token_id":int(seq[0][0]),
      "native_full_logits_sha256":digest,
      "native_full_logits_size_bytes":file.stat().st_size,
      "native_full_logits_binary_filename":file.name,
      "native_logits_finite":True,"native_logits_argmax_match_greedy_token":True,
      "observed_admission_ms_not_suitable_for_speed_benchmark":wall}

def run():
    original,inputs,prereg,exe_sha=before()
    records=[]
    root=ROOT/"workers"/"p16-exact-p15-inputs"
    if root.exists():raise RuntimeError("P16 no reusing old/foreign GPU worker root")
    old=(ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO)
    ps.PERSISTENT_BINARY=ROOT/"build/qwen_infer_gpu"
    ps.EXPECTED_PERSISTENT_BINARY_SHA=exe_sha
    ps.REQUEST_TIMEOUT_SECONDS=max(180,int(ps.REQUEST_TIMEOUT_SECONDS))
    base.REPO=ROOT/"src"
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root)
    original_env=worker._env
    def env():
        e=original_env()
        e.update({"QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"1",
          "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2",
          "BEGLIN_HCP_P9_PREPARED_CACHE":"1",
          "BEGLIN_P16_LOGITS_DIR":str(ROOT/"capture")})
        return e
    worker._env=env
    try:
        first=worker.start()
        if (not first.get("alive") or int(first["rss_bytes"])>=MAX_RSS or
            grc.read_runtime_ack(worker.ack_path)["active_policy_hash"]!=pc.policy_hash(A)):
            raise RuntimeError("P16 exact startup policy/RSS check failed")
        for entry in prereg["native_order"]:
            i=entry["request_seq"]
            if i!=len(records):raise RuntimeError("P16 frozen run order changed")
            name=entry["prompt_index"];pol=entry["policy"]
            cursor=len(worker.log_path.read_text(errors="replace"))
            existing={p.name for p in (ROOT/"capture").iterdir()}
            row=capture_one(worker,inputs[name],pol,i,cursor,existing)
            records.append(row)
            print("P16_NATIVE_FULL_LOGITS_CAPTURE",json.dumps({
              "step":i+1,"total":len(prereg["native_order"]),
              "policy":pol,"prompt":name,"generated_token":row["native_real_first_generated_token_id"],
              "logits_SHA256":row["native_full_logits_sha256"],
              "worker_rss_GiB":round(row["max_worker_rss_observed_bytes"]/1024**3,3)},sort_keys=True),flush=True)
        assert len(records)==8
        by={}
        for r in records:
            by.setdefault((r["input_name"],r["policy"]),[]).append(r)
        if len(by)!=4 or any(len(x)!=2 for x in by.values()):
            raise RuntimeError("P16 missing duplicate native precision cell")
        consistency={}
        for (name,policy),items in by.items():
            ok=(items[0]["native_full_logits_sha256"]==items[1]["native_full_logits_sha256"] and
                items[0]["native_real_first_generated_token_id"]==items[1]["native_real_first_generated_token_id"])
            consistency[name+"_"+policy]=ok
        status="ALL_NATIVE_FIRST_LOGITS_CAPTURED_STABLE" if all(consistency.values()) else "CAPTURED_NATIVE_LOGITS_BUT_REPEATED_OUTPUTS_DIFFER"
        end=original_live()
        if end!=original:raise RuntimeError("P16 original live stage/route/PIDs changed during run")
        result={"schema":"beglin-hcp-p16-original-teacher-matched-native-A-B-full-logits-v1",
          "status":status,"source_prereg_commit":PREREG_HEAD,
          "source_model_index_SHA256":MODEL_INDEX_SHA,"source_C_p16_observer_SHA256":P16_CAPTURE_C_SHA,
          "source_P9_MLX_SHA256":P9_MLX_SHA,"exact_frozen_reference_SHA256":REFERENCE_SHA,
          "instrumented_native_binary_sha256":exe_sha,"single_new_native_worker":True,
          "native_requests_total":8,"distinct_input_policy_cells":4,
          "same_input_policy_repeated_twice":True,
          "native_full_logits_captured_each_request":True,
          "all_input_token_SHA_matching_P15":True,
          "all_real_persistent_precision_epoch_ACKs_verified":True,
          "all_captured_full_logits_argmax_equal_generated_first_token":True,
          "duplicate_capture_byte_equal_by_cell":consistency,
          "capture_rows":records,"full_teacher_comparison_run_here":False,
          "original_stage_before":original,"original_stage_after":end,
          "quality_noninferiority_certified":False,
          "speedup_certified_with_SHA_observer":False,
          "production_touched":False,"automatic_live_promotion":False}
        out=ROOT/"evidence/P16_NATIVE_8_CAPTURE.json"
        with out.open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
        print("P16_NATIVE_8_COMPLETE",json.dumps({"status":status,"rows":len(records),
            "binary_SHA256":exe_sha,"record_SHA256":sha(out),
            "every_source_matches_P15":True,"main_unchanged":True},sort_keys=True),flush=True)
    finally:
        try:worker.stop(force=True)
        finally:ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO=old

if __name__=="__main__":run()
