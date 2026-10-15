#!/usr/bin/env python3
"""P17 real XOX MLX source: 31 gold-prefix full logits × A/B and 12 greedy tasks × A/B.

Separate run-owned P9 GPU workers, never simultaneous, always exact original
P5 C CB128 + P9 MLX KV128. Data capture optin only in prefix phase. A/B
policy epoch and terminal ACK validated on every request. No actual labels
fabricated; bad gold completions are observations, not implementation failures.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,re,shutil,subprocess,sys,time
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import precision_context as pc
import gpu_runtime_control as grc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_native_policies_ab import A,B

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p17-unseen-multitoken-20261015")
FROZEN="a8e6e42b11a0f881317ab8afddebf7b266a965f4100bfddd7342183e73728521"
C_SHA="d724af1eb97ee715e206188fa120547f4cc1f18f55a3c37938b6b5a4ce9986d2"
MLX_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
ORIG_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
PAT=re.compile(r"^P17_NATIVE_GOLD_PREFIX_LOGITS rid=([a-f0-9]{32}) inputsha=([a-f0-9]{64}) plen=([0-9]+) pos=([0-9]+) next=([0-9]+) vocab=(102400) bytes=(409600) floatsha=([a-f0-9]{64})$")
LIMIT=14*1024**3
POLS={"A":A,"B":B}

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def htok(ids):return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def live():
    text=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=5)
    return {"stage_sha256":sha(STAGE),"route_sha256":sha(ROUTE),
           "stage_pids":sorted(int(line.split()[0]) for line in text.splitlines() if str(STAGE) in line)}
def preflight():
    if (ROOT.is_symlink() or ROOT.resolve(strict=True)!=ROOT or
        ROOT.name!="hcp-p17-unseen-multitoken-20261015" or ROOT.parent.name!=".tailnet-tmp"):
        raise RuntimeError("P17 wrong native run-owned scratch path")
    if shutil.disk_usage(ROOT).free<20_000_000_000:
        raise RuntimeError("P17 XOX disk physical reserve below 20GB")
    doc=ROOT/"input/P17_FROZEN_TOKENS.json"
    if sha(doc)!=FROZEN:raise RuntimeError("P17 inputs changed")
    original=live()
    if original!={"stage_sha256":ORIG_SHA,"route_sha256":ROUTE_SHA,"stage_pids":[35481,35493]}:
        raise RuntimeError("P17 original active serving changed")
    model=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors/model.safetensors.index.json")
    if sha(model)!=MODEL_SHA:raise RuntimeError("P17 original model checkpoint changed")
    src=ROOT/"native_src"
    if sha(src/"qwen_infer.c")!=C_SHA or sha(src/"mlx_moe.cpp")!=MLX_SHA:
        raise RuntimeError("P17 wrong pinned native C/MLX source")
    binary=ROOT/"native_build/qwen_infer_gpu"
    if not binary.is_file():raise RuntimeError("P17 native binary not built")
    bank=json.loads(doc.read_text())
    if bank["question_count"]!=12 or bank["gold_token_prefix_count"]!=31:
        raise RuntimeError("P17 missing one of 12 original questions")
    return bank,original,sha(binary)
def save(path,data):
    with pathlib.Path(path).open("x") as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def run(phase):
    frozen,before,exe=preflight()
    if phase not in ("prefix","greedy"):raise ValueError("Unsupported P17 phase")
    root=ROOT/"native"/("capture-worker" if phase=="prefix" else "greedy-worker")
    if root.exists():raise RuntimeError("P17 refusing to reuse existing native worker state")
    old=(ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO)
    ps.PERSISTENT_BINARY=ROOT/"native_build/qwen_infer_gpu"
    ps.EXPECTED_PERSISTENT_BINARY_SHA=exe
    ps.REQUEST_TIMEOUT_SECONDS=max(180,int(ps.REQUEST_TIMEOUT_SECONDS))
    base.REPO=ROOT/"native_src"
    w=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root)
    original_env=w._env
    def env():
        val=original_env()
        val.update({"QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"1",
                    "BEGLIN_HCP_P9_PREPARED_CACHE":"1",
                    "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"})
        if phase=="prefix":
            val["BEGLIN_P17_LOGITS_DIR"]=str(ROOT/"capture")
        else:
            val.pop("BEGLIN_P17_LOGITS_DIR",None)
        return val
    w._env=env
    rows=[];last_policy="A";old_epoch=None
    try:
        st=w.start();ack=grc.read_runtime_ack(w.ack_path)
        pid=int(st["pid"])
        if (not st.get("alive") or st["rss_bytes"]>LIMIT or
            ack["active_policy_hash"]!=pc.policy_hash(A)):
            raise RuntimeError("P17 startup policy/RSS invalid")
        old_epoch=int(ack["weight_epoch"])
        for idx,item in enumerate(frozen["items"]):
            gold=item["gold_answer_ids"]
            if phase=="prefix":
                tasks=[(step,stepdoc["gold_next_token_id"],
                        item["prompt_ids"]+gold[:step],stepdoc["prefix_token_sha256"])
                       for step,stepdoc in enumerate(item["forced_prefixes"])]
            else:
                tasks=[(0,None,item["prompt_ids"],item["prompt_token_sha256"])]
            for step,target,ids,prefix_hash in tasks:
                # Paired A/B order alternates by question, so thermal/order drift is not perfectly confounded.
                order=("A","B") if idx%2==0 else ("B","A")
                for policy in order:
                    start=time.perf_counter_ns()
                    before_log_size=len(w.log_path.read_text(errors="replace"))
                    previous_files={p.name for p in (ROOT/"capture").iterdir()} if phase=="prefix" else set()
                    result=w.submit_with_precision_policy([(ids,1 if phase=="prefix" else 8)],
                         target_policy=POLS[policy],
                         admission_id=f"p17-{phase}-{idx:02d}-{step}-{policy}")
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    responses=result.get("responses")
                    native_ack=grc.read_runtime_ack(w.ack_path)
                    health=w.health()
                    transition=result.get("precision_epoch") or {}
                    if (result.get("finite_logits") is not True or not isinstance(responses,list)
                        or len(responses)!=1 or not responses[0] or
                        len(responses[0])>(1 if phase=="prefix" else 8) or
                        not health.get("alive") or int(health["pid"])!=pid or
                        health["rss_bytes"]>LIMIT or
                        native_ack["active_policy_hash"]!=pc.policy_hash(POLS[policy])):
                        raise RuntimeError("P17 genuine native output/ACK/PID/RSS invalid")
                    is_switch=last_policy!=policy
                    if bool(transition.get("transitioned"))!=is_switch:
                        raise RuntimeError("P17 native transition history contradicted actual policy order")
                    if is_switch and (native_ack["status"]!="REBIND_SET_APPLIED" or
                        native_ack["txn_id"]!=transition.get("txn_id") or
                        len(transition.get("changed_targets",[]))!=2 or
                        int(transition["after_epoch"])!=int(transition["before_epoch"])+1):
                        raise RuntimeError("P17 2-target REBIND_SET actual terminal ACK invalid")
                    if int(native_ack["weight_epoch"])!=old_epoch+int(is_switch):
                        raise RuntimeError("P17 native epoch not monotonic by transaction")
                    old_epoch=int(native_ack["weight_epoch"])
                    last_policy=policy
                    row={"question_id":item["id"],"domain":item["domain"],
                        "step":step,"phase":phase,"policy":policy,
                        "input_sha256":htok(ids),"source_frozen_prefix_sha256":prefix_hash,
                        "gold_next_token_id":target,
                        "generated_token_ids":responses[0],
                        "generated_token_sha256":htok(responses[0]),
                        "terminal_ack_status":native_ack["status"],
                        "terminal_ack_digest":native_ack["ack_sha256"],
                        "precision_epoch":old_epoch,"worker_pid":pid,
                        "changed_policy":is_switch,"rss_bytes":int(health["rss_bytes"]),
                        "observed_wall_ms_not_speed_certified":elapsed}
                    if htok(ids)!=prefix_hash:raise RuntimeError("P17 prefix source preimage changed")
                    if phase=="prefix":
                        trace=[s.strip() for s in w.log_path.read_text(errors="replace")[before_log_size:].splitlines()
                               if "P17_NATIVE_GOLD_PREFIX_LOGITS" in s]
                        if len(trace)!=1:raise RuntimeError("P17 required one actual C logit capture per prefix")
                        m=PAT.fullmatch(trace[0])
                        if m is None:raise RuntimeError("P17 native capture metadata source malformed")
                        rid,tok,plen,pos,first,vocab,bytes_,digest=m.groups()
                        if (tok!=prefix_hash or int(plen)!=len(ids) or
                            int(pos)!=len(ids)-1 or int(first)!=responses[0][0]):
                            raise RuntimeError("P17 C vs Python exact prefix/argmax not matching")
                        path=ROOT/"capture"/(rid+".f32")
                        files={p.name for p in (ROOT/"capture").iterdir()}
                        if ((files-previous_files)!={path.name} or path.stat().st_size!=409600 or
                            path.is_symlink() or sha(path)!=digest):
                            raise RuntimeError("P17 original XOX C source emitted missing/unsealed file")
                        logits=np.fromfile(path,dtype="<f4")
                        if (logits.shape!=(102400,) or not np.isfinite(logits).all() or
                            int(np.argmax(logits))!=int(responses[0][0])):
                            raise RuntimeError("P17 C full logits not genuine greedy source")
                        row.update({"raw_float_logits_filename":path.name,"native_full_logits_sha256":digest,
                          "native_vocab":102400,"native_NLL_gold_computed_later":True})
                    else:
                        row.update({"strict_gold_token_prefix_match":
                            len(responses[0])>=len(gold) and responses[0][:len(gold)]==gold,
                            "scoring_gold_token_count":len(gold)})
                    rows.append(row)
            print("P17_REAL_NATIVE_DONE",json.dumps({"phase":phase,"questions_completed":idx+1,
                 "total":12,"requests":len(rows),"worker_rss_GiB":round(health["rss_bytes"]/1024**3,3)},sort_keys=True),flush=True)
        expected=62 if phase=="prefix" else 24
        if len(rows)!=expected:raise RuntimeError("P17 real policy requests incomplete")
        after=live()
        if after!=before:raise RuntimeError("P17 mutated original production live stage")
        report={"schema":"beglin-hcp-p17-real-native-BF16-gold-prefix-full-logits-v1",
            "status":"P17_REAL_NATIVE_ALL_REQUESTS_COMPLETE",
            "phase":phase,"source_frozen_sha256":FROZEN,
            "native_C_sha256":C_SHA,"native_MLX_source_SHA256":MLX_SHA,
            "native_binary_sha256":exe,"native_requests":len(rows),
            "source_input_cases":12,"source_gold_tokens":31,
            "single_real_native_worker_pid":pid,"policy_A_and_B_applied":True,
            "all_real_native_policy_ACKs_verified":True,"original_live_before":before,
            "original_live_after":after,"records":rows,
            "quality_noninferiority_certified":False,
            "production_touched":False,"automatic_live_promotion":False}
        dest=ROOT/"evidence"/("P17_NATIVE_FORCED_PREFIX_FULL_LOGITS.json" if phase=="prefix" else
                             "P17_NATIVE_GREEDY_EIGHT_TOKEN_RESULTS.json")
        save(dest,report)
        print("P17_REAL_NATIVE_PHASE_COMPLETE",json.dumps({"phase":phase,"requests":expected,
               "sha256":sha(dest),"worker_pid":pid,"production_unchanged":True},sort_keys=True),flush=True)
    finally:
        try:w.stop(force=True)
        finally:
            ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO=old
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("prefix","greedy"),required=True)
    a=p.parse_args();run(a.phase)
