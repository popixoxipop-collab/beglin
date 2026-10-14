#!/usr/bin/env python3
"""P13 XOX real native MLX P9 cache SHA vs original BF16 C 5/6-bit packing.

ONE isolated native GPU request worker at any moment, dual OFF/ON study in same
compiled sha-instrumented binary. The trace is observational only. No raw model
weights, token IDs, or full logits are exported; no speedup claims.
"""
from __future__ import annotations
import contextlib,hashlib,json,math,os,pathlib,re,shutil,statistics,subprocess,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import precision_context as pc
import gpu_runtime_control as grc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,sha

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p13-native-byte-parity-20261014-2148")
P6=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104")
MODEL=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
HEAD="5418310d178f42d68dde09fa398414ab91fb38d4"
REF_SHA="50d222534ab1324b990a2c4d83cc227e0ca2ba53c2037b500d688f742caeb078"
C_SHA="26e542118302a5a82f147ef1678ec0f6e4516b467d0a4166cc1bc7dce29fa8c6"
MLX_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
P6_QUALITY="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
TRACE=re.compile(r"^P13_NATIVE_PACKED_SHA role=(shared_up|shared_down) layer=(3|26) bits=(5|6) phase=(store|hit) packed=([0-9a-f]{64}) scales=([0-9a-f]{64}) packed_bytes=([0-9]+) scales_bytes=([0-9]+)$")
POLICIES={"A":A,"B":B}
MAX_RSS=14*(1024**3)

def seqsha(seq):return hashlib.sha256(",".join(map(str,seq)).encode()).hexdigest()
def live():
    txt=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True)
    return {"stage_sha256":sha(STAGE),"route_sha256":sha(ROUTE),
        "source_worker_pids":sorted(int(l.split()[0]) for l in txt.splitlines() if str(STAGE) in l)}
def assert_live(orig):
    if live()!=orig:raise RuntimeError("P13 original production worker identity changed")
def preflight():
    assert ROOT.is_dir() and not ROOT.is_symlink() and ROOT.resolve(strict=True)==ROOT
    if shutil.disk_usage(ROOT).free <20_300_000_000:
        raise RuntimeError("P13 SSD reserve below 20GB+margin")
    p=ROOT/"src"
    head=subprocess.check_output(["/usr/bin/git","-C",str(p),"rev-parse","HEAD"],text=True).strip()
    if head!=HEAD or sha(p/"qwen_infer.c")!=C_SHA or sha(p/"mlx_moe.cpp")!=MLX_SHA:
        raise RuntimeError("P13 native patched source does not match frozen source")
    if sha(p/"experiments/hcp_p13/P13_REFERENCE_CODES.json")!=REF_SHA:
        raise RuntimeError("P13 original BF16 packed SHA reference changed")
    if sha(MODEL/"model.safetensors.index.json")!=MODEL_SHA:
        raise RuntimeError("P13 model index differs")
    if sha(P6/"repository/experiments/hcp_p6/evidence/quality.json")!=P6_QUALITY:
        raise RuntimeError("P13 original P6 generated token SHA source changed")
    orig=live()
    if orig!={"stage_sha256":STAGE_SHA,"route_sha256":ROUTE_SHA,
              "source_worker_pids":[35481,35493]}:
        raise RuntimeError("P13 original live route/stage/PID differ")
    exe=ROOT/"build/qwen_infer_gpu"
    if not exe.is_file() or exe.is_symlink():raise RuntimeError("P13 run-owned binary not built")
    ref=json.loads((p/"experiments/hcp_p13/P13_REFERENCE_CODES.json").read_text())
    frozen={}
    for row in ref["replacements"]:
        key=("shared_up" if row["target"]=="shared_up_L3" else "shared_down",
             3 if row["target"]=="shared_up_L3" else 26,row["bits"])
        frozen[key]=row
    if len(frozen)!=4:raise RuntimeError("P13 4 frozen original BF16 references missing")
    corpus=prepare(P6/"inputs/bank",MODEL)["prompts"]
    selected=[corpus[3*i+i%3] for i in range(6)]
    quality=json.loads((P6/"repository/experiments/hcp_p6/evidence/quality.json").read_text())
    truth={q["prompt_id"]:q for q in quality["records"]}
    expected={row["id"]:{pol:truth[row["id"]][pol]["sequence_sha256"] for pol in ("A","B")} for row in selected}
    return orig,sha(exe),selected,expected,frozen

@contextlib.contextmanager
def worker(name,cache_enabled,exe_sha):
    wroot=ROOT/"workers"/name
    if wroot.exists():raise ValueError("P13 duplicate owned worker")
    old=(ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,ps.REQUEST_TIMEOUT_SECONDS,base.REPO)
    ps.PERSISTENT_BINARY=ROOT/"build/qwen_infer_gpu"
    ps.EXPECTED_PERSISTENT_BINARY_SHA=exe_sha
    ps.REQUEST_TIMEOUT_SECONDS=max(180,int(ps.REQUEST_TIMEOUT_SECONDS))
    base.REPO=ROOT/"src"
    w=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=wroot)
    envfn=w._env
    def env():
        e=envfn()
        e["QWEN_MOE_CB_SLOTS"]="1";e["QWEN_MOE_CB_REQS"]="1"
        e["OMP_NUM_THREADS"]="2";e["OPENBLAS_NUM_THREADS"]="2"
        e["BEGLIN_HCP_P9_PREPARED_CACHE"]="1"
        if cache_enabled:e["BEGLIN_P13_PACKED_SHA"]="1"
        else:e.pop("BEGLIN_P13_PACKED_SHA",None)
        return e
    w._env=env
    try:
        s=w.start()
        ack=grc.read_runtime_ack(w.ack_path)
        if (not s.get("alive") or ack["active_policy_hash"]!=pc.policy_hash(A)
            or int(s["rss_bytes"])>=MAX_RSS):raise RuntimeError("P13 wrong initial policy or RSS")
        w.p13_pid=int(s["pid"])
        yield w,wroot
    finally:
        try:w.stop(force=True)
        finally:
            (ps.PERSISTENT_BINARY,ps.EXPECTED_PERSISTENT_BINARY_SHA,
             ps.REQUEST_TIMEOUT_SECONDS,base.REPO)=old

def run():
    production,exe_sha,prompts,expected,frozen=preflight()
    rows=[];traces=[]
    for enabled in (False,True):
        name="p13-sha-"+("on" if enabled else "off")
        with worker(name,enabled,exe_sha) as (w,wroot):
            for i,row in enumerate(prompts):
                for j,pol in enumerate(("B","A","B","A")):
                    started=time.perf_counter_ns()
                    response=w.submit_with_precision_policy([(row["token_ids"],4)],
                        target_policy=POLICIES[pol],admission_id=f"p13-{name}-{i}-{j}")
                    elapsed=(time.perf_counter_ns()-started)/1e6
                    sequences=response.get("responses")
                    ack=grc.read_runtime_ack(w.ack_path)
                    epoch=response.get("precision_epoch") or {}
                    health=w.health()
                    if (response.get("finite_logits") is not True or
                       not isinstance(sequences,list) or len(sequences)!=1 or
                       seqsha(sequences[0])!=expected[row["id"]][pol] or
                       not epoch.get("transitioned") or
                       ack.get("status")!="REBIND_SET_APPLIED" or
                       ack.get("txn_id")!=epoch.get("txn_id") or
                       ack["active_policy_hash"]!=pc.policy_hash(POLICIES[pol]) or
                       len(epoch.get("changed_targets",[]))!=2 or
                       int(epoch["after_epoch"])!=int(epoch["before_epoch"])+1 or
                       int(health["pid"])!=w.p13_pid or
                       not health.get("alive") or int(health["rss_bytes"])>=MAX_RSS):
                        raise RuntimeError("P13 native SHA trace changed actual generation or terminal ACK")
                    rows.append({"mode":"on" if enabled else "off","domain":row["domain"],
                      "prompt_id":row["id"],"policy":pol,"generated_sequence_sha256":expected[row["id"]][pol],
                      "native_ACK_verified":True,"latency_observation_only_ms":elapsed})
                print("P13_REAL_GENERATION_DOMAIN",json.dumps({"mode":name,"domains_done":i+1,
                      "total":6,"real_terminal_ACKs":(i+1)*4}),flush=True)
            log=wroot/"worker.log"
            if not log.is_file():raise RuntimeError("P13 native worker log missing")
            for line in log.read_text(errors="replace").splitlines():
                if "P13_NATIVE_HASH_CONTRACT_FAIL" in line:raise RuntimeError("P13 source digest contract violated")
                if "P13_NATIVE_PACKED_SHA" not in line:continue
                m=TRACE.fullmatch(line.strip())
                if m is None:raise RuntimeError("P13 malformed or unexpected native qNg64 SHA log")
                role,layer,n,phase,packed,scales,pbytes,sbytes=m.groups()
                key=(role,int(layer),int(n))
                if not enabled:raise RuntimeError("P13 SHA trace ran when env was OFF")
                if key not in frozen:raise RuntimeError("unexpected qNg64 cached tensor hash")
                gold=frozen[key]
                if (packed!=gold["packed_SHA256"] or scales!=gold["scales_SHA256"] or
                    int(pbytes)!=gold["packed_bytes"] or int(sbytes)!=gold["scales_bytes"]):
                    raise RuntimeError(f"P13 NATIVE_BF16_PACKED_CODE_MISMATCH {key}")
                traces.append({"role":role,"layer":int(layer),"bits":int(n),
                   "phase":phase,"packed_SHA256":packed,"scales_SHA256":scales,
                   "packed_bytes":int(pbytes),"scales_bytes":int(sbytes)})
        assert_live(production)
    seen={(r["role"],r["layer"],r["bits"]) for r in traces}
    if seen!=set(frozen):raise RuntimeError(f"P13 native cached qNg64 representation coverage {seen}")
    paired={}
    for r in rows:
        key=(r["prompt_id"],r["policy"])
        modes=paired.setdefault(key,[])
        modes.append(r["mode"])
    if len(paired)!=12 or any(sorted(v)!=["off","off","on","on"] for v in paired.values()):
        raise RuntimeError("P13 real on/off generated outputs pairing incomplete")
    report={
      "schema":"beglin-hcp-p13-real-native-original-BF16-packed-byte-parity-v1",
      "status":"FOUR_P9_REAL_MLX_QNG64_NATIVE_PACKED_AND_SCALES_SHA_EQUAL_P12_BF16_ORACLE",
      "source_head_prereg":HEAD,"four_frozen_reference_SHA256":REF_SHA,
      "native_traced_binary_SHA256":exe_sha,"P13_C_source_SHA256":C_SHA,"P9_MLX_source_SHA256":MLX_SHA,
      "native_4_unique_tensor_bitwidth_pairs_exact":True,
      "native_unique_pairs":len(seen),"actual_native_hash_lines":len(traces),
      "native_hash_log_events":traces,
      "real_native_generation_admissions":len(rows),
      "source_P6_generated_sequences_SHA_matched":True,
      "real_native_terminal_txns_verified":len(rows),
      "cache_tracing_OFF_generations":sum(r["mode"]=="off" for r in rows),
      "cache_tracing_ON_generations":sum(r["mode"]=="on" for r in rows),
      "no_speed_comparison_with_tracing":True,
      "bf16_full_model_logits_not_measured":True,
      "source_input_token_ids_not_exported":True,
      "production_before":production,"production_after":live(),
      "production_touched":False,"automatic_live_promotion":False}
    if report["production_before"]!=report["production_after"]:raise RuntimeError("P13 production preimage drifted")
    path=ROOT/"results/real_native_pack_sha.json"
    with path.open("x") as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    print("P13_NATIVE_BF16_TO_RUNTIME_SHA_PASS",json.dumps({"distinct_four":len(seen),
       "hash_log_events":len(traces),"generated":len(rows),
       "real_ack":len(rows),"source_reference_SHA":REF_SHA,
       "result_sha256":sha(path),"production_touched":False},sort_keys=True))
if __name__=="__main__":run()
