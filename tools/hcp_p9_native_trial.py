#!/usr/bin/env python3
"""P9 run-owned native DeepSeek on/off MLX repack-cache crossover.

Exact P6 prompt/model/source/output identities, true REBIND_SET ACK/PID,
one GPU slot, bounded RSS, no production policy change or live activation.
Precommitted alternating block order; no raw token/logit export.
"""
from __future__ import annotations
import argparse,contextlib,hashlib,json,math,os,pathlib,random,shutil
import statistics,subprocess,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import precision_context as pc
import gpu_runtime_control as grc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps
from hcp_p4_long_prompt_registry import prepare
from hcp_p4_native_policies_ab import A,B,sha

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p9-mlxcache-20261014-1350")
P6=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104")
STAGE=pathlib.Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
TOK=pathlib.Path("/Users/xox/vdsp_local_data/deepseek_v2lite_bf16_safetensors")
SOURCE_COMMIT="a46154d8d3598ef1c7b0d51cd1d4447090c89ac4"
C128_SHA="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
MLX_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
P4_SHA="c3b54b1ebe41d9a85ed94ddd0da87cde0de86ecf0af56d2e621733e439944f53"
P6_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
MAX_RSS=14*1024**3
SEED=20261014
POLICIES={"A":A,"B":B}
def sseq(a):return hashlib.sha256(",".join(map(str,a)).encode()).hexdigest()
def livepids():
 txt=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True)
 return sorted(int(line.split()[0]) for line in txt.splitlines() if str(STAGE) in line)
def liveproof():
 return {"stage":sha(STAGE),"route":sha(ROUTE),"pids":livepids()}
def save(path,doc):
 path=pathlib.Path(path)
 with path.open("x") as f:json.dump(doc,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def percentile(v,p):
 a=sorted(float(q) for q in v)
 pos=(len(a)-1)*p;i=math.floor(pos);j=math.ceil(pos)
 return a[i] if i==j else float(a[i]+(a[j]-a[i])*(pos-i))
def stat(v):
 v=list(v)
 return {"count":len(v),"mean":statistics.mean(v),"p50":percentile(v,.5),
         "p95":percentile(v,.95),"max":max(v)}
def preflight():
 assert ROOT.is_dir() and ROOT.resolve(strict=True)==ROOT and not ROOT.is_symlink()
 assert sha(STAGE)==STAGE_SHA and sha(ROUTE)==ROUTE_SHA
 assert sha(TOK/"model.safetensors.index.json")==MODEL_SHA
 assert sha(ROOT/"src/qwen_infer.c")==C128_SHA
 assert sha(ROOT/"src/mlx_moe.cpp")==MLX_SHA
 assert ROOT.joinpath("build-plain/qwen_infer_gpu").is_file()
 assert ROOT.joinpath("preflight.json").is_file()
 pre=json.loads((ROOT/"preflight.json").read_text())
 assert pre["base_commit"]==SOURCE_COMMIT and pre["p9_cache_mlx_sha"]==MLX_SHA
 assert subprocess.check_output(["/usr/bin/git","-C",str(ROOT/"src"),"rev-parse","HEAD"],text=True).strip()==SOURCE_COMMIT
 assert int(subprocess.check_output(["/usr/sbin/sysctl","-n","hw.memsize"]))>=40*1024**3
 assert shutil.disk_usage(ROOT).free > 20*1000**3+300*1000**2
 feat=P6/"inputs/p4_derived_long_features.json"
 quality=P6/"repository/experiments/hcp_p6/evidence/quality.json"
 assert sha(feat)==P4_SHA and sha(quality)==P6_SHA
 q=json.loads(quality.read_text())
 expected={rec["prompt_id"]:{x:rec[x]["sequence_sha256"] for x in ("A","B")} for rec in q["records"]}
 corpus=prepare(P6/"inputs/bank",TOK)
 prompts=[corpus["prompts"][3*i+i%3] for i in range(12)]
 freeze={x["prompt_id"]:x for x in json.loads(feat.read_text())["runs"]}
 assert len(prompts)==len(expected)==12 and len({x["domain"] for x in prompts})==12
 for row in prompts:
  obj=freeze[row["id"]]
  assert obj["token_sha256"]==row["token_sha256"] and obj["prompt_length"]==row["tokens"]
  assert 104<=row["tokens"]<=112 and row["tokens"]+4<128
 identity=liveproof()
 assert identity=={"stage":STAGE_SHA,"route":ROUTE_SHA,"pids":[35481,35493]}
 return prompts,expected,identity,sha(ROOT/"build-plain/qwen_infer_gpu")
@contextlib.contextmanager
def native(label,enabled,binary_hash):
 workerroot=ROOT/"workers"/label
 if workerroot.exists():raise ValueError("worker root reused")
 oldbin=ps.PERSISTENT_BINARY;oldsha=ps.EXPECTED_PERSISTENT_BINARY_SHA
 oldrepo=base.REPO;oldtimeout=ps.REQUEST_TIMEOUT_SECONDS
 ps.PERSISTENT_BINARY=ROOT/"build-plain/qwen_infer_gpu"
 ps.EXPECTED_PERSISTENT_BINARY_SHA=binary_hash
 ps.REQUEST_TIMEOUT_SECONDS=max(int(ps.REQUEST_TIMEOUT_SECONDS),180)
 base.REPO=ROOT/"src"
 w=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=workerroot)
 native_env=w._env
 def env():
  e=native_env()
  e.update({"QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"1",
            "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"})
  if enabled:e["BEGLIN_HCP_P9_PREPARED_CACHE"]="1"
  else:e.pop("BEGLIN_HCP_P9_PREPARED_CACHE",None)
  return e
 w._env=env
 try:
  health=w.start()
  ack=grc.read_runtime_ack(w.ack_path)
  if (not health.get("alive") or ack["active_policy_hash"]!=pc.policy_hash(A)
      or int(health["rss_bytes"])>=MAX_RSS):
   raise ValueError("P9 native startup invalid")
  w.p9_pid=int(health["pid"])
  yield w
 finally:
  try:w.stop(force=True)
  finally:
   ps.PERSISTENT_BINARY=oldbin;ps.EXPECTED_PERSISTENT_BINARY_SHA=oldsha
   ps.REQUEST_TIMEOUT_SECONDS=oldtimeout;base.REPO=oldrepo
def request(w,row,pol,admission):
 begin=time.perf_counter_ns()
 result=w.submit_with_precision_policy([(row["token_ids"],4)],
     target_policy=POLICIES[pol],admission_id=admission)
 wall=(time.perf_counter_ns()-begin)/1e6
 seq=result.get("responses")
 if (not isinstance(seq,list) or len(seq)!=1 or not (1<=len(seq[0])<=4)
     or result.get("finite_logits") is not True or result.get("inference_passes",1)!=1):
  raise ValueError("P9 actual inference invalid")
 ack=grc.read_runtime_ack(w.ack_path)
 health=w.health()
 if (not health.get("alive") or int(health["pid"])!=w.p9_pid or
     int(health["rss_bytes"])>=MAX_RSS or
     ack["active_policy_hash"]!=pc.policy_hash(POLICIES[pol])):
  raise ValueError("P9 runtime ACK hash/PID/RSS mismatch")
 epoch=result.get("precision_epoch") or {}
 moved=bool(epoch.get("transitioned"))
 if moved and (ack.get("status")!="REBIND_SET_APPLIED" or
        ack.get("txn_id")!=epoch.get("txn_id") or
        len(epoch.get("changed_targets",[]))!=2 or
        int(epoch["after_epoch"])!=int(epoch["before_epoch"])+1):
  raise ValueError("P9 true two-target REBIND_SET terminal ACK failure")
 values={"admission_ms":wall,"roundtrip_ms":float(result["roundtrip_ms"]),
         "engine_ms":float(result["engine_wall_ms"]),
         "transition_ms":float((epoch.get("transition_cost") or {}).get("transition_wall_ms",0))}
 if any(not math.isfinite(v) or v<0 for v in values.values()) or min(wall,values["engine_ms"],values["roundtrip_ms"])<=0:
  raise ValueError("invalid P9 timestamps")
 return {"policy":pol,"seq_sha256":sseq(seq[0]),"generated_count":len(seq[0]),
         "switched":moved,"rss_bytes":int(health["rss_bytes"]),
         "native_ack_status":ack.get("status"),**values}
def cache_events(log):
 text=pathlib.Path(log).read_text(errors="replace") if pathlib.Path(log).is_file() else ""
 return {tag:text.count("P9_PREPARED_CACHE_"+tag) for tag in
         ("STORE name=","HIT name=","SOURCE_MISMATCH","BUDGET_SKIP","STORE_SKIPPED")}
def block_mode(phase,block,enabled,prompts,expected,binary_hash,limit):
 label=f"p9-{phase}-b{block}-"+("on" if enabled else "off")
 result=[]
 with native(label,enabled,binary_hash) as w:
  # Prime all four L3/L26 n5/n6 bind choices, source SHA verified each time.
  warmup=["A","B","A","B"]
  for j,pol in enumerate(warmup):
   row=prompts[0]
   o=request(w,row,pol,f"{label}-warm-{j}")
   if o["seq_sha256"]!=expected[row["id"]][pol]:
    raise ValueError("cache miss/warmup changed generation against P6")
  # ABBA/BAAB matched for both on/off, rotating order with block number.
  for i,row in enumerate(prompts[:limit]):
   order=("A","B","B","A") if (block+i)%2==0 else ("B","A","A","B")
   for ordinal,pol in enumerate(order):
    o=request(w,row,pol,f"{label}-{i}-{ordinal}-{pol}")
    if o["seq_sha256"]!=expected[row["id"]][pol]:
     raise ValueError("P9 cache enabled changed generated token sequence")
    result.append({"block":block,"domain":row["domain"],"prompt_id":row["id"],
                   "position":ordinal,"policy_order":"".join(order),
                   "cache_enabled":enabled,"input_tokens":row["tokens"],
                   "verified_original_P6_sequence":True,**o})
 events=cache_events(ROOT/"workers"/label/"worker.log")
 if enabled and (events["STORE name="]<4 or events["HIT name="]<4):
  raise ValueError("P9 prepared cache never stored/reused the four permitted tensors")
 if not enabled and (events["STORE name="] or events["HIT name="]):
  raise ValueError("P9 cache disabled but used")
 if any(events[k] for k in ("SOURCE_MISMATCH","BUDGET_SKIP","STORE_SKIPPED")):
  raise ValueError("P9 cache source or budget integrity failure")
 return result,events
def paired(records,phase):
 by={}
 for row in records:
  k=(row["block"],row["prompt_id"],row["position"],row["policy"])
  by.setdefault(k,{})["on" if row["cache_enabled"] else "off"]=row
 if len(by)*2!=len(records) or any(set(v)!={"on","off"} for v in by.values()):
  raise ValueError("cache ON/OFF paired identity incomplete")
 pairs=[]
 for k,v in sorted(by.items()):
  a,b=v["off"],v["on"]
  if (a["policy_order"]!=b["policy_order"] or a["seq_sha256"]!=b["seq_sha256"] or
      a["switched"]!=b["switched"]):
   raise ValueError("P9 ON/OFF incomparable native outputs/policies")
  pairs.append({"block":k[0],"domain":a["domain"],"position":k[2],"policy":k[3],
          "switched":a["switched"],
          "off_admission_ms":a["admission_ms"],"on_admission_ms":b["admission_ms"],
          "delta_on_minus_off_ms":b["admission_ms"]-a["admission_ms"]})
 return pairs
def summary(records,pairs,blocks):
 d={}
 for mode in ("off","on"):
  for switched in (False,True):
   a=[r for r in records if r["cache_enabled"]==(mode=="on") and r["switched"]==switched]
   if not a:raise ValueError("P9 missing steady/switch cells")
   d[mode+"_"+("switch" if switched else "steady")]={"admission_ms":stat(x["admission_ms"] for x in a),
       "engine_ms":stat(x["engine_ms"] for x in a),"rss_bytes":stat(x["rss_bytes"] for x in a)}
 diff=[x["delta_on_minus_off_ms"] for x in pairs if x["switched"]]
 rng=random.Random(SEED)
 bdiff={b:[x["delta_on_minus_off_ms"] for x in pairs if x["block"]==b and x["switched"]] for b in range(blocks)}
 if any(not a for a in bdiff.values()):raise ValueError("P9 empty switching block")
 means=[statistics.mean(x) for x in bdiff.values()]
 draw=10000
 sims=sorted(statistics.mean([means[rng.randrange(blocks)] for _ in range(blocks)]) for _ in range(draw))
 d["switch_paired_delta_on_minus_off_ms"]={"count":len(diff),"mean":statistics.mean(diff),
        "temporal_block_bootstrap_ci95_ms":[sims[int(draw*.025)],sims[int(draw*.975)]],
        "seed":SEED,"draws":draw,"blocks":blocks}
 return d
def main():
 arg=argparse.ArgumentParser()
 arg.add_argument("--phase",choices=("smoke","full"),required=True)
 args=arg.parse_args()
 prompts,expected,before,binsha=preflight()
 blocks,limit=(2,2) if args.phase=="smoke" else (4,12)
 records=[];events=[];started=time.monotonic()
 try:
  for block in range(blocks):
   modes=(False,True) if block%2==0 else (True,False)
   for enabled in modes:
    cells,counts=block_mode(args.phase,block,enabled,prompts,expected,binsha,limit)
    records.extend(cells)
    events.append({"block":block,"enabled":enabled,**counts})
    print("P9_REAL_NATIVE_BLOCK",json.dumps({"phase":args.phase,"block":block,
      "cache_enabled":enabled,"complete_admissions":len(cells),
      "cache_stores":counts["STORE name="],"cache_hits":counts["HIT name="],
      "elapsed_s":round(time.monotonic()-started,1)}),flush=True)
 finally:
  # Scratch worker logs/requests contain raw token IDs; erase only this run's
  # exact-named P9 worker roots even on failed native study.
  for p in (ROOT/"workers").glob("p9-"+args.phase+"-*"):
   if p.is_symlink():raise ValueError("unsafe symlink worker")
   shutil.rmtree(p)
 after=liveproof()
 if after!=before:raise ValueError("P9 mutated live binary/route/PID")
 pairs=paired(records,args.phase)
 d=summary(records,pairs,blocks)
 report={"schema":"beglin-hcp-p9-real-native-optin-packed-cache-v1",
  "status":"REAL_XOX_RESEARCH_PAIRED_CACHE_ON_OFF_ONLY",
  "phase":args.phase,"blocks":blocks,"prompt_domains":limit,
  "full_admissions":len(records),"paired_admissions":len(pairs),
  "model": "DeepSeek-V2-Lite","generated_token_budget":4,
  "model_index_sha256":MODEL_SHA,"original_P4_SHA256":P4_SHA,
  "P6_quality_SHA256":P6_SHA,"source_commit":SOURCE_COMMIT,
  "C128_source_SHA256":C128_SHA,"P9_MLX_source_SHA256":MLX_SHA,
  "one_built_binary_both_cache_conditions_SHA256":binsha,
  "warmup_admissions_excluded_per_worker":4,"native_events":events,
  "all_on_off_sequence_SHA_equal":True,"all_P6_original_A_B_sequence_SHA_verified":True,
  "all_real_ACKs_not_mocked":True,"summary":d,"records":records,
  "quality_authority_for_B_granted":False,"production_touched":False,
  "automatic_live_promotion":False,
  "speedup_proven_for_production":False,"full_benchmark_not_new_holdout":True,
  "live_preimage":before,"live_postimage":after}
 dest=ROOT/"results"/("p9_"+args.phase+".json")
 save(dest,report)
 print("P9_NATIVE_RUN_COMPLETE",json.dumps({"phase":args.phase,
   "actual_requests":len(records),"paired":len(pairs),
   "on_switch_ms":d["on_switch"]["admission_ms"]["mean"],
   "off_switch_ms":d["off_switch"]["admission_ms"]["mean"],
   "switch_delta_ci95":d["switch_paired_delta_on_minus_off_ms"]["temporal_block_bootstrap_ci95_ms"],
   "unchanged":True},sort_keys=True),flush=True)

if __name__=="__main__":main()
