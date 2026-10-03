#!/usr/bin/env python3
"""XOX scratch benchmark for P4 measured precision costs."""
from __future__ import annotations
import argparse, hashlib, json, math, shutil, statistics, subprocess
from pathlib import Path

import precision_context as pc
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps

DEFAULT_ROOT=Path("/Users/xox/vdsp_shadow_runs/precision_e2e_cost")
STARTUP=[
    {"role":"shared_up_proj","layer":3,"n":6},
    {"role":"shared_down_proj","layer":26,"n":5},
]
ALT_L3=[
    {"role":"shared_up_proj","layer":3,"n":5},
    {"role":"shared_down_proj","layer":26,"n":5},
]
ALT_L26=[
    {"role":"shared_up_proj","layer":3,"n":6},
    {"role":"shared_down_proj","layer":26,"n":6},
]
COMBINED=[
    {"role":"shared_up_proj","layer":3,"n":5},
    {"role":"shared_down_proj","layer":26,"n":6},
]

def pct(values,p=.5):
    values=sorted(float(v) for v in values)
    if not values:return None
    k=(len(values)-1)*p; lo=math.floor(k); hi=math.ceil(k)
    return values[lo] if lo==hi else values[lo]+(values[hi]-values[lo])*(k-lo)

def source_head():
    return subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def check_root(path):
    root=Path(path).expanduser().resolve()
    allowed=DEFAULT_ROOT.resolve()
    if root!=allowed and allowed not in root.parents:
        raise SystemExit(f"root must stay under {allowed}")
    return root
def summarize_runs(rows):
    return {
        "count":len(rows),
        "p50_engine_ms":pct([r["engine_wall_ms"] for r in rows]),
        "p95_engine_ms":pct([r["engine_wall_ms"] for r in rows],.95),
        "p50_roundtrip_ms":pct([r["roundtrip_ms"] for r in rows]),
        "p95_roundtrip_ms":pct([r["roundtrip_ms"] for r in rows],.95),
    }

def transition_summary(rows, cache_state):
    costs=[r["precision_epoch"]["transition_cost"] for r in rows]
    return {
        "from_n":None,
        "cache_state":cache_state,
        "samples":len(rows),
        "p50_transition_ms":pct([r["transition_wall_ms"] for r in costs]),
        "p95_transition_ms":pct([r["transition_wall_ms"] for r in costs],.95),
        "cache_hits":int(round(statistics.mean([r["cache_hits"] for r in costs]))),
        "cache_misses":int(round(statistics.mean([r["cache_misses"] for r in costs]))),
        "cache_bytes_added":int(round(statistics.mean([r["cache_bytes_added"] for r in costs]))),
        "resident_cache_bytes_p50":int(pct([r["resident_cache_bytes"] for r in costs])),
    }

def run_noop(worker,prompt,policy,count,prefix):
    rows=[]
    for i in range(count):
        got=worker.submit_with_precision_policy(
            [(prompt,10)],target_policy=policy,admission_id=f"{prefix}-{i}")
        if not got.get("finite_logits"):raise RuntimeError("non-finite logits")
        rows.append(got)
    return rows

def profile_target(binary,root,role,layer,base_n,alt_n,alt_policy,steady_n,warm_cycles):
    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root)
    prompt=base._read_first_certified_prompt()
    try:
        worker.start(); pid=worker.health()["pid"]
        base_rows=run_noop(worker,prompt,STARTUP,steady_n,"base-steady")
        cold=worker.submit_with_precision_policy(
            [(prompt,10)],target_policy=alt_policy,admission_id="alt-cold")
        alt_rows=run_noop(worker,prompt,alt_policy,steady_n,"alt-steady")
        restore_rows=[]; warm_alt=[]
        for i in range(warm_cycles):
            restore_rows.append(worker.submit_with_precision_policy(
                [(prompt,10)],target_policy=STARTUP,admission_id=f"restore-{i}"))
            warm_alt.append(worker.submit_with_precision_policy(
                [(prompt,10)],target_policy=alt_policy,admission_id=f"alt-warm-{i}"))
        health=worker.health()
        if health["pid"]!=pid:raise RuntimeError("worker PID changed")
        cold_sum=transition_summary([cold],"cold"); cold_sum["from_n"]=base_n
        warm_sum=transition_summary(warm_alt,"warm"); warm_sum["from_n"]=base_n
        restore_sum=transition_summary(restore_rows,"warm"); restore_sum["from_n"]=alt_n
        base_steady=summarize_runs(base_rows); alt_steady=summarize_runs(alt_rows)
        return [
            {
                "role":role,"layer":layer,"n":base_n,
                "steady_engine_p50_ms":base_steady["p50_engine_ms"],
                "steady_roundtrip_p50_ms":base_steady["p50_roundtrip_ms"],
                "rss_bytes":int(health["rss_bytes"]),
                "expected_inference_passes":1,
                "transitions":[restore_sum],
                "context_policy_hash":pc.policy_hash(STARTUP),
            },
            {
                "role":role,"layer":layer,"n":alt_n,
                "steady_engine_p50_ms":alt_steady["p50_engine_ms"],
                "steady_roundtrip_p50_ms":alt_steady["p50_roundtrip_ms"],
                "rss_bytes":int(health["rss_bytes"]),
                "expected_inference_passes":1,
                "transitions":[cold_sum,warm_sum],
                "context_policy_hash":pc.policy_hash(alt_policy),
            },
        ], {
            "role":role,"layer":layer,"pid":pid,
            "base_steady":base_steady,"alt_steady":alt_steady,
            "cold_transition":cold["precision_epoch"]["transition_cost"],
            "warm_alt_transitions":[r["precision_epoch"]["transition_cost"] for r in warm_alt],
            "warm_restore_transitions":[r["precision_epoch"]["transition_cost"] for r in restore_rows],
        }
    finally:
        worker.stop(force=True)
def profile_combined(binary,root,steady_n,warm_cycles):
    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root)
    prompt=base._read_first_certified_prompt()
    try:
        worker.start(); pid=worker.health()["pid"]
        base_rows=run_noop(worker,prompt,STARTUP,steady_n,"combo-base")
        cold=worker.submit_with_precision_policy(
            [(prompt,10)],target_policy=COMBINED,admission_id="combo-cold")
        alt_rows=run_noop(worker,prompt,COMBINED,steady_n,"combo-alt")
        restore=[]; warm=[]
        for i in range(warm_cycles):
            restore.append(worker.submit_with_precision_policy(
                [(prompt,10)],target_policy=STARTUP,admission_id=f"combo-restore-{i}"))
            warm.append(worker.submit_with_precision_policy(
                [(prompt,10)],target_policy=COMBINED,admission_id=f"combo-warm-{i}"))
        return {
            "pid":pid,
            "startup_policy_hash":pc.policy_hash(STARTUP),
            "target_policy_hash":pc.policy_hash(COMBINED),
            "startup_steady":summarize_runs(base_rows),
            "target_steady":summarize_runs(alt_rows),
            "cold_transition":cold["precision_epoch"]["transition_cost"],
            "warm_transitions":[r["precision_epoch"]["transition_cost"] for r in warm],
            "warm_restores":[r["precision_epoch"]["transition_cost"] for r in restore],
            "final_health":worker.health(),
        }
    finally:
        worker.stop(force=True)
def profile_adaptive_recovery(binary, root, iterations=3):
    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(
        route=base.candidate_route(), root=root
    )
    prompt=base._read_first_certified_prompt()
    rows=[]
    try:
        worker.start(); pid=worker.health()["pid"]
        for i in range(int(iterations)):
            got=worker.submit([(prompt,10)])
            adaptive=got.get("adaptive_precision") or {}
            if adaptive.get("action")!="RECOVERY_N6":
                raise RuntimeError("adaptive benchmark did not exercise recovery")
            if int(got.get("inference_passes",0))!=2:
                raise RuntimeError("adaptive recovery did not report two inference passes")
            tokens=got.get("responses",[[]])[0]
            if len(tokens)<=8 or int(tokens[8])!=1224:
                raise RuntimeError("adaptive recovery reference mismatch")
            rows.append({
                "iteration":i,
                "inference_passes":int(got["inference_passes"]),
                "engine_wall_ms":float(got["engine_wall_ms"]),
                "roundtrip_ms":float(got["roundtrip_ms"]),
                "worker_epoch":int(worker.health()["weight_epoch"]),
                "worker_pid":int(worker.health()["pid"]),
            })
        if any(row["worker_pid"]!=pid for row in rows):
            raise RuntimeError("adaptive benchmark PID changed")
        return {
            "schema":"beglin-adaptive-inference-pass-profile-v1",
            "status":"PASS",
            "same_pid":True,
            "pid":pid,
            "samples":len(rows),
            "observed_inference_passes":sorted({r["inference_passes"] for r in rows}),
            "expected_inference_passes":2,
            "p50_engine_ms":pct([r["engine_wall_ms"] for r in rows]),
            "p50_roundtrip_ms":pct([r["roundtrip_ms"] for r in rows]),
            "rows":rows,
        }
    finally:
        worker.stop(force=True)


def aggregate_cost_rows(rows, from_hash, cache_state):
    return {
        "from_policy_hash": from_hash,
        "cache_state": cache_state,
        "samples": len(rows),
        "p50_transition_ms": pct([r["transition_wall_ms"] for r in rows]),
        "p95_transition_ms": pct([r["transition_wall_ms"] for r in rows], .95),
        "cache_hits": int(round(statistics.mean([r["cache_hits"] for r in rows]))),
        "cache_misses": int(round(statistics.mean([r["cache_misses"] for r in rows]))),
        "cache_bytes_added": int(round(statistics.mean([r["cache_bytes_added"] for r in rows]))),
    }


def accepted_policy_profiles(combo):
    startup_hash = combo["startup_policy_hash"]
    target_hash = combo["target_policy_hash"]
    cold = dict(combo["cold_transition"])
    cold_row = {
        "from_policy_hash": startup_hash,
        "cache_state": "cold",
        "samples": 1,
        "p50_transition_ms": float(cold["transition_wall_ms"]),
        "p95_transition_ms": float(cold["transition_wall_ms"]),
        "cache_hits": int(cold["cache_hits"]),
        "cache_misses": int(cold["cache_misses"]),
        "cache_bytes_added": int(cold["cache_bytes_added"]),
    }
    return [
        {
            "policy": STARTUP,
            "policy_hash": startup_hash,
            "steady_engine_p50_ms": combo["startup_steady"]["p50_engine_ms"],
            "steady_roundtrip_p50_ms": combo["startup_steady"]["p50_roundtrip_ms"],
            "expected_inference_passes": 1,
            "transitions": [
                aggregate_cost_rows(combo["warm_restores"], target_hash, "warm")
            ],
        },
        {
            "policy": COMBINED,
            "policy_hash": target_hash,
            "steady_engine_p50_ms": combo["target_steady"]["p50_engine_ms"],
            "steady_roundtrip_p50_ms": combo["target_steady"]["p50_roundtrip_ms"],
            "expected_inference_passes": 1,
            "transitions": [
                cold_row,
                aggregate_cost_rows(combo["warm_transitions"], startup_hash, "warm"),
            ],
        },
    ]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",required=True)
    ap.add_argument("--root",default=str(DEFAULT_ROOT/"p4-20261003"))
    ap.add_argument("--steady",type=int,default=5)
    ap.add_argument("--warm-cycles",type=int,default=3)
    args=ap.parse_args()
    binary=Path(args.binary).resolve(strict=True); root=check_root(args.root)
    if root.exists():shutil.rmtree(root)
    root.mkdir(parents=True)
    p1,e1=profile_target(binary,root/"l3","shared_up_proj",3,6,5,ALT_L3,args.steady,args.warm_cycles)
    p2,e2=profile_target(binary,root/"l26","shared_down_proj",26,5,6,ALT_L26,args.steady,args.warm_cycles)
    combo=profile_combined(binary,root/"combined",args.steady,args.warm_cycles)
    adaptive=profile_adaptive_recovery(binary,root/"adaptive",iterations=max(3,args.warm_cycles))
    out={
        "schema":"beglin-precision-e2e-cost-v1",
        "status":"PASS",
        "source_head":source_head(),
        "production_touched":False,
        "binary_sha256":sha(binary),
        "startup_policy":STARTUP,
        "target_profiles":p1+p2,
        "accepted_policy_profiles":accepted_policy_profiles(combo),
        "policy_profiles":{"combined":combo},
        "adaptive_recovery_profile":adaptive,
        "raw_evidence":{"l3":e1,"l26":e2},
    }
    raw=json.dumps(out,indent=2,sort_keys=True)+"\n"
    (root/"result.json").write_text(raw)
    print(raw,end="")
    print("RESULT_SHA256",hashlib.sha256(raw.encode()).hexdigest())
    return 0

if __name__=="__main__":
    raise SystemExit(main())
