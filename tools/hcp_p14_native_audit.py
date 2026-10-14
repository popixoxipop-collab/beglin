#!/usr/bin/env python3
"""P14 independent read-only whole-record audit; export only aggregate receipt.

This auditor reads the original 48 native response rows + peer-crash receipts
locally on XOX, checks frozen P6 reference SHA for each inference, and hashes
the completed raw result. It does not copy raw request times, native PIDs,
original command plaintext, raw token IDs or checkpoint bytes outside XOX.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,pathlib,subprocess

ROOT=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p11-inflight-p14-multiworker-20261014")
P6=pathlib.Path("/Users/xox/mcp-sandbox/tailnet-commander/hcp-p6-af-logits-20261014-1104/repository/experiments/hcp_p6/evidence/quality.json")
STAGE_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE_SHA="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
P6_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
BASE_SHA="103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd"
FAULT_SHA="a5fd3f590ad6bbddcb69e690ccf738b711b8f7ceb3319f99ee933123f5ac7e78"
SRC="93ac73bb6feefe4bf020b7882fbc0bf3d8d977e7"

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def audit():
    datafile=ROOT/"evidence/real_dual_worker_and_crash.json"
    if not datafile.is_file() or datafile.is_symlink() or sha(P6)!=P6_SHA:
        raise ValueError("P14 missing raw actual native evidence or original P6 authority")
    data=json.loads(datafile.read_text())
    truth={r["prompt_id"]:r for r in json.loads(P6.read_text())["records"]}
    pre=json.loads((ROOT/"preflight.json").read_text())
    if data.get("schema")!="beglin-hcp-p14-two-real-worker-native-overlap-and-peer-inflight-crash-v1":
        raise ValueError("wrong P14 source native contract")
    checks={
      "source_preregistered_commit":SRC,
      "native_P9_SHA256":BASE_SHA,
      "native_fault_only_P11_SHA256":FAULT_SHA,
      "real_separate_controller_processes":2,
      "real_simultaneously_alive_P9_native_workers":2,
      "real_gpu_slots_total":2,
      "real_concurrent_paired_calls":24,
      "real_concurrent_native_generated_responses":48,
      "actual_matched_same_prompt_opposite_policy_pairs":24,
      "full_BF16_teacher_logits_measured":False,
      "production_touched":False,
      "automatic_live_promotion":False,
    }
    for key,value in checks.items():
        if data.get(key)!=value:raise ValueError(f"P14 native report scope contract mismatch: {key}")
    if (pre["source_commit"]!=SRC or pre["P9_original_C_SHA256"]!="bbb43d6144b1656e2f170079e21623b7bce29ccc5923b16f3558f4fc64fb7612"
        or pre["P9_MLX128_SHA256"]!="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
        or pre["P11_test_crash_inject_C_SHA256"]!="3d5a6d9f2f63d53e960c14a88dc4892ffa27cfd67b6d15258b6c930a7c17f2bd"):
        raise ValueError("P14 source preflight untrusted")
    live={"stage_sha256":STAGE_SHA,"route_sha256":ROUTE_SHA,"source_worker_pids":[35481,35493]}
    if data["original_service_before"]!=live or data["original_service_after"]!=live:
        raise ValueError("P14 original live worker/binary/route identity drift")
    pids=data["simultaneous_native_PIDs"]
    if (len(pids)!=2 or len(set(pids))!=2 or set(pids)&{35481,35493}):
        raise ValueError("P14 concurrent actual GPU worker PIDs not independent")
    cells=set();counts=collections.Counter();overlap=0;maxrss=0
    for pair in data["real_concurrent_pair_records"]:
        left,right=pair["left"],pair["right"]
        if (left["PID"]!=pids[0] or right["PID"]!=pids[1] or
            left["controller"]!="alpha" or right["controller"]!="beta" or
            left["prompt_id"]!=right["prompt_id"] or
            left["domain"]!=right["domain"] or
            left["policy"]==right["policy"] or
            left["status"]!=right["status"] or
            left["status"]!="inference_ok" or
            left["end_ns"]<=left["start_ns"] or right["end_ns"]<=right["start_ns"]):
            raise ValueError("P14 simultaneous pairing identity invalid")
        k=(left["prompt_id"],pair["position"])
        if k in cells or not 0<=pair["position"]<4:raise ValueError("P14 duplicate native prompt")
        cells.add(k);counts[left["domain"]]+=1
        overlap_real=max(left["start_ns"],right["start_ns"])<min(left["end_ns"],right["end_ns"])
        if overlap_real!=pair["simultaneous_calls_interval_overlapped"]:
            raise ValueError("P14 incorrect overlapping native request window")
        overlap+=int(overlap_real)
        rss_sum=left["rss_bytes"]+right["rss_bytes"]
        maxrss=max(maxrss,rss_sum)
        if rss_sum>=12*(1024**3) or max(left["rss_bytes"],right["rss_bytes"])>=7*(1024**3):
            raise ValueError("P14 violated preregistered GPU worker RAM")
        for row in (left,right):
            original=truth.get(row["prompt_id"])
            if (original is None or row["domain"]!=original["domain"] or
                row["seq_sha256"]!=original[row["policy"]]["sequence_sha256"]):
                raise ValueError("P14 actual generated response not the frozen original P6")
            if row["switched"] and row["terminal_ACK"]!="REBIND_SET_APPLIED":
                raise ValueError("P14 true native rebind should carry terminal ACK")
            if row["epoch"]<2:raise ValueError("P14 native epoch invalid")
    if (len(cells)!=24 or len(counts)!=6 or set(counts.values())!={4} or
        overlap<1 or overlap!=data["overlapped_admission_time_windows"] or
        maxrss>data["peak_combined_worker_RSS_bytes"] or
        data["peak_combined_worker_RSS_bytes"]>=12*(1024**3)):
        raise ValueError("P14 coverage overlap count / max RSS discrepancy")
    survivors=data["survivor_records"]
    if len(survivors)!=6 or not data.get("survivor_native_PID_never_restarted"):
        raise ValueError("P14 original independent healthy worker did not survive peer crash")
    for r in survivors:
        original=truth.get(r["prompt_id"])
        if (r["PID"]!=pids[0] or original is None or
            r["seq_sha256"]!=original[r["policy"]]["sequence_sha256"] or
            r["status"]!="inference_ok"):
            raise ValueError("P14 peer survivor result or PID wrong")
    fault=data["real_peer_crash_and_quarantine"]
    if (fault["status"]!="actual_midtxn_fault_quarantined_and_recovered" or
        fault["original_outcome"]!="UNKNOWN_BEFORE_DURABLE_ACK" or
        fault["old_terminal_ACK_for_crashed_txn"] is not False or
        fault["recovered_P6_sha_requests"]!=3 or
        fault["crashed_worker_pid"]==pids[0] or
        fault["crashed_worker_pid"]==fault["fresh_worker_pid"] or
        not fault.get("source_worker_unchanged")):
        raise ValueError("P14 native crash/peer isolation incomplete")
    recovered=fault["recovery"]
    if [z["policy"] for z in recovered]!=["B","A","B"]:
        raise ValueError("P14 newly authorized B/A/B failover requests missing")
    for r in recovered:
        original=truth.get(r["prompt_id"])
        if (r["PID"]!=fault["fresh_worker_pid"] or
            original is None or
            r["seq_sha256"]!=original[r["policy"]]["sequence_sha256"] or
            r["status"]!="inference_ok" or r["terminal_ACK"]!="REBIND_SET_APPLIED"):
            raise ValueError("P14 new worker did not truly recover P6 response and ACK")
    marker=ROOT/"before_ack.fault.marker"
    journal=ROOT/"workers/worker-fault/quarantine"
    if any(p.is_symlink() or not p.is_file() for p in (
        marker,journal/"INTENT.json",journal/"VERIFIED.json")):
        raise ValueError("P14 original fsynced marker/journal not preserved")
    m=marker.read_text().split()
    verified=json.loads((journal/"VERIFIED.json").read_text())
    intent=json.loads((journal/"INTENT.json").read_text())
    command_candidates=list(journal.glob("epoch-*.cmd"))
    if len(command_candidates)!=1:raise ValueError("P14 missing exact original crashed command")
    command=command_candidates[0]
    cmd=command.read_text().split()
    if (len(m)!=2 or m[0]!="before_ack" or
        cmd[0]!="REBIND_SET" or cmd[1]!=m[1] or
        verified["txn_id"]!=m[1] or intent["txn_id"]!=m[1] or
        verified["source_command_sha256"]!=sha(command) or
        verified["source_command_sha256"]!=fault["original_command_sha256"] or
        verified["source_marker_sha256"]!=sha(marker) or
        verified["source_marker_sha256"]!=fault["marker_sha256"] or
        verified["terminal_ACK_for_interrupted_transaction"] is not False or
        verified["no_automatic_retry"] is not True or
        verified["journaled_fsynced"] is not True or
        sha(journal/"VERIFIED.json")!=fault["quarantine_VERIFIED_sha256"]):
        raise ValueError("P14 exact original unknown command/marker/intent/receipt not cryptographically bound")
    for k in ("txn_id","source_command_sha256","source_marker_sha256",
              "preimage_ack_sha256","expected_epoch","preimage_policy_hash"):
        if intent[k]!=verified[k]:raise ValueError("P14 durable original intent disagrees with verified")
    return {"schema":"beglin-hcp-p14-readonly-native-dual-worker-audit-v1",
        "status":"REAL_TWO_NATIVE_WORKERS_P6_SHA_ACK_AND_PEER_FAULT_VALIDATED",
        "raw_native_result_sha256":sha(datafile),
        "source_preregistered_commit":SRC,
        "two_distinct_real_MLX_native_workers":True,
        "actual_same_host_concurrent_pairs":len(cells),
        "actual_overlapping_admission_intervals":overlap,
        "native_generation_responses_verified_against_P6":len(cells)*2,
        "six_distinct_P6_domains":len(counts),
        "peak_combined_native_worker_RSS_GiB":round(data["peak_combined_worker_RSS_bytes"]/(1024**3),5),
        "native_peer_survivor_new_requests_P6_SHA":len(survivors),
        "surviving_worker_PID_unchanged":True,
        "real_before_ack_crash_escrow_original_outcome":"UNKNOWN",
        "quarantined_native_command_SHA256":sha(command),
        "quarantine_original_marker_SHA256":sha(marker),
        "quarantine_original_INTENT_SHA256":sha(journal/"INTENT.json"),
        "quarantine_original_VERIFIED_SHA256":sha(journal/"VERIFIED.json"),
        "fresh_post_fault_B_A_B_new_requests_verified":len(recovered),
        "full_BF16_original_teacher_not_loaded":True,
        "production_untouched":True,
        "generalized_multi_worker_certification":False,
        "automatic_live_promotion":False}
if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--out",required=True)
    a=args.parse_args()
    doc=audit()
    path=pathlib.Path(a.out)
    if path.exists():raise RuntimeError("P14 original evidence audit immutable output exists")
    with path.open("x") as f:json.dump(doc,f,indent=2,sort_keys=True);f.write("\n")
    print("P14_INDEPENDENT_TWO_GPU_WORKER_AUDIT_PASS",
        json.dumps({"actual_concurrent_native_responses":doc["native_generation_responses_verified_against_P6"],
            "overlap":doc["actual_overlapping_admission_intervals"],
            "survivor_requests":doc["native_peer_survivor_new_requests_P6_SHA"],
            "unknown_original_quarantined":True,
            "receipt_sha256":sha(path),"production_touched":False},sort_keys=True),flush=True)
