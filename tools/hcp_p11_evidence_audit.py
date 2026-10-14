#!/usr/bin/env python3
"""P11 independent, sanitized evidence recheck — no native work or network.

Rejects missing/modified crash markers, uncommitted transaction escrow, wrong
epoch/policy preimage, mismatched recovery SHA, or false BF16 whole-teacher claim.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,re

P6_QUALITY_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
EXPECT_BIN="a5fd3f590ad6bbddcb69e690ccf738b711b8f7ceb3319f99ee933123f5ac7e78"
EXPECT_SOURCE="6a3790590321b35e353f0b06978517e5407dcce9"
EXPECTED=["B","A","B"]
WINDOWS=("first_bind","before_ack")
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def audit(root,quality):
    root,quality=pathlib.Path(root),pathlib.Path(quality)
    if not root.is_dir() or root.is_symlink() or sha(quality)!=P6_QUALITY_SHA:
        raise ValueError("P11 evidence root or original P6 reference not pinned")
    q=json.loads(quality.read_text())
    truth={z["prompt_id"]:z for z in q["records"]}
    if "p000" not in truth:raise ValueError("P11 original reference prompt absent")
    original={p:truth["p000"][p]["sequence_sha256"] for p in ("A","B")}
    checked={}
    txnids=set()
    for w in WINDOWS:
        report=root/(w+".json")
        cmd=root/(w+"_quarantined_transaction.cmd")
        marker=root/(w+".marker")
        if any(not p.is_file() or p.is_symlink() for p in (report,cmd,marker)):
            raise ValueError("P11 missing real run receipt or source-marker/command evidence")
        r=json.loads(report.read_text())
        if (r.get("schema")!="beglin-hcp-p11-real-midtxn-crash-reconcile-v1" or
            r.get("status")!="TRUE_INFLIGHT_WINDOW_WITNESSED_AND_QUARANTINED" or
            r.get("phase")!=w or r.get("source_commit")!=EXPECT_SOURCE or
            r.get("isolated_binary_sha256")!=EXPECT_BIN or
            r.get("old_transaction_outcome")!="UNKNOWN_CRASHED_BEFORE_DURABLE_ACK" or
            r.get("native_old_terminal_ack_for_txn") is not False or
            r.get("old_transaction_never_replayed") is not True or
            r.get("fresh_B_A_B_accepted_with_P6_output_SHA") is not True or
            r.get("production_touched") is not False or
            r.get("automatic_live_promotion") is not False or
            r.get("production_automatic_reconciler_installed") is not False):
            raise ValueError("P11 native crash/ACK/reconcile outcome contract not met")
        if (not r.get("controlled_PID_kill_after_fsynced_marker") or
            not r.get("startup_fresh_pid") or not r.get("startup_fresh_policy_A") or
            r.get("crashed_admission_thread_error_types")!=["PrecisionEpochSchedulerError"]):
            raise ValueError("P11 real interruption/matching PID witness missing")
        if sha(cmd)!=r["native_cmd_SHA256"] or sha(marker)!=r["native_crash_marker_sha256"]:
            raise ValueError("P11 source-verified crash evidence changed")
        mark=marker.read_text().strip().split()
        toks=cmd.read_text().strip().split()
        if (len(mark)!=2 or mark[0]!=w or
            len(toks)!=13 or toks[0]!="REBIND_SET" or toks[1]!=mark[1] or toks[3]!="2" or
            not re.fullmatch(r"epoch-[0-9a-f]{32}",mark[1])):
            raise ValueError("P11 marker/command are not same exact native transaction")
        if set([(toks[5],int(toks[6]),int(toks[7]),int(toks[8])),
                (toks[9],int(toks[10]),int(toks[11]),int(toks[12]))])!={
            ("shared_down_proj",26,5,6),("shared_up_proj",3,6,5)}:
            raise ValueError("P11 source transaction is not original P6 policy A→B")
        if mark[1] in txnids:raise ValueError("P11 reused crashed txn across experiments")
        txnids.add(mark[1])
        receipt=r["quarantine_receipt"]
        if (receipt.get("status")!="CRASH_AMBIGUOUS_UNCOMMITTED_TRANSACTION_QUARANTINED" or
            receipt.get("source_command_sha256")!=sha(cmd) or
            receipt.get("source_marker_sha256")!=sha(marker) or
            receipt.get("txn_id")!=mark[1] or
            receipt.get("terminal_ACK_for_interrupted_transaction") is not False or
            receipt.get("preimage_epoch")!=int(toks[2]) or
            receipt.get("preimage_policy_hash")!=toks[4] or
            not receipt.get("journaled_fsynced") or
            not receipt.get("no_automatic_retry")):
            raise ValueError("P11 uncommitted transaction proof not bound to cmd/ACK")
        rec=r["fresh_recovery"]
        if len(rec)!=3 or [e["policy"] for e in rec]!=EXPECTED:
            raise ValueError("P11 B-A-B recovery incomplete")
        if any(z["sequence_sha256"]!=original[z["policy"]] or not z["worker_pid_stable"] for z in rec):
            raise ValueError("P11 real recovery tokens diverged from sealed P6")
        checked[w]={"report_sha256":sha(report),
            "command_sha256":sha(cmd),"marker_sha256":sha(marker),
            "command_quarantined":True,
            "fresh_B_A_B_original_sequence_SHA_matched":True,
            "old_result_classification":"UNKNOWN"}
    abort=root/"first_bind_attempt1_unmarked.json"
    if not abort.is_file():raise ValueError("P11 initial premarker timeout disclosure missing")
    first=json.loads(abort.read_text())
    if (first.get("status")!="NOT_A_WITNESSED_NATIVE_CRASH" or
        first.get("native_marker_exists") is not False):
        raise ValueError("P11 incorrectly claims premarker timeout was successful crash")
    bf=root/"bf16_source_tiles.json"
    if not bf.is_file() or bf.is_symlink():raise ValueError("P11 partial BF16 reference missing")
    b=json.loads(bf.read_text())
    expected_names={"model.layers.3.mlp.shared_experts.up_proj.weight",
                    "model.layers.26.mlp.shared_experts.down_proj.weight"}
    if (b.get("schema")!="beglin-hcp-p11-readonly-bf16-weight-slice-oracle-v1" or
        b.get("status")!="ORIGINAL_BF16_TILE_PROJECTION_ONLY" or
        len(b["samples"])!=2 or {z["tensor_key"] for z in b["samples"]}!=expected_names or
        b["total_BF16_tile_value_bytes"]!=16384 or
        b["whole_BF16_model_forward_executed"] is not False or
        b["BF16_reference_logits_measured"] is not False or
        b["quantized_A_or_B_noninferiority_proven"] is not False or
        b["quality_certified"] is not False or
        b["RSS_peak_bytes"]>=2*1024**3):
        raise ValueError("P11 partial BF16 tile may not be represented as full teacher")
    for s in b["samples"]:
        if s["source_dtype"]!="BF16" or s["tile_shape"]!=[16,256] or s["slice_bytes_read_for_values"]!=8192:
            raise ValueError("P11 unbounded or wrong BF16 data type")
    reread=root/"bf16_tile_reread.json"
    if not reread.is_file() or reread.is_symlink():
        raise ValueError("P11 independent BF16 source re-read missing")
    readback=json.loads(reread.read_text())
    original_shas={x["tensor_key"]:x for x in b["samples"]}
    if (readback.get("schema")!="beglin-hcp-p11-independent-original-BF16-source-tile-reread-v1" or
        readback.get("original_probe_sha256")!=sha(bf) or
        readback.get("source_index_sha256")!=b["source_index_sha256"] or
        readback.get("raw_BF16_tile_bytes_covered")!=16384 or
        readback.get("full_teacher_inference") is not False or
        readback.get("BF16_model_logits_measured") is not False or
        len(readback.get("per_tensor_recheck",[]))!=2):
        raise ValueError("P11 BF16 tile original source re-read receipt invalid")
    for row in readback["per_tensor_recheck"]:
        k=row["tensor_key"]
        if (k not in original_shas or row["matching"] is not True or
            row["readback_tile_weight_sha256"]!=original_shas[k]["tile_weight_sha256"] or
            row["readback_projection_sha256"]!=original_shas[k]["tile_matvec_result_sha256"]):
            raise ValueError("P11 independent BF16 tile sample changed")
    return {"schema":"beglin-hcp-p11-independent-native-audit-v1",
        "status":"REAL_CRASH_MARKER_AND_COMMAND_ESCROW_MATCHED",
        "windows":checked,
        "unique_ambiguous_transaction_count":2,
        "total_real_recovered_inferences":6,
        "bad_early_test_not_counted_as_crash":True,
        "new_study_source_commit":EXPECT_SOURCE,
        "read_only_BF16_slice_probe_SHA256":sha(bf),
        "read_only_BF16_weight_bytes_checked":16384,
        "independent_BF16_source_tile_readback_sha256":sha(reread),
        "independent_BF16_source_tile_readback_matched":True,
        "full_BF16_teacher_not_executed":True,
        "exactly_once_old_requests_not_proven":True,
        "automatic_prod_reconciler_not_installed":True,
        "production_promotion_allowed":False}
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",required=True)
    ap.add_argument("--quality",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    result=audit(a.evidence,a.quality)
    dest=pathlib.Path(a.output)
    with dest.open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    print("P11_INDEPENDENT_NATIVE_AUDIT_PASS",json.dumps({"windows":list(result["windows"]),
        "unique_unknown_txn":2,"actual_recovered_sequences":6,
        "bf16_teacher_not_run":True,"report_sha256":sha(dest),
        "promotion_allowed":False},sort_keys=True),flush=True)
