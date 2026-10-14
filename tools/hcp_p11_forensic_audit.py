#!/usr/bin/env python3
"""P11 independent source-pinned forensic audit of durable quarantine files.

Extends the existing two-window independent auditor. Requires original
XOX INTENT / VERIFIED receipts and an independent BF16 tile reread. It never
starts a worker, replays a transaction or assumes the crash outcome committed.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib
from hcp_p11_evidence_audit import audit,sha,WINDOWS,STAGE,ROUTE,EXPECT_BIN,EXPECT_SOURCE

P11_FAULT_C_SHA="3d5a6d9f2f63d53e960c14a88dc4892ffa27cfd67b6d15258b6c930a7c17f2bd"
P9_MLX_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
P11_PATCHER_SHA="c1572f94d411b1246efca443cc9b7a384fb19e298c0e0871eed91088d49b984f"

def review(evidence,quality):
    folder=pathlib.Path(evidence)
    base=audit(folder,quality)
    p=folder/"preflight.json"
    if not p.is_file() or p.is_symlink():raise ValueError("original XOX preflight not present")
    info=json.loads(p.read_text())
    if (info.get("source_commit")!=EXPECT_SOURCE
        or info.get("fault_binary_sha")!=EXPECT_BIN
        or info.get("fault_C_sha")!=P11_FAULT_C_SHA
        or info.get("P9_packed_cache_mlx128_sha")!=P9_MLX_SHA
        or info.get("C_string_escape_correction_SHA256")!=P11_PATCHER_SHA
        or info.get("stage_sha")!=STAGE or info.get("route_sha")!=ROUTE
        or info.get("production_touched") is not False
        or info.get("promotion_enabled") is not False):
        raise ValueError("P11 preimage / corrected fault-only binary no longer matches")
    forensic={}
    for window in WINDOWS:
        orig=folder/(window+".json")
        r=json.loads(orig.read_text())
        i=folder/"forensic"/(window+"_INTENT.json")
        f=folder/"forensic"/(window+"_VERIFIED.json")
        if any(not x.is_file() or x.is_symlink() for x in (i,f)):
            raise ValueError("P11 original durable quarantine receipts not copied")
        intent=json.loads(i.read_text())
        verified=json.loads(f.read_text())
        if verified!=r["quarantine_receipt"]:
            raise ValueError("P11 original VERIFIED receipt differs from native report")
        original_fields=("schema","window","txn_id","expected_epoch",
            "source_command_sha256","source_marker_sha256","preimage_ack_sha256",
            "preimage_policy_hash","preimage_epoch",
            "terminal_ACK_for_interrupted_transaction","old_native_worker_verified_dead")
        if any(intent.get(k)!=verified.get(k) for k in original_fields):
            raise ValueError("P11 durable INTENT journal disagrees with VERIFIED")
        if (verified["source_command_sha256"]!=sha(folder/(window+"_quarantined_transaction.cmd"))
            or verified["source_marker_sha256"]!=sha(folder/(window+".marker"))
            or verified["no_automatic_retry"] is not True
            or verified["terminal_ACK_for_interrupted_transaction"] is not False):
            raise ValueError("P11 retained native command not provably quarantined")
        forensic[window]={
           "INTENT_sha256":sha(i),"VERIFIED_sha256":sha(f),
           "source_command_sha256":verified["source_command_sha256"],
           "original_ambiguous_txn_outcome":"UNKNOWN"}
    bf_source=folder/"bf16_source_tiles.json"
    bf_recheck=folder/"bf16_tile_reread.json"
    if not bf_recheck.is_file() or bf_recheck.is_symlink():
        raise ValueError("P11 independent BF16 source reread missing")
    main=json.loads(bf_source.read_text())
    repeat=json.loads(bf_recheck.read_text())
    if (repeat.get("schema")!="beglin-hcp-p11-independent-original-BF16-source-tile-reread-v1"
        or repeat.get("original_probe_sha256")!=sha(bf_source)
        or repeat.get("raw_BF16_tile_bytes_covered")!=16384
        or repeat.get("source_index_sha256")!=main["source_index_sha256"]
        or repeat.get("BF16_model_logits_measured") is not False
        or repeat.get("full_teacher_inference") is not False
        or repeat.get("two_independent_reads_from_original_shards") is not True
        or repeat.get("max_rss_bytes",2**32)>=2*1024**3):
        raise ValueError("P11 BF16 independent source tile reread invalid/overstated")
    rows={s["tensor_key"]:s for s in repeat.get("per_tensor_recheck",[])}
    if len(rows)!=2 or set(rows)!={x["tensor_key"] for x in main["samples"]}:
        raise ValueError("P11 BF16 tensor set changed")
    for rec in main["samples"]:
        comparison=rows[rec["tensor_key"]]
        if (comparison.get("matching") is not True or
            comparison.get("readback_tile_weight_sha256")!=rec["tile_weight_sha256"] or
            comparison.get("readback_projection_sha256")!=rec["tile_matvec_result_sha256"]):
            raise ValueError("independent BF16 BF16-tile / CPU projection mismatch")
    return {
       "schema":"beglin-hcp-p11-independent-forensic-supplement-v1",
       "status":"SIGNED_PREIMAGE_NATIVE_DURABLE_INTENT_VERIFIED_MATCH_AND_BF16_REREAD_PASS",
       "native_original_source_commit":EXPECT_SOURCE,
       "native_fault_source_sha256":P11_FAULT_C_SHA,
       "native_fault_binary_sha256":EXPECT_BIN,
       "P11_original_preflight_SHA256":sha(p),
       "first_bind_before_ack_forensic_sha256":forensic,
       "base_audit_windows":list(base["windows"]),
       "new_successful_recovery_request_count":base["total_real_recovered_inferences"],
       "original_interrupted_transaction_outcome_not_claimed_successful":True,
       "BF16_first_original_tile_SHA256":sha(bf_source),
       "BF16_second_original_tile_reread_SHA256":sha(bf_recheck),
       "BF16_source_tile_bytes_verified":16384,
       "BF16_reference_full_model_logits_measured":False,
       "promotion_certified":False,
       "production_touched":False}
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",required=True)
    ap.add_argument("--quality",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    result=review(args.evidence,args.quality)
    target=pathlib.Path(args.output)
    with target.open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    print("P11_FORENSIC_SUPPLEMENT_PASS",
          json.dumps({"SHA256":sha(target),"recovered":6,
            "quarantined_uncommitted":2,"full_teacher":False},sort_keys=True))
