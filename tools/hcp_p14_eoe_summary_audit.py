#!/usr/bin/env python3
"""P14 independent EOE audit of sanitized XOX two-worker result + BF16 gate.

Actual per-admission raw JSON remains on XOX because the transfer service
rejected the full artifact. The original 48 generation SHA checks were made by
the separate XOX auditor; here we independently verify its immutable checksum
and the original fsynced escrow marker, command, intent and receipt, plus the
model's 4 safetensors header-derived BF16 resource envelope.

Do not construe this as a second full per-response audit or teacher forward.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib

EXPECTED_RAW="98aa76b19129a7634061595e98463267f506435069c3714db7f2b0a3ff7e9a71"
SRC="93ac73bb6feefe4bf020b7882fbc0bf3d8d977e7"
BF16_SOURCE_INDEX="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
RESULT_SHA="236475031894ea9ef1df87ff7c9a46a36f770fa89e0cc15373254927aedee470"
BF16_GATE_SHA="14ca8badf6221b4d144049a74203bb6b8d4cacd047b861a1922ad3434abac9ec"
CMD_SHA="6463a471f5bb77b60f6699483c139a2bf4cda654cabe7467d3f60adf43ebb9db"
INTENT_SHA="8e4b5e47a74420ab011deb74ac72518cb28c1ad481ef07a44e6ee29febac41b5"
VERIFIED_SHA="02d779515be3d45dc29059270a13b1e5547ee22381ff81ae6778e8fa19426cc9"
MARKER_SHA="b57807c4cef1d1e53f40d248476139f7c1aefe4870c52e70bed1dace640a5da5"

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open("rb") as f:
        for b in iter(lambda:f.read(512*1024),b""):h.update(b)
    return h.hexdigest()
def audit(root):
    root=pathlib.Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("P14 EOE audited evidence not a regular directory")
    required={
       "P14_AUDIT_SUMMARY.json":RESULT_SHA,
       "BF16_TEACHER_FEASIBILITY.json":BF16_GATE_SHA,
       "before_ack.fault.marker":MARKER_SHA,
       "quarantine/original_unknown.cmd":CMD_SHA,
       "quarantine/INTENT.json":INTENT_SHA,
       "quarantine/VERIFIED.json":VERIFIED_SHA
    }
    for name,d in required.items():
        p=root/name
        if not p.is_file() or p.is_symlink() or sha(p)!=d:
            raise ValueError(f"P14 independent EOE evidence hash mismatch: {name}")
    s=json.loads((root/"P14_AUDIT_SUMMARY.json").read_text())
    envelope=json.loads((root/"BF16_TEACHER_FEASIBILITY.json").read_text())
    verified=json.loads((root/"quarantine/VERIFIED.json").read_text())
    intent=json.loads((root/"quarantine/INTENT.json").read_text())
    cmd=(root/"quarantine/original_unknown.cmd").read_text().strip().split()
    marker=(root/"before_ack.fault.marker").read_text().strip().split()
    if (s.get("schema")!="beglin-hcp-p14-readonly-native-dual-worker-audit-v1" or
        s.get("status")!="REAL_TWO_NATIVE_WORKERS_P6_SHA_ACK_AND_PEER_FAULT_VALIDATED" or
        s.get("raw_native_result_sha256")!=EXPECTED_RAW or
        s.get("source_preregistered_commit")!=SRC or
        s.get("actual_same_host_concurrent_pairs")!=24 or
        s.get("actual_overlapping_admission_intervals")!=24 or
        s.get("native_generation_responses_verified_against_P6")!=48 or
        s.get("native_peer_survivor_new_requests_P6_SHA")!=6 or
        s.get("fresh_post_fault_B_A_B_new_requests_verified")!=3 or
        s.get("two_distinct_real_MLX_native_workers") is not True or
        s.get("surviving_worker_PID_unchanged") is not True or
        s.get("production_untouched") is not True or
        s.get("automatic_live_promotion") is not False or
        not 0<float(s.get("peak_combined_native_worker_RSS_GiB",999))<12):
        raise ValueError("P14 XOX original independent conclusion unsupported")
    if (len(cmd)!=13 or cmd[0]!="REBIND_SET" or cmd[3]!="2" or
        len(marker)!=2 or marker[0]!="before_ack" or marker[1]!=cmd[1] or
        verified.get("txn_id")!=cmd[1] or intent.get("txn_id")!=cmd[1] or
        verified.get("terminal_ACK_for_interrupted_transaction") is not False or
        intent.get("terminal_ACK_for_interrupted_transaction") is not False or
        verified.get("journaled_fsynced") is not True or
        verified.get("no_automatic_retry") is not True or
        verified.get("status")!="CRASH_AMBIGUOUS_UNCOMMITTED_TRANSACTION_QUARANTINED" or
        s.get("real_before_ack_crash_escrow_original_outcome")!="UNKNOWN" or
        verified.get("source_command_sha256")!=CMD_SHA or
        verified.get("source_marker_sha256")!=MARKER_SHA):
        raise ValueError("P14 native crash command must remain UNKNOWN and hash-bound")
    if set([(cmd[5],int(cmd[6]),int(cmd[7]),int(cmd[8])),
            (cmd[9],int(cmd[10]),int(cmd[11]),int(cmd[12]))])!={
        ("shared_down_proj",26,5,6),("shared_up_proj",3,6,5)}:
        raise ValueError("P14 original command not exact A->B two-tensor policy")
    for k in ("txn_id","source_command_sha256","source_marker_sha256","expected_epoch",
              "preimage_policy_hash","preimage_ack_sha256"):
        if intent[k]!=verified[k]:
            raise ValueError("P14 original fsynced command intent and final receipt disagree")
    if (envelope.get("schema")!="beglin-hcp-p14-fully-original-BF16-header-only-resource-admission-v1"
        or envelope.get("status")!="FULL_BF16_TEACHER_ENGINE_NOT_YET_STREAMING_CERTIFIED"
        or envelope.get("model_index_sha256")!=BF16_SOURCE_INDEX or
        envelope.get("model_type")!="deepseek_v2" or
        envelope.get("model_decoder_layer_count")!=27 or
        envelope.get("original_all_tensors_indexed")!=5291 or
        envelope.get("original_shard_total_file_bytes")!=31413626576 or
        len(envelope.get("original_per_shard_headers_checked",{}))!=4 or
        len(envelope.get("layer_weight_bytes_by_index",{}))!=27 or
        not 0<envelope.get("largest_layer_weight_gib",999)<1.2 or
        envelope.get("RSS_peak_bytes",2**50)>=512*(1024**2) or
        envelope.get("whole_BF16_model_loaded") is not False or
        envelope.get("full_BF16_teacher_forward_executed") is not False or
        envelope.get("exact_whole_BF16_teacher_logits_produced") is not False):
        raise ValueError("P14 header-only BF16 gate falsely claims full teacher quality")
    pre=json.loads((root/"preflight.json").read_text())
    if (pre.get("source_commit")!=SRC or pre.get("original_stage_SHA256")!=STAGE or
        pre.get("original_route_SHA256")!=ROUTE or pre.get("production_touched") is not False):
        raise ValueError("P14 original research preimage/route changed")
    return {
      "schema":"beglin-hcp-p14-EOE-sanitized-summary-independent-audit-v1",
      "status":"P14_XOX_FULL_RECORD_AUDIT_SHA_AND_INDEPENDENT_ESCROW_BF16_PROVENANCE_VERIFIED",
      "P14_raw_full_native_result_SHA256":EXPECTED_RAW,
      "P14_full_record_independent_XOX_audit_SHA256":RESULT_SHA,
      "P14_EOE_rechecked_original_quarantine_SHA256":VERIFIED_SHA,
      "true_dual_native_responses_passed_in_original_XOX_audit":48,
      "original_XOX_time_overlap_pairs":24,
      "native_survivor_real_requests":6,
      "native_crashed_original_result":"UNKNOWN_NO_AUTO_RETRY",
      "native_new_B_A_B_recovered_requests":3,
      "native_dual_worker_peak_RSS_GiB":s["peak_combined_native_worker_RSS_GiB"],
      "exact_BF16_metadata_model_layers":27,
      "exact_BF16_metadata_tensors":5291,
      "largest_layer_weight_gib":envelope["largest_layer_weight_gib"],
      "full_teacher_logits_measured":False,
      "independent_48_response_raw_evidence_recomputed_on_EOE":False,
      "full_raw_per_admission_evidence_remains_on_XOX":True,
      "promote_production_allowed":False}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    d=audit(a.dir)
    target=pathlib.Path(a.output)
    if target.exists():raise ValueError("P14 audit result already exists")
    with target.open("x") as f:json.dump(d,f,indent=2,sort_keys=True);f.write("\n")
    print("P14_EOE_SANITIZED_FORENSIC_AUDIT_PASS",json.dumps({
        "result_SHA256":sha(target),"native_responses":48,
        "paired_overlap":24,"survivor":6,"unknown_txn_protected":True,
        "BF16_teacher_not_run":True},sort_keys=True))
