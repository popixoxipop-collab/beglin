#!/usr/bin/env python3
"""Re-run existing P4 qNg64 A/B with *scratch-compiled* REBIND_SET-capable C.

The old pinned persistent-stage binary is READ ONLY and never overwritten.
The only override is Python module state within this isolated test process:
ps.PERSISTENT_BINARY and ps.EXPECTED_PERSISTENT_BINARY_SHA. The worker's
run-specific queue, promotion map, transaction file, ACK, and log are placed
under RUNNER_TEMP. Native gpu_runtime_control exact txn/hash/epoch checks
remain ENABLED, with no mocking and no fallback if an ACK is missing.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib
import production_serving_supervisor_persistent as ps
from hcp_p4_native_24token_compat import run as run_existing_native_ab

SCHEMA="beglin-hcp-p5-scratch-rebind-set-v1"
MAX_AB=12
LIVE_ROUTE=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")

def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def check_scratch_binary(path,*,run_root):
    binpath=pathlib.Path(path).resolve(strict=True)
    root=pathlib.Path(run_root).resolve(strict=True)
    if (not binpath.is_file() or binpath.is_symlink() or
        root not in binpath.parents or
        binpath.relative_to(root)!=pathlib.Path("build/qwen_infer_gpu")):
        raise ValueError("P5 binary not the exact scratch build of this run")
    return digest(binpath)

def validate_prior_stage(*,stage,expected_sha):
    stage=pathlib.Path(stage)
    if (not stage.is_file() or stage.is_symlink() or
        digest(stage)!=expected_sha):
        raise ValueError("pinned immutable stage binary identity changed")
    return expected_sha

def run(*,binary,source,source_c_sha,root,bank,tokenizer,index,
        output,sample_count):
    if sample_count not in (2,12):
        raise ValueError("P5 allows exact 2-prompt smoke or 12-domain comparison")
    run_root=pathlib.Path(root).resolve(strict=True)
    base=pathlib.Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    expected=base/("hcp-p4-native-short-"+os.environ["GITHUB_RUN_ID"])
    if run_root!=expected or not run_root.is_dir():
        raise ValueError("P5 run root must be its own runner scratch")
    old_bin=ps.PERSISTENT_BINARY
    old_digest=ps.EXPECTED_PERSISTENT_BINARY_SHA
    stage_hash=validate_prior_stage(stage=old_bin,expected_sha=old_digest)
    real_bin_hash=check_scratch_binary(binary,run_root=run_root)
    c=pathlib.Path(source).resolve(strict=True)
    if not c.is_file() or c.is_symlink() or digest(c)!=source_c_sha:
        raise ValueError("P5 scratch source C identity does not match build pin")
    code=c.read_text()
    if ("REBIND_SET" not in code or "REBIND_SET_APPLIED" not in code or
        "MOE_CBATCH_MAXPOS 32" not in code):
        raise ValueError("scratch source lacks certified exact transaction and capacity contract")
    if os.path.exists("/Users/xox/vdsp_serving/active-route.json"):
        live_before=digest(LIVE_ROUTE)
    else:
        live_before=None
    # No import or assignment in the user-facing server process. Only this
    # subprocess sees these module-level values.
    try:
        ps.PERSISTENT_BINARY=pathlib.Path(binary).resolve(strict=True)
        ps.EXPECTED_PERSISTENT_BINARY_SHA=real_bin_hash
        ps.REQUEST_TIMEOUT_SECONDS=max(int(ps.REQUEST_TIMEOUT_SECONDS),120)
        target_root=run_root/("smoke_worker" if sample_count==2 else "measure_worker")
        result=run_existing_native_ab(
            bank=bank,tokenizer=tokenizer,index=index,root=target_root,
            output=output,sample_count=sample_count)
    finally:
        ps.PERSISTENT_BINARY=old_bin
        ps.EXPECTED_PERSISTENT_BINARY_SHA=old_digest
    if digest(old_bin)!=stage_hash:
        raise ValueError("old production stage binary changed")
    live_after=digest(LIVE_ROUTE) if LIVE_ROUTE.is_file() else None
    if live_before!=live_after:
        raise ValueError("active route was changed")
    if result.get("status")!="REAL_NATIVE_GPU_SHORT24_COMPATIBILITY_ONLY":
        raise ValueError("did not complete strict qNg64 native AB")
    if not result.get("production_route_hash_unchanged") or result.get("production_touched") is not False:
        raise ValueError("actual runtime/route integrity not established")
    report={"schema":SCHEMA,
        "status":"SCRATCH_REBIND_SET_ACTUAL_ACK_AB_VERIFIED",
        "sample_pairs":sample_count,
        "source_pinned_c_sha256":source_c_sha,
        "new_scratch_binary_sha256":real_bin_hash,
        "old_persistent_stage_binary_sha256":stage_hash,
        "old_persistent_stage_unchanged":True,
        "live_active_route_unchanged":True,
        "native_beglin_precision_epoch_ack_not_mocked":True,
        "no_live_precision_rebind":True,
        "no_production_model_changes":True,
        "worker_root_is_run_local":True,
        "native_ab_result_sha256":digest(output),
        "first_token_A_B_agreement_rate":result["first_token_A_B_agreement_rate"],
        "exact_sequence_A_B_agreement_rate":result["exact_sequence_A_B_agreement_rate"],
        "native_112_token_capacity_certified":False,
        "native_24_token_policy_A_B_evaluated":True,
        "automatic_live_promotion":False}
    receipt=pathlib.Path(output).with_name("p5_"+("smoke" if sample_count==2 else "12domain")+"_receipt.json")
    receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("P5_NATIVE_SCRATCH_ACK_AB_PASS",json.dumps(report,sort_keys=True))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("binary","source","root","bank","tokenizer","index","output"):
        p.add_argument("--"+name,required=True,type=pathlib.Path)
    p.add_argument("--source-c-sha",required=True)
    p.add_argument("--sample-count",type=int,default=2)
    a=p.parse_args()
    run(binary=a.binary,source=a.source,source_c_sha=a.source_c_sha,
        root=a.root,bank=a.bank,tokenizer=a.tokenizer,index=a.index,
        output=a.output,sample_count=a.sample_count)
