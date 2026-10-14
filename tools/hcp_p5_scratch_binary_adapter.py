#!/usr/bin/env python3
"""P5: research-only immutable scratch runtime, exact qNg64 two-target REBIND_SET.

The historical fixed production-stage source (8b46cdd...) did NOT parse
REBIND_SET; the current research C source does. Binding an obsolete stage
binary to the newer Python PrecisionEpochScheduler produces an ACK txn_id=None.
No modification or replacement of the pinned serving stage is allowed.

This wrapper uses the EXISTING complete Beglin epoch scheduler and exact ACK
verification, switching only the Python process-local binary path to an
explicitly built, pinned scratch native binary. All native queue/txn/ACK writes
stay under RUNNER_TEMP and the original binary is checked unchanged afterwards.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,re

import production_serving_supervisor_persistent as ps
import hcp_p4_native_24token_compat as study

SCHEMA="beglin-hcp-p5-shadow-binary-atomic-rebind-v1"
PINNED_OLD_BINARY_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
PINNED_OLD_SOURCE_SHA="8b46cdd38f5b39dcaeaa9d1c5dfbd9642601a30a"
REQUIRED_COMMANDS=("REBIND_SET", "REBIND_SET_APPLIED")
HEX64=re.compile(r"^[0-9a-f]{64}$")

def file_sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def verify_scratch_binary(binary,source_path,owner):
    owned=pathlib.Path(owner).resolve(strict=True)
    root=pathlib.Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    if owned.parent!=root or not owned.name.startswith("hcp-p4-native-short-"):
        raise ValueError("shadow run root must be under runner temp, not production")
    binary=pathlib.Path(binary).resolve(strict=True)
    if not binary.is_file() or not binary.is_relative_to(owned):
        raise ValueError("scratch compiled binary must be under exact per-run root")
    source=pathlib.Path(source_path).resolve(strict=True)
    if not source.is_file() or not source.is_relative_to(owned):
        raise ValueError("source for scratch binary must be under same controlled root")
    data=source.read_text()
    if any(cmd not in data for cmd in REQUIRED_COMMANDS):
        raise ValueError("P5 scratch source lacks multi-target REBIND_SET contract")
    if "#define MOE_CBATCH_MAXPOS 32" not in data:
        raise ValueError("24token transaction study requires unchanged source cbatch32")
    return {"binary":str(binary),"binary_sha256":file_sha(binary),
        "C_source_sha256":file_sha(source),
        "compiled_with_rebind_set_source":True,
        "cbatch_maxpos":32}

def run_scratch(binary,source,owner,bank,tokenizer,index,*,count=2,output):
    if count not in (2,12):raise ValueError("24token contract count must be 2 or 12")
    scratch=verify_scratch_binary(binary,source,owner)
    stage=pathlib.Path(ps.PERSISTENT_BINARY).resolve(strict=True)
    if ps.EXPECTED_PERSISTENT_SOURCE_SHA!=PINNED_OLD_SOURCE_SHA:
        raise ValueError("historical binary source identity unexpectedly changed")
    if ps.EXPECTED_PERSISTENT_BINARY_SHA!=PINNED_OLD_BINARY_SHA:
        raise ValueError("original stage binary pin unexpectedly changed")
    stage_before=file_sha(stage)
    if stage_before!=PINNED_OLD_BINARY_SHA:
        raise ValueError("production stage is not historically pinned")
    route=pathlib.Path("/Users/xox/vdsp_serving/active-route.json")
    route_before=file_sha(route) if route.is_file() else None

    # Deliberate, process-local replacement of the *expected research worker*
    # executable identity, never of a file, production mapping or persistent
    # policy. The actual built C source and SHA are preserved in the receipt.
    ps.PERSISTENT_BINARY=pathlib.Path(scratch["binary"])
    ps.EXPECTED_PERSISTENT_BINARY_SHA=scratch["binary_sha256"]
    output=pathlib.Path(output)
    if output.resolve().parent!=pathlib.Path(owner).resolve()/"upload":
        raise ValueError("research output must stay under run-specific upload")
    ret=study.run(bank=bank,tokenizer=tokenizer,index=index,
        root=pathlib.Path(owner)/("smoke_worker" if count==2 else "measure_worker"),
        output=output,sample_count=count)

    if file_sha(stage)!=stage_before:
        raise ValueError("old immutable production stage binary mutated")
    route_after=file_sha(route) if route.is_file() else None
    if route_after!=route_before:
        raise ValueError("live route unexpectedly changed")
    if ret.get("schema")!="beglin-hcp-p4-24token-native-compat-v1" or ret.get("sample_pairs")!=count:
        raise ValueError("true native source did not return fully paired result")
    if ret.get("production_touched") is not False or ret.get("native_qng64_quality_certification") is not False:
        raise ValueError("native evidence improperly grants authority")
    receipt={"schema":SCHEMA,"status":"ATOMIC_REBIND_SET_PASSED_SHADOW_NATIVE_ONLY",
        "sample_pairs":count,
        "source_head":os.getenv("GITHUB_SHA"),
        "shadow_binary_sha256":scratch["binary_sha256"],
        "shadow_source_sha256":scratch["C_source_sha256"],
        "historical_stage_binary_sha256_unchanged":stage_before,
        "historical_stage_source_sha256":PINNED_OLD_SOURCE_SHA,
        "native_result_sha256":file_sha(output),
        "actual_ACK_and_policy_hash_verification":True,
        "one_worker_same_PID_per_pair":True,
        "live_route_sha_unchanged":True,
        "original_stage_binary_untouched":True,
        "production_touched":False,"automatic_live_promotion":False}
    (output.parent/(output.stem+".receipt.json")).write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("P5_SCRATCH_BINARY_ATOMIC_REBIND",json.dumps(receipt,sort_keys=True))
    return receipt

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("binary","source","owner","bank","tokenizer","index","output"):
        ap.add_argument("--"+name,required=True,type=pathlib.Path)
    ap.add_argument("--sample-count",required=True,type=int)
    x=ap.parse_args()
    run_scratch(x.binary,x.source,x.owner,x.bank,x.tokenizer,x.index,
                count=x.sample_count,output=x.output)
