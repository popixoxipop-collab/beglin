#!/usr/bin/env python3
"""JE01 actual physical same-byte 8-Expert payload bundles, research-only.

Reads SHA-attested BF16-source native 2/6/8bit artifacts from pinned GitHub
Actions run 44172058412. Creates *real* contiguous 43,253,760-byte files,
each containing 8 frozen Expert slots × gate/up/down packed+f32 scales.
Frequency heuristic, independent BF16-weight reconstruction proxy, and
deterministic permutation only. No model integration or gold-token NLL.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_native_exact_byte_presolve import (
    canonical, read_verified_ledger, validate_ledger, projected_slot_bytes
)

SOURCE_RUN=44172058412
SOURCE_COMMIT="62f9aba7a8e328099136875ad601c64546ddf5d0"
SOURCE_RECEIPT="0ddc760bbc33777e1558fbe163847379a68bd3fe80f98d642846a66835a685ee"
BF16_IDENTITY="36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
BASE_BYTES=43_253_760
TOTAL_ALT_BYTES={"2":21626880,"6":56229888,"8":73531392}
SCHEMA="beglin-je01-real-packed-8expert-samebyte-2x3-6x5-research-payload/1"
ROLES=("gate_proj","up_proj","down_proj")
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
SOURCE_MANIFEST="JE01_HF_RANGE_NATIVE_BF16_72_ARTIFACTS.json"

def verify_artifacts(root: Path):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("JE01_ORIGINAL_TIER_ARTIFACT_DIRECTORY_MISSING")
    file=root/SOURCE_MANIFEST
    if not file.is_file() or file.is_symlink():
        raise ValueError("JE01_REAL_TIER_RECEIPT_MISSING")
    raw=json.loads(file.read_text())
    receipt=raw.pop("receipt_sha256",None)
    if receipt!=SOURCE_RECEIPT or canonical(raw)!=SOURCE_RECEIPT:
        raise ValueError("JE01_REAL_BF16_TIER_RECEIPT_SHA_DIFFERS")
    if raw.get("status")!="REAL_ORIGINAL_BF16_SHA_ATTESTED_24_TENSORS_72_NATIVE_TIERS_PACKED" or raw.get("real_role_tier_count")!=72:
        raise ValueError("JE01_REAL_BF16_TIER_STATUS_INVALID")
    if raw.get("source_original_bf16_checkpoint_sha256")!=BF16_IDENTITY:
        raise ValueError("JE01_REAL_BF16_CHECKPOINT_NOT_ORIGINAL")
    if raw.get("real_physical_byte_totals_all8")!=TOTAL_ALT_BYTES:
        raise ValueError("JE01_REAL_BF16_PHYSICAL_TIER_BYTES_DRIFT")
    if raw.get("actual_gold_token_NLL_improvement")!="NOT_MEASURED" or raw.get("original_full_model_mixed_precision_forward")!="NOT_RUN":
        raise ValueError("JE01_SOURCE_TIER_SCOPE_CHANGED")
    ledger=read_verified_ledger(LEDGER)
    slots=validate_ledger(ledger)
    expected={(s["layer"],s["expert"],role,tier) for s in slots for role in ROLES for tier in (2,6,8)}
    seen={}
    for item in raw["records"]:
        k=(item["layer"],item["expert"],item["role"],item["tier"])
        if k not in expected or k in seen: raise ValueError("JE01_SOURCE_ROLE_TIER_DUPLICATE_OR_UNKNOWN")
        for tag in ("packed","scale"):
            spec=item[tag]
            path=root/spec["filename"]
            if path.parent!=root or not path.is_file() or path.is_symlink():
                raise ValueError("JE01_UNSAFE_ARTIFACT_PATH")
            if path.stat().st_size!=spec["actual_bytes"]:
                raise ValueError("JE01_REAL_FILE_SIZE_MISMATCH")
            sha=hashlib.sha256()
            with path.open("rb") as f:
                for buf in iter(lambda:f.read(1<<20),b""): sha.update(buf)
            if sha.hexdigest()!=spec["sha256"]:
                raise ValueError("JE01_REAL_FILE_SHA_MISMATCH")
        if item["saved_physical_bytes"]!=sum(item[tag]["actual_bytes"] for tag in ("packed","scale")):
            raise ValueError("JE01_REAL_TIER_BYTES_INCONSISTENT")
        if item["tier"] not in (2,6,8) or not 0<=item["relative_rmse_bf16"]<10:
            raise ValueError("JE01_BF16_ROUNDTRIP_METRIC_INVALID")
        seen[k]=item
    if set(seen)!=expected:
        raise ValueError("JE01_ORIGINAL_SELECTED_EIGHT_BF16_TIER_INCOMPLETE")
    for tier in (2,6,8):
        observed=sum(v["saved_physical_bytes"] for k,v in seen.items() if k[3]==tier)
        if observed!=TOTAL_ALT_BYTES[str(tier)]:
            raise ValueError("JE01_TIER_PHYSICAL_BYTES_NOT_EQUAL")
    return slots,seen

def all_equal_byte_maps(slots,records):
    # All 56 = C(8,3) possibilities; physical tier sizes included, no q4
    # baseline file references because these 2/6 bundles are complete.
    options=[]
    keys=[(x["layer"],x["expert"]) for x in slots]
    for twos in itertools.combinations(keys,3):
        two_set=set(twos)
        bits={key:2 if key in two_set else 6 for key in keys}
        physical=sum(sum(records[(l,e,role,bits[(l,e)])]["saved_physical_bytes"]
                     for role in ROLES) for (l,e) in keys)
        if physical!=BASE_BYTES:
            raise ValueError("JE01_ACTUAL_BUNDLED_BYTES_NOT_MATCHED")
        metric=sum(
            sum(records[(l,e,role,bits[(l,e)])]["relative_rmse_bf16"]**2
                for role in ROLES)/3
            for l,e in keys
        )/len(keys)
        options.append((metric,tuple(sorted(twos)),bits))
    if len(options)!=56:
        raise ValueError("JE01_56_EQUALBYTE_PLANS_NOT_FOUND")
    return options

def plans(slots,records):
    options=all_equal_byte_maps(slots,records)
    keys=[(s["layer"],s["expert"]) for s in slots]
    ordered=sorted(slots,key=lambda s:(-s["calibration_last_token_routed_calls"],s["layer"],s["expert"]))
    freq_twos=frozenset((s["layer"],s["expert"]) for s in ordered[-3:])
    # Best BF16 reconstruction proxy after quantizing REAL original weights;
    # NOT the Bayesian NIG posterior and NOT real target-token NLL.
    best=min(options,key=lambda x:(x[0],x[1]))
    proxy_twos=frozenset(best[1])
    # Non-oracle deterministic unrelated Expert permutation, fixed before gold.
    placebo_order=sorted(keys,key=lambda k:hashlib.sha256(
        ("JE01_FIXED_2BIT_PLACEBO_20261118|"+str(k)).encode()).digest())
    placebo_twos=frozenset(placebo_order[:3])
    if placebo_twos in (freq_twos,proxy_twos):
        for delta in range(1,8):
            candidate=frozenset(placebo_order[delta:delta+3])
            if len(candidate)==3 and candidate not in (freq_twos,proxy_twos):
                placebo_twos=candidate
                break
    strategy={"frequency_prelabel":freq_twos,"BF16_reconstruction_proxy":proxy_twos,
              "deterministic_permutation":placebo_twos}
    assert all(len(v)==3 for v in strategy.values())
    return strategy,options

def copy_sha_into(src,dst,hasher):
    offset=dst.tell()
    with src.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            dst.write(part);hasher.update(part)
    if dst.tell()-offset!=src.stat().st_size:
        raise ValueError("JE01_BUNDLE_COPY_BYTES_CHANGED")
    return [offset,dst.tell()]

def create_one(root,output,name,bits,slots,records):
    path=output/(name+".packed-and-scales.bin")
    if path.exists() or path.is_symlink():
        raise ValueError("JE01_BUNDLE_REFUSE_OVERWRITE")
    hasher=hashlib.sha256(); segments=[]
    with path.open("xb") as out:
        for s in slots:
            l,e=s["layer"],s["expert"]
            bit=bits[(l,e)]
            for role in ROLES:
                item=records[(l,e,role,bit)]
                for k in ("packed","scale"):
                    src=root/item[k]["filename"]
                    offsets=copy_sha_into(src,out,hasher)
                    segments.append({"layer":l,"expert":e,"role":role,"bit":bit,
                                     "kind":k,"source_sha256":item[k]["sha256"],
                                     "offsets":offsets})
        out.flush();os.fsync(out.fileno())
    if path.stat().st_size!=BASE_BYTES:
        raise ValueError("JE01_BUNDLE_REAL_FILE_NOT_EXACT_43253760_BYTES")
    return {"name":name,"file_name":path.name,"file_sha256":hasher.hexdigest(),
            "physical_binary_file_bytes":path.stat().st_size,
            "role_files":len(segments),
            "bit_assignment":[{"layer":s["layer"],"expert":s["expert"],
                               "bits":bits[(s["layer"],s["expert"])]} for s in slots],
            "file_regions":segments}

def run(source,output):
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("JE01_OUTPUT_DIRECTORY_MUST_BE_NEW")
    slots,records=verify_artifacts(source)
    strategies,options=plans(slots,records)
    output.mkdir(mode=0o700)
    result=[]
    for name,twos in strategies.items():
        bits={(s["layer"],s["expert"]):(2 if (s["layer"],s["expert"]) in twos else 6) for s in slots}
        result.append(create_one(source,output,name,bits,slots,records))
        print(json.dumps({"name":name,"actual_bytes":result[-1]["physical_binary_file_bytes"],
                          "sha":result[-1]["file_sha256"],"2bit_experts":sorted(twos)}),flush=True)
    minima=min(options,key=lambda x:(x[0],x[1]))
    doc={"schema":SCHEMA,"status":"THREE_ACTUAL_8EXPERT_EQUALBYTE_43253760_PAYLOADS_CONSTRUCTED",
         "source_real72_workflow_run":SOURCE_RUN,"source_real72_git_commit":SOURCE_COMMIT,
         "source_real72_receipt_sha256":SOURCE_RECEIPT,
         "native_original_AF_exact_baseline_expert_bytes":BASE_BYTES,
         "exact_physical_2x3_6x5_feasible_combinations":len(options),
         "bf16_weight_proxy_minimum_mean_relative_MSE":minima[0],
         "case_specific_Bayesian_NLL_effects":"NOT_MEASURED",
         "source_original_AF4bit_vs_BF16_requantized_parity":"NOT_TESTED",
         "full_original_model_routed_mixed_precision_forward":"NOT_RUN",
         "heldout_and_ood_not_opened":True,"promotion_allowed":False,
         "bundles":result}
    doc["receipt_sha256"]=canonical(doc)
    (output/"JE01_REAL3_EQUALBYTE_BUNDLES_RECEIPT.json").write_text(
        json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return doc

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        doc=run(a.source,a.output)
        print(json.dumps({"status":doc["status"],"receipt_sha256":doc["receipt_sha256"],
                          "bundle_count":len(doc["bundles"]),"bytes_each":BASE_BYTES},sort_keys=True))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error":str(e)[:300]}))
        raise SystemExit(2)
