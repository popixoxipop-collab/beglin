#!/usr/bin/env python3
"""JE01 preregistered B/C counterfactual physical 43.25MB role bundle.

Research scratch ONLY. Reuses exact signed training-240 original BF16 source
and pinned original native codec, never reselects by observed TRAIN16 NLL.
Changes only six predeclared Q4 role choices, fixes all six Q5 role choices
and total 12 AF4 / 6 Q4 / 6 Q5 cost. The source SSE-DP assignment remains
frozen as source evidence: B/C are hypotheses, not inferred winners.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
PRE=BASE/"docs/research/JE01_REAL27_NLL_POST_FAILURE_EARLY_EXPERT_SWAP_PREREG_20261122.json"
BUILDER=BASE/"tools/je01_eoe_cal240_frozen_sse_exactbyte_role_bundle_20261122.py"
PIN_PRE="08c493215078b83481c871203d1e08232f2646e3"
PIN_BUILDER="89f05ad396b3face37da585b024f23e31ab706e6"
ROOT=Path("/Volumes/D50/TailnetCommander/work")
ARMS={"B":"B_PROTECT_EARLY_GATE","C":"C_PROTECT_EARLY_UP"}
NEW_STATUS="JE01_PREREG_BC_ACTUAL_NATIVE_Q4Q5_PHYSICAL_43MB_BUILT"


def gitblob(p):
    b=p.read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()


def canonical(d):
    return hashlib.sha256(json.dumps(
        d,sort_keys=True,separators=(",",":"),allow_nan=False
    ).encode()).hexdigest()


def frozen_policy(arm):
    if gitblob(PRE)!=PIN_PRE or gitblob(BUILDER)!=PIN_BUILDER:
        raise ValueError("JE01_BC_IMMUTABLE_POLICY_OR_NATIVE_BUILDER_DRIFT")
    p=json.loads(PRE.read_text())
    if p["schema"]!="beglin-je01-post-real-original27-train16-exploratory-fixedbudget-next-ablation-preregister/1" or p["stage"]!="PRE_REGISTERED_CONDITIONAL_UNRUN":
        raise ValueError("JE01_BC_NO_FROZEN_PREREGISTRATION")
    if arm not in ARMS:
        raise ValueError("JE01_ONLY_PREREG_B_C_ARMS_ALLOWED")
    a=next((a for a in p["arms"] if a["id"]==ARMS[arm]),None)
    if a is None or a["counts"]!={"AF4":12,"Q4":6,"Q5":6} or a["actual_43mb_bytes"]!=43253760:
        raise ValueError("JE01_BC_ARM_BYTES_OR_TIERS_CHANGED")
    Q4={tuple((int(v) if k<2 else v) for k,v in enumerate(t.split("/"))) for t in a["q4"]}
    Q5={tuple((int(v) if k<2 else v) for k,v in enumerate(t.split("/"))) for t in p["budget"]["Q5_FROZEN"]}
    frozen={tuple((int(v) if k<2 else v) for k,v in enumerate(t.split("/"))) for t in p["budget"]["original_role_universe"]}
    if len(Q4)!=6 or len(Q5)!=6 or Q4&Q5 or len(frozen)!=24 or not Q4|Q5 <= frozen:
        raise ValueError("JE01_BC_ROLE_SET_NOT_24_OR_OVERLAPPING")
    assigned={t:"Q4" if t in Q4 else "Q5" if t in Q5 else "AF4" for t in frozen}
    return p,a,assigned


def build(arm,dest):
    p,a,assignment=frozen_policy(arm)
    if dest.exists() or dest.is_symlink() or dest.parent!=ROOT or "je01" not in dest.name.lower():
        raise ValueError("JE01_BC_FRESH_ISOLATED_D50_RESEARCH_DEST_REQUIRED")
    spec=importlib.util.spec_from_file_location("je01_cal240_pin_original_real_builder",BUILDER)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    baseline,per=original.load_ledger() # MUST validate original fixed SSE policy before any modification
    if set(baseline)!=set(assignment) or any(k not in per for k in assignment):
        raise ValueError("JE01_BC_UNORIGINAL_FROZEN_ROLE_SOURCE_SET")
    base_sse=sum(per[k]["route_weighted_output_SSE_AF4"] for k in baseline)
    new_sse=sum(per[k]["route_weighted_output_SSE_"+assignment[k]] for k in assignment)
    expected_gain=a["train240_weighted_output_SSE_gain_vs_AF4_pct"]
    if abs(100*(1-new_sse/base_sse)-expected_gain)>1e-9:
        raise ValueError("JE01_BC_PREREG_PROXIES_NOT_REPRODUCED")
    original.load_ledger=lambda:(assignment,per) # immutable source, only alternate hypothesis assignment
    original.build(dest) # exact native source BF16/AF4 and pinned codec, SHA checks
    original_name="JE01_CAL240_FROZEN_SSE_12AF4_6Q4_6Q5_EXACT43MB.bin"
    manifest_name="JE01_CAL240_FROZEN_SSE_24ROLE_SOURCE.tsv"
    binary=dest/original_name
    manifest=dest/manifest_name
    original_receipt=dest/"JE01_CAL240_FROZEN_SSE_PHYSICAL_SOURCE_RECEIPT.json"
    underlying=json.loads(original_receipt.read_text())
    if underlying["counts"]!={"AF4":12,"Q4":6,"Q5":6} or underlying["physical_file_size_bytes"]!=43253760 or underlying["heldout_OOD_gold_accessed"] is not False:
        raise ValueError("JE01_BC_CODEC_AND_PROVENANCE_MISMATCH")
    renamed=dest/("JE01_POSTNLL_"+arm+"_EXACT43MB.bin")
    tsv=dest/("JE01_POSTNLL_"+arm+"_24ROLES.tsv")
    binary.rename(renamed)
    manifest.rename(tsv)
    if original.sha_file(renamed)!=underlying["physical_file_sha256"] or original.sha_file(tsv)!=underlying["role_manifest_sha256"]:
        raise ValueError("JE01_BC_PHYSICAL_RENAME_OR_SHA_MISMATCH")
    receipt={
        "schema":"beglin-je01-counterfactual-bc-pre_registered-physical-policy/1",
        "status":NEW_STATUS,
        "arm":ARMS[arm],
        "prereg_git_blob_sha1":PIN_PRE,
        "source_native_builder_git_blob_sha1":PIN_BUILDER,
        "actual_training240_SSE_receipt_sha256":original.LEDGER_SHA,
        "actual_original_AF4_sha256":original.AF_SHA,
        "original_builder_canonical_receipt":underlying["receipt_sha256"],
        "binary_path":str(renamed),
        "physical_file_sha256":original.sha_file(renamed),
        "physical_file_size_bytes":renamed.stat().st_size,
        "role_manifest_path":str(tsv),
        "role_manifest_sha256":original.sha_file(tsv),
        "q4_roles":sorted("/".join(map(str,k)) for k,v in assignment.items() if v=="Q4"),
        "q5_roles":sorted("/".join(map(str,k)) for k,v in assignment.items() if v=="Q5"),
        "counts":{"AF4":12,"Q4":6,"Q5":6},
        "routed_TRAIN240_proxy_SSE":new_sse,
        "routed_TRAIN240_proxy_SSE_gain_vs_AF4_pct":100*(1-new_sse/base_sse),
        "per_role_raw_source_sha256_verified":True,
        "native_original27_true_gold_NLL":"NOT_RUN",
        "actual_fisher_or_parameter_gradients":"NOT_MEASURED",
        "heldout_validation_OOD_used":False,
        "production_checkpoint_modified":False,
        "source_research_only":True,
    }
    receipt["receipt_sha256"]=canonical(receipt)
    receipt_path=dest/("JE01_POSTNLL_"+arm+"_SOURCE_RECEIPT.json")
    receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("JE01_POSTNLL_BC_ACTUAL_SOURCE_NATIVE_PACKED_BYTES_PASS",json.dumps({
        "arm":arm,"file_sha256":receipt["physical_file_sha256"],
        "bytes":43253760,"manifest_sha256":receipt["role_manifest_sha256"],
        "receipt_sha256":receipt["receipt_sha256"],
        "routed_TRAIN240_proxy_SSE":new_sse
    },sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--arm",choices=("B","C"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    build(a.arm,a.output)
