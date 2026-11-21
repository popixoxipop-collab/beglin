#!/usr/bin/env python3
"""JE01 physical, SHA-attested NIG/3 controls mixed precision ROLE-level payloads.

Use frozen 32-train-input NIG role choices and its three controls from the
immutable original BF16-source q3/q5 receipt. DO NOT optimize on real NLL,
synthetic test scores, or withheld WikiText/HellaSwag. Construct four actual
43,253,760-byte E1 role payload files; no native 27layer forwarding.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_native_exact_byte_presolve import canonical,read_verified_ledger,validate_ledger
SOURCE_Q35="04ddca2f84367f41098357b5fae905e76c1207ef023d648a9f388aaddb3bb0ca"
SOURCE_NIG="72769af77d5cfbce683333bd4f1fe1d8aeb728a196e9d477df7ca851fe1c602c"
SOURCE_MODEL="36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
SOURCE_SCOPE="REAL_BF16_SOURCE_ROLE_LEVEL_NIG_PROXY_TRAIN32_TEST32_COMPLETE"
BASE=43_253_760
Q3_ROLE_BYTES=1_261_568
Q5_ROLE_BYTES=1_982_464
ROLES=("gate_proj","up_proj","down_proj")
EXPECTED_POLICIES=("NIG_proxy_upper_6roles","weight_MSE_proxy_6roles",
                    "prelabel_frequency_2experts_x3","fixed_role_permutation_6roles")
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
SCHEMA="beglin-je01-real-originalBF16-nig-role4-actual-equalbyte-files/1"

def read_signed(path,expected):
    if not path.is_file() or path.is_symlink():raise ValueError("JE01_SIGNED_SOURCE_MISSING")
    d=json.loads(path.read_text())
    signed=d.pop("receipt_sha256",None)
    if signed!=expected or canonical(d)!=expected:
        raise ValueError("JE01_SIGNED_RESEARCH_SOURCE_CHANGED")
    return d

def read_sources(q3root,nig_path):
    source=read_signed(q3root/"JE01_ORIGINAL_BF16_E8_Q3Q5_PHYSICAL_RECEIPT.json",SOURCE_Q35)
    nig=read_signed(nig_path,SOURCE_NIG)
    if source["status"]!="REAL_SOURCE_BF16_8EXPERT_48_NATIVE_Q3Q5_ROLE_TIERS_GENERATED":
        raise ValueError("JE01_REAL_SOURCE_Q35_NOT_COMPLETE")
    if source["source_original_BF16_identity_SHA256"]!=SOURCE_MODEL:
        raise ValueError("JE01_REAL_ORIGINAL_BF16_SOURCE_SHA_WRONG")
    if source["actual_full8_physical_tier_bytes"]!={"3":30_277_632,"5":47_579_136}:
        raise ValueError("JE01_REAL_Q35_PHYSICAL_BYTES_CHANGED")
    if nig["status"]!=SOURCE_SCOPE or nig["original_source_SHA256"]!=SOURCE_MODEL or \
       nig["source_native_role_receipt"]!=SOURCE_Q35 or \
       nig["train_synthetic_seed"]!=20261121 or nig["test_synthetic_seed"]!=20261122:
        raise ValueError("JE01_NIG_SOURCE_OR_SEEDS_CHANGED")
    if nig["actual_role3_count"]!=6 or nig["actual_role5_count"]!=18 or \
       nig["baseline_original_AF8_byte_budget"]!=BASE:
        raise ValueError("JE01_NIG_SELECTED_ROLE_BUDGET_DRIFT")
    if set(nig["synthetic_test_policy_results"])!=set(EXPECTED_POLICIES):
        raise ValueError("JE01_NIG_POLICY_NAMES_DRIFT")
    frozen=validate_ledger(read_verified_ledger(LEDGER))
    allroles={(s["layer"],s["expert"],role) for s in frozen for role in ROLES}
    records={}
    for entry in source["records"]:
        key=(entry["layer"],entry["expert"],entry["role"],entry["bits"])
        if key in records or key[:3] not in allroles or key[3] not in (3,5):
            raise ValueError("JE01_REAL_Q35_ROLE_UNKNOWN_OR_DUPLICATE")
        for kind in ("packed","scale"):
            spec=entry["files"][kind]
            f=q3root/spec["filename"]
            if f.parent!=q3root or not f.is_file() or f.is_symlink() or f.stat().st_size!=spec["actual_bytes"]:
                raise ValueError("JE01_Q35_INPUT_REAL_FILE_INVALID")
            h=hashlib.sha256()
            with f.open("rb") as source_fd:
                for chunk in iter(lambda:source_fd.read(1<<20),b""):h.update(chunk)
            if h.hexdigest()!=spec["sha256"]:raise ValueError("JE01_Q35_INPUT_SHA_CHANGED")
        records[key]=entry
    if set(records)!={(l,e,role,bit) for l,e,role in allroles for bit in (3,5)}:
        raise ValueError("JE01_Q35_48_ROLE_TIER_COVERAGE_DRIFT")
    choice={}
    for policy in EXPECTED_POLICIES:
        rows=nig["synthetic_test_policy_results"][policy]["selected_n3_roles"]
        selected={tuple(row) for row in rows}
        if len(rows)!=6 or len(selected)!=6 or not selected.issubset(allroles):
            raise ValueError("JE01_ROLE_NIG_FROZEN_SELECTION_INVALID")
        choice[policy]=selected
    if 6*Q3_ROLE_BYTES+18*Q5_ROLE_BYTES!=BASE:
        raise ValueError("JE01_EQUAL_BYTE_ROLE_FORMULA_CHANGED")
    return frozen,records,choice

def build(root,nig_path,out):
    if not root.is_dir() or root.is_symlink() or not nig_path.is_file() or \
       out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError("JE01_ROLE_BUNDLE_INPUT_OUTPUT_PATH_INVALID")
    frozen,rows,choices=read_sources(root,nig_path)
    out.mkdir(mode=0o700)
    constructed=[]
    for policy in EXPECTED_POLICIES:
        lowered=choices[policy]
        path=out/(policy+".bin")
        sha=hashlib.sha256();segments=[]
        with path.open("xb") as dst:
            for expert in frozen:
                layer,e=expert["layer"],expert["expert"]
                for role in ROLES:
                    bit=3 if (layer,e,role) in lowered else 5
                    rec=rows[(layer,e,role,bit)]
                    if rec["physical_bytes"]!=(Q3_ROLE_BYTES if bit==3 else Q5_ROLE_BYTES):
                        raise ValueError("JE01_REAL_ROLE_PACKET_BYTES_DRIFT")
                    for kind in ("packed","scale"):
                        item=rec["files"][kind]
                        src=root/item["filename"]
                        begin=dst.tell()
                        with src.open("rb") as fd:
                            for chunk in iter(lambda:fd.read(1<<20),b""):
                                dst.write(chunk);sha.update(chunk)
                        end=dst.tell()
                        if end-begin!=item["actual_bytes"]:
                            raise ValueError("JE01_ROLE_BUNDLE_COPY_BYTES_CHANGED")
                        segments.append({"layer":layer,"expert":e,"role":role,"bits":bit,
                                         "kind":kind,"offsets":[begin,end],
                                         "source_file_sha256":item["sha256"]})
            dst.flush();os.fsync(dst.fileno())
        if len(segments)!=48 or path.stat().st_size!=BASE:
            raise ValueError("JE01_ROLE_NIG_ACTUAL_43MB_BUDGET_FAIL")
        constructed.append({"policy":policy,"file":path.name,
                            "sha256":sha.hexdigest(),"physical_bytes":path.stat().st_size,
                            "3bit_role_count":len(lowered),"5bit_role_count":18,
                            "3bit_role_ids":[list(k) for k in sorted(lowered)],
                            "source_role_regions":segments})
        print(json.dumps({"policy":policy,"sha256":sha.hexdigest(),
                          "bytes":BASE,"three_bit_roles":sorted(lowered)}),flush=True)
    receipt={"schema":SCHEMA,"status":"FOUR_NIG_AND_CONTROL_REAL_BF16_43MB_ROLE_LEVEL_BUNDLES_PASS",
             "frozen_source_BF16_identity_sha256":SOURCE_MODEL,
             "frozen_real_q3q5_source_receipt_sha256":SOURCE_Q35,
             "frozen_synthetic_NIG_selection_receipt_sha256":SOURCE_NIG,
             "same_physical_bytes_per_policy":BASE,
             "total_policies":len(constructed),"bundles":constructed,
             "original_quantized_AF_q4_byte_identity":"NOT_VERIFIED",
             "real_native_full_gold_NLL":"NOT_MEASURED",
             "real_full_27layer_model_forward":"NOT_RUN",
             "heldout_WikiText128_or_HellaSwag128_labels_accessed":False,
             "bayesian_model_NLL_improvement_claim":False,
             "production_modified":False}
    receipt["receipt_sha256"]=canonical(receipt)
    (out/"JE01_REAL_Q35_FOUR_ROLE_POLICIES_PHYSICAL_RECEIPT.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return receipt

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    ap.add_argument("--nig-receipt",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args()
    try:
        res=build(a.source,a.nig_receipt,a.output)
        print(json.dumps({"status":res["status"],"receipt_sha256":res["receipt_sha256"],
                          "policy_count":len(res["bundles"])}))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:260]}))
        raise SystemExit(2)
