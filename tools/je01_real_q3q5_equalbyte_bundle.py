#!/usr/bin/env python3
"""JE01: SHA-attested actual BF16 n3/n5 file bundles, 3×43,253,760B."""
import argparse,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_native_exact_byte_presolve import canonical,read_verified_ledger,validate_ledger
SOURCE="04ddca2f84367f41098357b5fae905e76c1207ef023d648a9f388aaddb3bb0ca"
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
BASE=43253760
ROLES=("gate_proj","up_proj","down_proj")
def build(source,dest):
    if not source.is_dir() or source.is_symlink() or dest.exists() or dest.is_symlink() or not dest.parent.is_dir():
        raise ValueError("JE01_BUNDLE_PATH_INVALID_OR_ALREADY_EXISTS")
    p=source/"JE01_ORIGINAL_BF16_E8_Q3Q5_PHYSICAL_RECEIPT.json"
    if not p.is_file() or p.is_symlink():raise ValueError("JE01_MISSING_PINNED_Q35_RECEIPT")
    meta=json.loads(p.read_text())
    sig=meta.pop("receipt_sha256",None)
    if sig!=SOURCE or canonical(meta)!=SOURCE or meta["actual_full8_physical_tier_bytes"]!={"3":30277632,"5":47579136}:
        raise ValueError("JE01_REAL_Q35_SOURCE_SHA_OR_SIZE_CHANGED")
    slots=validate_ledger(read_verified_ledger(LEDGER))
    valid={(s["layer"],s["expert"],r,b) for s in slots for r in ROLES for b in (3,5)}
    rows={}
    for rec in meta["records"]:
        k=(rec["layer"],rec["expert"],rec["role"],rec["bits"])
        if k not in valid or k in rows:raise ValueError("JE01_SOURCE_48_ROLE_COVERAGE_WRONG")
        for kind in ("packed","scale"):
            spec=rec["files"][kind]; f=source/spec["filename"]
            if f.parent!=source or not f.is_file() or f.is_symlink() or f.stat().st_size!=spec["actual_bytes"]:
                raise ValueError("JE01_ORIGINAL_PACKED_SOURCE_FILE_MISSING")
            if hashlib.sha256(f.read_bytes()).hexdigest()!=spec["sha256"]:
                raise ValueError("JE01_ORIGINAL_PACKED_SOURCE_FILE_SHA_CHANGED")
        rows[k]=rec
    if set(rows)!=valid:raise ValueError("JE01_Q3Q5_48_TIERS_INCOMPLETE")
    plans=meta["frozen_before_gold_three_candidate_assignments"]
    if set(plans)!={"route_frequency","BF16_weight_MSE_proxy","fixed_permutation"}:
        raise ValueError("JE01_LOCKED_MAPS_CHANGED")
    dest.mkdir(mode=0o700);bundles=[]
    for name,chosen in sorted(plans.items()):
        n3={tuple(x) for x in chosen}
        if len(n3)!=2:raise ValueError("JE01_EXPECT_EXACT_TWO_N3")
        target=dest/(name+".bin");h=hashlib.sha256();sections=[]
        with target.open("xb") as out:
            for slot in slots:
                l,e=slot["layer"],slot["expert"]
                bits=3 if (l,e) in n3 else 5
                for role in ROLES:
                    rec=rows[(l,e,role,bits)]
                    for kind in ("packed","scale"):
                        spec=rec["files"][kind];f=source/spec["filename"]
                        start=out.tell()
                        with f.open("rb") as fd:
                            for part in iter(lambda:fd.read(1<<20),b""):
                                out.write(part);h.update(part)
                        end=out.tell()
                        if end-start!=spec["actual_bytes"]:raise ValueError("JE01_SOURCE_REGION_LENGTH_DRIFT")
                        sections.append({"layer":l,"expert":e,"role":role,"bits":bits,
                                         "kind":kind,"offsets":[start,end],"sha":spec["sha256"]})
            out.flush();os.fsync(out.fileno())
        if target.stat().st_size!=BASE or len(sections)!=48:
            raise ValueError("JE01_EXACT_PHYSICAL_AF_BYTE_BUDGET_FAILED")
        bundles.append({"policy":name,"filename":target.name,"sha256":h.hexdigest(),
                        "real_physical_bytes":target.stat().st_size,
                        "n3_selected":[list(x) for x in sorted(n3)],"regions":sections})
        print(json.dumps({"policy":name,"real_physical_bytes":BASE,
                          "sha256":h.hexdigest()}),flush=True)
    doc={"schema":"beglin-je01-real-original-BF16-q3q5-3-physical-equalbyte-bundles/1",
         "status":"THREE_REAL_BF16_Q3Q5_EXACTBYTE_BUNDLES_PASS",
         "source_original_q3q5_receipt_sha":SOURCE,
         "exact_physical_bytes_per_bundle":BASE,"bundles":bundles,
         "original_AF4_weight_parity":"NOT_VERIFIED",
         "gold_token_NLL":"NOT_MEASURED","full_27layer_model":"NOT_RUN",
         "production_modified":False}
    doc["receipt_sha256"]=canonical(doc)
    (dest/"JE01_REAL_Q3Q5_EQUALBYTE_BUNDLE_RECEIPT.json").write_text(
        json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return doc
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        r=build(a.source,a.output)
        print(json.dumps({"status":r["status"],"sha":r["receipt_sha256"]}))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error":str(e)[:260]}))
        raise SystemExit(2)
