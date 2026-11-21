#!/usr/bin/env python3
"""JE01 original BF16 real q3/q5 exact-byte 24 role GPU input descriptor."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_native_exact_byte_presolve import canonical,read_verified_ledger,validate_ledger
RECEIPT="61d71b136ea0cce2618dff2d0fac112d55f9e3f3fcf1cd60f61b3a7e74343b6b"
BLOB_SHA="f0604982875e7bc38f64978332e1ab6fef5997c4dd7fedc6cc8ac57c35a1786e"
BUDGET=43253760
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
def verify(root):
    if not root.is_dir() or root.is_symlink():raise ValueError("JE01_Q35_REAL_FOLDER_MISSING")
    f=root/"JE01_REAL_Q3Q5_EQUALBYTE_BUNDLE_RECEIPT.json"
    if not f.is_file() or f.is_symlink():raise ValueError("JE01_Q35_ORIGINAL_RECEIPT_MISSING")
    meta=json.loads(f.read_text())
    got=meta.pop("receipt_sha256",None)
    if got!=RECEIPT or canonical(meta)!=RECEIPT:
        raise ValueError("JE01_Q35_REAL_ORIGINAL_RECEIPT_SHA_CHANGED")
    choices=[x for x in meta["bundles"] if x["policy"]=="route_frequency"]
    if len(choices)!=1 or choices[0]["sha256"]!=BLOB_SHA:
        raise ValueError("JE01_Q35_REAL_FREQUENCY_SOURCE_SHA_CHANGED")
    blob=choices[0]
    p=root/blob["filename"]
    if p.parent!=root or not p.is_file() or p.is_symlink() or p.stat().st_size!=BUDGET:
        raise ValueError("JE01_Q35_REAL_BLOB_PATH_OR_SIZE_INVALID")
    if hashlib.sha256(p.read_bytes()).hexdigest()!=BLOB_SHA:
        raise ValueError("JE01_Q35_REAL_BLOB_SHA_CHANGED")
    frozen=validate_ledger(read_verified_ledger(LEDGER))
    expected={(s["layer"],s["expert"],r,k) for s in frozen for r in ("gate_proj","up_proj","down_proj") for k in ("packed","scale")}
    spans={}
    for r in blob["regions"]:
        k=(r["layer"],r["expert"],r["role"],r["kind"])
        if k not in expected or k in spans or r["bits"] not in (3,5):
            raise ValueError("JE01_Q35_REGION_DUPLICATE_OR_UNKNOWN")
        a,b=r["offsets"]
        if not isinstance(a,int) or not isinstance(b,int) or not 0<=a<b<=BUDGET:
            raise ValueError("JE01_Q35_REGION_OUT_OF_BOUNDS")
        spans[k]=r
    if set(spans)!=expected:raise ValueError("JE01_Q35_48_REQUIRED_REGIONS_MISSING")
    records=[]
    for s in frozen:
        l,e=s["layer"],s["expert"]
        for role in ("gate_proj","up_proj","down_proj"):
            p=spans[(l,e,role,"packed")]
            sc=spans[(l,e,role,"scale")]
            if p["bits"]!=sc["bits"]:raise ValueError("JE01_Q35_MIXED_ROLE_BITS_FORBIDDEN")
            out,ins=(2048,1408) if role=="down_proj" else (1408,2048)
            bits=p["bits"]
            if p["offsets"][1]-p["offsets"][0]!=out*ins*bits//8 or \
               sc["offsets"][1]-sc["offsets"][0]!=out*(ins//64)*4:
                raise ValueError("JE01_Q35_ACTUAL_PACKED_LAYOUT_MISMATCH")
            records.append(f"{l}\t{e}\t{role}\t{bits}\t{out}\t{ins}\t{p['offsets'][0]}\t{p['offsets'][1]}\t{sc['offsets'][0]}\t{sc['offsets'][1]}")
    if len(records)!=24 or sum(int(x.split("\t")[3])==3 for x in records)!=6:
        raise ValueError("JE01_REAL_Q35_ROLE_BIT_COUNT")
    return records
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True,type=Path)
    p.add_argument("--tsv",required=True,type=Path)
    a=p.parse_args()
    try:
        if a.tsv.exists():raise ValueError("JE01_OUTPUT_ALREADY_EXISTS")
        rows=verify(a.source)
        a.tsv.write_text("\n".join(rows)+"\n")
        print(json.dumps({"status":"PASS_REAL_24_Q3Q5_DESCRIPTOR","rows":len(rows)}))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:260]}))
        raise SystemExit(2)
