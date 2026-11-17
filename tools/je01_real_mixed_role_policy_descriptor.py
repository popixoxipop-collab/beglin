#!/usr/bin/env python3
"""JE01 43MB original BF16-source weight-only Q3Q5 role-policy Metal input.

Verifies exact sealed source and creates 24 role descriptors including
mixed bits inside ONE Expert (L24 E26 gate/up=3 down=5).
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_native_exact_byte_presolve import canonical,read_verified_ledger,validate_ledger
RECEIPT="8d3be1509b0484b9f97687c98d0eaceeea3a6e2bde836c94e0a30ac242898d58"
SOURCE_SHA="9f0475000b33dc6015405dc785f1277572089f9bad0f6d63481280748e8ad3d8"
POLICY="weight_MSE_proxy_6roles"
BASE=43253760
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"

def resolve(root):
    if not root.is_dir() or root.is_symlink():raise ValueError("JE01_ROLE_SOURCE_DIR_INVALID")
    p=root/"JE01_REAL_Q35_FOUR_ROLE_POLICIES_PHYSICAL_RECEIPT.json"
    if not p.is_file() or p.is_symlink():raise ValueError("JE01_SOURCE_RECEIPT_MISSING")
    d=json.loads(p.read_text());sig=d.pop("receipt_sha256",None)
    if sig!=RECEIPT or canonical(d)!=RECEIPT or \
       d.get("status")!="FOUR_NIG_AND_CONTROL_REAL_BF16_43MB_ROLE_LEVEL_BUNDLES_PASS":
        raise ValueError("JE01_SOURCE_FROZEN_RECEIPT_CHANGED")
    matches=[v for v in d["bundles"] if v["policy"]==POLICY]
    if len(matches)!=1 or matches[0]["sha256"]!=SOURCE_SHA:
        raise ValueError("JE01_WEIGHT_ROLE_POLICY_WRONG")
    selected=matches[0]
    file=root/selected["file"]
    if file.parent!=root or file.is_symlink() or not file.is_file() or file.stat().st_size!=BASE:
        raise ValueError("JE01_WEIGHT_SOURCE_FILE_SIZE_MISSING")
    if hashlib.sha256(file.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("JE01_WEIGHT_SOURCE_FULL_SHA_CHANGED")
    roles=validate_ledger(read_verified_ledger(LEDGER))
    expected={(s["layer"],s["expert"],role,kind) for s in roles
              for role in ("gate_proj","up_proj","down_proj") for kind in ("packed","scale")}
    spans={}
    for v in selected["source_role_regions"]:
        k=(v["layer"],v["expert"],v["role"],v["kind"])
        if k not in expected or k in spans or v["bits"] not in (3,5):
            raise ValueError("JE01_REAL_48_REGION_DUPLICATE_OR_UNKNOWN")
        a,b=v["offsets"]
        if not isinstance(a,int) or not isinstance(b,int) or not 0<=a<b<=BASE:
            raise ValueError("JE01_REGION_BYTES_OUT_OF_BOUNDS")
        spans[k]=v
    if set(spans)!=expected:raise ValueError("JE01_ALL_48_REAL_REGIONS_REQUIRED")
    lines=[]
    for slot in roles:
        l,e=slot["layer"],slot["expert"]
        for role in ("gate_proj","up_proj","down_proj"):
            packed=spans[(l,e,role,"packed")]
            scale=spans[(l,e,role,"scale")]
            bit=packed["bits"]
            if bit!=scale["bits"]:raise ValueError("JE01_MIXED_BITS_WITHIN_ROLE_NOT_SUPPORTED")
            out,inp=(2048,1408) if role=="down_proj" else (1408,2048)
            if packed["offsets"][1]-packed["offsets"][0]!=out*inp*bit//8 or \
               scale["offsets"][1]-scale["offsets"][0]!=out*(inp//64)*4:
                raise ValueError("JE01_REAL_WEIGHT_ROLE_PHYSICAL_BYTES_DRIFT")
            lines.append(f"{l}\t{e}\t{role}\t{bit}\t{out}\t{inp}\t{packed['offsets'][0]}\t{packed['offsets'][1]}\t{scale['offsets'][0]}\t{scale['offsets'][1]}")
    target=sorted((l,e,r,int(bit)) for l,e,r,bit,*_ in [x.split("\t") for x in lines] if int(l)==24 and int(e)==26)
    if target!=[(24,26,"down_proj",5),(24,26,"gate_proj",3),(24,26,"up_proj",3)]:
        raise ValueError("JE01_MIXED_REAL_E26_ROLE_PRESELECTION_DRIFT")
    return file,lines

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--tsv",type=Path,required=True)
    a=p.parse_args()
    try:
        if a.tsv.exists() or a.tsv.is_symlink():raise ValueError("JE01_OUTPUT_EXISTS")
        _,rows=resolve(a.source)
        a.tsv.write_text("\n".join(rows)+"\n")
        print(json.dumps({"status":"PASS_REAL_BF16_MIXED_ROLE_DESCRIPTOR","roles":len(rows),
                          "selected_layer":24,"selected_expert":26,"selected_role_bits":[3,3,5]}))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error":str(e)[:260]}))
        raise SystemExit(2)
