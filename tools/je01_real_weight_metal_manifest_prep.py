#!/usr/bin/env python3
"""Proof-only immutable real BF16-tier 8-Expert GPU probe descriptor writer.

No re-quantizing, generation, model logits, or production file changes.
"""
import argparse,hashlib,json,pathlib,sys

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from je01_native_exact_byte_presolve import canonical,read_verified_ledger,validate_ledger
EXPECTED_RECEIPT="60c6b13650f64fba23fe077d14df97bcf076e119e788a6e31f3f2be266a1867f"
EXPECTED_BLOB_SHA="5cea38b1b525b1456b699a4b3a3c06cb46d97dc7faf37d7ac24ea55707762b49"
BASE_BYTES=43253760

def load(root):
    p=root/"JE01_REAL3_EQUALBYTE_BUNDLES_RECEIPT.json"
    if not p.is_file() or p.is_symlink():raise ValueError("JE01_REAL_BUNDLE_RECEIPT_MISSING")
    d=json.loads(p.read_text())
    asserted=d.pop("receipt_sha256",None)
    if asserted!=EXPECTED_RECEIPT or canonical(d)!=EXPECTED_RECEIPT:
        raise ValueError("JE01_REAL_BUNDLE_ORIGINAL_SHA_CHANGED")
    if d["status"]!="THREE_ACTUAL_8EXPERT_EQUALBYTE_43253760_PAYLOADS_CONSTRUCTED" or        d["case_specific_Bayesian_NLL_effects"]!="NOT_MEASURED":
        raise ValueError("JE01_REAL_BUNDLE_SCOPE_CHANGED")
    choices=[x for x in d["bundles"] if x["name"]=="frequency_prelabel"]
    if len(choices)!=1:raise ValueError("JE01_FREQUENCY_CHOICE_NOT_UNIQUE")
    c=choices[0]
    if c["file_sha256"]!=EXPECTED_BLOB_SHA or c["physical_binary_file_bytes"]!=BASE_BYTES:
        raise ValueError("JE01_REAL_BUNDLE_UNEXPECTED_SOURCE")
    raw=root/c["file_name"]
    if raw.parent!=root or not raw.is_file() or raw.is_symlink() or raw.stat().st_size!=BASE_BYTES:
        raise ValueError("JE01_REAL_BUNDLE_FILE_NOT_PRESENT")
    sha=hashlib.sha256(raw.read_bytes()).hexdigest()
    if sha!=EXPECTED_BLOB_SHA:raise ValueError("JE01_REAL_BUNDLE_PHYSICAL_SHA_DIFFERS")
    ledger=read_verified_ledger(pathlib.Path(__file__).resolve().parents[1]/
        "docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json")
    selected=validate_ledger(ledger)
    ids={(x["layer"],x["expert"]) for x in selected}
    if {(x["layer"],x["expert"]) for x in c["bit_assignment"]}!=ids:
        raise ValueError("JE01_ORIGINAL_SELECTED_EIGHT_DRIFT")
    bits={(x["layer"],x["expert"]):x["bits"] for x in c["bit_assignment"]}
    if len(bits)!=8 or sorted(bits.values())!=[2,2,2,6,6,6,6,6]:
        raise ValueError("JE01_REAL_BITS_NOT_2x3_6x5")
    parts={}
    for r in c["file_regions"]:
        key=(r["layer"],r["expert"],r["role"],r["kind"])
        if key in parts or r["role"] not in ("gate_proj","up_proj","down_proj") or            r["kind"] not in ("packed","scale") or (r["layer"],r["expert"]) not in ids or            r["bit"]!=bits[(r["layer"],r["expert"])]:
            raise ValueError("JE01_BUNDLE_REGION_MAP_INVALID")
        start,end=r["offsets"]
        if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=BASE_BYTES:
            raise ValueError("JE01_BUNDLE_REGION_OUT_OF_RANGE")
        parts[key]=(start,end)
    if len(parts)!=48:raise ValueError("JE01_8_EXPERT_3_ROLE_2_PARTS_INCOMPLETE")
    output=[]
    for s in selected:
        l,e=s["layer"],s["expert"]
        for role in ("gate_proj","up_proj","down_proj"):
            p=parts[(l,e,role,"packed")]
            sc=parts[(l,e,role,"scale")]
            out,in_dim=(2048,1408) if role=="down_proj" else (1408,2048)
            if p[1]-p[0]!=out*in_dim*bits[(l,e)]//8 or sc[1]-sc[0]!=out*(in_dim//64)*4:
                raise ValueError("JE01_REAL_NBIT_NATIVE_MATRIX_SHAPE_BYTES_DRIFT")
            output.append(f"{l}\t{e}\t{role}\t{bits[(l,e)]}\t{out}\t{in_dim}\t{p[0]}\t{p[1]}\t{sc[0]}\t{sc[1]}")
    assert len(output)==24
    return raw,output

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=pathlib.Path,required=True)
    p.add_argument("--tsv",type=pathlib.Path,required=True)
    a=p.parse_args()
    try:
        raw,rows=load(a.source)
        if a.tsv.exists() or a.tsv.is_symlink():raise ValueError("JE01_NO_OVERWRITE")
        a.tsv.write_text("\n".join(rows)+"\n")
        print(json.dumps({"source_blob":str(raw),"source_sha256":EXPECTED_BLOB_SHA,
                          "total_bytes":BASE_BYTES,"role_binding_rows":len(rows)}))
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:270]}))
        raise SystemExit(2)
