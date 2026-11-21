#!/usr/bin/env python3
"""JE01 real n2/6 versus n3/5, same new input and common real q8 reference.

Actual BF16-source 24 roles; no original AF4 NLL or model serving inference.
"""
import argparse,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_real_72_to_equalbyte_bundle import verify_artifacts,plans
from je01_native_exact_byte_presolve import read_verified_ledger,validate_ledger,canonical
from je01_selected_bf16_native_artifacts import SHAPES
OLD_REC="0ddc760bbc33777e1558fbe163847379a68bd3fe80f98d642846a66835a685ee"
NEW_REC="04ddca2f84367f41098357b5fae905e76c1207ef023d648a9f388aaddb3bb0ca"
MODEL="36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
SEED=20261120
CASES=12
ROLES=("gate_proj","up_proj","down_proj")
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"

def verify_new(root,slots,old):
    p=root/"JE01_ORIGINAL_BF16_E8_Q3Q5_PHYSICAL_RECEIPT.json"
    if p.is_symlink() or not p.is_file():raise ValueError("JE01_NEW_SOURCE_MANIFEST_MISSING")
    d=json.loads(p.read_text())
    sha=d.pop("receipt_sha256",None)
    if sha!=NEW_REC or canonical(d)!=NEW_REC or d["source_original_BF16_identity_SHA256"]!=MODEL:
        raise ValueError("JE01_NEW_SOURCE_FULL_SHA_NOT_PINNED")
    if d["status"]!="REAL_SOURCE_BF16_8EXPERT_48_NATIVE_Q3Q5_ROLE_TIERS_GENERATED" or len(d["records"])!=48:
        raise ValueError("JE01_NEW_48_ROLE_TIERS_NOT_COMPLETE")
    if d["actual_full8_physical_tier_bytes"]!={"3":30277632,"5":47579136}:
        raise ValueError("JE01_NEW_SOURCE_PHYSICAL_BYTES_DRIFT")
    expected={(s["layer"],s["expert"],r,b) for s in slots for r in ROLES for b in (3,5)}
    out={}
    for v in d["records"]:
        key=(v["layer"],v["expert"],v["role"],v["bits"])
        if key not in expected or key in out or v["BF16_tensor_sha256"]!=old[(key[0],key[1],key[2],2)]["source_exact_bf16_tensor_sha256"]:
            raise ValueError("JE01_NEW_ORIGINAL_SOURCE_TENSOR_SHA_MISMATCH")
        out[key]=v
    if set(out)!=expected:raise ValueError("JE01_NEW_TIERS_SOURCE_MISSING")
    return out,d["frozen_before_gold_three_candidate_assignments"]

def real_matrix(root,row,bit,is_new):
    sources=row["files"] if is_new else row
    bufs=[]
    for kind in ("packed","scale"):
        f=sources[kind];p=root/f["filename"]
        if p.parent!=root or not p.is_file() or p.is_symlink() or p.stat().st_size!=f["actual_bytes"]:
            raise ValueError("JE01_REAL_FILE_MISSING_OR_SIZE_CHANGED")
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=f["sha256"]:
            raise ValueError("JE01_REAL_FILE_SHA_MISMATCH")
        bufs.append(data)
    rows,cols=SHAPES[row["role"]]
    groups=cols//64
    s=np.frombuffer(bufs[1],dtype="<f4")
    if s.size!=rows*groups:raise ValueError("JE01_SCALE_LENGTH")
    if bit==8:
        c=np.frombuffer(bufs[0],dtype=np.int8)
        if c.size!=rows*cols:raise ValueError("JE01_Q8_SHAPE")
        codes=c.astype(np.float32).reshape(rows,groups,64)
    else:
        c=np.frombuffer(bufs[0],dtype=np.uint8)
        if c.size!=rows*cols*bit//8:raise ValueError("JE01_BITPLANE_SHAPE")
        planes=np.unpackbits(c.reshape(rows,groups,bit,8),axis=-1,bitorder="little")
        word=np.zeros((rows,groups,64),dtype=np.int16)
        for j in range(bit):word|=planes[:,:,j,:].astype(np.int16)<<j
        codes=(word-(1<<(bit-1))).astype(np.float32)
    return (codes*s.reshape(rows,groups,1)).reshape(rows,cols)

def ffn(x,t):
    g=x@t["gate_proj"].T
    u=x@t["up_proj"].T
    return (g/(1+np.exp(-np.clip(g,-40,40)))*u)@t["down_proj"].T

def run(oldroot,newroot):
    slots=validate_ledger(read_verified_ledger(LEDGER))
    oldslots,old=verify_artifacts(oldroot)
    if [(s["layer"],s["expert"]) for s in slots]!=[(s["layer"],s["expert"]) for s in oldslots]:
        raise ValueError("JE01_FROZEN_EIGHT_DRIFT")
    new,newmaps=verify_new(newroot,slots,old)
    oldmaps,_=plans(slots,old)
    rng=np.random.default_rng(SEED);by={}
    for s in slots:
        k=(s["layer"],s["expert"])
        x=rng.normal(size=(CASES,2048)).astype(np.float32)
        x/=np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-12)
        ys={}
        for bits in (2,3,5,6,8):
            roles={}
            for role in ROLES:
                source=old if bits in (2,6,8) else new
                root=oldroot if bits in (2,6,8) else newroot
                roles[role]=real_matrix(root,source[(*k,role,bits)],bits,bits in (3,5))
            ys[bits]=ffn(x,roles).astype(np.float64)
        ref=np.mean(ys[8]**2,axis=1)
        if not np.isfinite(ref).all() or np.any(ref<=1e-25):
            raise ValueError("JE01_Q8_REFERENCE_NONFINITE")
        by[k]={str(b):float(np.mean(np.mean((ys[b]-ys[8])**2,axis=1)/ref)) for b in (2,3,5,6)}
        if not np.isfinite(list(by[k].values())).all():raise ValueError("JE01_FFN_MSE_NAN")
        print(json.dumps({"layer":k[0],"expert":k[1],"MSE_vs_native_q8":by[k]}),flush=True)
    keys=list(by)
    def score(chosen,low,high):
        lowered=set(map(tuple,chosen))
        return sum(by[k][str(low if k in lowered else high)] for k in keys)/8
    old56=sorted((score(c,2,6),tuple(sorted(c))) for c in itertools.combinations(keys,3))
    new28=sorted((score(c,3,5),tuple(sorted(c))) for c in itertools.combinations(keys,2))
    def report(maps,low,high,grid):
        out={}
        for name,selection in maps.items():
            c=tuple(sorted(map(tuple,selection)))
            if c not in {x for _,x in grid}:raise ValueError("JE01_PREREG_POLICY_NOT_ON_GRID")
            value=score(c,low,high)
            out[name]={"low_bits_experts":[list(x) for x in c],"mean_FFN_relative_MSE_to_q8":value,
                       "posthoc_rank_within_equalbyte_tier":1+sum(v<value-1e-15 for v,_ in grid)}
        return out
    result={"schema":"beglin-je01-original-real-weight-FFN-q26-vs-q35-independent/1",
            "status":"REAL_ORIGINAL_BF16_Q26_Q35_FFN_COMMON_Q8_REFERENCE_PASS",
            "original_source_SHA256":MODEL,
            "common_reference":"ALL8_Q8_NATIVE_BF16_REQUANTIZED_NOT_ORIGINAL_AF4",
            "original_AF8_budget_bytes":43253760,
            "old_q2x3_q6x5":report(oldmaps,2,6,old56),
            "new_q3x2_q5x6":report(newmaps,3,5,new28),
            "independent_seed":SEED,"synthetic_activations_per_expert":CASES,
            "posthoc_min_old56":old56[0][0],"posthoc_min_new28":new28[0][0],
            "real_role_MSE":[{"layer":k[0],"expert":k[1],"by_bit":by[k]} for k in keys],
            "native_original_AF4_logits":"NOT_MEASURED",
            "real_gold_NLL":"NOT_MEASURED","native_27layer_forward":"NOT_RUN",
            "OOD_holdout_labels_accessed":False,"production_changed":False}
    result["receipt_sha256"]=canonical(result)
    return result

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--old",required=True,type=Path)
    a.add_argument("--new",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    x=a.parse_args()
    try:
        if x.output.exists():raise ValueError("JE01_OUTPUT_OVERWRITE_REFUSED")
        d=run(x.old,x.new)
        x.output.write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"status":d["status"],"old":d["old_q2x3_q6x5"],
                          "new":d["new_q3x2_q5x6"],"receipt_sha256":d["receipt_sha256"]},
                         sort_keys=True),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:300]}),flush=True)
        raise SystemExit(2)
