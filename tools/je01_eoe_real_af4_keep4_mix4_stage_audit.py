#!/usr/bin/env python3
"""Independent EOE read-only original AF4 keep4 + real BF16 Q45 mix4 stage audit."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

ROOT=Path("/Volumes/D50/TailnetCommander/work/je01_eoe_real_af4_q4q5_1266_20261122_v1")
BASE=Path("/Volumes/D50/TailnetCommander/work")
SOURCE_SHA="24cb38797cddcf6e6a3363df54a24d5d4665a39a3be03739eadbf837830e291a"
SOURCE_RECEIPT_SHA="4f3fa3ebc443bd9974b53f4ed2f804a9c31994e316fa59d099c9955e167f767b"
BINARY_SHA="856749a39a393a9a2aff011a5ac590a1a82382a3bea209a75839b10cd4eb9a15"
AF_SHA="9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
BUDGET=43253760
Q4={(13,22),(24,60)}
Q5={(12,13),(14,63)}
AF={(13,28),(20,17),(21,61),(24,26)}
ROLES=("gate_proj","up_proj","down_proj")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(2<<20),b""):h.update(buf)
    return h.hexdigest()
def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def run():
    if not ROOT.is_dir() or ROOT.is_symlink():
        raise ValueError("JE01_AFFINE_MIX4_SOURCE_ROOT_MISSING")
    report=ROOT/"JE01_AF4_Q4_Q5_EXACTBYTE_RECEIPT.json"
    if report.is_symlink() or not report.is_file():
        raise ValueError("JE01_AFFINE_MIX4_SOURCE_RECEIPT_MISSING")
    r=json.loads(report.read_text())
    claim=r.pop("receipt_sha256",None)
    if claim!=SOURCE_RECEIPT_SHA or canonical(r)!=SOURCE_RECEIPT_SHA or \
       r["candidate_full_file_sha256"]!=SOURCE_SHA or \
       r["candidate_full_file_bytes"]!=BUDGET or r["physical_segments"]!=60 or \
       r["AF4_preserved_role_count"]!=12 or r["Q4_symmetric_role_count"]!=6 or \
       r["Q5_symmetric_role_count"]!=6 or \
       not r["source_AF4_pass_through_roles_physically_included_but_not_rebound_in_experiment"] or \
       not r["AF4_experts_keep_original_native_64way_router_unchanged"] or \
       r["production_modified"] or r["heldout_OOD_labels_accessed"]:
        raise ValueError("JE01_REAL_SOURCE_1266_CANONICAL_RECEIPT_CHANGED")
    bundle=ROOT/"JE01_REAL_AF4_Q4_Q5_12_6_6_EXACT_BYTE.bin"
    tsv=ROOT/"JE01_REAL_AF4_Q4_Q5_24_ROLES.tsv"
    exe=ROOT/"bin"/"je01_eoe_original384_affine_keep4_mix4_fullgold"
    binr=ROOT/"bin"/"JE01_ORIGINAL384_AF4_KEEP4_Q4Q5_NATIVE_BUILD_RECEIPT.json"
    if any(x.is_symlink() or not x.is_file() for x in (bundle,tsv,exe,binr)) or \
       bundle.stat().st_size!=BUDGET or sha(bundle)!=SOURCE_SHA or \
       sha(tsv)!=r["tsv_sha256"] or sha(exe)!=BINARY_SHA:
        raise ValueError("JE01_MIX4_NATIVE_PHYSICAL_CHECKPOINT_OR_BINARY_SHA_DRIFT")
    b=json.loads(binr.read_text())
    if b["actual_executable_sha256"]!=BINARY_SHA or \
       b["status"]!="JE01_REAL_ORIGINAL384_KEEP4_AFFINE4_MIX4_TRUE_GOLD_NATIVE_COMPILED" or \
       b["source_patch"]["status"]!="JE01_ORIGINAL27_AFFINE4_KEEP4_Q4Q5_MIX4_NATIVE_CODE_STAGED" or \
       b["source_patch"]["registered_experts"]!=4 or \
       b["source_patch"]["native_original_AF4_preserved_experts"]!=4 or \
       b["production_modified"] or b["model_gold_NLL"]!="NOT_MEASURED":
        raise ValueError("JE01_REAL_MIX4_NATIVE_BUILD_SOURCE_RECEIPT_CHANGED")
    # Independently ensure original AF4 file digest and every physical
    # affine passthrough role's packed, scale and bias equal native source.
    original=BASE/"hcp-p18-external-multimetric-20261015/native_moe_base_source_copy/weights_moe/deepseek_moe_af.bin"
    layoutfile=original.parent/"layout_af.txt"
    if not original.is_file() or original.is_symlink() or \
       original.stat().st_size!=9814343680 or sha(original)!=AF_SHA or \
       sha(layoutfile)!="ba3898f3bf366b1aec8471d9896885f101bf64b78c816828ea82bd7d58851c3e":
        raise ValueError("JE01_REAL_AF4_ORIGINAL9GB_SHA_CHANGED")
    layout={}
    for line in layoutfile.read_text().splitlines():
        f=line.split()
        if len(f)!=9:raise ValueError("JE01_NATIVE_ORIGINAL_LAYOUT_FIELD_COUNT")
        layout[f[0]]=list(map(int,f[1:]))
    rows=[];spans=[]
    source=ROOT/"JE01_REAL_AF4_Q4_Q5_12_6_6_EXACT_BYTE.bin"
    with original.open("rb") as orig,source.open("rb") as candidate:
        for line in tsv.read_text().splitlines():
            t=line.split("\t")
            if len(t)!=13:raise ValueError("JE01_SOURCE_TSV_13_FIELDS_INVALID")
            layer,expert,role,bits,out,inn,pb,pe,sb,se,bb,be,codec=t
            layer,expert,bits,out,inn,pb,pe,sb,se,bb,be=map(int,(layer,expert,bits,out,inn,pb,pe,sb,se,bb,be))
            ident=(layer,expert)
            if role not in ROLES or ident not in AF|Q4|Q5 or \
               (out,inn)!=((2048,1408) if role=="down_proj" else (1408,2048)) or \
               (ident in AF and (codec!="AF4" or bits!=4)) or \
               (ident in Q4 and (codec!="Q4" or bits!=4)) or \
               (ident in Q5 and (codec!="Q5" or bits!=5)):
                raise ValueError("JE01_FROZEN_ORIGINAL_AF4_AND_Q45_ROLE_ASSIGNMENT_CHANGED")
            if not 0<=pb<pe<=BUDGET or not 0<=sb<se<=BUDGET or \
               pe-pb!=out*inn*bits//8 or se-sb!=out*(inn//64)*4:
                raise ValueError("JE01_1266_PACKED_SCALE_SOURCE_RANGE_CHANGED")
            spans.extend(((pb,pe),(sb,se)))
            if ident in AF:
                if not 0<=bb<be<=BUDGET or be-bb!=out*(inn//64)*4:
                    raise ValueError("JE01_AFFINE_NATIVE_BIAS_REQUIRED")
                spans.append((bb,be))
                name=f"model.layers.{layer}.mlp.switch_mlp.{role}"
                E,rout,rin,ng,po,plen,so,bo=layout[name]
                if E!=64 or (rout,rin)!=(out,inn) or ng!=inn//64:
                    raise ValueError("JE01_ORIGINAL_AF4_ROLE_SOURCE_LAYOUT_DRIFT")
                packedlen=out*inn//2;groupbytes=out*ng*4
                for off1,off2,total in ((po+expert*packedlen,pb,packedlen),
                        (so+expert*groupbytes,sb,groupbytes),
                        (bo+expert*groupbytes,bb,groupbytes)):
                    orig.seek(off1);candidate.seek(off2)
                    if orig.read(total)!=candidate.read(total):
                        raise ValueError("JE01_ORIGINAL_AF4_PRESERVED_ROLE_BYTES_DO_NOT_MATCH")
            elif bb!=0 or be!=0:
                raise ValueError("JE01_NO_EXTRA_BIAS_FOR_NATIVE_SYMMETRIC_Q4_OR_Q5")
            rows.append((layer,expert,role,codec))
    spans.sort()
    if len(rows)!=24 or len(set(x[:3] for x in rows))!=24 or len(spans)!=60 or \
       spans[0][0]!=0 or spans[-1][1]!=BUDGET or \
       any(a[1]!=b[0] for a,b in zip(spans[:-1],spans[1:])):
        raise ValueError("JE01_TRUE_REAL_1266_PACKED_SCALE_BIAS_REGIONS_NOT_CONTIGUOUS")
    import os
    os.chmod(exe,0o700)
    stage={"schema":"beglin-je01-eoe-real-original-AF4-keep4-Q45mix4-native-stage/1",
      "status":"JE01_EOE_ORIGINAL_AF4_KEEP4_Q4Q5_MIX4_REAL_SOURCE_NATIVE_STAGE_PASS",
      "source_file_sha256":SOURCE_SHA,"source_receipt_sha256":SOURCE_RECEIPT_SHA,
      "native_executable_sha256":BINARY_SHA,"original_AF4_source_sha256":AF_SHA,
      "physical_budget_bytes":BUDGET,"roles":24,"physical_segments":60,
      "AF4_original_roles":12,"sym_Q4_roles":6,"sym_Q5_roles":6,
      "AF4_pass_through_original_bytes_independently_compared":True,
      "original_AF4_native_64way_GPU_paths_unchanged":True,
      "research_binary_path":str(exe),"research_bundle_path":str(bundle),
      "research_tsv_path":str(tsv),"research_tsv_sha256":r["tsv_sha256"],
      "full_27layer_NLL":"NOT_MEASURED","heldout_OOD_labels_accessed":False,
      "production_modified":False}
    stage["receipt_sha256"]=canonical(stage)
    p=ROOT/"JE01_EOE_ORIGINAL_AF4_KEEP4_Q45MIX4_STAGE.json"
    with p.open("x") as f:
        f.write(json.dumps(stage,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return stage

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    parser.parse_args()
    try:
        d=run()
        print(json.dumps({"status":d["status"],"receipt_sha256":d["receipt_sha256"],
             "physical_bytes":d["physical_budget_bytes"],"AF4_roles":d["AF4_original_roles"],
             "Q4_roles":d["sym_Q4_roles"],"Q5_roles":d["sym_Q5_roles"],
             "real_native_binary_sha":d["native_executable_sha256"]},sort_keys=True),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:400]},sort_keys=True),flush=True)
        raise SystemExit(2)
