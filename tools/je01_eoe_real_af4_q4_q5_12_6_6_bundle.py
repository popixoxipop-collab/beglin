#!/usr/bin/env python3
"""JE01 actual original AF4 pass-through (top4) plus original-BF16 Q4/Q5 (low4).

Writes one sealed 43,253,760-byte 8-expert/24-role alternative with:
  12 original AFFINE_AF4 (packed nibble + signed f32 scales + real f32 bias);
   6 original BF16-derived Q4_SYM (packed nibble + f32 scales);
   6 original BF16-derived Q5_SYM (packed bitplanes + f32 scales).
The original 27-layer serving model keeps the four AF4 important Experts
UNMODIFIED and the research overlay binds only the other 4 experts. The
12 AF4 roles are physically included for exact *logical* budget audit, not
registered twice. No label lookups, model forward or production writes.
"""
from __future__ import annotations
import argparse,hashlib,json,math,struct
from pathlib import Path

SCHEMA="beglin-je01-original27-select4-affineAF4-q4sym-q5sym-equalbytes/1"
BUDGET=43_253_760
ROOT=Path("/Volumes/D50/TailnetCommander/work")
AF_ROOT=ROOT/"hcp-p18-external-multimetric-20261015/native_moe_base_source_copy/weights_moe"
Q45_ROOT=ROOT/"je01_eoe_real_q45_gold16_20261122_v1"
NIG_ROOT=ROOT/"je01_eoe_real_nig_gold16_20261122_v1"
AF_SHA="9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"
AF_LAYOUT_SHA="ba3898f3bf366b1aec8471d9896885f101bf64b78c816828ea82bd7d58851c3e"
Q45_FILE_SHA="b4bd15db43f60f5aeb3ae8d6b9af148fa38545e5289ac049cd26a57a82932e6b"
Q45_STAGE_SHA="8ef0ea13e92e0fd188e65f085cc8cc4f0c39d541cd8cf8bb1804f1cfa6a7ede0"
NIG_FILE_SHA="dab4f50b6233688d789facf37a7fc94e22b0f4aa33298c36e5f69d89063db8cd"
NIG_STAGE_SHA="736ca28a433a30bfba31e3d09720d07a91ce5e72b4aa9edcc2f0a8a9c60bc5ea"
ROLES=("gate_proj","up_proj","down_proj")
SHAPES={"gate_proj":(1408,2048),"up_proj":(1408,2048),"down_proj":(2048,1408)}
AF4_KEEP={(21,61),(24,26),(13,28),(20,17)}
Q4_SYM={(13,22),(24,60)}
Q5_SYM={(12,13),(14,63)}
FROZEN=AF4_KEEP|Q4_SYM|Q5_SYM
COSTS={"AF4":1_802_240,"Q4":1_622_016,"Q5":1_982_464}

def sha_file(path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(2<<20),b""):digest.update(part)
    return digest.hexdigest()
def sha_blob(raw):return hashlib.sha256(raw).hexdigest()
def canonical(doc):
    return hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def signed(path,expected):
    if not path.is_file() or path.is_symlink():raise ValueError("JE01_SOURCE_STAGE_MISSING:"+path.name)
    d=json.loads(path.read_text());s=d.pop("receipt_sha256",None)
    if s!=expected or canonical(d)!=expected:
        raise ValueError("JE01_SOURCE_STAGE_CANONICAL_SIGNATURE_MISMATCH:"+path.name)
    return d

def source_region_index(stage,root,kind,expected_stage,expected_file):
    d=signed(root/stage,expected_stage)
    if d["production_modified"] or d["full_43MB_byte_size"]!=BUDGET:
        raise ValueError("JE01_ORIGINAL_SOURCE_PREEXISTING_STAGE_INVALID")
    file=Path(d["bundle_path"]);tsv=Path(d["tsv_path"])
    allowed=(Q45_ROOT/"weights"/"je01_real_q45" if kind=="q45" else NIG_ROOT/"weights_single")
    if file.is_symlink() or not file.is_file() or file.resolve().parent!=allowed.resolve() or \
       file.stat().st_size!=BUDGET or sha_file(file)!=expected_file or \
       tsv.is_symlink() or not tsv.is_file() or tsv.parent!=file.parent and kind=="q45":
        raise ValueError("JE01_REAL_SOURCE_BUNDLE_REGION_SHA_FAIL:"+kind)
    if kind=="nig" and tsv.parent!=NIG_ROOT/"tsv":
        raise ValueError("JE01_NIG_REGION_TSV_NOT_IN_SOURCE_ROOT")
    if (d["source_Q45_SHA256"] if kind=="q45" else d["source_NIG_SHA256"])!=expected_file:
        raise ValueError("JE01_SOURCE_STAGE_NIG_Q45_CHECKPOINT_ID_CHANGED")
    regions={}
    for line in tsv.read_text().splitlines():
        t=line.split("\t")
        if len(t)!=10:raise ValueError("JE01_SOURCE_NATIVE_ROLE_FIELD_COUNT_WRONG")
        l,e,role,bits,out,inn,pb,pe,sb,se=t
        l,e,bits,out,inn,pb,pe,sb,se=map(int,(l,e,bits,out,inn,pb,pe,sb,se))
        key=(l,e,role)
        if key in regions or (l,e) not in FROZEN or role not in ROLES or \
           (out,inn)!=SHAPES[role] or bits not in (3,4,5) or \
           not 0<=pb<pe<=BUDGET or not 0<=sb<se<=BUDGET or \
           pe-pb!=out*inn*bits//8 or se-sb!=out*(inn//64)*4:
            raise ValueError("JE01_NATIVE_SOURCE_TSV_LAYOUT_MISMATCH:"+str(key))
        regions[key]={"bits":bits,"packed":(pb,pe),"scale":(sb,se)}
    if len(regions)!=24:raise ValueError("JE01_ORIGINAL_PREEXISTING_SOURCE_24_ROLES_MISSING")
    return file,regions

def original_AF_layout():
    lp=AF_ROOT/"layout_af.txt";ap=AF_ROOT/"deepseek_moe_af.bin"
    if lp.is_symlink() or ap.is_symlink() or not lp.is_file() or not ap.is_file() or \
       ap.stat().st_size!=9_814_343_680 or sha_file(lp)!=AF_LAYOUT_SHA or sha_file(ap)!=AF_SHA:
        raise ValueError("JE01_9POINT8GB_ORIGINAL_AF4_FULL_SHA_CHANGED")
    layout={}
    for raw in lp.read_text().splitlines():
        v=raw.split()
        if len(v)!=9 or v[0] in layout:
            raise ValueError("JE01_ORIGINAL_AF4_SOURCE_LAYOUT_INVALID")
        layout[v[0]]=list(map(int,v[1:]))
    return ap,layout

def af4_expert(f,layout,layer,expert,role):
    name=f"model.layers.{layer}.mlp.switch_mlp.{role}"
    if name not in layout:raise ValueError("JE01_ORIGINAL_AF4_ROLE_NOT_PRESENT")
    E,out,inn,ng,po,plen,so,bo=layout[name]
    if E!=64 or (out,inn)!=SHAPES[role] or ng!=inn//64 or plen!=E*out*inn//2:
        raise ValueError("JE01_ORIGINAL_AF4_ROLE_SHAPE_OR_GROUP_INVALID")
    nc=out*inn//2;ns=out*ng*4
    with f.open("rb") as inp:
        sections={}
        for rolepart,begin,count in (("packed",po+expert*nc,nc),
                                      ("scale",so+expert*ns,ns),
                                      ("bias",bo+expert*ns,ns)):
            if begin<0 or begin+count>f.stat().st_size:
                raise ValueError("JE01_ORIGINAL_SOURCE_REGION_OVERFLOW")
            inp.seek(begin);chunk=inp.read(count)
            if len(chunk)!=count:raise ValueError("JE01_ORIGINAL_AF4_SOURCE_REGION_SHORT")
            sections[rolepart]=chunk
    if sum(map(len,sections.values()))!=COSTS["AF4"]:
        raise ValueError("JE01_ORIGINAL_AF4_ROLE_PHYSICAL_BYTES_DRIFT")
    scales=sections["scale"];bias=sections["bias"]
    if not all(math.isfinite(x[0]) for x in struct.iter_unpack("<f",scales+bias)):
        raise ValueError("JE01_ORIGINAL_AF4_SCALES_OR_BIASES_NONFINITE")
    return sections

def run(dest):
    if dest.exists() or dest.is_symlink() or dest.parent!=ROOT:
        raise ValueError("JE01_NEW_D50_RESEARCH_OUTPUT_ONLY")
    if len(FROZEN)!=8 or len(AF4_KEEP)!=4 or len(Q4_SYM)!=2 or len(Q5_SYM)!=2 or \
       12*COSTS["AF4"]+6*COSTS["Q4"]+6*COSTS["Q5"]!=BUDGET:
        raise ValueError("JE01_FROZEN_AFFINE_Q4_Q5_ROLES_OR_EXACT_BYTE_MATH_INVALID")
    original,layout=original_AF_layout()
    q45file,q45regs=source_region_index("JE01_REAL_Q45_EOE_SOURCE_STAGE.json",
              Q45_ROOT,"q45",Q45_STAGE_SHA,Q45_FILE_SHA)
    nigfile,nigregs=source_region_index("JE01_REAL_NIG_EOE_SOURCE_STAGE.json",
              NIG_ROOT,"nig",NIG_STAGE_SHA,NIG_FILE_SHA)
    dest.mkdir(mode=0o700)
    bundle=dest/"JE01_REAL_AF4_Q4_Q5_12_6_6_EXACT_BYTE.bin"
    tsv=dest/"JE01_REAL_AF4_Q4_Q5_24_ROLES.tsv"
    records=[];spans=[];n={"AF4":0,"Q4":0,"Q5":0}
    with bundle.open("xb") as out, q45file.open("rb") as q45, nigfile.open("rb") as nig:
        hasher=hashlib.sha256()
        for layer,expert in sorted(FROZEN):
            codec="AF4" if (layer,expert) in AF4_KEEP else ("Q4" if (layer,expert) in Q4_SYM else "Q5")
            for role in ROLES:
                shape=SHAPES[role]
                if codec=="AF4":
                    blocks=af4_expert(original,layout,layer,expert,role)
                else:
                    if codec=="Q4":spec=q45regs[(layer,expert,role)];source=q45
                    elif (layer,expert)==(12,13):spec=q45regs[(layer,expert,role)];source=q45
                    else:spec=nigregs[(layer,expert,role)];source=nig
                    if spec["bits"]!=(4 if codec=="Q4" else 5):
                        raise ValueError("JE01_EXPECTED_REAL_Q4_Q5_VERIFIED_SOURCE_BIT_NOT_PRESENT")
                    blocks={}
                    for kind in ("packed","scale"):
                        start,end=spec[kind]
                        source.seek(start);blocks[kind]=source.read(end-start)
                        if len(blocks[kind])!=end-start:
                            raise ValueError("JE01_Q45_NIG_VERIFIED_ROLE_SHORT")
                segs={}
                for kind in ("packed","scale","bias"):
                    if kind not in blocks:continue
                    blob=blocks[kind]
                    begin=out.tell();out.write(blob);hasher.update(blob);end=out.tell()
                    segs[kind]=(begin,end)
                    spans.append((begin,end))
                if sum(end-beg for beg,end in segs.values())!=COSTS[codec]:
                    raise ValueError("JE01_ACTUAL_12_6_6_SOURCE_ROLE_BYTE_DRIFT")
                p=segs["packed"];s=segs["scale"];b=segs.get("bias",(0,0))
                n[codec]+=1
                records.append({"layer":layer,"expert":expert,"role":role,"codec":codec,
                  "bits":4 if codec!="Q5" else 5,"out":shape[0],"in":shape[1],
                  "packed":list(p),"scale":list(s),"bias":list(b),
                  "physical_bytes":COSTS[codec],
                  "source_sha256":{kind:sha_blob(blocks[kind]) for kind in blocks},
                  "original_AF4_routed_calls":{(21,61):97,(24,26):91,(13,28):87,
                      (20,17):86,(12,13):81,(24,60):81,(14,63):80,(13,22):79}[(layer,expert)]})
            print(json.dumps({"completed_layer":layer,"expert":expert,"codec":codec,"finished_roles":len(records)},
                             sort_keys=True),flush=True)
        out.flush()
        import os
        os.fsync(out.fileno())
    spans.sort()
    if n!={"AF4":12,"Q4":6,"Q5":6} or len(records)!=24 or len(spans)!=60 or \
       spans[0][0]!=0 or spans[-1][1]!=BUDGET or \
       any(spans[i][1]!=spans[i+1][0] for i in range(len(spans)-1)) or \
       bundle.stat().st_size!=BUDGET or sha_file(bundle)!=hasher.hexdigest():
        raise ValueError("JE01_AF4_12_Q4_6_Q5_6_REAL_PHYSICAL_FILE_BYTES_OR_SOURCE_GAP_DRIFT")
    lines=[]
    for v in records:
        p=v["packed"];s=v["scale"];b=v["bias"]
        lines.append("\t".join(str(x)for x in (
          v["layer"],v["expert"],v["role"],v["bits"],v["out"],v["in"],
          *p,*s,*b,v["codec"])))
    tsv.write_text("\n".join(lines)+"\n")
    doc={"schema":SCHEMA,"status":"REAL_ORIGINAL_AF4_12_ROLE_PASS_THROUGH_Q4_6_Q5_6_PRETRAINED_EXACT43MB_CREATED",
         "scientific_scope":"POST_Q45_AND_Q35_TRAIN16_EXPLORATORY_PILOT_NO_HOLDOUT",
         "frozen_important_original_affine_AF4_experts":[list(x) for x in sorted(AF4_KEEP)],
         "frozen_symmetric_q4_experts":[list(x) for x in sorted(Q4_SYM)],
         "frozen_symmetric_q5_experts":[list(x) for x in sorted(Q5_SYM)],
         "original_full_AF4_source_SHA256":AF_SHA,
         "original_full_AF4_layout_SHA256":AF_LAYOUT_SHA,
         "real_prior_Q45_SHA256":Q45_FILE_SHA,"real_prior_NIG_SHA256":NIG_FILE_SHA,
         "AF4_experts_keep_original_native_64way_router_unchanged":True,
         "source_AF4_pass_through_roles_physically_included_but_not_rebound_in_experiment":True,
         "logical_eight_expert_original_AF4_budget_bytes":BUDGET,
         "candidate_full_file_bytes":bundle.stat().st_size,
         "candidate_full_file_sha256":hasher.hexdigest(),
         "tsv_sha256":sha_file(tsv),"role_count":len(records),
         "AF4_preserved_role_count":n["AF4"],"Q4_symmetric_role_count":n["Q4"],
         "Q5_symmetric_role_count":n["Q5"],"physical_segments":len(spans),
         "per_role":records,"real_27layer_fullgold_NLL":"NOT_MEASURED",
         "heldout_OOD_labels_accessed":False,"production_modified":False}
    doc["receipt_sha256"]=canonical(doc)
    (dest/"JE01_AF4_Q4_Q5_EXACTBYTE_RECEIPT.json").write_text(
      json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return doc

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        d=run(args.output)
        print(json.dumps({"status":d["status"],"sha256":d["candidate_full_file_sha256"],
          "receipt_sha256":d["receipt_sha256"],"bytes":d["candidate_full_file_bytes"],
          "AF4":d["AF4_preserved_role_count"],"Q4":d["Q4_symmetric_role_count"],
          "Q5":d["Q5_symmetric_role_count"]},sort_keys=True),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error_type":type(e).__name__,
             "error":str(e)[:400]},sort_keys=True),flush=True)
        raise SystemExit(2)
