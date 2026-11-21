#!/usr/bin/env python3
"""JE01 physically build REAL original Chat BF16 Q4/Q5 exact-byte 24-role challenger.

Q4 is original native gguf_quantize_q4g64_error_feedback, packed nibble + group
float32 scales (SYMMETRIC, implicit GPU bias=-8*scale). Q5 is original
SHA-attested qNg64 n5 bitplanes reused byte-exact from sealed real NIG bundle.
12 q4 + 12 q5 = 43,253,760 true packed+scale bytes (no bias file).
This 16-train-pilot-adapted candidate is EXPLORATORY, not preregistered gold.
No 128+128 heldout labels or model inference, production untouched.
"""
import argparse,ctypes,hashlib,importlib.util,json,math,os
from pathlib import Path
import numpy as np

SOURCE_AUDIT=Path("/Volumes/D50/TailnetCommander/work/je01_affine4_bf16_source_parity_20261122_v1/je01_original_af4_vs_chat_bf16_selected24_weight_audit_v2.py")
NIG_DIR=Path("/Volumes/D50/TailnetCommander/work/je01_eoe_real_nig_gold16_20261122_v1")
SOURCE_AUDIT_SHA="a16828e9b17db0d16d75946a54fdad943938398c"
SOURCE_AF_BF16_RECEIPT="be513c64e6af0a91613abdc34f1051a666023870561c143e4ad231e93a6e9233"
NIG_SRC_SHA="dab4f50b6233688d789facf37a7fc94e22b0f4aa33298c36e5f69d89063db8cd"
NIG_STAGE_SHA="736ca28a433a30bfba31e3d09720d07a91ce5e72b4aa9edcc2f0a8a9c60bc5ea"
C_SHA1="ff739d886f55b5a35843cf82fb55c9cd8f76104f"
H_SHA1="2f8767868cb5576044c1b20e174855ade13ecf33"
SCHEMA="beglin-je01-original-chat-BF16-real-Q4g64-Q5g64-12plus12-equalbyte-source/1"
BUDGET=43253760
Q4_PRICE=1622016
Q5_PRICE=1982464
FROZEN_Q4_EXPERTS={(13,28),(24,60),(13,22),(14,63)}
# Chosen after prior NIG train16 pilot; never claim confirmatory / preregistered!
FROZEN_ROUTE_COUNTS={(13,28):87,(24,60):81,(13,22):79,(14,63):80}
ROLES=("gate_proj","up_proj","down_proj")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):h.update(part)
    return h.hexdigest()
def canonical(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def selected_source():
    if not SOURCE_AUDIT.is_file() or SOURCE_AUDIT.is_symlink():
        raise ValueError("JE01_PINNED_ORIGINAL_AF_BF16_AUDIT_SCRIPT_MISSING")
    text=SOURCE_AUDIT.read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(text)).encode()+b"\0"+text).hexdigest()
    if blob!=SOURCE_AUDIT_SHA:
        raise ValueError("JE01_SEALED_BF16_SOURCE_AUDIT_SCRIPT_SHA_DRIFT")
    spec=importlib.util.spec_from_file_location("je01_q45_original_bf16_source",SOURCE_AUDIT)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
def verify_nig(mod):
    r=NIG_DIR/"JE01_REAL_NIG_EOE_SOURCE_STAGE.json"
    if not r.is_file() or r.is_symlink():raise ValueError("JE01_RESEARCH_NIG_SOURCE_STAGE_MISSING")
    data=json.loads(r.read_text())
    sig=data.pop("receipt_sha256",None)
    if sig!=NIG_STAGE_SHA or mod.canonical(data)!=NIG_STAGE_SHA or \
       data["status"]!="JE01_EOE_REAL_NIG8_SOURCE_AND_GOLD_NATIVE_BINARY_SHA_STAGED_PASS" or \
       data["source_NIG_SHA256"]!=NIG_SRC_SHA or \
       data["full_43MB_byte_size"]!=BUDGET or data["real_role_count"]!=24:
        raise ValueError("JE01_NIG_FULL_SOURCE_RECEIPT_DRIFT")
    p=Path(data["bundle_path"])
    t=Path(data["tsv_path"])
    if p.is_symlink() or not p.is_file() or p.resolve().parent!=NIG_DIR/"weights_single" or \
       p.stat().st_size!=BUDGET or sha(p)!=NIG_SRC_SHA:
        raise ValueError("JE01_NIG_REAL_PHYSICAL_BLOB_SHA_MISMATCH")
    if t.is_symlink() or not t.is_file() or t.resolve().parent!=NIG_DIR/"tsv":
        raise ValueError("JE01_NIG_FROZEN_24_ROLE_TSV_MISSING")
    rows={}
    for line in t.read_text().splitlines():
        a=line.split("\t")
        if len(a)!=10:raise ValueError("JE01_NIG_REAL_TSV_TEN_FIELDS")
        l,e,role,bits,out,inn,pb,pe,sb,se=a
        l,e,bits,out,inn,pb,pe,sb,se=map(int,(l,e,bits,out,inn,pb,pe,sb,se))
        key=l,e,role
        if key in rows or (l,e) not in mod.FROZEN_SLOTS or role not in ROLES:
            raise ValueError("JE01_NIG_ROLE_NOT_FROZEN_OR_DUPLICATE")
        if (out,inn)!=mod.SHAPES[role] or bits not in (3,5) or \
           not 0<=pb<pe<=BUDGET or not 0<=sb<se<=BUDGET or \
           pe-pb!=out*inn*bits//8 or se-sb!=out*(inn//64)*4:
            raise ValueError("JE01_REAL_NIG_24_ROLE_LAYOUT_DRIFT")
        rows[key]=(bits,out,inn,pb,pe,sb,se)
    expected={(l,e,r) for l,e in mod.FROZEN_SLOTS for r in ROLES}
    if set(rows)!=expected:raise ValueError("JE01_NIG_FROZEN_ROLE_IDS_CHANGED")
    if sum(x[0]==3 for x in rows.values())!=6:
        raise ValueError("JE01_NIG_SIX_LOWBIT_ROLES_DRIFT")
    return p.read_bytes(),rows

def quant4(lib,w):
    out,inn=w.shape
    p=np.zeros(w.size//2,dtype=np.uint8)
    s=np.zeros((out,inn//64),dtype=np.float32)
    lib.gguf_quantize_q4g64_error_feedback(
       w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),out,inn,
       ctypes.c_void_p(p.ctypes.data),s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if not np.isfinite(s).all() or np.any(s<=0) or w.size%64:
        raise ValueError("JE01_Q4G64_SCALES_INVALID")
    c=np.empty(p.size*2,dtype=np.int16)
    c[0::2]=(p&15).astype(np.int16)
    c[1::2]=(p>>4).astype(np.int16)
    y=((c.reshape(out,inn//64,64)-8)*s.reshape(out,inn//64,1)).reshape(w.shape)
    if not np.isfinite(y).all():raise ValueError("JE01_Q4G64_RECONSTRUCTION_NONFINITE")
    return p.tobytes(),s.tobytes(),y

def dequant5(blob,row):
    bit,out,inn,pb,pe,sb,se=row
    if bit!=5:raise ValueError("JE01_SELECTED_Q5_FROM_OLD_NIG_NOT_FIVE")
    groups=inn//64
    raw=np.frombuffer(blob,dtype=np.uint8,count=pe-pb,offset=pb)
    scales=np.frombuffer(blob,dtype="<f4",count=(se-sb)//4,offset=sb)
    if raw.size!=out*inn*5//8 or scales.size!=out*groups or \
       not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError("JE01_ORIGINAL_Q5_SOURCE_INVALID")
    planes=np.unpackbits(raw.reshape(out,groups,5,8),axis=-1,bitorder="little")
    codes=np.zeros((out,groups,64),dtype=np.int16)
    for i in range(5):codes|=planes[:,:,i,:].astype(np.int16)<<i
    y=((codes-16)*scales.reshape(out,groups,1)).reshape(out,inn)
    if not np.isfinite(y).all():raise ValueError("JE01_Q5_SOURCE_NAN")
    return blob[pb:pe],blob[sb:se],y

def relative_error(decoded,source):
    a=decoded.astype(np.float64)-source.astype(np.float64)
    b=source.astype(np.float64)
    energy=float(np.sum(b*b))
    if energy<=1e-20:raise ValueError("JE01_BF16_SOURCE_ZERO")
    return float(np.sum(a*a)/energy)

def run(out,codec_c,codec_h):
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError("JE01_OUTPUT_NEW_FOLDER_ONLY")
    if any(not f.is_file() or f.is_symlink() for f in (codec_c,codec_h)):
        raise ValueError("JE01_NATIVE_Q4_C_CODEC_SOURCES_NOT_FOUND")
    if hashlib.sha1(b"blob "+str(codec_c.stat().st_size).encode()+b"\0"+codec_c.read_bytes()).hexdigest()!=C_SHA1 or \
       hashlib.sha1(b"blob "+str(codec_h.stat().st_size).encode()+b"\0"+codec_h.read_bytes()).hexdigest()!=H_SHA1:
        raise ValueError("JE01_NATIVE_EXISTING_BEGLIN_Q4_C_CODEC_SHA_CHANGED")
    mod=selected_source()
    old=json.loads((Path("/Volumes/D50/TailnetCommander/work/je01_affine4_bf16_source_parity_20261122_v1")/"all24_roles.json").read_text())
    sig=old.pop("receipt_sha256",None)
    if sig!=SOURCE_AF_BF16_RECEIPT or mod.canonical(old)!=SOURCE_AF_BF16_RECEIPT or \
       old["role_comparison_count"]!=24 or old["compare_coefficients"]!=69206016:
        raise ValueError("JE01_BF16_ORIGINAL_AFFINE_NUMERIC_AUDIT_SOURCE_CHANGED")
    old_map={(x["layer"],x["expert"],x["role"]):x for x in old["per_role"]}
    data,old_roles=verify_nig(mod)
    if len(FROZEN_Q4_EXPERTS)!=4 or \
       not FROZEN_Q4_EXPERTS.issubset(set(mod.FROZEN_SLOTS)):
        raise ValueError("JE01_Q4_SELECTED_FOUR_EXPERTS_INVALID")
    if 12*Q4_PRICE+12*Q5_PRICE!=BUDGET:
        raise ValueError("JE01_EXACT_BYTES_FORMULA_FAIL")
    # SHA all four exact original BF16 safetensors 31GB before reading roles.
    wmap,shards,heads=mod.read_bf16_headers()
    out.mkdir(mode=0o700)
    so=out/"original_bf16_q45_native.dylib"
    import subprocess
    subprocess.run(["/usr/bin/clang","-O2","-std=c11","-fPIC",
       "-dynamiclib",str(codec_c),"-o",str(so),"-lm"],check=True,timeout=120)
    lib=ctypes.CDLL(str(so))
    lib.gguf_quantize_q4g64_error_feedback.argtypes=[
       ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int,
       ctypes.c_void_p,ctypes.POINTER(ctypes.c_float)]
    lib.gguf_quantize_q4g64_error_feedback.restype=None
    target=out/"JE01_REAL_BF16_Q4G64_Q5G64_EQUALBYTE.bin"
    h=hashlib.sha256();parts=[];role_results=[];q4=0;q5=0
    with target.open("xb") as file:
      for layer,expert in mod.FROZEN_SLOTS:
        for role in ROLES:
          key=layer,expert,role
          source,src=mod.bf16_selected(wmap,shards,heads,layer,expert,role)
          if src["original_bf16_tensor_sha256"]!=old_map[key]["original_bf16_tensor_sha256"]:
              raise ValueError("JE01_SAME_REAL_ORIGINAL_CHAT_BF16_TENSOR_SHA_DRIFT")
          bit=4 if (layer,expert) in FROZEN_Q4_EXPERTS else 5
          if bit==4:
              p,s,y=quant4(lib,source)
              q4+=1
          else:
              p,s,y=dequant5(data,old_roles[key])
              q5+=1
          expected_price=Q4_PRICE if bit==4 else Q5_PRICE
          if len(p)+len(s)!=expected_price:
              raise ValueError("JE01_SAVED_REAL_Q45_ROLE_COST_MISMATCH")
          for kind,blob in (("packed",p),("scale",s)):
              begin=file.tell();file.write(blob);h.update(blob);end=file.tell()
              parts.append({"layer":layer,"expert":expert,"role":role,
                  "bits":bit,"kind":kind,"offsets":[begin,end],
                  "source_region_sha256":hashlib.sha256(blob).hexdigest()})
          error=relative_error(y,source)
          role_results.append({"layer":layer,"expert":expert,"role":role,
              "bits":bit,"actual_physical_bytes":len(p)+len(s),
              "original_BF16_source_tensor_sha256":src["original_bf16_tensor_sha256"],
              "source_BF16_weight_relative_MSE":error,
              "original_affine_AF4_source_relative_RMSE":
                 old_map[key]["relative_weight_RMSE_AF4_vs_trueBF16"]})
        print(json.dumps({"source_verified_layer":layer,"expert":expert,
              "real_roles_generated":len(role_results)},sort_keys=True),flush=True)
      file.flush();os.fsync(file.fileno())
    if q4!=12 or q5!=12 or len(parts)!=48 or target.stat().st_size!=BUDGET:
        raise ValueError("JE01_Q4_Q5_ACTUAL_43MB_PHYSICAL_BUDGET_NOT_EXACT")
    # One full SHA from disk, independent of streaming digest.
    if sha(target)!=h.hexdigest():
        raise ValueError("JE01_Q45_PHYSICAL_FILE_REOPEN_SHA_CHANGED")
    tsv=out/"JE01_REAL_Q45_ORIGINAL_24_ROLES.tsv"
    rows=[]
    for layer,expert in mod.FROZEN_SLOTS:
      for role in ROLES:
        bit=4 if (layer,expert) in FROZEN_Q4_EXPERTS else 5
        shape=mod.SHAPES[role]
        matches=[z for z in parts if (z["layer"],z["expert"],z["role"],z["bits"])==(layer,expert,role,bit)]
        if len(matches)!=2:raise ValueError("JE01_Q45_ROLE_REGION_NOT_TWO")
        a=next(z["offsets"] for z in matches if z["kind"]=="packed")
        b=next(z["offsets"] for z in matches if z["kind"]=="scale")
        rows.append(f"{layer}\t{expert}\t{role}\t{bit}\t{shape[0]}\t{shape[1]}\t{a[0]}\t{a[1]}\t{b[0]}\t{b[1]}")
    tsv.write_text("\n".join(rows)+"\n")
    q4_source_error=float(np.mean([z["source_BF16_weight_relative_MSE"] for z in role_results if z["bits"]==4]))
    q5_source_error=float(np.mean([z["source_BF16_weight_relative_MSE"] for z in role_results if z["bits"]==5]))
    receipt={"schema":SCHEMA,"status":"ACTUAL_ORIGINAL_BF16_Q4G64_Q5G64_12_12_EXACTBYTE_FILE_AND_SHA_PASS",
       "status_scope":"POST_NIG_TRAIN16_EXPLORATORY_CANDIDATE_NOT_PRE_REGISTERED",
       "original_source_deepseek_chat_index_sha":mod.INDEX_SHA,
       "original_four_BF16_source_shard_fullSHA_reverified":True,
       "prior_signed_AF4_BF16_alignment_receipt":SOURCE_AF_BF16_RECEIPT,
       "actual_source_NIG_43MB_file_sha":NIG_SRC_SHA,
       "q4_codec":"existing_BEG_LIN_gguf_quantize_q4g64_error_feedback_symmetric_nibbles",
       "q5_codec":"existing_BEG_LIN_gguf_quantize_qNg64_5bit_bitplanes",
       "q4_requires_synthesized_GPU_bias_minus_8times_scale":True,
       "original_baseline_AF4_affine_bias_not_reused":True,
       "q4_expert_IDs":[list(x) for x in sorted(FROZEN_Q4_EXPERTS)],
       "q4_roles":q4,"q5_roles":q5,
       "actual_physical_file_bytes":target.stat().st_size,
       "actual_full_file_sha256":h.hexdigest(),
       "native_24_role_tsv_sha256":sha(tsv),"per_role":role_results,
       "full_48_physical_regions":parts,
       "q4_role_mean_source_relative_MSE":q4_source_error,
       "q5_role_mean_source_relative_MSE":q5_source_error,
       "original_27_layer_model_gold_NLL":"NOT_MEASURED",
       "original_replay_gold_heldout128_and_ood128_used":False,
       "operating_model_modified":False}
    receipt["receipt_sha256"]=canonical(receipt)
    (out/"JE01_REAL_BF16_Q45_12_12_RECEIPT.json").write_text(
      json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return receipt
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--codec-c",required=True,type=Path)
    ap.add_argument("--codec-h",required=True,type=Path)
    opts=ap.parse_args()
    try:
        r=run(opts.output,opts.codec_c,opts.codec_h)
        print(json.dumps({"status":r["status"],
             "real_sha256":r["actual_full_file_sha256"],
             "bytes":r["actual_physical_file_bytes"],
             "q4":r["q4_roles"],"q5":r["q5_roles"],
             "receipt_sha256":r["receipt_sha256"],
             "q4_source_MSE":r["q4_role_mean_source_relative_MSE"],
             "q5_source_MSE":r["q5_role_mean_source_relative_MSE"]},sort_keys=True),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","error":str(e)[:340],
               "exception":type(e).__name__},sort_keys=True),flush=True)
        raise SystemExit(2)
