#!/usr/bin/env python3
"""Real 196 Qwen BF16 tensor q4g64 vs qNg64 n8 weight distortion audit.

Research-only, original BEGLIN encoder C via ctypes; no target/test labels.
Proxy metric is weight Frobenius relative error, NOT task sensitivity.
Select four source-only 8-bit upgrades out of 28 layers per fixed 7 roles.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_tensor_proxy_20261021"
SNAP=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
SRC=SNAP/"model.safetensors"
SOURCEREV="4ae77cb209f06199b8df1c94e21ff341332a3a89"
ROLES=(
("q_proj","self_attn.q_proj.weight"),
("k_proj","self_attn.k_proj.weight"),
("v_proj","self_attn.v_proj.weight"),
("o_proj","self_attn.o_proj.weight"),
("gate_proj","mlp.gate_proj.weight"),
("up_proj","mlp.up_proj.weight"),
("down_proj","mlp.down_proj.weight"),
)
NUM_LAYERS=28
UPGRADES_PER_ROLE=4
SEED=73

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

def verify_codecs():
    targets=["gguf_transcode.c","qng64_dense.c"]
    names=[ROOT/f for f in targets]
    for f in names:
        if not f.is_file():raise RuntimeError("source encoder missing "+str(f))
    so=LAB/"libbeglin_real_q4_qng64_n8.dylib"
    subprocess.run(["clang","-dynamiclib","-O3","-fPIC","-I",str(ROOT),
                    str(names[0]),str(names[1]),"-o",str(so),"-lm"],check=True)
    lib=ctypes.CDLL(str(so))
    lib.gguf_quantize_q4g64_error_feedback.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    lib.gguf_quantize_q4g64_error_feedback.restype=None
    lib.qng64_quantize_f32.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    lib.qng64_quantize_f32.restype=ctypes.c_int
    return lib,{f.name:sha(f.read_bytes()) for f in names}

def original_tensors():
    with SRC.open("rb") as fp:
        n=struct.unpack("<Q",fp.read(8))[0]
        if not 1024<=n<=2_000_000:raise RuntimeError("unexpected safetensors header size")
        hdr=json.loads(fp.read(n))
    names=[f"model.layers.{layer}.{role}" for role in [r[1] for r in ROLES] for layer in range(NUM_LAYERS)]
    if len(names)!=196 or len(set(names))!=196:raise RuntimeError("invalid source scope")
    for name in names:
        node=hdr.get(name)
        if node is None or node.get("dtype")!="BF16" or len(node.get("shape",[]))!=2:
            raise RuntimeError("source tensor missing/not BF16: "+name)
        if node["shape"][1]%64:raise RuntimeError("role matrix group alignment drift")
    return hdr,8+n

def decode_q4(packed,scales,out,dim):
    per=dim//64
    b=packed.reshape(out,per,32)
    codes=np.empty((out,per,64),dtype=np.int16)
    codes[:,:,0::2]=(b&15).astype(np.int16)-8
    codes[:,:,1::2]=(b>>4).astype(np.int16)-8
    return (codes.astype(np.float32)*scales.reshape(out,per,1)).reshape(out,dim)

def decode_n8(planes,scales,out,dim):
    per=dim//64
    b=planes.reshape(out,per,8,8)
    u=np.zeros((out,per,64),dtype=np.uint16)
    for j in range(8):
        bit=np.unpackbits(b[:,:,j,:],axis=-1,bitorder="little")
        u|=(bit.astype(np.uint16)<<j)
    return ((u.astype(np.int16)-128).astype(np.float32)*scales.reshape(out,per,1)).reshape(out,dim)

def float_from_bf16(mm,node,offset):
    st,en=node["data_offsets"]
    count=(en-st)//2
    shape=node["shape"]
    if en-st!=shape[0]*shape[1]*2:raise RuntimeError("bad BF16 element length")
    bits=np.frombuffer(mm,dtype="<u2",count=count,offset=offset+st)
    return ((bits.astype("<u4")<<16).view("<f4")).reshape(shape).copy()

def analyze_role(name,layer,node,mm,offset,lib):
    W=float_from_bf16(mm,node,offset)
    out,dim=W.shape
    ng=out*(dim//64)
    bfbytes=out*dim*2
    w4=np.empty((out*dim//2,),dtype=np.uint8)
    s4=np.empty(ng,dtype=np.float32)
    w8=np.empty((out*dim,),dtype=np.uint8)
    s8=np.empty(ng,dtype=np.float32)
    t=time.perf_counter()
    lib.gguf_quantize_q4g64_error_feedback(W.ctypes.data,out,dim,w4.ctypes.data,s4.ctypes.data)
    t4=time.perf_counter()-t
    t=time.perf_counter()
    rc=lib.qng64_quantize_f32(W.ctypes.data,out,dim,8,w8.ctypes.data,s8.ctypes.data)
    t8=time.perf_counter()-t
    if rc!=0:raise RuntimeError("native n8 encoder rejected "+name)
    decoded4=decode_q4(w4,s4,out,dim)
    decoded8=decode_n8(w8,s8,out,dim)
    f32=W.astype(np.float64)
    denom=float(np.square(f32).sum())
    e4=float(np.square(decoded4.astype(np.float64)-f32).sum())
    e8=float(np.square(decoded8.astype(np.float64)-f32).sum())
    if denom<=1e-10 or e4<0 or e8<0 or not np.isfinite([denom,e4,e8]).all():
        raise RuntimeError("bad numeric distortions")
    row={
      "role":name,"layer":layer,"weight_shape":[out,dim],"parameter_count":out*dim,
      "bf16_weight_bytes":bfbytes,"q4_with_scales_bytes":len(w4)+len(s4.tobytes()),
      "qng64_8_with_scales_bytes":len(w8)+len(s8.tobytes()),
      "q4_relative_l2":float(np.sqrt(e4/denom)),
      "qng64_8_relative_l2":float(np.sqrt(e8/denom)),
      "squared_normalized_error_reduction_q4_minus_n8":float((e4-e8)/denom),
      "encoder_q4_ms":round(t4*1000,3),"encoder_q8_ms":round(t8*1000,3),
    }
    if name=="q_proj" and layer==0:
        # Exact codec identity to the previous independent real tensor probe.
        row["q4_codes_sha256"]=sha(w4.tobytes())
        row["qng64_n8_planes_sha256"]=sha(w8.tobytes())
        row["actual_codec_source_identity"]="checked_at_scan"
    return row

def policy_candidates(rows):
    policies={}
    rng=np.random.default_rng(SEED)
    for role,_ in ROLES:
        subset=[r for r in rows if r["role"]==role]
        if len(subset)!=NUM_LAYERS:raise RuntimeError("need 28 layers per role")
        ranking=sorted(subset,key=lambda r:(-r["squared_normalized_error_reduction_q4_minus_n8"],r["layer"]))
        policies[role]={
         "proxy_n8_layers":sorted(r["layer"] for r in ranking[:UPGRADES_PER_ROLE]),
         "seed73_random_n8_layers":sorted(int(t) for t in rng.choice(NUM_LAYERS,UPGRADES_PER_ROLE,replace=False)),
         "min_proxy_improvement":ranking[UPGRADES_PER_ROLE-1]["squared_normalized_error_reduction_q4_minus_n8"],
         "max_proxy_improvement":ranking[0]["squared_normalized_error_reduction_q4_minus_n8"],
        }
    keys={"proxy":set(),"random":set()}
    for role,d in policies.items():
        for label,layers in [("proxy",d["proxy_n8_layers"]),("random",d["seed73_random_n8_layers"])]:
            for layer in layers:keys[label].add((role,layer))
    if len(keys["proxy"])!=28 or len(keys["random"])!=28:raise RuntimeError("not exactly 28 n8 projections")
    totalbytes={}
    for label,keys_ in keys.items():
        totalbytes[label]=sum(r["qng64_8_with_scales_bytes"] if (r["role"],r["layer"]) in keys_ else r["q4_with_scales_bytes"] for r in rows)
    if totalbytes["proxy"]!=totalbytes["random"]:
        raise RuntimeError("role-matched random control has unequal logical storage")
    return policies,totalbytes

def main():
    if LAB.exists():raise FileExistsError("original scan dir exists (will not reselect)")
    if not SRC.is_file() or not (SNAP/"config.json").is_file():
        raise RuntimeError("pinned BF16 model missing")
    os.umask(0o077)
    LAB.mkdir(parents=True,mode=0o700)
    lib,source_hashes=verify_codecs()
    hdr,offset=original_tensors()
    start=time.perf_counter()
    rows=[]
    with SRC.open("rb") as fp:
        with mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ) as mm:
            for role,rolepart in ROLES:
                for layer in range(NUM_LAYERS):
                    name=f"model.layers.{layer}.{rolepart}"
                    rows.append(analyze_role(role,layer,hdr[name],mm,offset,lib))
                    if len(rows)%14==0:
                        print("BEGLIN_Q_PROXY_PROGRESS",len(rows),"of 196",role,"elapsed",round(time.perf_counter()-start,2),flush=True)
    ordered,totalbytes=policy_candidates(rows)
    counts={role:4 for role,_ in ROLES}
    output={
      "schema":"oml-acs-q-native-196-tensor-proxy/1",
      "status":"PASS_WEIGHT_DISTORTION_PROXY_NOT_TASK_SENSITIVITY",
      "model":"Qwen/Qwen2.5-1.5B-Instruct",
      "source_bf16_snapshot_rev":SOURCEREV,
      "native_BEGILN_quantizer_source_sha256":source_hashes,
      "selected_roles_order":[r[0] for r in ROLES],
      "source_tensors":196,"seed":SEED,"n8_per_role":4,
      "role_to_layer_selection":ordered,"rows":rows,
      "actual_196_q4_weight_bytes":sum(x["q4_with_scales_bytes"] for x in rows),
      "actual_196_q8_weight_bytes":sum(x["qng64_8_with_scales_bytes"] for x in rows),
      "actual_196_bf16_weight_bytes":sum(x["bf16_weight_bytes"] for x in rows),
      "mixed_logical_weight_bytes":totalbytes,
      "method":"native C encoder and independent NumPy packed decode; relative Frobenius weight error proxy only",
      "proxy_does_not_use_ppl_or_target_labels":True,
      "model_behavior_task_sensitive_measured":False,
      "final_Q_acceptance":False,
      "production_modified":False,
      "scan_elapsed_seconds":round(time.perf_counter()-start,2),
    }
    txt=json.dumps(output,sort_keys=True,indent=2)+"\n"
    path=LAB/"q_196_proxy_results_private.json"
    with path.open("xb") as fd:
        fd.write(txt.encode());fd.flush();os.fsync(fd.fileno())
    path.chmod(0o600)
    for label in ("proxy","random"):
        lines=[f"{role} {layer}\n" for role,_ in ROLES for layer in
          (ordered[role]["proxy_n8_layers"] if label=="proxy" else ordered[role]["seed73_random_n8_layers"])]
        p=LAB/f"n8_{label}_role_layer_policy.txt"
        with p.open("xb") as fd:
            fd.write("".join(lines).encode());fd.flush();os.fsync(fd.fileno())
        p.chmod(0o600)
    print("BEGLIN_Q_PROXY_196_RECEIPT",
        json.dumps(dict(source_count=196,role_budget=counts,
             proxy_layers={role:data["proxy_n8_layers"] for role,data in ordered.items()},
             random_layers={role:data["seed73_random_n8_layers"] for role,data in ordered.items()},
             full_q4_weight_bytes=output["actual_196_q4_weight_bytes"],
             full_q8_weight_bytes=output["actual_196_q8_weight_bytes"],
             proxy_and_random_bytes=totalbytes,
             source_C_sha256=source_hashes,
             source_file_sha256=sha(path.read_bytes()),
             elapsed_seconds=output["scan_elapsed_seconds"]),sort_keys=True),
             flush=True)

if __name__=="__main__":
    main()
