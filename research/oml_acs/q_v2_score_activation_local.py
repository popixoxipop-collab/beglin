#!/usr/bin/env python3
"""BEGLIN Qv2 diagonal activation-aware LOCAL projection error estimator.

Works only with original BF16 Qwen (pinned), source calibration second
moments, original native BEGLIN q4 and n8 encoders compiled for WSL x86-64.
No full-model output interventions, test labels, or evaluation prompts read.

Each of 196 tensors is scheduled to 3 bounded concurrent CPU workers.
Candidate 8bit benefit = weighted local projection reconstruction error
q4 minus q8, using E[a_j^2] from unmodified BF16 CUDA teacher hooks.
This is a diagonal-covariance proxy, not true Hessian/Fisher or causal NLL.
"""
from __future__ import annotations
import ctypes,hashlib,json,mmap,os,pathlib,struct,subprocess,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
SRC=pathlib.Path("/home/alienware-r13/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors")
STAT=pathlib.Path.home()/".local/share/beglin/oml_acs/20261022_qv2_retry1"
LAB=pathlib.Path.home()/".local/share/beglin/oml_acs/20261022_qv2_quality_scores"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
EXPECTED_ACTIV_SHA="ee7b177b4dde984a5d83e0cdb5983d2ae00c34ee752fef9113a79568a1d12663"
EXPECTED_PROTOCOL_SHA="0d9177b152286ccd413028e74838529e2a5aad57b42ec793df3095f99208d4af"
EXPECTED_ENCODER={
 "gguf_transcode.c":"7371cc352226e912dbf42762e5e63a70e91acb5c90691d90e32401e18af7ec5f",
 "qng64_dense.c":"cb985ec48cfa1f1f9ebe8411d12da5d3f73720c8acdcad51fc279a2b25ff8956",
}
MAX_WORKERS=3

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()

def original_header():
    with SRC.open("rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        if not 1000<n<2_000_000:raise RuntimeError("original safetensors index invalid")
        h=json.loads(f.read(n))
    return h,8+n

def prepare():
    if LAB.exists():raise FileExistsError("refuse to overwrite old quality source run")
    if not SRC.is_file() or not STAT.is_dir():raise RuntimeError("original BF16 or source calibration missing")
    if sha((STAT/"bf16_calibration_196_input_diag_moments.npz").read_bytes())!=EXPECTED_ACTIV_SHA:
        raise RuntimeError("original source BF16 activation stats changed")
    p=ROOT/"configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json"
    if sha(p.read_bytes())!=EXPECTED_PROTOCOL_SHA:raise RuntimeError("quality/calibration/test prelock changed")
    for filename,dig in EXPECTED_ENCODER.items():
        if sha((ROOT/filename).read_bytes())!=dig:raise RuntimeError("native quantizer source changed")
    os.umask(0o077)
    LAB.mkdir(parents=True,mode=0o700)
    so=LAB/"libbeglin_quant_source_x86.so"
    subprocess.run(["gcc","-O3","-fPIC","-shared","-I",str(ROOT),
        str(ROOT/"gguf_transcode.c"),str(ROOT/"qng64_dense.c"),
        "-o",str(so),"-lm"],check=True)
    manifest=json.loads((STAT/"activation_diagonal_manifest.json").read_text())
    if manifest["source_snapshot_rev"]!=SRC.parent.name or manifest["module_count"]!=196:
        raise RuntimeError("activation module source manifest mismatch")
    return so,sha(so.read_bytes())

def compute_one(task):
    name,shape,offset,length,moments,so_path=task
    role=name.rsplit(".",2)[-2].split(".")[-1] if False else name.split(".")[-2]
    layer=int(name.split(".")[2])
    # Actually name=model.layers.L.self_attn.q_proj.weight: role index -2
    if role not in ROLES or not (0<=layer<28):
        raise RuntimeError("untrusted source role "+name)
    if len(shape)!=2 or shape[1]%64:raise RuntimeError("invalid original BF16 matrix")
    with SRC.open("rb") as fp:
        with mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ) as mm:
            raw=bytes(mm[offset:offset+length])
    if len(raw)!=length or length!=shape[0]*shape[1]*2:raise RuntimeError("safetensors weights truncated")
    weight=(np.frombuffer(raw,dtype="<u2").astype("<u4")<<16).view("<f4").reshape(shape).copy()
    if not np.isfinite(weight).all():raise RuntimeError("nonfinite original source BF16")
    n=weight.size;out,inp=shape;ng=n//64
    p4=np.empty(n//2,dtype=np.uint8);s4=np.empty(ng,dtype=np.float32)
    p8=np.empty(n,dtype=np.uint8);s8=np.empty(ng,dtype=np.float32)
    lib=ctypes.CDLL(so_path);pt=ctypes.c_void_p
    a=lib.gguf_quantize_q4g64_error_feedback
    a.argtypes=[pt,ctypes.c_int,ctypes.c_int,pt,pt];a.restype=None
    b=lib.qng64_quantize_f32
    b.argtypes=[pt,ctypes.c_int,ctypes.c_int,ctypes.c_int,pt,pt];b.restype=ctypes.c_int
    a(weight.ctypes.data,out,inp,p4.ctypes.data,s4.ctypes.data)
    if b(weight.ctypes.data,out,inp,8,p8.ctypes.data,s8.ctypes.data)!=0:
        raise RuntimeError("native n8 quant source error "+name)
    group=inp//64
    nib=p4.reshape(out,group,32)
    c4=np.empty((out,group,64),dtype=np.int8)
    c4[:,:,0::2]=(nib&15).astype(np.int8)-8
    c4[:,:,1::2]=(nib>>4).astype(np.int8)-8
    w4=(c4.astype(np.float32)*s4.reshape(out,group,1)).reshape(shape)
    bitplanes=p8.reshape(out,group,8,8)
    u=np.zeros((out,group,64),dtype=np.uint16)
    for shift in range(8):
        code=np.unpackbits(bitplanes[:,:,shift,:],bitorder="little",axis=-1)
        u|=(code.astype(np.uint16)<<shift)
    w8=((u.astype(np.int16)-128).astype(np.float32)*s8.reshape(out,group,1)).reshape(shape)
    a2=np.asarray(moments,dtype=np.float64)
    if a2.shape!=(inp,) or not np.isfinite(a2).all() or min(a2)<0:
        raise RuntimeError("missing moment for "+name)
    den=float(np.dot(np.square(weight.astype(np.float64)).sum(axis=0),a2))
    if den<=0 or not np.isfinite(den):raise RuntimeError("zero/no original BF16 activation-weighted energy")
    e4=float(np.dot(np.square((w4-weight).astype(np.float64)).sum(axis=0),a2)/den)
    e8=float(np.dot(np.square((w8-weight).astype(np.float64)).sum(axis=0),a2)/den)
    if e4<0 or e8<0 or not np.isfinite([e4,e8]).all():
        raise RuntimeError("invalid source-only quality proxy")
    if e8>=e4:raise RuntimeError("native 8bit unexpectedly not better than q4 for "+name)
    return dict(name=name,role=role,layer=layer,weight_shape=list(shape),
        diag_activation_local_error_q4=e4,
        diag_activation_local_error_n8=e8,
        source_activation_quality_upgrade_gain=e4-e8,
        original_BF16_weight_sha256=sha(raw),
        encoded_q4_SHA256=sha(p4.tobytes()),
        encoded_n8_bitplanes_SHA256=sha(p8.tobytes()),
        Q4_with_scales_bytes=p4.nbytes+s4.nbytes,
        N8_with_scales_bytes=p8.nbytes+s8.nbytes,
        source_position_count=461,
        method="sum_j E[x_j^2] sum_o ((W-Wb)_oj)^2 / sum_j E[x_j^2] sum_o W_oj^2")
def main():
    start=time.monotonic()
    so,binary_hash=prepare()
    moments=np.load(STAT/"bf16_calibration_196_input_diag_moments.npz",allow_pickle=False)
    if len(moments.files)!=196:raise RuntimeError("not 196 original BF16 module input moments")
    h,offset=original_header()
    tasks=[]
    for role in ROLES:
        group="self_attn" if role in ROLES[:4] else "mlp"
        for l in range(28):
            name=f"model.layers.{l}.{group}.{role}.weight"
            node=h[name]
            moment_key=f"layer{l}__{group}__{role}"
            start_off,end_off=node["data_offsets"]
            tasks.append((name,node["shape"],offset+start_off,end_off-start_off,
                       moments[moment_key].copy(),str(so)))
    assert len(tasks)==196
    observations=[]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as workers:
        submitted=[workers.submit(compute_one,t) for t in tasks]
        for i,future in enumerate(as_completed(submitted),1):
            row=future.result()
            observations.append(row)
            if i%14==0:
                print("BEGLIN_ACTIVATION_AWARE_CPU_SHARDS",i,196,round(time.monotonic()-start,2),flush=True)
    observations.sort(key=lambda z:(ROLES.index(z["role"]),z["layer"]))
    assert len(observations)==196 and len({(x["role"],x["layer"]) for x in observations})==196
    byrole={role:[round(item["source_activation_quality_upgrade_gain"],12) for item in observations if item["role"]==role] for role in ROLES}
    compact={
       "schema":"BEGLIN-Q-v2-activation-conditioned-local-error-benefits/1",
       "source_BF16_revision":SRC.parent.name,
       "preregistered_protocol_sha256":EXPECTED_PROTOCOL_SHA,
       "private_activation_moment_sha256":EXPECTED_ACTIV_SHA,
       "encoder_original_source_sha256":EXPECTED_ENCODER,
       "native_CPU_quant_encoder_library_sha256":binary_hash,
       "moment_tokens":461,
       "quality_is_local_diag_covariance_not_NLL":True,
       "quality_upgrade_by_role_layer":byrole,
       "tensor_count":196,"test_prompts_not_used":True,
       "final_Q_quality_claim":"NO_VERDICT",
       "metric":"fraction of source activation-scaled local squared weight-error energy improved from native Q4G64 to qNg64 n8"
    }
    full=dict(compact,rows=observations,status="PASS_CONCURRENT_3_CPU_WORKERS_ORIGINAL_BF16_SOURCE_ONLY",
              workers=MAX_WORKERS,elapsed_seconds=round(time.monotonic()-start,2))
    privatefile=LAB/"quality_score_196_full_private.json"
    with privatefile.open("xb") as out:
        out.write((json.dumps(full,sort_keys=True,indent=2)+"\n").encode());out.flush();os.fsync(out.fileno())
    privatefile.chmod(0o600)
    publicfile=LAB/"quality_score_196_aggregate_public.json"
    with publicfile.open("xb") as out:
        out.write((json.dumps(compact,sort_keys=True,indent=2)+"\n").encode());out.flush();os.fsync(out.fileno())
    publicfile.chmod(0o600)
    print("BEGLIN_ACTIVATION_AWARE_SCORES_COMPLETE",json.dumps({
       "private_receipt_sha256":sha(privatefile.read_bytes()),
       "aggregate_sha256":sha(publicfile.read_bytes()),
       "source_tensors":196,"worker_count":MAX_WORKERS,
       "smallest_improvement":min(x["source_activation_quality_upgrade_gain"] for x in observations),
       "largest_improvement":max(x["source_activation_quality_upgrade_gain"] for x in observations),
       "elapsed_seconds":round(time.monotonic()-start,2)},sort_keys=True),flush=True)
if __name__=="__main__":main()
