#!/usr/bin/env python3
"""Real original-BF16, unmodified-model answer-token empirical directional Fisher.

For each of 196 projection tensors:
 score_n = mean_train [ (grad_W L_correct_answer · (native_Qn(W)-W)) ** 2 ]
 Qv5_gain = score_Q4 - score_n8
The gradient is obtained by backpropagating the original BF16 model exactly
once per source example; NEVER run a forward with a quantized intermediate.
This is the empirical *directional* Fisher quadratic proxy, not true Fisher,
exact Hessian, causal intervention, or final task NLL.
"""
from __future__ import annotations
import ctypes,hashlib,json,math,os,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv5_fisher_20261022"
PROTO=ROOT/"configs/research/BEGLIN_QV5_FISHER_GRAD_PRELOCK_20261022.json"
SOURCE=LAB/"source_train_frozen.json"
DEV=LAB/"development_val_frozen.json"
LIB=LAB/"libqv5_native_quant.dylib"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PUB=ROOT/"docs/research/BEGLIN_QV5_FISHER_SOURCE_AGGREGATE_20261022.json"
OUT=LAB/"fisher_196_train_private_receipt.json"
PRELOCK_SHA="a886418affc9f665d64ce4ca9eb0b5ea2aef4ae97bc59bf58ea1affb53bf19ef"
SOURCE_SHA="631bb02d22b696ddcabfd3d7ab759218fd75e1e8f3d8384cbf4a5df4de3f6994"
DEV_SHA="a00985cc73f80518f448c2b8a1295cf894283723f87b5af53321d486ce0187ae"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha_file(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for data in iter(lambda:f.read(1024*1024),b""):h.update(data)
    return h.hexdigest()
def private(path,obj):
    data=(json.dumps(obj,indent=2,sort_keys=True)+"\n").encode()
    with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
    return sha(data)
def check_source():
    if OUT.exists() or PUB.exists():
        raise FileExistsError("fisher source already calibrated; refuse re-fit after potential eval")
    if sha(PROTO.read_bytes())!=PRELOCK_SHA or sha(SOURCE.read_bytes())!=SOURCE_SHA or sha(DEV.read_bytes())!=DEV_SHA:
        raise RuntimeError("frozen original source/development identity changed")
    c=json.loads(PROTO.read_text())
    if c["production_write_allowed"] or c["auto_promotion"] or c["source_questions"]!=24:
        raise RuntimeError("source-only contract violated")
    train=json.loads(SOURCE.read_text())
    if train["question_count"]!=24 or train["dataset_sha"]!=c["dataset_sha"]:
        raise RuntimeError("wrong pinned source training data")
    if not LIB.is_file():raise RuntimeError("native original C quantizer unavailable")
    for f,dig in c["original_C_quant_source_hashes"].items():
        if sha((ROOT/f).read_bytes())!=dig:raise RuntimeError("C quantizer source changed")
    if not torch.backends.mps.is_available():raise RuntimeError("requires original-BF16 Apple MPS")
    return c,train

def lib_quant():
    dll=ctypes.CDLL(str(LIB))
    ptr=ctypes.c_void_p
    dll.gguf_quantize_q4g64_error_feedback.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ptr,ptr]
    dll.gguf_quantize_q4g64_error_feedback.restype=None
    dll.qng64_quantize_f32.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr]
    dll.qng64_quantize_f32.restype=ctypes.c_int
    return dll

def quant_errors(dll,w:np.ndarray):
    if w.ndim!=2 or w.dtype!=np.float32 or not w.flags.c_contiguous:
        raise RuntimeError("original BF16 tensor not a contiguous FP32 CPU view")
    if not np.isfinite(w).all():raise RuntimeError("BF16 source weight nonfinite")
    out,inp=map(int,w.shape)
    if inp%64 or out<1:raise RuntimeError("requires authentic group64 tensor")
    nc=out*inp; ng=inp//64; groups=nc//64
    packed4=np.empty(nc//2,dtype=np.uint8);scale4=np.empty(groups,dtype=np.float32)
    packed8=np.empty(nc,dtype=np.uint8);scale8=np.empty(groups,dtype=np.float32)
    dll.gguf_quantize_q4g64_error_feedback(w.ctypes.data,out,inp,packed4.ctypes.data,scale4.ctypes.data)
    if dll.qng64_quantize_f32(w.ctypes.data,out,inp,8,packed8.ctypes.data,scale8.ctypes.data)!=0:
        raise RuntimeError("native 8-bit bitplane quantizer rejected BF16 tensor")
    if not np.isfinite(scale4).all() or not np.isfinite(scale8).all():
        raise RuntimeError("native quantization scale nonfinite")
    q4=np.empty((out,ng,64),dtype=np.int8)
    code=packed4.reshape(out,ng,32)
    q4[:,:,0::2]=(code&15).astype(np.int8)-8
    q4[:,:,1::2]=(code>>4).astype(np.int8)-8
    restored4=(q4.astype(np.float32)*scale4.reshape(out,ng,1)).reshape(out,inp)
    planes=packed8.reshape(out,ng,8,8)
    bits=np.zeros((out,ng,64),dtype=np.uint16)
    for bit in range(8):
        bits |= np.unpackbits(planes[:,:,bit,:],bitorder="little",axis=-1).astype(np.uint16)<<bit
    restored8=((bits.astype(np.int16)-128).astype(np.float32)*scale8.reshape(out,ng,1)).reshape(out,inp)
    d4=(restored4-w).astype(np.float16)
    d8=(restored8-w).astype(np.float16)
    if not np.isfinite(d4).all() or not np.isfinite(d8).all():
        raise RuntimeError("quantizer residual overflow")
    if not 0<np.sum(d8.astype(np.float64)**2)<np.sum(d4.astype(np.float64)**2):
        raise RuntimeError("counterintuitive quantizer weight residual, fail closed")
    checks={
      "original_BF16_widened_F32_weight_SHA256":sha(w.tobytes()),
      "Q4G64_native_code_SHA256":sha(packed4.tobytes()),
      "n8_native_bitplane_SHA256":sha(packed8.tobytes()),
      "Q4_scale_SHA256":sha(scale4.tobytes()),
      "n8_scale_SHA256":sha(scale8.tobytes()),
      "Q4_residual_F32_sum_sq":float(np.sum((restored4-w).astype(np.float64)**2)),
      "n8_residual_F32_sum_sq":float(np.sum((restored8-w).astype(np.float64)**2))
    }
    return d4,d8,checks

def diag_gradient_dot(g:np.ndarray,d4:np.ndarray,d8:np.ndarray):
    if g.shape!=d4.shape or g.shape!=d8.shape or not np.isfinite(g).all():
        raise RuntimeError("BF16 backward tensor gradient shape/finite mismatch")
    gv=g.ravel()
    # Float64 accumulation controls large cancellation between signed gradients.
    a=float(np.einsum("i,i->",gv,d4.reshape(-1).astype(np.float32),dtype=np.float64))
    b=float(np.einsum("i,i->",gv,d8.reshape(-1).astype(np.float32),dtype=np.float64))
    tr=float(np.einsum("i,i->",gv,gv,dtype=np.float64))
    return a*a,b*b,tr

def run():
    c,data=check_source()
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    torch.set_num_threads(2)
    torch.backends.mps.is_available() or (_ for _ in ()).throw(RuntimeError("MPS unavailable"))
    begin=time.monotonic()
    print("QV5_NATIVE_FISHER_LOADING_BF16",flush=True)
    model=AutoModelForCausalLM.from_pretrained(str(SNAP),dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False,attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)
    modules=dict(model.named_modules())
    chosen=[]
    for role in ROLES:
        for layer in range(28):
            kind="self_attn" if role in ROLES[:4] else "mlp"
            name=f"model.layers.{layer}.{kind}.{role}"
            if name not in modules or modules[name].weight.ndim!=2:
                raise RuntimeError("frozen 196 tensor role/layer missing "+name)
            modules[name].weight.requires_grad_(True)
            chosen.append((role,layer,name,modules[name].weight))
    if len(chosen)!=196 or len({x[2] for x in chosen})!=196:
        raise RuntimeError("frozen source tensor family mismatch")
    native=lib_quant()
    residual=[]
    for ix,(role,layer,name,param) in enumerate(chosen):
        w=param.detach().to(dtype=torch.float32,device="cpu").contiguous().numpy()
        d4,d8,check=quant_errors(native,w)
        residual.append((d4,d8,check))
        if (ix+1)%28==0:
            print("QV5_NATIVE_Q4N8_RESIDUALS",ix+1,"/196",round(time.monotonic()-begin,1),flush=True)
    print("QV5_FISHER_DELTA_CACHE_READY",len(residual),flush=True)
    model=model.to("mps")
    stats=[{"q4_sq":[],"n8_sq":[],"trace":[],"first_order_q4":[],"first_order_n8":[]} for _ in chosen]
    train_losses=[]
    for ix,row in enumerate(data["questions"]):
        model.zero_grad(set_to_none=True)
        ids=torch.tensor([row["prefix_ids"]],dtype=torch.long,device="mps")
        target=torch.tensor([row["suffix_ids"][row["correct"]]],dtype=torch.long,device="mps")
        output=model(input_ids=ids,use_cache=False,return_dict=True)
        loss=F.cross_entropy(output.logits[:,-1,:].float(),target)
        value=float(loss.detach().cpu())
        if not math.isfinite(value):raise RuntimeError("source BF16 original answer loss nonfinite")
        loss.backward()
        train_losses.append(value)
        for j,(_,_,name,param) in enumerate(chosen):
            if param.grad is None:raise RuntimeError("source BF16 gradient missing "+name)
            g=param.grad.detach().to(dtype=torch.float32,device="cpu").contiguous().numpy()
            d4,d8,_=residual[j]
            q4,n8,tr=diag_gradient_dot(g,d4,d8)
            stats[j]["q4_sq"].append(q4);stats[j]["n8_sq"].append(n8);stats[j]["trace"].append(tr)
        del output,loss
        torch.mps.synchronize()
        print("QV5_FISHER_SOURCE_TOKEN",ix+1,"/",len(data["questions"]),"original_bf16_correct_answer_nll",round(value,6),"elapsed",round(time.monotonic()-begin,1),flush=True)
    # No validation/dev labels or model outputs have been accessed by this module.
    records=[];gain={r:[0.]*28 for r in ROLES}
    for ix,(role,layer,name,param) in enumerate(chosen):
        q4=float(np.mean(stats[ix]["q4_sq"]));n8=float(np.mean(stats[ix]["n8_sq"]))
        g=q4-n8
        if not all(math.isfinite(z) for z in (q4,n8,g)) or q4<0 or n8<0:
            raise RuntimeError("nonfinite Fisher direction")
        gain[role][layer]=g
        records.append({"role":role,"layer":layer,"q4_directional_Fisher":q4,
          "n8_directional_Fisher":n8,"gain_8bit_vs_4bit":g,
          "original_BF16_gradient_trace_mean":float(np.mean(stats[ix]["trace"])),
          "per_source_item_q4_sq":stats[ix]["q4_sq"],
          "per_source_item_n8_sq":stats[ix]["n8_sq"],
          "native_codes_and_BF16_weight_sha256":residual[ix][2]})
    from scipy.stats import spearmanr
    odd=[];even=[]
    for ix in range(196):
        odd.append(float(np.mean(np.array(stats[ix]["q4_sq"])[::2]-np.array(stats[ix]["n8_sq"])[::2])))
        even.append(float(np.mean(np.array(stats[ix]["q4_sq"])[1::2]-np.array(stats[ix]["n8_sq"])[1::2])))
    rho=float(spearmanr(odd,even).statistic)
    score_vec=np.asarray([gain[role][layer] for role in ROLES for layer in range(28)])
    if not np.isfinite(score_vec).all() or not math.isfinite(rho):
        raise RuntimeError("Fisher score/rank split-half unstable or nonfinite")
    receipt={"schema":"BEGLIN-Qv5-original-BF16-directional-empirical-Fisher-196/1",
      "status":"SOURCE_TRAIN_ONLY_ALL_196_NATIVE_GRADIENTS_AND_QUANT_ERRORS",
      "prelock_sha256":PRELOCK_SHA,"source_train_SHA256":SOURCE_SHA,"unopened_dev_cohort_sha256":DEV_SHA,
      "original_HF_rev":c["bf16_official_rev"],"bf16_cache_rev":SNAP.name,
      "native_original_C_source_SHA256":c["original_C_quant_source_hashes"],
      "native_C_quant_dynamic_library_sha256":sha_file(LIB),
      "BF16_model_forward_unchanged":True,"no_quantized_forward_or_per_tensor_downstream_interventions":True,
      "grad_dtype":"BF16 param/backward, converted FP32 with float64 directional dot accumulation",
      "native_residual_dtype":"Q4G64/n8 original C code and F32 scales, compressed to FP16 memory residuals",
      "source_dataset":"ARC-Challenge train","train_case_count":24,
      "source_full_correct_answer_avg_nll":sum(train_losses)/len(train_losses),
      "train_original_NLL_per_case":train_losses,
      "split_half_odd_even_spearman":rho,
      "all_tensors_with_finite_gradients":len(records),
      "source_score_rows":records,
      "quality_upgrade_by_role_layer":gain,
      "negative_gain_tensors":int((score_vec<0).sum()),
      "no_dev_labels_accessed":True,"no_original_model_or_production_mutation":True,
      "elapsed_seconds":round(time.monotonic()-begin,2),
      "scientific_claim":"ONLY_FIRST_ORDER_DIRECTIONAL_EMPIRICAL_FISHER_PROXY_NOT_ACTUAL_QUANTIZED_NLL"}
    os.umask(0o077)
    pr=private(OUT,receipt)
    aggregate={k:v for k,v in receipt.items() if k not in ("source_score_rows","train_original_NLL_per_case")}
    aggregate["private_full_source_receipt_SHA256"]=pr
    aggregate["source_196_grad_score_min_max"]=[float(score_vec.min()),float(score_vec.max())]
    pub=(json.dumps(aggregate,sort_keys=True,indent=2)+"\n").encode()
    with PUB.open("xb") as f:f.write(pub);f.flush();os.fsync(f.fileno())
    print("QV5_BF16_FISHER_196_REAL_SUCCESS",json.dumps({
      "receipt_SHA":pr,"aggregate_SHA":sha(pub),"spearman_odd_even":rho,
      "positive_gain":int((score_vec>0).sum()),"negative_gain":int((score_vec<0).sum()),
      "original_source_NLL":aggregate["source_full_correct_answer_avg_nll"],
      "seconds":round(time.monotonic()-begin,1)},sort_keys=True),flush=True)
if __name__=="__main__":run()
