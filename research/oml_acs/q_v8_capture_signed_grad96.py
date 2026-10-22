#!/usr/bin/env python3
"""96 original BF16 task gradients × 196 real native Q4/n8 *signed* residuals.

Key difference from Qv5–7: preserve g·delta *signs* for the same complete
source question, enabling global cross-tensor interaction of first-order
correct-answer loss displacement. No full-model quantized forward, no
per-tensor final-output intervention, no HellaSwag outputs.
"""
from __future__ import annotations
import hashlib,importlib.util,json,math,os,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
ORIG=OLD/"research/oml_acs/q_v5_original_bf16_fisher_source.py"
OLDRECEIPT=OLD/"lab_local/qv5_fisher_20261022/fisher_196_train_private_receipt.json"
SOURCE=ROOT/"lab_local/qv8_joint_signed_20261022/pinned_original_ARC_train96.json"
HELD=ROOT/"lab_local/qv8_joint_signed_20261022/pinned_HellaSwag_val48.json"
PRE=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_SOURCE_PRELOCK_20261022.json"
LAB=ROOT/"lab_local/qv8_joint_signed_20261022"
CASE=LAB/"signed_source96_case_receipts"
OUT=LAB/"original_BF16_signed_196x96_source_private.json"
PUB=ROOT/"docs/research/BEGLIN_QV8_BF16_SIGNED_CROSS_TENSOR_SOURCE_PUBLIC_20261022.json"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
PRE_SHA="9b833f5fc2729d13320c87e368baaff952c890ed068586622badfc7da63b4db1"
SOURCE_SHA="8da2f96b6ec02a5a45a2308e84074a1de14e7a93dc30e9517960596e3b2cee0c"
HELD_SHA="45c95170a2eb01b4c79ad44719ab9a1877e568f2583a90d84340d7448e0d1a33"
Q5PRIVATE_SHA="6d87dfef842fcce932cdb85c9a42bc3ede380493917ae608663e2072f8143417"

def sha(b):return hashlib.sha256(b).hexdigest()
def private(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private source/checkpoint missing or unsafe "+str(p))
    return p.read_bytes()
def write(p,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    return sha(b)
def read_case(ix,c):
    p=CASE/f"signed_{ix:03d}.json"
    if not p.exists():return None
    x=json.loads(private(p))
    if x["schema"]!="BEGLIN-Qv8-original-BF16-source-signed-grad-196/1" or \
       x["index"]!=ix or x["id_sha256"]!=sha(c["id"].encode()) or \
       len(x["signed_q4_per_tensor"])!=196 or len(x["signed_n8_per_tensor"])!=196:
        raise RuntimeError("source signed gradient checkpoint altered")
    return x
def inspect():
    if OUT.exists() or PUB.exists():raise FileExistsError("source calibration finalized; no refitting")
    if sha(PRE.read_bytes())!=PRE_SHA or sha(private(SOURCE))!=SOURCE_SHA or sha(private(HELD))!=HELD_SHA or \
       sha(private(OLDRECEIPT))!=Q5PRIVATE_SHA:
        raise RuntimeError("original BF16 Qv5/Qv8 model/dataset identities changed")
    d=json.loads(private(SOURCE))["source_train96"]
    if len(d)!=96 or len({v["id"] for v in d})!=96:
        raise RuntimeError("96 calibration questions invalid")
    if not torch.backends.mps.is_available():raise RuntimeError("BF16 original Apple MPS unavailable")
    return d,json.loads(private(OLDRECEIPT))
def begin_native(model,old):
    spec=importlib.util.spec_from_file_location("qv5_original_native_quantizer",ORIG)
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    dll=q.lib_quant(); module=dict(model.named_modules());chosen=[];cache=[]
    for i,role in enumerate(ROLES):
        for layer in range(28):
            name=f"model.layers.{layer}.{'self_attn' if i<4 else 'mlp'}.{role}"
            if name not in module:raise RuntimeError("unexpected official BF16 projection role "+name)
            p=module[name].weight
            p.requires_grad_(True)
            w=p.detach().float().cpu().contiguous().numpy()
            d4,d8,h=q.quant_errors(dll,w)
            old_row=old["source_score_rows"][i*28+layer]
            previous=old_row["native_codes_and_BF16_weight_sha256"]
            if old_row["role"]!=role or old_row["layer"]!=layer or \
               any(previous[k]!=h[k] for k in (
                  "original_BF16_widened_F32_weight_SHA256",
                  "Q4G64_native_code_SHA256","n8_native_bitplane_SHA256")):
                raise RuntimeError("original BF16 weight / native C Q4n8 encoder identity mismatch "+name)
            chosen.append((role,layer,p));cache.append((d4,d8))
        print("QV8_ORIGINAL_QWEN_SOURCE_NATIVE_QUANT_DELTA_VERIFIED",role,len(chosen),flush=True)
    if len(chosen)!=196:raise RuntimeError("source has wrong 196 projection tensor family")
    return chosen,cache
def run():
    train,old=inspect()
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    torch.set_num_threads(2)
    os.umask(0o077);CASE.mkdir(mode=0o700,exist_ok=True)
    if CASE.is_symlink() or (CASE.stat().st_mode&0o777)!=0o700:
        raise RuntimeError("signed task gradient checkpoint directory not owner-private")
    t=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(str(SNAP),dtype=torch.bfloat16,local_files_only=True,
        trust_remote_code=False,attn_implementation="eager").eval()
    model.requires_grad_(False)
    selected,residuals=begin_native(model,old)
    model=model.to("mps")
    for ix,c in enumerate(train):
        if read_case(ix,c):
            print("QV8_SIGNED_SOURCE_RESUME_SHA_VERIFIED",ix+1,"/96",flush=True);continue
        model.zero_grad(set_to_none=True)
        inp=torch.tensor([c["prefix_ids"]],dtype=torch.long,device="mps")
        target=torch.tensor([c["suffix_ids"][c["correct"]]],dtype=torch.long,device="mps")
        f=model(input_ids=inp,use_cache=False)
        loss=F.cross_entropy(f.logits[:,-1,:].float(),target)
        value=float(loss.detach().cpu())
        if not math.isfinite(value):raise RuntimeError("original BF16 answer task loss nonfinite")
        loss.backward()
        q4=[];n8=[];genergy=[]
        for j,(role,layer,p) in enumerate(selected):
            if p.grad is None:raise RuntimeError("original source BF16 projection gradient missing")
            g=p.grad.detach().float().cpu().contiguous().numpy().ravel()
            w4,w8=residuals[j]
            a=float(np.einsum("i,i->",g,w4.ravel().astype(np.float32),dtype=np.float64))
            b=float(np.einsum("i,i->",g,w8.ravel().astype(np.float32),dtype=np.float64))
            if not (math.isfinite(a) and math.isfinite(b)):
                raise RuntimeError("original native Q4/n8 signed task-error nonfinite")
            q4.append(a);n8.append(b)
            genergy.append(float(np.einsum("i,i->",g,g,dtype=np.float64)))
        record={
         "schema":"BEGLIN-Qv8-original-BF16-source-signed-grad-196/1",
         "index":ix,"id_sha256":sha(c["id"].encode()),"original_BF16_NLL":value,
         "source_original_96_SHA256":SOURCE_SHA,"native_BF16_widened_F32_quant_encoder":"original C Q4G64/n8",
         "signed_q4_per_tensor":q4,"signed_n8_per_tensor":n8,
         "source_BF16_gradient_squared_norm":genergy,
         "no_quantized_full_model_forward":True,"no_crosscorpus_eval_data_loaded":True}
        proof=write(CASE/f"signed_{ix:03d}.json",record)
        del f,loss
        torch.mps.synchronize()
        if (ix+1)%8==0:
            print("QV8_NEW_SIGNED_BF16_SOURCE_PROGRESS",ix+1,"/96",round(time.monotonic()-t,1),"case_SHA",proof[:16],flush=True)
    cases=[]
    for ix,c in enumerate(train):
        x=read_case(ix,c)
        if x is None:raise RuntimeError("missing per-case original BF16 gradient")
        cases.append(x)
    a=np.array([c["signed_q4_per_tensor"] for c in cases],dtype=np.float64)
    b=np.array([c["signed_n8_per_tensor"] for c in cases],dtype=np.float64)
    if a.shape!=(96,196) or b.shape!=(96,196) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("signed native source gradient x projection incomplete")
    df=b-a
    corr=np.corrcoef(df,rowvar=False)
    # Correlation matrix is diagnostic; no down-stream input accessed.
    offdiag=np.abs(corr-np.eye(196));offdiag=offdiag[np.triu_indices(196,1)]
    if not np.isfinite(offdiag).all():raise RuntimeError("signed source covariance degenerate")
    priv={
        "schema":"BEGLIN-Qv8-original-BF16-signed-joint-error-196x96/1",
        "status":"SOURCE_ONLY_196x96_SIGNED_NATIVE_Q4N8_DELTA_VERIFIED",
        "prelock_sha256":PRE_SHA,"ARC96_source_SHA256":SOURCE_SHA,
        "unopened_HellaSwag48_SHA256":HELD_SHA,
        "Qv5_original_196_tensor_quant_native_SHA_source":Q5PRIVATE_SHA,
        "original_model_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "source_loss_original_BF16_mean_NLL":float(np.mean([c["original_BF16_NLL"] for c in cases])),
        "original_tensor_grad_dot_q4":a.tolist(),
        "original_tensor_grad_dot_n8":b.tolist(),
        "first_order_joint_Q4_source_rms":float(np.sqrt(np.mean(a.sum(axis=1)**2))),
        "mean_cross_tensor_abs_correlation":float(np.mean(offdiag)),
        "private_per_case_SHA256":[sha(private(CASE/f"signed_{ix:03d}.json")) for ix in range(96)],
        "scalar_signed_directional_dot_dtype":"original BF16 MPS gradient promoted FP32, source native residual FP16 promoted FP32, FP64 signed dot",
        "eval_logits_unopened":True,
        "no_model_weights_mutated":True,"no_operational_state_changed":True,
        "elapsed_s":round(time.monotonic()-t,2)}
    digest=write(OUT,priv)
    pub={k:v for k,v in priv.items() if k not in (
       "original_tensor_grad_dot_q4","original_tensor_grad_dot_n8","private_per_case_SHA256")}
    pub["full_source_private_SHA256"]=digest
    doc=(json.dumps(pub,indent=2,sort_keys=True)+"\n").encode()
    with PUB.open("xb") as f:f.write(doc);f.flush();os.fsync(f.fileno())
    print("QV8_ORIGINAL_BF16_196x96_SIGNED_CROSSTENSOR_SOURCE_PASS",
        json.dumps({"private_SHA":digest,"public_SHA":sha(doc),
          "source_rms":priv["first_order_joint_Q4_source_rms"],
          "abs_offdiag":priv["mean_cross_tensor_abs_correlation"],
          "elapsed_s":priv["elapsed_s"]},sort_keys=True),flush=True)
if __name__=="__main__":run()
