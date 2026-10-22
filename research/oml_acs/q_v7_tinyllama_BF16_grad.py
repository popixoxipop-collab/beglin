#!/usr/bin/env python3
"""Original pinned TinyLlama BF16 154-tensor directional Fisher on 64 ARC cases.

All original Llama model weights stay unchanged. Real BEGLIN group64 native C
Q4G64 and qNg64-n8 quant residuals are contracted against 64 original
per-item BF16 gradients from teacher-forced one-letter answers. This measures
a first-order directional squared-gradient proxy, not actual quantized-model
task loss or a causal single-tensor intervention.
"""
from __future__ import annotations
import hashlib,importlib.util,json,math,os,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
BASE=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
SOURCE_CODE=BASE/"research/oml_acs/q_v5_original_bf16_fisher_source.py"
MODEL=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv7-tinyllama-model-20261022")
PRE=ROOT/"configs/research/BEGLIN_QV7_CROSSFAMILY_CALIBRATION_PRELOCK_20261022.json"
LAB=ROOT/"lab_local/qv7_tinyllama_20261022"
SOURCE=LAB/"source_arc64_tinyllama_frozen.json"
HELD=LAB/"eval_openbookqa_test48_tinyllama_frozen.json"
OUT=LAB/"TinyLlama_original_BF16_154_directional_Fisher_source64_private.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV7_TINYLLAMA_SOURCE_154_FISHER_PUBLIC_20261022.json"
PRE_SHA="1b5fa904cf11f25292720efc75f658fc10cedc8923236b4d5b0ae7e70d3e88c6"
SOURCE_SHA="59c1d22197ba83ac3769049c01de4ae98414c678066ef6faa48fa2a0859bbe41"
HELD_SHA="8807002e79afadcb2d150b32dffdefa204afb0d70678375a608b7900f18d0ab0"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
LAYERS=22
def sha(b):return hashlib.sha256(b).hexdigest()
def seal(p,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
    return sha(b)
def read_locked():
    if OUT.exists() or PUBLIC.exists():raise FileExistsError("TinyLlama source Fisher already measured; no post-eval refit")
    if sha(PRE.read_bytes())!=PRE_SHA or sha(SOURCE.read_bytes())!=SOURCE_SHA or sha(HELD.read_bytes())!=HELD_SHA:
        raise RuntimeError("source-only original BF16 Llama source or held inputs changed")
    cfg=json.loads(PRE.read_text())
    if cfg["replication_revision"]!="fe8a4ea1ffedaf415f4da2f062534de366a451e6" or \
       cfg["total_projections"]!=154 or cfg["source_count"]!=64:
        raise RuntimeError("cross-family 154 exact target mismatch")
    d=json.loads(SOURCE.read_text())["source64"]
    if len(d)!=64 or len({x["id"] for x in d})!=64:
        raise RuntimeError("64 source inputs missing/unfrozen")
    if not torch.backends.mps.is_available():raise RuntimeError("original BF16 MPS not available")
    return cfg,d
def native_quant():
    spec=importlib.util.spec_from_file_location("qv5_fisher_source_verified",SOURCE_CODE)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m,m.lib_quant()
def evaluate():
    cfg,train=read_locked()
    torch.set_num_threads(2)
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    t=time.monotonic()
    print("QV7_TINYLLAMA_REAL_BF16_GRAD_SOURCE_LOADING",flush=True)
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,
       local_files_only=True,trust_remote_code=False,attn_implementation="eager").eval()
    if model.config.model_type!="llama" or model.config.num_hidden_layers!=LAYERS:
        raise RuntimeError("original Llama family not loaded")
    model.requires_grad_(False)
    modules=dict(model.named_modules());tensor=[]
    for ri,role in enumerate(ROLES):
        for layer in range(LAYERS):
            key=f"model.layers.{layer}.{'self_attn' if ri<4 else 'mlp'}.{role}"
            if key not in modules or modules[key].weight.ndim!=2:raise RuntimeError("missing original Llama projection "+key)
            p=modules[key].weight
            p.requires_grad_(True)
            tensor.append((role,layer,key,p))
    if len(tensor)!=154:raise RuntimeError("not exact 154 original TinyLlama tensors")
    f,quant=native_quant()
    cache=[]
    for j,(role,layer,key,param) in enumerate(tensor):
        w=param.detach().float().cpu().contiguous().numpy()
        d4,d8,check=f.quant_errors(quant,w)
        cache.append((d4,d8,check))
        if (j+1)%22==0:
            print("QV7_TINYLLAMA_C_ORIGINAL_NATIVE_Q4N8_RESIDUALS",j+1,"/154",flush=True)
    model=model.to("mps")
    measures4=[[] for _ in tensor]
    measures8=[[] for _ in tensor]
    traces=[[] for _ in tensor]
    source_losses=[]
    for ix,c in enumerate(train):
        model.zero_grad(set_to_none=True)
        ids=torch.tensor([c["prefix_ids"]],dtype=torch.long,device="mps")
        label=torch.tensor([c["suffix_ids"][c["correct"]]],dtype=torch.long,device="mps")
        prediction=model(input_ids=ids,use_cache=False,return_dict=True)
        loss=F.cross_entropy(prediction.logits[:,-1,:].float(),label)
        nll=float(loss.detach().cpu())
        if not math.isfinite(nll):raise RuntimeError("original BF16 source NLL nonfinite")
        loss.backward()
        for j,(role,layer,key,p) in enumerate(tensor):
            if p.grad is None:raise RuntimeError("autograd missing original BF16 gradient "+key)
            g=p.grad.detach().float().cpu().contiguous().numpy()
            d4,d8,_=cache[j]
            x,y,tr=f.diag_gradient_dot(g,d4,d8)
            if not all(math.isfinite(v) and v>=0 for v in (x,y,tr)):
                raise RuntimeError("TinyLlama original BF16 Fisher nonfinite")
            measures4[j].append(x);measures8[j].append(y);traces[j].append(tr)
        source_losses.append(nll)
        del prediction,loss
        torch.mps.synchronize()
        if (ix+1)%8==0:
            print("QV7_TINYLLAMA_BF16_SOURCE_ITEM",ix+1,"/64","elapsed_s",round(time.monotonic()-t,1),flush=True)
    m4=np.asarray(measures4,dtype=np.float64)
    m8=np.asarray(measures8,dtype=np.float64)
    upgrade=m4-m8
    avg=upgrade.mean(axis=1)
    odd=upgrade[:,::2].mean(axis=1)
    even=upgrade[:,1::2].mean(axis=1)
    rho=float(spearmanr(odd,even).statistic)
    if m4.shape!=(154,64) or m8.shape!=(154,64) or \
       not np.isfinite(avg).all() or not math.isfinite(rho) or max(abs(v) for v in avg)<=0:
        raise RuntimeError("154x64 original BF16 Fisher source incomplete")
    quality={role:[float(v) for v in avg[j*LAYERS:(j+1)*LAYERS]] for j,role in enumerate(ROLES)}
    data={"schema":"BEGLIN-Qv7-real-original-TinyLlama-154-projection-64-source-directional-Fisher/1",
      "status":"SOURCE_ONLY_REAL_LLAMA_ORIGINAL_BF16_64_GRAD_AND_NATIVE_C_QUANT",
      "prelock_sha256":PRE_SHA,"frozen_train64_sha256":SOURCE_SHA,
      "frozen_eval48_not_opened_sha256":HELD_SHA,
      "official_model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "official_rev":cfg["replication_revision"],"architecture":"llama",
      "source_cases":64,"original_model_layer_count":LAYERS,
      "original_model_projection_count":154,
      "per_case_source_loss":source_losses,
      "source_bf16_train_mean_correct_label_nll":float(np.mean(source_losses)),
      "per_source_case_Fisher_Q4_by_tensor":m4.tolist(),
      "per_source_case_Fisher_n8_by_tensor":m8.tolist(),
      "per_source_case_BF16_gradient_trace_by_tensor":traces,
      "source_weight_and_quant_code_sha":[c[2] for c in cache],
      "quality_upgrade_by_role_layer":quality,
      "source_split_half_odd_even_spearman":rho,
      "num_positive_upgrade":int((avg>0).sum()),
      "num_negative_upgrade":int((avg<0).sum()),
      "forward_always_original_unquantized_BF16":True,
      "no_separate_tensor_fullmodel_interventions":True,
      "BF16_MPS_grad_precision_approximation":True,
      "FP16_quant_error_cache":True,
      "original_C_encoded_native_Q4_and_n8_codes":True,
      "crosscorpus_labels_or_logits_unopened":True,
      "no_main_or_live_model_mutation":True,
      "elapsed_s":round(time.monotonic()-t,2)}
    os.umask(0o077)
    h=seal(OUT,data)
    public={k:v for k,v in data.items() if k not in (
        "per_case_source_loss","per_source_case_Fisher_Q4_by_tensor",
        "per_source_case_Fisher_n8_by_tensor","per_source_case_BF16_gradient_trace_by_tensor",
        "source_weight_and_quant_code_sha")}
    public["private_original_source_SHA256"]=h
    b=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as fp:fp.write(b);fp.flush();os.fsync(fp.fileno())
    print("QV7_PINNED_TINYLLAMA_REAL_154x64_SOURCE_GRAD_PASS",json.dumps({
      "rank_rho":rho,"positive_gain":int((avg>0).sum()),"negative_gain":int((avg<0).sum()),
      "src_private_sha":h,"public_sha":sha(b),"time":round(time.monotonic()-t,1)},sort_keys=True),flush=True)
if __name__=="__main__":evaluate()
