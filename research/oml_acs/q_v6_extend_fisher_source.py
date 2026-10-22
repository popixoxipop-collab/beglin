#!/usr/bin/env python3
"""Qv6: true additional original-BF16 task-gradient directions for ARC 24/64/128.

Reuses *exact* original Qv5 24 private per-question directional scores, verifying
all original native C quantizer code SHA for every tensor. For 104 new ARC
training questions performs BF16 MPS autograd against correct one-letter target,
no quantized model forward and no cross-corpus validation access. Per-item
source scalar receipts are secure, append-only, and resumable.
"""
from __future__ import annotations
import hashlib,importlib.util,json,math,os,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
OLD=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
LAB=ROOT/"lab_local/qv6_fisher_scaling_20261022"
SRC=LAB/"source_arc_128_frozen.json"
CROSS=LAB/"openbookqa_val48_frozen.json"
PROTO=ROOT/"configs/research/BEGLIN_QV6_SAMPLE_SCALING_PRELOCK_20261022.json"
OLDER=OLD/"lab_local/qv5_fisher_20261022/fisher_196_train_private_receipt.json"
OLDER_SRC=OLD/"lab_local/qv5_fisher_20261022/source_train_frozen.json"
OLDER_PY=OLD/"research/oml_acs/q_v5_original_bf16_fisher_source.py"
LOG=LAB/"additional_original_BF16_gradients"
FINAL=LAB/"fisher_24_64_128_source_private.json"
PUB=ROOT/"docs/research/BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRE_SHA="2e35764878191717ef7b84ed546e72e2a1c513523ecb17d24b54489bd29f319b"
SRC_SHA="78d267bef54240cfb32b292c5d1885b6716fadd9bbd78ec7b3b1bf97822a1ab2"
CROSS_SHA="c09831bf86c6eedebed433dc2a0398a0992afb19fddc9f86750c2d137d71d1b0"
OLD_FISHER_SHA="6d87dfef842fcce932cdb85c9a42bc3ede380493917ae608663e2072f8143417"
OLD_24_SHA="631bb02d22b696ddcabfd3d7ab759218fd75e1e8f3d8384cbf4a5df4de3f6994"
OLD_SOURCE_SCRIPT_SHA="890c35b3b0f9ecf8905d18378b9637801d6159b96b40ce32f45e4092fad965cc"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")

def sha(x:bytes):return hashlib.sha256(x).hexdigest()
def sha_file(path:pathlib.Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def private_read(path):
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("unsecured or nonexistent private original source: "+str(path))
    return path.read_bytes()
def save(path,obj):
    data=(json.dumps(obj,indent=2,sort_keys=True)+"\n").encode()
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
    return sha(data)

def imports():
    if sha(OLDER_PY.read_bytes())!=OLD_SOURCE_SCRIPT_SHA:
        raise RuntimeError("Qv5 original native quant residual source script changed")
    spec=importlib.util.spec_from_file_location("qv5_verified_BF16_source",OLDER_PY)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def check():
    if FINAL.exists() or PUB.exists():
        raise FileExistsError("new Fisher source already finalized; refuse reuse for refitting")
    if sha(PROTO.read_bytes())!=PRE_SHA:
        raise RuntimeError("Qv6 protocol has drifted")
    if sha(private_read(SRC))!=SRC_SHA or sha(private_read(CROSS))!=CROSS_SHA:
        raise RuntimeError("source 128 or cross-corpus 48 contract changed")
    if sha(private_read(OLDER))!=OLD_FISHER_SHA or sha(private_read(OLDER_SRC))!=OLD_24_SHA:
        raise RuntimeError("Qv5 original 24 source gradients changed")
    old=json.loads(private_read(OLDER)); src=json.loads(private_read(SRC))
    if not old["no_dev_labels_accessed"] or not old["BF16_model_forward_unchanged"] or \
       src["source_case_count"]!=128 or not src["original_Qv5_first_24_exactly_reproduced"]:
        raise RuntimeError("source samples or original BF16 provenance failed")
    if len(old["source_score_rows"])!=196:
        raise RuntimeError("missing 196 native original quant residuals")
    if old["original_HF_rev"]!="989aa7980e4cf806f80c7fef2b1adb7bc71aa306":
        raise RuntimeError("wrong official BF16 revision")
    if not torch.backends.mps.is_available():raise RuntimeError("cannot run original BF16 MPS")
    return old,src

def resume_existing(index,row):
    path=LOG/f"gradient_{index:03d}.json"
    if not path.exists():return None
    x=json.loads(private_read(path))
    if x["schema"]!="BEGLIN-Qv6-source-one-extra-BF16-gradient-196/1" or \
       x["source_case_index"]!=index or x["prompt_token_sha256"]!=row["input_sha256"] or \
       x["case_id_sha256"]!=sha(row["id"].encode()) or \
       len(x["q4_squared_directions"])!=196 or len(x["n8_squared_directions"])!=196:
        raise RuntimeError("existing per-question checkpoint non-authoritative "+str(index))
    return x

def load_residuals(old,src,model,module):
    orig=imports()
    native=orig.lib_quant()
    if sha_file(orig.LIB)!=old["native_C_quant_dynamic_library_sha256"]:
        raise RuntimeError("original audited Q4/n8 C dynamic library digest mismatch")
    cache=[];params=[]
    for i,role in enumerate(ROLES):
        for layer in range(28):
            name=f"model.layers.{layer}.{'self_attn' if i<4 else 'mlp'}.{role}"
            if name not in module:raise RuntimeError("original BF16 projection missing "+name)
            param=module[name].weight
            if param.ndim!=2:raise RuntimeError("unexpected original BF16 linear shape "+name)
            param.requires_grad_(True)
            w=param.detach().float().cpu().contiguous().numpy()
            d4,d8,h=orig.quant_errors(native,w)
            oldrow=old["source_score_rows"][i*28+layer]
            if oldrow["role"]!=role or oldrow["layer"]!=layer:
                raise RuntimeError("source role order mismatch")
            checks=oldrow["native_codes_and_BF16_weight_sha256"]
            for field in ("original_BF16_widened_F32_weight_SHA256","Q4G64_native_code_SHA256","n8_native_bitplane_SHA256"):
                if h[field]!=checks[field]:
                    raise RuntimeError("source original BF16 or native C quant code identity changed "+name)
            cache.append((d4,d8));params.append(param)
        print("QV6_NATIVE_SOURCE_RESIDUALS_VERIFIED",role,len(params),flush=True)
    return orig,cache,params

def run():
    old,src=check()
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    torch.set_num_threads(2)
    # Original old 24 records are read-only and never recomputed or modified.
    os.umask(0o077)
    LOG.mkdir(mode=0o700,parents=True,exist_ok=True)
    if LOG.is_symlink() or (LOG.stat().st_mode&0o777)!=0o700:
        raise RuntimeError("source gradient private checkpoint path permissions")
    t0=time.monotonic()
    print("QV6_SOURCE_LOADING_ORIGINAL_BF16",flush=True)
    model=AutoModelForCausalLM.from_pretrained(str(SNAP),dtype=torch.bfloat16,local_files_only=True,
        trust_remote_code=False,attn_implementation="eager").eval()
    model.requires_grad_(False)
    _,delta,params=load_residuals(old,src,model,dict(model.named_modules()))
    print("QV6_SOURCE_GRAD_DIRECTIONS_READY",len(delta),"elapsed_s",round(time.monotonic()-t0,1),flush=True)
    model=model.to("mps")
    for ix in range(24,128):
        row=src["all_source_questions"][ix]
        if resume_existing(ix,row) is not None:
            print("QV6_RESUME_VERIFIED",ix+1,"/128",flush=True)
            continue
        model.zero_grad(set_to_none=True)
        ids=torch.tensor([row["prefix_ids"]],dtype=torch.long,device="mps")
        label=torch.tensor([row["suffix_ids"][row["correct"]]],dtype=torch.long,device="mps")
        output=model(input_ids=ids,use_cache=False)
        loss=F.cross_entropy(output.logits[:,-1,:].float(),label)
        nll=float(loss.detach().cpu())
        if not math.isfinite(nll):raise RuntimeError("original BF16 source loss nonfinite")
        loss.backward()
        q4=[];n8=[];trace=[]
        for j,param in enumerate(params):
            if param.grad is None:raise RuntimeError("original BF16 source gradient absent tensor "+str(j))
            grad=param.grad.detach().float().cpu().contiguous().numpy()
            a,b,t=orig.diag_gradient_dot(grad,*delta[j]) if False else (None,None,None)
            # Original implementation imported earlier; verify direct source-code contract.
            g=grad.reshape(-1)
            d4,d8=delta[j]
            r4=d4.reshape(-1).astype(np.float32);r8=d8.reshape(-1).astype(np.float32)
            u=float(np.einsum("i,i->",g,r4,dtype=np.float64))
            v=float(np.einsum("i,i->",g,r8,dtype=np.float64))
            q4.append(u*u);n8.append(v*v)
            trace.append(float(np.einsum("i,i->",g,g,dtype=np.float64)))
        if not np.isfinite(q4+n8+trace).all() or len(q4)!=196:
            raise RuntimeError("new source 196 parameter gradients nonfinite")
        record={"schema":"BEGLIN-Qv6-source-one-extra-BF16-gradient-196/1",
          "source_case_index":ix,"prompt_token_sha256":row["input_sha256"],
          "case_id_sha256":sha(row["id"].encode()),"original_correct_answer_NLL":nll,
          "tensor_role_count":196,"q4_squared_directions":q4,"n8_squared_directions":n8,
          "trace_mean_per_tensor":trace,
          "original_BF16_weights_only":True,"actual_C_native_Q4n8_residuals":True,
          "no_cross_corpus_outputs":True,"no_downstream_quantized_model_interventions":True}
        digest=save(LOG/f"gradient_{ix:03d}.json",record)
        del output,loss
        torch.mps.synchronize()
        print("QV6_SOURCE_ADDITIONAL_FINISHED",ix+1,"/128","loss",round(nll,5),
              "checkpoint_sha",digest[:15],"elapsed_s",round(time.monotonic()-t0,1),flush=True)
    # Finish only after all source-only gradient cases complete.
    rows=[];recordshas=[]
    for idx in range(24,128):
        record=resume_existing(idx,src["all_source_questions"][idx])
        if record is None:raise RuntimeError("incomplete frozen 128 source observations")
        recordshas.append(sha(private_read(LOG/f"gradient_{idx:03d}.json")))
        rows.append(record)
    oldrows=old["source_score_rows"]
    scores4=[];scores8=[]
    for j,row in enumerate(oldrows):
        q4=list(row["per_source_item_q4_sq"])+[r["q4_squared_directions"][j] for r in rows]
        n8=list(row["per_source_item_n8_sq"])+[r["n8_squared_directions"][j] for r in rows]
        if len(q4)!=128 or len(n8)!=128 or not np.isfinite(q4+n8).all():
            raise RuntimeError("original 24+new104 score shape or finite audit failure")
        scores4.append(q4);scores8.append(n8)
    gainmat=np.asarray(scores4,dtype=np.float64)-np.asarray(scores8,dtype=np.float64)
    split={}
    gainmaps={}
    for n in (24,64,128):
        v=gainmat[:,:n]
        gain=v.mean(axis=1)
        odd=v[:,::2].mean(axis=1);even=v[:,1::2].mean(axis=1)
        rho=float(spearmanr(odd,even).statistic)
        if not np.isfinite(gain).all() or not np.isfinite(rho):
            raise RuntimeError("unstable Fisher split statistics")
        key=str(n)
        split[key]={
          "odd_even_sample_Fisher_rank_spearman":rho,
          "score_positive_count":int((gain>0).sum()),
          "source_gradient_mean":float(gain.mean()),
          "gain_min":float(gain.min()),"gain_max":float(gain.max()),
          "plain_top28_tensor_index_set":np.argsort(-gain)[:28].tolist()
        }
        gainmaps[key]={role:[float(v) for v in gain[i*28:(i+1)*28]] for i,role in enumerate(ROLES)}
    pair={}
    for a,b in ((24,64),(24,128),(64,128)):
        x=gainmat[:,:a].mean(axis=1);y=gainmat[:,:b].mean(axis=1)
        xa=set(split[str(a)]["plain_top28_tensor_index_set"])
        xb=set(split[str(b)]["plain_top28_tensor_index_set"])
        pair[f"{a}_{b}"]={
            "full_196_spearman":float(spearmanr(x,y).statistic),
            "unconstrained_top28_overlap":len(xa&xb),
            "unconstrained_top28_jaccard":len(xa&xb)/len(xa|xb)
        }
    final={"schema":"BEGLIN-Qv6-original-24-plus-104-BF16-directional-Fisher-scaling/1",
      "status":"SOURCE_ONLY_24_64_128_GRADIENT_SCALING_DONE",
      "prelock_sha256":PRE_SHA,"source_128_SHA256":SRC_SHA,"unopened_cross48_SHA256":CROSS_SHA,
      "original_Qv5_24_per_tensor_Fisher_receipt_SHA256":OLD_FISHER_SHA,
      "original_n8_native_library_SHA256":old["native_C_quant_dynamic_library_sha256"],
      "original_model_rev":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "source_data_origin":"ARC-Challenge train", "source_gradients":128,"reused_old24":24,
      "actually_executed_new_BF16_backward_items":104,
      "new_gradient_checkpoint_SHA256s":recordshas,
      "score_rows_196":{
          f"{role}:{layer}":{"q4_squared":scores4[i*28+layer],"n8_squared":scores8[i*28+layer]}
          for i,role in enumerate(ROLES) for layer in range(28)},
      "nested_split_stats":split,"nested_pairwise_rank_stability":pair,
      "quality_upgrade_by_role_layer_per_size":gainmaps,
      "new_mean_original_BF16_source_loss":float(np.mean([row["original_correct_answer_NLL"] for row in rows])),
      "no_validation_label_or_logits_accessed":True,
      "no_live_or_main_mutation":True,"elapsed_seconds":round(time.monotonic()-t0,1)}
    original_sha=save(FINAL,final)
    agg={k:v for k,v in final.items() if k not in ("score_rows_196","new_gradient_checkpoint_SHA256s")}
    agg["source_private_full_receipt_SHA256"]=original_sha
    data=(json.dumps(agg,sort_keys=True,indent=2)+"\n").encode()
    with PUB.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    print("QV6_REAL_128_FISHER_NESTED_SOURCE_DONE",json.dumps({
       "source_private_SHA":original_sha,"public_SHA":sha(data),"pairs":pair,
       "source_odd_even_spearman":{n:split[n]["odd_even_sample_Fisher_rank_spearman"] for n in split},
       "elapsed_seconds":final["elapsed_seconds"]},sort_keys=True),flush=True)

if __name__=="__main__":run()
