#!/usr/bin/env python3
"""Real output-choice Gauss-Newton source Jacobian by 3 BF16 backward per ARC item.

For 4 exact answer letter logits A/B/C/D, compute 3 independent relative
logit gradients ∂(z_A-z_D), ∂(z_B-z_D), ∂(z_C-z_D) against every one of
196 original BF16 projection tensors. Dot those gradients with *actual
native C* Q4G64 and qNg64-n8 quantization residuals, preserving sign.
p=softmax(BF16 z_A..z_D) gives conditional choice-space curvature
diag(p)-ppT. This is NOT the full-vocab Hessian or native quantized loss.

Both train24 and independently held source diagnostic24 are SHA frozen;
diagnostic may run only after source-only policy is frozen, never fit on it.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
Q5=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
Q5SRC=Q5/"research/oml_acs/q_v5_original_bf16_fisher_source.py"
ORIG_NQUANT_SHA="890c35b3b0f9ecf8905d18378b9637801d6159b96b40ce32f45e4092fad965cc"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
LAB=ROOT/"lab_local/qv9_ggn_20261023"
PRE=ROOT/"configs/research/BEGLIN_QV9_GGN_SOURCE_PRELOCK_20261023.json"
PUBLIC_INPUT=ROOT/"docs/research/BEGLIN_QV9_INPUT_PREMODEL_FREEZE_20261023.json"
POLICY=ROOT/"docs/research/BEGLIN_QV9_ROBUST_CONDITIONAL_GGN_SOURCE_POLICY_20261023.json"
SOURCES={"train":LAB/"ARC_train24_GGN_source_frozen.json",
 "diagnostic":LAB/"ARC_diag24_independent_frozen.json"}
SOURCE_HASH={"train":"87ed0d782636470b4e31db8bd6b69f0fd069dc30b167bdcb7aed1a81b53fe254",
 "diagnostic":"4714dc2be1ef4708c0099f153b15a1bd0f36b2143b006d000781bc592bf9be79"}
RESULTS={"train":LAB/"original_BF16_choice_GGN_train24_private.json",
 "diagnostic":LAB/"original_BF16_choice_GGN_diagnostic24_private.json"}
PUBLIC={"train":ROOT/"docs/research/BEGLIN_QV9_TRAIN24_CONDITIONAL_GGN_PUBLIC_20261023.json",
 "diagnostic":ROOT/"docs/research/BEGLIN_QV9_DIAGNOSTIC24_CONDITIONAL_GGN_PUBLIC_20261023.json"}
PRE_SHA="73dc2829da79637a97d68f903f72fed753eace2b154caec111c1fe60a7f4f983"
INPUT_SHA="2b770befceeed2b3670900d96cc2b36b01ff71b161758bec49a53765f2b48808"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(b):return hashlib.sha256(b).hexdigest()
def secure(path:pathlib.Path):
    if not path.is_file() or path.is_symlink() or path.stat().st_mode&0o777!=0o600:
        raise RuntimeError("unprotected original BF16 GGN private evidence "+str(path))
    return path.read_bytes()
def put(path,obj):
    data=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
    with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
    return sha(data)
def validate(stage):
    if sha(PRE.read_bytes())!=PRE_SHA or sha(PUBLIC_INPUT.read_bytes())!=INPUT_SHA:
        raise RuntimeError("original Qv9 prereg/frozen input SHA changed")
    if sha(secure(SOURCES[stage]))!=SOURCE_HASH[stage]:
        raise RuntimeError("source 24 original Qwen tokenizer IDs modified")
    if stage=="diagnostic":
        if not POLICY.is_file():raise RuntimeError("cannot open original diagnostic before policy frozen")
        pol=json.loads(POLICY.read_text())
        if pol.get("status")!="GGN_SOURCE_ONLY_FROZEN_BEFORE_DIAG_OR_MMLU":
            raise RuntimeError("diagnostic cohort not read before source policy sealed")
    if RESULTS[stage].exists() or PUBLIC[stage].exists():
        raise FileExistsError("original conditional GGN source cohort already finalized")
    if sha(Q5SRC.read_bytes())!=ORIG_NQUANT_SHA:
        raise RuntimeError("original audited native C quantizer loader source drift")
    x=json.loads(secure(SOURCES[stage]))["questions"]
    if len(x)!=24:raise RuntimeError("original Qv9 GGN cohort count not 24")
    return x
def native_quant_residuals(model):
    imp=importlib.util.spec_from_file_location("qv5_native_real_encoder",str(Q5SRC))
    module=importlib.util.module_from_spec(imp);imp.loader.exec_module(module)
    lib=module.lib_quant()
    layers=dict(model.named_modules())
    named=[];deltas=[]
    for j,role in enumerate(ROLES):
        for layer in range(28):
            key=f"model.layers.{layer}.{'self_attn' if j<4 else 'mlp'}.{role}"
            param=layers[key].weight
            param.requires_grad_(True)
            weight=param.detach().float().cpu().contiguous().numpy()
            d4,d8,proof=module.quant_errors(lib,weight)
            named.append((role,layer,param))
            deltas.append((d4,d8,proof))
        print("QV9_REAL_C_NATIVE_QUANT_RESIDUALS",role,len(named),"/196",flush=True)
    if len(named)!=196:raise RuntimeError("wrong BF16 Qwen GGN dimension")
    return named,deltas
def case_resume(folder,idx,row):
    path=folder/f"choice_jacobian_{idx:02d}.json"
    if not path.exists():return None
    x=json.loads(secure(path))
    if x.get("schema")!="BEGLIN-Qv9-real-original-BF16-4-choice-Jacobian-case/1" or \
       x.get("index")!=idx or x.get("source_question_id_SHA256")!=sha(row["id"].encode()) or \
       len(x["signed_Q4_relative_jacobian_196x3"])!=196 or \
       len(x["signed_n8_relative_jacobian_196x3"])!=196:
        raise RuntimeError("existing per-case original BF16 curvature data modified")
    return x
def run(stage):
    samples=validate(stage)
    os.umask(0o077)
    folder=LAB/("train24_choice_Jacobians" if stage=="train" else "diag24_choice_Jacobians")
    folder.mkdir(mode=0o700,parents=True,exist_ok=True)
    if folder.is_symlink() or (folder.stat().st_mode&0o777)!=0o700:
        raise RuntimeError("original per-case gradient folder must be 0700")
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError("Apple BF16 MPS required")
    began=time.monotonic()
    print("QV9_ORIGINAL_BF16_GGN_MODEL_LOADING",stage,flush=True)
    model=AutoModelForCausalLM.from_pretrained(str(SNAP),dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False,attn_implementation="eager").eval()
    model.requires_grad_(False)
    named,residuals=native_quant_residuals(model)
    model=model.to("mps")
    for index,row in enumerate(samples):
        if case_resume(folder,index,row) is not None:
            print("QV9_GGN_SHA_REUSE",stage,index+1,flush=True);continue
        ids=torch.tensor([row["prefix_ids"]],dtype=torch.long,device="mps")
        valid=[row["suffix_ids"][k] for k in "ABCD"]
        if len(set(valid))!=4:raise RuntimeError("not distinct source A/B/C/D true tokens")
        # One full BF16 forward per question, 3 relative-choice-logit backwards.
        model.zero_grad(set_to_none=True)
        outputs=model(input_ids=ids,use_cache=False)
        zl=outputs.logits[0,-1,:].float()[valid]
        p=F.softmax(zl,dim=0).detach().float().cpu().numpy().astype(np.float64)
        if not np.isfinite(p).all() or np.min(p)<0 or abs(float(p.sum())-1)>1e-5:
            raise RuntimeError("original BF16 choice-probability source not normalized")
        h4=np.empty((196,3),dtype=np.float64)
        h8=np.empty((196,3),dtype=np.float64)
        for alt in range(3):
            model.zero_grad(set_to_none=True)
            (zl[alt]-zl[3]).backward(retain_graph=(alt<2))
            for j,(_,_,param) in enumerate(named):
                if param.grad is None:raise RuntimeError("original BF16 contrastive logit gradient missing")
                grad=param.grad.detach().float().cpu().contiguous().numpy()
                e4,e8,_=residuals[j]
                g=grad.ravel()
                h4[j,alt]=np.einsum("i,i->",g,e4.ravel().astype(np.float32),dtype=np.float64)
                h8[j,alt]=np.einsum("i,i->",g,e8.ravel().astype(np.float32),dtype=np.float64)
        if not np.isfinite(h4).all() or not np.isfinite(h8).all():
            raise RuntimeError("source conditional GGN original Jacobian contracted residual nonfinite")
        # Signed relative logit contractions are protected; do not save raw model logits.
        case={"schema":"BEGLIN-Qv9-real-original-BF16-4-choice-Jacobian-case/1",
            "index":index,"source_question_id_SHA256":sha(row["id"].encode()),
            "probability_of_4_options_original_BF16":p.tolist(),
            "signed_Q4_relative_jacobian_196x3":h4.tolist(),
            "signed_n8_relative_jacobian_196x3":h8.tolist(),
            "GGN_relative_logit_anchor":"D", "original_Llama_or_Qwen":"qwen2",
            "original_BF16_original_weights_only":True,
            "num_full_BF16_forwards":1,"num_relative_BF16_backwards":3,
            "full_vocab_GGN_is_not_computed":True,"quantized_final_model_forward_not_executed":True}
        h=put(folder/f"choice_jacobian_{index:02d}.json",case)
        torch.mps.synchronize()
        if (index+1)%4==0:
            print("QV9_REAL_CHOICE_GGN_SOURCE_CASE",stage,index+1,"/24",
                 "case_SHA",h[:16],"elapsed_s",round(time.monotonic()-began,1),flush=True)
        del outputs,zl
    cases=[]
    casehashes=[]
    for ix,q in enumerate(samples):
        item=case_resume(folder,ix,q)
        if item is None:raise RuntimeError("incomplete original BF16 input choice Jacobians")
        cases.append(item);casehashes.append(sha(secure(folder/f"choice_jacobian_{ix:02d}.json")))
    q4=np.asarray([c["signed_Q4_relative_jacobian_196x3"] for c in cases],dtype=np.float64)
    n8=np.asarray([c["signed_n8_relative_jacobian_196x3"] for c in cases],dtype=np.float64)
    ps=np.asarray([c["probability_of_4_options_original_BF16"] for c in cases],dtype=np.float64)
    if q4.shape!=(24,196,3) or n8.shape!=(24,196,3) or ps.shape!=(24,4):
        raise RuntimeError("wrong GGN choice contrastive source shape")
    def risk_single(d,probs):
        z=np.concatenate([d,np.zeros(d.shape[:-1]+(1,),dtype=np.float64)],axis=-1)
        mu=np.sum(z*probs,axis=-1)
        return 0.5*(np.sum(z*z*probs,axis=-1)-mu*mu)
    allq4_shifts=np.sum(q4,axis=1)
    alln8_shifts=np.sum(n8,axis=1)
    source_base=float(np.mean(risk_single(allq4_shifts,ps)))
    if not np.isfinite([source_base,float(np.mean(risk_single(alln8_shifts,ps)))]).all():
        raise RuntimeError("nonfinite original BF16 conditional GGN curvature scalar")
    receipt={
      "schema":"BEGLIN-Qv9-real-BF16-conditional-4choice-output-GGN24/1",
      "status":"ORIGINAL_BF16_24x196x3_CHOICE_LOGIT_CONTRASTIVE_JACOBIAN_COMPLETE",
      "source_group":stage,"original_checkpoint_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "prelock_sha256":PRE_SHA,"frozen_source_token_input_SHA256":SOURCE_HASH[stage],
      "frozen_other_source_group_untouched_SHA256":SOURCE_HASH["diagnostic" if stage=="train" else "train"],
      "q4_signed_relative_logit_delta":q4.tolist(),
      "n8_signed_relative_logit_delta":n8.tolist(),
      "original_conditional_four_letter_probabilities":ps.tolist(),
      "per_case_original_BF16_forward_and_backward_receipts_sha256":casehashes,
      "n_tokens_choice":4,"n_contrastive_logits_backprop":3,
      "n_source_examples":24,"model_projection_tensors":196,
      "GGN_fullvocab_loss_not_claimed":True,
      "GGN_true_full_Hessian_not_claimed":True,
      "GGN_conditional_4choice_0p5_var_correct":True,
      "source_mean_Q4_conditional_GGN":source_base,
      "no_single_tensor_downstream_output_intervention":True,
      "no_model_weight_mutation":True,"no_live_runtime_mutation":True,
      "elapsed_s":round(time.monotonic()-began,2)
    }
    fullsha=put(RESULTS[stage],receipt)
    public={k:v for k,v in receipt.items() if k not in (
        "q4_signed_relative_logit_delta","n8_signed_relative_logit_delta",
        "original_conditional_four_letter_probabilities",
        "per_case_original_BF16_forward_and_backward_receipts_sha256")}
    public["full_private_receipt_sha256"]=fullsha
    b=(json.dumps(public,indent=2,sort_keys=True)+"\n").encode()
    with PUBLIC[stage].open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    print("QV9_OUTPUT_CONDITIONAL_GGN_REAL_BF16_SUCCESS",
          json.dumps({"stage":stage,"public_sha":sha(b),"private_sha":fullsha,
           "Q4_choice_GGN":source_base,"model_calls":24,"backwards":72},sort_keys=True),flush=True)
if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--stage",choices=("train","diagnostic"),required=True)
    run(arg.parse_args().stage)
