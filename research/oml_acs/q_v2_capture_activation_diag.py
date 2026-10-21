#!/usr/bin/env python3
"""BEGLIN Q v2 BF16 activation-only calibration capture (no output interventions).

Use original pinned HF Qwen2.5-1.5B on one CUDA GPU, local-only. Register
196 forward prehooks for *unmodified* source BF16 tensors. Aggregate each
projection's input-channel squared second moments over exactly the public
calibration prompts prelocked in source JSON; NO logits, labels, gradients,
row text, raw activations, test prompts, old V6 or model modifications.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,sys,time
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=pathlib.Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json"
MODEL=pathlib.Path("/home/alienware-r13/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
OUT=pathlib.Path.home()/".local/share/beglin/oml_acs/20261022_qv2_retry1"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def digest(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def atomic_private(path,data:bytes):
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def main():
    os.umask(0o077)
    if not torch.cuda.is_available():raise RuntimeError("PINNED BF16 cuda required")
    if not MODEL.is_dir() or not CONFIG.is_file():raise RuntimeError("pinned model or preregistration missing")
    cfgraw=CONFIG.read_bytes();config=json.loads(cfgraw)
    if config["status"]!="LOCKED_BEFORE_NEW_ACTIVATION_EXTRACTION_OR_TEST_NLL":raise RuntimeError("prereg not frozen")
    if config["model"]["revision"]!=MODEL.name:raise RuntimeError("model revision mismatch")
    texts=config["calibration_public_only"]
    assert len(texts)==12 and len(set(texts))==12
    if OUT.exists():raise FileExistsError("refuse to overwrite captured private calibration")
    OUT.mkdir(parents=True,mode=0o700)
    tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=False,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,
        trust_remote_code=False,local_files_only=True,attn_implementation="sdpa")
    model=model.eval().to("cuda:0")
    named=dict(model.named_modules())
    required={}
    for layer in range(28):
        for role in ROLES:
            segment="self_attn" if role in ROLES[:4] else "mlp"
            key=f"model.layers.{layer}.{segment}.{role}"
            if key not in named or not hasattr(named[key],"weight"):
                raise RuntimeError("missing original Qwen module "+key)
            required[key]=named[key]
    if len(required)!=196:raise RuntimeError("not 196 original projections")
    accum={name:torch.zeros(int(mod.weight.shape[1]),device="cuda",dtype=torch.float64)
           for name,mod in required.items()}
    count={name:0 for name in required}
    hooks=[]
    for name,module in required.items():
        def make(name):
            def hook(module,inputs):
                x=inputs[0]
                if x is None or x.shape[-1]!=accum[name].shape[0]:
                    raise RuntimeError("input dimension drift "+name)
                xx=x.detach().reshape(-1,x.shape[-1]).to(torch.float64)
                accum[name].add_(torch.sum(xx*xx,dim=0))
                count[name]+=int(xx.shape[0])
            return hook
        hooks.append(module.register_forward_pre_hook(make(name)))
    t0=time.monotonic()
    token_lengths=[]
    try:
        with torch.inference_mode():
            for idx,text in enumerate(texts):
                encoded=tok.apply_chat_template([{"role":"user","content":text}],
                   tokenize=True,add_generation_prompt=True,return_tensors="pt")
                # Transformers 5.12 returns BatchEncoding rather than bare tensor.
                ids=encoded["input_ids"] if hasattr(encoded,"__getitem__") and not torch.is_tensor(encoded) else encoded
                if not torch.is_tensor(ids) or ids.ndim!=2 or ids.shape[0]!=1 or not (8<=ids.shape[1]<=90):
                    raise RuntimeError("Qwen template returned invalid token tensor")
                token_lengths.append(int(ids.shape[1]))
                model(input_ids=ids.to("cuda:0"),use_cache=False,return_dict=True)
                print("ACTIVATION_CAPTURE_PROGRESS",idx+1,len(texts),
                      "tokens",token_lengths[-1],flush=True)
    finally:
        for hook in hooks:hook.remove()
    torch.cuda.synchronize()
    if len(count)!=196 or set(count.values())!={sum(token_lengths)}:
        raise RuntimeError("calibration input sample counts differ across Qwen 196 modules")
    stats={};size=0
    for name in sorted(accum):
        v=(accum[name]/count[name]).float().cpu().numpy().astype("<f4")
        if not (np.isfinite(v).all() and (v>=0).all() and v.sum()>0):
            raise RuntimeError("bad squared source activation moment "+name)
        key=name.replace("model.layers.","layer").replace(".","__")
        stats[key]=v
        size+=v.size
    arrays=OUT/"bf16_calibration_196_input_diag_moments.npz"
    with arrays.open("xb") as f:
        np.savez_compressed(f,**stats)
        f.flush();os.fsync(f.fileno())
    arrays.chmod(0o600)
    manifest={
      "schema":"beglin-Q-bf16-activation-input-second-moments/1",
      "status":"PASS_SOURCE_ORIGINAL_BF16_HOOKS_NO_QUANT_INTERVENTIONS",
      "source_snapshot_rev":MODEL.name,
      "preregistered_config_sha256":digest(cfgraw),
      "calibration_prompts_count":len(texts),
      "source_public_prompt_sha256":digest(json.dumps(texts,sort_keys=True).encode()),
      "token_lengths":token_lengths,
      "observed_tokens_per_module":sum(token_lengths),
      "module_count":len(required),
      "sum_feature_count":size,
      "arrays_sha256":digest(arrays.read_bytes()),
      "arrays_private_0600":True,
      "GPU":torch.cuda.get_device_name(0),
      "elapsed_seconds":round(time.monotonic()-t0,2),
      "type":"activation diagonal input variance only; no full Hessian, no logits, no target labels",
      "calibration_only_not_evaluation":True,
      "model_original_unchanged":True,"main_unchanged":True,
      "production_unchanged":True,
      "V6_24_holdout_untouched":True
    }
    target=OUT/"activation_diagonal_manifest.json"
    atomic_private(target,(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_Q_V2_ACTIVATION_SOURCE_PASS",json.dumps(manifest,sort_keys=True),flush=True)
if __name__=="__main__":main()
