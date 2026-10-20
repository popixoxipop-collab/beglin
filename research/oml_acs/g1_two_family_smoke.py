"""EXPLORATORY Qwen->Mistral G1 technical transfer, NOT preregistered ACS.
Cached BF16 only; no network calls, official revisions pinned. Synthetic
anchors/tests prohibit statistical/scientific generalization claims.
"""
import os, time, gc, json
from pathlib import Path
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hf_adapter import ModelSpec, PinnedHF
from acs_core import AnchorProjector, source_axis_direction, unit
t0=time.perf_counter()
QREV="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MREV="c170c708c41dac9275d15a8fff4eca08d52bab71"
CACHE=Path.home()/".cache/huggingface/hub"
QPATH=CACHE/"models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"/QREV
MPATH=CACHE/"models--mistralai--Mistral-7B-Instruct-v0.3/snapshots"/MREV
if not (QPATH/"model.safetensors").is_file() or len(list(MPATH.glob("*.safetensors")))!=3:
    raise RuntimeError("expected pinned locally cached full checkpoints missing")
A=[
"Describe rainfall in one sentence.",
"Explain a classic physics experiment.",
"What does photosynthesis mean?",
"Compare the integers six and nine.",
"Write a poem about a sunrise.",
"Explain why animals need water.",
"Describe a simple bread recipe.",
"What is the purpose of a library?",
"How is ice different from steam?",
"Summarize a short bicycle journey.",
"Describe the color of the ocean.",
"What is an elementary triangle?"
]
POS=[
"This film was a brilliant masterpiece.",
"The food and service were excellent.",
"The product works and I really love it.",
"The book was exciting and delightful."
]
NEG=[
"This film was a boring disappointment.",
"The food and service were terrible.",
"The product failed and I really hate it.",
"The book was tedious and unpleasant."
]
TEST_POS=["The warm meal was unexpectedly delicious.","The game was exciting and fantastic.",
          "I adored the beautiful performance.","The trip was pleasant and relaxing."]
TEST_NEG=["The cold meal was absolutely disgusting.","The game was dull and frustrating.",
          "I disliked the terrible performance.","The trip was unpleasant and exhausting."]
print("STAGE source_loading",flush=True)
src=PinnedHF(ModelSpec("Qwen/Qwen2.5-1.5B-Instruct",QREV,28,14),
             local_snapshot_path=str(QPATH),device="cuda")
anchor_s=np.stack([src.last_token(v) for v in A])
p_s=np.stack([src.last_token(v) for v in POS])
n_s=np.stack([src.last_token(v) for v in NEG])
projector_s=AnchorProjector.fit(anchor_s)
native_s=source_axis_direction(p_s,n_s)
canonical=unit(projector_s.project(native_s,treat_as_point=False))
print("STAGE source_projection",anchor_s.shape,"canonical_dim",len(canonical),
      "elapsed",round(time.perf_counter()-t0,1),flush=True)
del src,anchor_s,p_s,n_s,projector_s,native_s
gc.collect()
torch.cuda.empty_cache()
print("STAGE target_loading",flush=True)
tok=AutoTokenizer.from_pretrained(str(MPATH),local_files_only=True,trust_remote_code=False)
target=AutoModelForCausalLM.from_pretrained(str(MPATH),
    local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,
    device_map="auto",max_memory={0:"12GiB","cpu":"20GiB"},low_cpu_mem_usage=True).eval()
block=target.model.layers[16]
print("STAGE target_loaded",round(time.perf_counter()-t0,1),
      "device_map",str(getattr(target,"hf_device_map",None))[:200],flush=True)
def encode(q):
    inp=tok.apply_chat_template([{"role":"user","content":q}],
                tokenize=True,return_tensors="pt",return_dict=True,add_generation_prompt=True)
    return {k:v.to(target.device) for k,v in inp.items() if k in ("input_ids","attention_mask")}
def extract(q):
    captured=[]
    def read(_m,_i,out):
        h=out[0] if isinstance(out,tuple) else out
        captured.append(h[0,-1,:].detach().float().cpu().numpy().copy())
    hook=block.register_forward_hook(read)
    try:
        with torch.inference_mode():
            target(**encode(q),use_cache=False)
    finally: hook.remove()
    if len(captured)!=1 or captured[0].shape!=(4096,):
        raise RuntimeError("Mistral activation shape or hook count incorrect")
    return captured[0]
anchor_t=np.stack([extract(v) for v in A])
proj_t=AnchorProjector.fit(anchor_t)
transferred=proj_t.backproject(canonical)
hpos=np.stack([extract(v) for v in TEST_POS])
hneg=np.stack([extract(v) for v in TEST_NEG])
def signal(p, c):
    return np.array([float(p.project(h,treat_as_point=True)@c) for h in hpos]),np.array([
                     float(p.project(h,treat_as_point=True)@c) for h in hneg])
p_true,n_true=signal(proj_t,canonical)
rng=np.random.default_rng(42)
permuted=AnchorProjector(proj_t.mu,proj_t.ahat[rng.permutation(len(A))])
p_shuf,n_shuf=signal(permuted,canonical)
def sep(p,n):return int(sum(float(a>b) for a in p for b in n)),int(len(p)*len(n))
print("STAGE crossfamily_activations",anchor_t.shape, "rank",sep(p_true,n_true),
      "shuffled",sep(p_shuf,n_shuf), "elapsed",round(time.perf_counter()-t0,1),flush=True)
generation_prompt="Describe a typical movie in one short sentence."
def gen(vec=None,alpha=0):
    inputs=encode(generation_prompt)
    hook=None
    if vec is not None:
        v=torch.tensor(unit(vec),dtype=torch.float32)
        def add(_m,_i,out):
            h=out[0] if isinstance(out,tuple) else out
            updated=h+(float(alpha)*v).to(h.device,dtype=h.dtype).view(1,1,-1)
            return (updated,*out[1:]) if isinstance(out,tuple) else updated
        hook=block.register_forward_hook(add)
    try:
        with torch.inference_mode():
            g=target.generate(**inputs,do_sample=False,max_new_tokens=10,
                       pad_token_id=tok.eos_token_id)
    finally:
        if hook is not None:hook.remove()
    return tok.decode(g[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
base=gen()
steered=gen(transferred,3.)
torch.cuda.synchronize()
print("G1_CROSS_FAMILY_TECHNICAL_RECEIPT "+json.dumps({
"schema":"oml-acs-g1-two-family-technical-smoke/1",
"verdict":"ENGINEERING_ONLY_NONPREREGISTERED",
"source":"Qwen/Qwen2.5-1.5B-Instruct","source_revision":QREV,
"target":"mistralai/Mistral-7B-Instruct-v0.3","target_revision":MREV,
"precision":"BF16","anchor_kind":"synthetic_not_HELM","anchor_count":len(A),
"source_contrast_pairs":len(POS),"target_eval_positives":len(TEST_POS),
"target_eval_negatives":len(TEST_NEG),"target_hidden_dim":4096,
"target_layer_0based":16,"target_anchor_shape":list(anchor_t.shape),
"pairwise_rank_target":list(sep(p_true,n_true)),
"pairwise_rank_shuffled":list(sep(p_shuf,n_shuf)),
"positive_scores":p_true.tolist(),"negative_scores":n_true.tolist(),
"control_generation":base,"steered_generation_alpha3":steered,
"generated_text_changed":base!=steered,
"peak_cuda_mb":round(torch.cuda.max_memory_allocated()/2**20,1),
"elapsed_seconds":round(time.perf_counter()-t0,1),
"notes":["Original 300 HELM anchors not used","Two-family single-source feasibility only; not the planned three-model leave-one-out experiment",
"Source and target synthetic examples are different, no target labels used to learn transferred direction",
"No holdout inferential claim, no A/B/C hypothesis verdict"]},ensure_ascii=False,sort_keys=True),flush=True)
