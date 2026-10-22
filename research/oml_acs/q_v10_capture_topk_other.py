#!/usr/bin/env python3
"""Original BF16 Qwen 7-group top2+letters+OTHER signed logit Jacobians.

Original output full vocabulary logits are partitioned without truncation into
7 exact probability masses: A,B,C,D, two top-scoring nonchoice tokens and
OTHER=logsumexp(all remaining logits). Original Qv9 A/B/C-D native-residual
Jacobian columns are reused by SHA and independently compared to BF16 choice
probabilities. New columns use 3 true BF16 backprops per example:
top1-D, top2-D, logsumexp(OTHER)-D.
One original C native quantized residual per tensor is projected into each
signed BF16 Jacobian. All output data private, no quantized model forward.

The OTHER grouped logit gradient is a probability-weighted mean of many logits
and collapsing them discards within-bin Fisher curvature. This is not an exact
full-vocabulary Hessian or exact quantized-model loss. Diagnostic 24 is never
opened before new source-only mask is frozen.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
ROOT=pathlib.Path(__file__).resolve().parents[2]
QV9=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv9-ggn-20261023")
QV5=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
LIB_PY=QV5/"research/oml_acs/q_v5_original_bf16_fisher_source.py"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
LAB=ROOT/"lab_local/qv10_topk_trust_20261023"
PRE=ROOT/"configs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_PRELOCK_20261023.json"
FREEZE=ROOT/"docs/research/BEGLIN_QV10_RACE_INPUT_PREMODEL_FREEZE_20261023.json"
POLICY=ROOT/"docs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_SOURCE_POLICY_20261023.json"
OUTPUT={x:LAB/f"original_BF16_top2other_{x}24_private.json" for x in ("train","diagnostic")}
PUBLIC={x:ROOT/f"docs/research/BEGLIN_QV10_TOP2OTHER_{x.upper()}24_PUBLIC_20261023.json" for x in ("train","diagnostic")}
OLD_SOURCE={x:QV9/f"lab_local/qv9_ggn_20261023/ARC_{'train24_GGN_source' if x=='train' else 'diag24_independent'}_frozen.json" for x in ("train","diagnostic")}
OLD_SIGNED={x:QV9/f"lab_local/qv9_ggn_20261023/original_BF16_choice_GGN_{x}24_private.json" for x in ("train","diagnostic")}
OLD_SOURCE_SHA={"train":"87ed0d782636470b4e31db8bd6b69f0fd069dc30b167bdcb7aed1a81b53fe254",
                "diagnostic":"4714dc2be1ef4708c0099f153b15a1bd0f36b2143b006d000781bc592bf9be79"}
OLD_JAC_SHA={"train":"3d1c9bc2ff023b0b1b2efad0116a69cdb0d3ea2e0c025a6a82b7e8492548feb3",
             "diagnostic":"62d6b1be35f2034d059ee6e4486676afddda51125b26366c46585a56732a9e57"}
PRE_SHA="4ef4d31a4b9d0be2e015ffaac84a8ee86feb86047a8a8aae4a50318144d5d207"
FREEZE_SHA="3d8883792f3ba709035a0197232fdec74b6854d7d5b45081a6e89d228e9e794c"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(path):
 if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
  raise RuntimeError("private original Qv9 BF16 scalar gradient or case receipt unsealed "+str(path))
 return path.read_bytes()
def private_write(p,data):
 raw=(json.dumps(data,sort_keys=True,indent=2)+"\n").encode()
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
 return sha(raw)
def inputs(stage):
 if OUTPUT[stage].exists() or PUBLIC[stage].exists():
  raise FileExistsError("cannot recompute finalized grouped fullvocab source after evaluation")
 if sha(PRE.read_bytes())!=PRE_SHA or sha(FREEZE.read_bytes())!=FREEZE_SHA:
  raise RuntimeError("new RACE and source prior preregistration changed")
 if stage=="diagnostic":
  if not POLICY.is_file() or json.loads(POLICY.read_text())["status"]!="SOURCE_ONLY_REAL_TOPK_OTHER_GROUPED_KL_28N8_MASK_FROZEN_BEFORE_DIAG_OR_RACE":
   raise RuntimeError("diagnostic source not admissible before mask freeze")
 if sha(sealed(OLD_SOURCE[stage]))!=OLD_SOURCE_SHA[stage] or \
    sha(sealed(OLD_SIGNED[stage]))!=OLD_JAC_SHA[stage]:
  raise RuntimeError("original Qv9 shared raw BF16 choice Jacobian/answers changed")
 rows=json.loads(sealed(OLD_SOURCE[stage]))["questions"]
 old=json.loads(sealed(OLD_SIGNED[stage]))
 if old["source_group"]!=stage or len(rows)!=24 or old["n_source_examples"]!=24:
  raise RuntimeError("not 24 original source/diag BF16 Qwen frozen cases")
 oldq4=np.asarray(old["q4_signed_relative_logit_delta"],dtype=np.float64)
 oldn8=np.asarray(old["n8_signed_relative_logit_delta"],dtype=np.float64)
 oldp=np.asarray(old["original_conditional_four_letter_probabilities"],dtype=np.float64)
 if oldq4.shape!=(24,196,3) or oldn8.shape!=(24,196,3) or oldp.shape!=(24,4):
  raise RuntimeError("original previous BF16 A/B/C-D signed Q4/n8 Jacobian shape wrong")
 return rows,oldq4,oldn8,oldp
def original_delta(model):
 imp=importlib.util.spec_from_file_location("qv10_unchanged_native_C_encoder",LIB_PY)
 mod=importlib.util.module_from_spec(imp);imp.loader.exec_module(mod)
 lib=mod.lib_quant()
 module=dict(model.named_modules())
 selected=[];cache=[]
 for group,role in enumerate(ROLES):
  for layer in range(28):
   name=f"model.layers.{layer}.{'self_attn' if group<4 else 'mlp'}.{role}"
   param=module[name].weight;param.requires_grad_(True)
   w=param.detach().float().cpu().contiguous().numpy()
   d4,d8,proof=mod.quant_errors(lib,w)
   selected.append(param);cache.append((d4,d8,proof))
  print("QV10_UNMODIFIED_NATIVE_C_Q4N8_RESIDUALS",role,len(selected),flush=True)
 if len(selected)!=196:raise RuntimeError("not 28x7 real source Qwen projections")
 return selected,cache
def verify_checkpoint(file,index,row):
 if not file.is_file():return None
 old=json.loads(sealed(file))
 if (old["schema"]!="BEGLIN-Qv10-original-BF16-top2other-7group-source-case/1" or
     old["index"]!=index or old["id_sha256"]!=sha(row["id"].encode()) or
     len(old["Q4_signed_top2other_196x3"])!=196 or len(old["n8_signed_top2other_196x3"])!=196 or
     len(old["output_group_mass_A_B_C_T1_T2_OTHER_D"])!=7):
  raise RuntimeError("per-case grouped output trust input altered")
 return old
def run(stage):
 rows,old4,old8,oldp=inputs(stage)
 os.environ["TOKENIZERS_PARALLELISM"]="false"
 torch.set_num_threads(2)
 if not torch.backends.mps.is_available():raise RuntimeError("BF16 Apple MPS unavailable")
 t0=time.monotonic()
 os.umask(0o077)
 folder=LAB/f"{stage}24_top2other_native_signed"
 folder.mkdir(parents=True,mode=0o700,exist_ok=True)
 if folder.is_symlink() or (folder.stat().st_mode&0o777)!=0o700:
  raise RuntimeError("new original BF16 grouped output per-case receipts not 0700")
 print("QV10_LOAD_ORIGINAL_BF16_TOP2OTHER",stage,flush=True)
 model=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,
  local_files_only=True,trust_remote_code=False,attn_implementation="eager").eval()
 model.requires_grad_(False)
 weights,deltas=original_delta(model)
 model=model.to("mps")
 for i,row in enumerate(rows):
  receipt=folder/f"group_{i:02d}.json"
  if verify_checkpoint(receipt,i,row):
   print("QV10_RESUME_ORIGINAL_GROUPED_BF16_CASE",stage,i+1,flush=True)
   continue
  model.zero_grad(set_to_none=True)
  tokens=torch.tensor([row["prefix_ids"]],dtype=torch.long,device="mps")
  choices=[row["suffix_ids"][l] for l in "ABCD"]
  if len(set(choices))!=4:raise RuntimeError("wrong Qwen original 4 answer tokens")
  out=model(input_ids=tokens,use_cache=False,return_dict=True)
  logits=out.logits[0,-1,:].float()
  if logits.numel()<100000:raise RuntimeError("not native Qwen original full vocabulary")
  current4=torch.softmax(logits[choices],dim=0).detach().float().cpu().numpy().astype(np.float64)
  if np.max(np.abs(current4-oldp[i]))>5.e-4:
   raise RuntimeError("old original Qv9 source BF16 4-choice output does not match new BF16 source; abort")
  exclude=torch.zeros_like(logits,dtype=torch.bool)
  exclude[choices]=True
  top_ids=torch.topk(logits.masked_fill(exclude,float("-inf")),2).indices
  if top_ids.numel()!=2 or int(top_ids[0]) in choices or int(top_ids[1]) in choices:
   raise RuntimeError("Qv10 actual new top2 outside answer candidates not distinct")
  exclude[top_ids]=True
  other=torch.logsumexp(logits.masked_fill(exclude,float("-inf")),dim=0)
  # Ordering: A,B,C,T1,T2,OTHER,D (the D token anchors all six relative logits).
  group_logits=torch.stack((logits[choices[0]],logits[choices[1]],logits[choices[2]],
      logits[top_ids[0]],logits[top_ids[1]],other,logits[choices[3]]))
  p=torch.softmax(group_logits,dim=0).detach().float().cpu().numpy().astype(np.float64)
  if (not np.isfinite(p).all() or np.min(p)<0 or abs(p.sum()-1)>5.e-5):
   raise RuntimeError("new grouped full-vocab exact probability masses not normalized")
  # Verify merging just the four answer bins reproduces the previous exact 4-way probabilities.
  answerp=p[[0,1,2,6]];answerp=answerp/answerp.sum()
  if np.max(abs(answerp-oldp[i]))>5e-4:raise RuntimeError("group answer mass not original Qv9 BF16 source")
  results4=np.zeros((196,3),dtype=np.float64);results8=np.zeros((196,3),dtype=np.float64)
  for col,z in enumerate((logits[top_ids[0]],logits[top_ids[1]],other)):
   model.zero_grad(set_to_none=True)
   (z-logits[choices[3]]).backward(retain_graph=(col<2))
   for j,param in enumerate(weights):
    if param.grad is None:raise RuntimeError("Qv10 top-k OTHER BF16 backward missing")
    g=param.grad.detach().float().cpu().contiguous().numpy().ravel()
    q4,q8,_=deltas[j]
    results4[j,col]=np.einsum("i,i->",g,q4.ravel().astype(np.float32),dtype=np.float64)
    results8[j,col]=np.einsum("i,i->",g,q8.ravel().astype(np.float32),dtype=np.float64)
  if not np.isfinite(results4).all() or not np.isfinite(results8).all():raise RuntimeError("grouped BF16 native signed gradients not finite")
  obj={"schema":"BEGLIN-Qv10-original-BF16-top2other-7group-source-case/1",
       "stage":stage,"index":i,"id_sha256":sha(row["id"].encode()),
       "output_group_mass_A_B_C_T1_T2_OTHER_D":p.tolist(),
       "top2_fullvocab_nonchoice_token_ids":[int(top_ids[0]),int(top_ids[1])],
       "Q4_signed_top2other_196x3":results4.tolist(),
       "n8_signed_top2other_196x3":results8.tolist(),
       "old_Qv9_4choice_group_mass_agreed":True,
       "full_vocab_OTHER_exact_group_logsumexp":True,
       "original_BF16_forward_calls":1,"new_BF16_backward_calls":3,
       "grouped_OTHER_curvature_withinbin_missing":True,
       "no_native_fullmodel_quantized_forward":True}
  h=private_write(receipt,obj)
  del out,logits
  torch.mps.synchronize()
  if (i+1)%4==0:print("QV10_ACTUAL_BF16_TOP2PLUS_OTHER",stage,i+1,"/24","sha",h[:14],
       "elapsed_s",round(time.monotonic()-t0,1),flush=True)
 cases=[];proofs=[]
 for i,row in enumerate(rows):
  path=folder/f"group_{i:02d}.json"
  obj=verify_checkpoint(path,i,row)
  if obj is None:raise RuntimeError("incomplete original output grouped BF16 samples")
  cases.append(obj);proofs.append(sha(sealed(path)))
 # Ordering 6 relative logits: (A-D,B-D,C-D,T1-D,T2-D,OTHER-D), D=0.
 v4=np.concatenate((old4,np.array([x["Q4_signed_top2other_196x3"] for x in cases],dtype=np.float64)),axis=-1)
 v8=np.concatenate((old8,np.array([x["n8_signed_top2other_196x3"] for x in cases],dtype=np.float64)),axis=-1)
 prob=np.array([x["output_group_mass_A_B_C_T1_T2_OTHER_D"] for x in cases],dtype=np.float64)
 if v4.shape!=(24,196,6) or v8.shape!=(24,196,6) or prob.shape!=(24,7):
  raise RuntimeError("original Qwen top2 and full-OTHER grouped source shape wrong")
 if not (np.isfinite(v4).all() and np.isfinite(v8).all() and np.isfinite(prob).all()):
  raise RuntimeError("nonfinite 7group original BF16 signed source")
 packaged={"schema":"BEGLIN-Qv10-original-BF16-Qwen24-case-top2-4letters-OTHER-grouped/1",
    "status":"SOURCE_ONLY_ACTUAL_ORIGINAL_BF16_24x196x6_TOP2_OTHER_SIGNED_DONE",
    "stage":stage,"original_Qv9_previous_source_id_SHA256":OLD_SOURCE_SHA[stage],
    "original_Qv9_signed3columns_SHA256":OLD_JAC_SHA[stage],
    "original_Qwen_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "prelock_SHA256":PRE_SHA,
    "new_RACE48_input_freeze_SHA256":FREEZE_SHA,
    "num_cases":24,"num_projected_tensors":196,"num_7group_nonanchor":6,
    "grouping_order":"A,B,C,original_fullvocab_top1,original_fullvocab_top2,all_OTHER_except_all_6,D",
    "signed_Q4_relative_logit_groups":v4.tolist(),
    "signed_n8_relative_logit_groups":v8.tolist(),
    "source_original_BF16_output_group_probabilities":prob.tolist(),
    "original_case_proof_SHA256":proofs,
    "mean_mass_letters":float(prob[:,[0,1,2,6]].sum(axis=1).mean()),
    "mean_mass_top2":float(prob[:,[3,4]].sum(axis=1).mean()),
    "mean_mass_OTHER":float(prob[:,5].mean()),
    "permitted_bit_policy_calibration":stage=="train",
    "not_fullvocab_Hessian_due_to_collapsed_OTHER":True,
    "numeric_group_probability_mass_original_fullvocab_exact":True,
    "no_model_weight_changes":True,"no_operational_promotion":True,
    "elapsed_seconds":round(time.monotonic()-t0,1)}
 h=private_write(OUTPUT[stage],packaged)
 pub={k:v for k,v in packaged.items() if k not in (
       "signed_Q4_relative_logit_groups","signed_n8_relative_logit_groups",
       "source_original_BF16_output_group_probabilities","original_case_proof_SHA256")}
 pub["private_full_grouped_source_SHA256"]=h
 b=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
 with PUBLIC[stage].open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 print("QV10_ORIGINAL_QWEN_GROUPED_FULLVOCAB_MASS_SIGNED_SOURCE_PASS",
   json.dumps({"stage":stage,"group_prob_mass":{k:round(pub[k],4) for k in (
      "mean_mass_letters","mean_mass_top2","mean_mass_OTHER")},
     "private_sha":h,"public_sha":sha(b),"elapsed_s":pub["elapsed_seconds"]},sort_keys=True),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--stage",choices=("train","diagnostic"),required=True)
 run(p.parse_args().stage)
