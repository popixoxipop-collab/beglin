#!/usr/bin/env python3
"""Frozen original-greedy token divergence and L0 Q/K/V/O role attribution.

The only C change is a gated dump of already-computed logits, built in a
separate Actions checkout. Candidate/Q4 outputs must bit-match the *previous*
unmodified binary's 12-token rollout. Every path and SHA is pinned.
"""
from __future__ import annotations
import array
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid

from tokenizers import Tokenizer

SCHEMA="beglin-qt-logit-first-divergence-role-isolation/1"
ROOT=Path.cwd().resolve()
PINNED_BASE="4c072d8b827c0e0c4806feb5b02832e281a2bb87"
ORIGINAL=Path("/tmp/beglin-qt-l0-fp32-control/build-qt-l0-fp32/qwen_infer_gpu")
ORIGINAL_SHA="a023fe2e15da39eddbf7b2a483d65a25e3897eb3c3d9825db211b4ec2adce96e"
DIAG=ROOT/"build-qt-logit-isolation/qwen_infer_gpu"
MODEL=Path("/tmp/qt-l0-fp32-cache/model.safetensors")
MODEL_SHA="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
CONFIG=Path("/tmp/qt-l0-fp32-cache/config.json")
CONFIG_SHA="18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
TOKENIZER=Path("/tmp/qt-natural-vfocus-tokenizer-7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json")
TOKENIZER_SHA="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
PREVIOUS=Path("/Users/xox/mcp-sandbox/tailnet-commander/qt-natural-vfocus-results/QT_NATURAL_VFOCUS_ada97a1056464cb2a0b43d17325cdc33.json")
PREVIOUS_SHA="025f81423f967eb0d4a4bad08f627331c6bdd4bac78b52ffb164225cb2db82a5"
PROMPTS_DIR=Path("/tmp/qt-natural-vfocus-ciftt2c_/prompts")
PRIOR_VFOCUS=Path("/tmp/qt-role-grid-ld4quo2c/bits_16_16_64_32")
RESULT_ROOT=Path("/Users/xox/mcp-sandbox/tailnet-commander/qt-logit-role-results")
ROLES=("q_proj","k_proj","v_proj","o_proj")
MAP_SHA={
 "q_proj":"a5c8ad63bd93746850d9035824f06128a1a536f9b8268633ed8b953c136f0d4d",
 "k_proj":"98eb5bc1cfbdff603c2bfd433909b15572730d196828f137dc58b35ab7e1ac4b",
 "v_proj":"8f272ed4639bc2b2427b9b40eeb9e1575a30980f3124cd345a08a8058042482f",
 "o_proj":"1c5dba1e57a4d8271932339d8e2736f9276873ad047d2022e93a19f25a6f0290",
}
BUDGET={"q_proj":16,"k_proj":16,"v_proj":64,"o_proj":32}
EXPECTED_CASES={
 "english-128":("4c3da8e626e825046412117df9e7d2c368674e8c258ea11facbe03c421d442a0",128),
 "english-512":("b6351c03dffb30107762e31b73b9c5f46c5b974b1493e1fe7705e4208b1da646",512),
 "english-1024":("0ca0c64767175a813574d9c111d3b9c90fab48f185c62897d3d834f0c20e73f0",1024)
}
VOCAB=151936
MAX_GEN=12

def sha(p:Path):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1048576),b""):
   h.update(block)
 return h.hexdigest()

def pin(path,expected):
 if not path.is_file() or sha(path)!=expected:
  raise RuntimeError("SHA_PIN_MISMATCH:"+str(path))
 return expected

def prepare(work):
 for p,h in ((ORIGINAL,ORIGINAL_SHA),(MODEL,MODEL_SHA),(CONFIG,CONFIG_SHA),
             (TOKENIZER,TOKENIZER_SHA),(PREVIOUS,PREVIOUS_SHA)):
  pin(p,h)
 if not DIAG.is_file() or DIAG.stat().st_size<200000:
  raise RuntimeError("DIAGNOSTIC_BINARY_MISSING")
 build=json.loads((ROOT/"QT_LOGIT_DIAG_BUILD.json").read_text())
 if build["diagnostic_binary_sha256"]!=sha(DIAG) or build["frozen_source_commit"]!=PINNED_BASE:
  raise RuntimeError("DIAGNOSTIC_BUILD_IDENTITY_MISMATCH")
 previous=json.loads(PREVIOUS.read_text())
 prompts={}
 for name,(expected_sha,n) in EXPECTED_CASES.items():
  p=PROMPTS_DIR/(name+".i32")
  pin(p,expected_sha)
  if p.stat().st_size!=n*4:raise RuntimeError("PROMPT_SIZE_MISMATCH")
  old=next((x for x in previous["prompt_cases"] if x["case"]==name),None)
  if not old or old["prompt_sha256"]!=expected_sha:
   raise RuntimeError("PRIOR_PROMPT_RECEIPT_MISMATCH")
  qa=next((x for x in previous["runs"] if x["case"]==name and x["position"]==n-1),None)
  if not qa or not qa["quality"]:raise RuntimeError("MISSING_PREVIOUS_GOLD_GENERATIONS")
  prompts[name]={"path":p,"sha256":expected_sha,"tokens":n,
      "prior_q4":qa["runs"]["q4"]["generated_tokens"],
      "prior_full":qa["runs"]["vfocus"]["generated_tokens"],
      "prior_gold":old["expected_answer"]}
 maps={}
 for role in ROLES:
  path=PRIOR_VFOCUS/("model_layers_0_self_attn_"+role+"_weight.bits")
  pin(path,MAP_SHA[role])
  raw=path.read_bytes()
  if set(raw)!={4,5} or raw.count(5)!=BUDGET[role]:
   raise RuntimeError("FROZEN_ROLE_BITMAP_INVALID_"+role)
  maps[role]=raw
 maproot=work/"maps"
 maproot.mkdir()
 def make(name,role_set):
  folder=maproot/name
  folder.mkdir()
  for role in ROLES:
   data=maps[role] if role in role_set else bytes([4])*len(maps[role])
   (folder/("model_layers_0_self_attn_"+role+"_weight.bits")).write_bytes(data)
  return folder
 out={"all_q4":make("all_q4",set()),"all_vfocus":make("all_vfocus",set(ROLES))}
 for role in ROLES:
  out["only_"+role]=make("only_"+role,{role})
  out["without_"+role]=make("without_"+role,set(ROLES)-{role})
 return build,previous,prompts,out

def run(work,tag,prompt,engine,manifest,steps,original=False):
 d=work/"runs"/tag
 d.mkdir(parents=True,exist_ok=False)
 dst=d/"logits.f32"
 env={k:v for k,v in os.environ.items() if not k.startswith("QWEN_")}
 env.update(QWEN_SAFETENSORS=str(MODEL),QWEN_HF_CONFIG=str(CONFIG),
            QWEN_PROMPT=str(prompt),QWEN_BASE=str(d))
 if manifest is not None:
  env["QWEN_QT_MIXED_MANIFEST"]=str(manifest)
 if not original:
  env["QWEN_QT_DIAG_LOGITS"]=str(dst)
 t0=time.monotonic()
 process=subprocess.run([str(engine),"greedy",str(steps)],
   cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
 elapsed=time.monotonic()-t0
 if process.returncode:
  raise RuntimeError("ENGINE_FAILED:"+tag+":"+str(process.returncode)+":"+process.stderr[-1100:])
 m=re.search(r"(?m)^greedy:((?:[ \t]+\d+)+)\s*$",process.stdout)
 if not m:
  raise RuntimeError("NO_RAW_TOKEN_OUTPUT:"+tag+":"+process.stdout[-350:])
 tokens=[int(s) for s in m.group(1).strip().split()]
 if len(tokens)!=steps or any(n>=VOCAB or n<0 for n in tokens):
  raise RuntimeError("BAD_GENERATION_LENGTH_OR_TOKEN_ID:"+tag)
 if manifest is not None:
  logs=process.stderr
  if sum("target=model.layers.0.self_attn." in x for x in logs.splitlines())<4:
   raise RuntimeError("ROLE_MAP_NOT_REGISTERED:"+tag)
 info={"tag":tag,"generated_tokens":tokens,"duration_seconds":round(elapsed,4),
       "stderr_sha256":hashlib.sha256(process.stderr.encode()).hexdigest(),
       "stdout_sha256":hashlib.sha256(process.stdout.encode()).hexdigest(),
       "engine_sha256":sha(engine),"map_directory":manifest.name if manifest else None}
 if original:
  if dst.exists():raise RuntimeError("DIAG_SIDE_EFFECT_ON_ORIGINAL")
  return None,info
 raw=dst.read_bytes()
 if len(raw)!=steps*VOCAB*4:
  raise RuntimeError("BAD_LOGIT_VECTOR_SIZE:"+tag+":"+str(len(raw)))
 floats=array.array("f");floats.frombytes(raw)
 if sys.byteorder!="little":floats.byteswap()
 if not all(math.isfinite(x) for x in floats):
  raise RuntimeError("NONFINITE_LOGITS:"+tag)
 info["logits_sha256"]=hashlib.sha256(raw).hexdigest()
 info["logits_size_bytes"]=len(raw)
 logits=floats[:VOCAB]
 top=heapq.nlargest(6,range(VOCAB),key=logits.__getitem__)
 if top[0]!=tokens[0]:
  raise RuntimeError("DUMP_LOGITS_DO_NOT_MATCH_GENERATION:"+tag+":"+str(top[0])+":"+str(tokens[0]))
 info["first_top6"]=[{"id":i,"logit":logits[i]} for i in top]
 info["first_top1_top2_margin"]=float(logits[top[0]]-logits[top[1]])
 return logits,info

def condition(logits,base,a_token,b_token,tok):
 ix=heapq.nlargest(6,range(len(logits)),key=logits.__getitem__)
 margin=logits[a_token]-logits[b_token]
 delta=math.sqrt(sum((x-y)**2 for x,y in zip(logits,base)))/(math.sqrt(sum(v*v for v in base))+1e-12)
 return {"top1":ix[0],"top1_text":tok.decode([ix[0]],skip_special_tokens=False),
     "top6":[{"token_id":i,"piece":tok.decode([i],skip_special_tokens=False),"logit":logits[i]} for i in ix],
     "q4_token_logit":logits[a_token],"vfocus_token_logit":logits[b_token],
     "decision_margin_q4minus_vfocus":margin,
     "relative_l2_logits_vs_q4":delta}

def write_receipt(payload):
 result=(json.dumps(payload,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
 RESULT_ROOT.mkdir(parents=True,exist_ok=True)
 target=RESULT_ROOT/("QT_FIRST_LOGIT_ROLE_"+uuid.uuid4().hex+".json")
 with target.open("xb") as f:
  f.write(result);f.flush();os.fsync(f.fileno())
 return {"path":str(target),"sha256":hashlib.sha256(result).hexdigest(),"bytes":len(result)}

def main():
 tok=Tokenizer.from_file(str(TOKENIZER))
 work=Path(tempfile.mkdtemp(prefix="qt-first-logit-role-",dir="/tmp"))
 build,previous,prompts,maps=prepare(work)
 analyses={}
 for name in EXPECTED_CASES:
  meta=prompts[name]
  orig_q4_info=run(work,name+"_original_q4",meta["path"],ORIGINAL,None,MAX_GEN,original=True)[1]
  orig_full_info=run(work,name+"_original_full",meta["path"],ORIGINAL,maps["all_vfocus"],MAX_GEN,original=True)[1]
  if orig_q4_info["generated_tokens"]!=meta["prior_q4"] or orig_full_info["generated_tokens"]!=meta["prior_full"]:
   raise RuntimeError("ORIGINAL_ENGINE_VS_PRIOR_RECEIPT_GENERATION_DRIFT:"+name)
  q4,base_info=run(work,name+"_diag_q4",meta["path"],DIAG,None,MAX_GEN)
  full,full_info=run(work,name+"_diag_full_vfocus",meta["path"],DIAG,maps["all_vfocus"],MAX_GEN)
  if base_info["generated_tokens"]!=orig_q4_info["generated_tokens"] or full_info["generated_tokens"]!=orig_full_info["generated_tokens"]:
   raise RuntimeError("INSTRUMENTATION_GENERATION_PARITY_FAILED:"+name)
  diverge=next((i for i,(a,b) in enumerate(zip(base_info["generated_tokens"],full_info["generated_tokens"])) if a!=b),None)
  top_a=base_info["generated_tokens"][0]
  top_b=full_info["generated_tokens"][0]
  data={"case":name,"prompt_sha256":meta["sha256"],"length":meta["tokens"],
    "previous_12step_generation_sha_check":True,
    "instrumented_12step_generation_parity":True,
    "original_q4":orig_q4_info,"original_vfocus":orig_full_info,
    "diagnostic_q4":base_info,"diagnostic_vfocus":full_info,
    "q4_first_top1":top_a,"vfocus_first_top1":top_b,
    "first_divergence_index_zero_based":diverge,
    "same_input_first_token":top_a==top_b,
    "q4_first_token_text":tok.decode([top_a],skip_special_tokens=False),
    "vfocus_first_token_text":tok.decode([top_b],skip_special_tokens=False)}
  if name=="english-128":
   if diverge!=0 or top_a!=785 or top_b!=6471:
    raise RuntimeError("KNOWN_FIRST_SPLIT_MISSING")
   arms={}
   for arm in ["all_q4"]+[f"only_{role}" for role in ROLES]+[f"without_{role}" for role in ROLES]:
    v,info=run(work,"english128_role_"+arm,meta["path"],DIAG,maps[arm],1)
    arms[arm]={"metrics":condition(v,q4,top_a,top_b,tok),"run":info}
   metrics={"q4":condition(q4,q4,top_a,top_b,tok),
            "vfocus":condition(full,q4,top_a,top_b,tok)}
   base_margin=metrics["q4"]["decision_margin_q4minus_vfocus"]
   full_margin=metrics["vfocus"]["decision_margin_q4minus_vfocus"]
   data["isolated_role_arms"]=arms
   data["first_step_pairwise_logit_metrics"]=metrics
   data["margin_shifts"]={
     role:{
      "single_role_delta_vs_q4":arms["only_"+role]["metrics"]["decision_margin_q4minus_vfocus"]-base_margin,
      "conditional_delta_in_full_vs_without_role":full_margin-arms["without_"+role]["metrics"]["decision_margin_q4minus_vfocus"]
     } for role in ROLES
   }
   data["all_q4_mixed_margin_drift"]=arms["all_q4"]["metrics"]["decision_margin_q4minus_vfocus"]-base_margin
   # Teacher forcing: same Q4 prefix in both arms isolates logits at next token
   # rather than comparing differently generated trajectories.
   teacher=[]
   raw=meta["path"].read_bytes()
   for n in (1,2):
    q4prefix=meta["prior_q4"][:n]
    pp=work/"teacher_prompts"/("q4prefix_"+str(n)+".i32")
    pp.parent.mkdir(parents=True,exist_ok=True)
    pp.write_bytes(raw+struct.pack("<%di"%n,*q4prefix))
    q,qi=run(work,f"english128_teacher_{n}_q4",pp,DIAG,None,1)
    z,zi=run(work,f"english128_teacher_{n}_vfocus",pp,DIAG,maps["all_vfocus"],1)
    teacher.append({"q4_prefix_length":n,"input_sha256":sha(pp),
      "input_suffix_token_ids":q4prefix,
      "q4_generated_first":qi["generated_tokens"][0],
      "vfocus_generated_first":zi["generated_tokens"][0],
      "q4_top1":qi["first_top6"][0],
      "vfocus_top1":zi["first_top6"][0],
      "relative_l2_logits":math.sqrt(sum((x-y)**2 for x,y in zip(q,z)))/(math.sqrt(sum(x*x for x in q))+1e-12),
      "q4_run":qi,"vfocus_run":zi})
   data["q4_prefix_teacher_forced"]=teacher
  analyses[name]=data
  print("QT_LOGIT_CASE_COMPLETE="+json.dumps({"case":name,"q4_first":top_a,
    "vfocus_first":top_b,"first_divergence":diverge,
    "q4_first_margin":base_info["first_top1_top2_margin"],
    "vfocus_first_margin":full_info["first_top1_top2_margin"]},sort_keys=True),flush=True)
 payload={
  "schema":SCHEMA,
  "status":"SUCCEEDED",
  "production_write_allowed":False,
  "automatic_live_promotion":False,
  "source_scope":"L0 Q/K/V/O QT mixed map only",
  "instrumentation":"first logits of actual greedy decoding, not the crashing dump mode; unmodified 12-token generation parity asserted for Q4 and V-focus across 128,512,1024",
  "reference_engine_sha256":ORIGINAL_SHA,
  "model_sha256":MODEL_SHA,
  "config_sha256":CONFIG_SHA,
  "tokenizer_sha256":TOKENIZER_SHA,
  "prior_natural_result_sha256":PREVIOUS_SHA,
  "diagnostic_build":build,
  "map_sha256":MAP_SHA,
  "vfocus_bit_budget":BUDGET,
  "cases":analyses,
  "disclaimer":"Single frozen natural-language cohort reused strictly for diagnosis after quality failure; not an independent generalization estimate. Role perturbations are not necessarily additive causal contributions."
 }
 receipt=write_receipt(payload)
 print("QT_LOGIT_ROLE_RESULT="+json.dumps({"schema":SCHEMA,"status":"SUCCEEDED",
  "receipt":receipt,"engine_sha256":ORIGINAL_SHA,
  "diagnostic_engine_sha256":sha(DIAG),
  "first_english128_split":analyses["english-128"]["first_divergence_index_zero_based"],
  "english128_margin_shifts":analyses["english-128"].get("margin_shifts",{}),
  "baseline_first_top1":analyses["english-128"]["q4_first_top1"],
  "vfocus_first_top1":analyses["english-128"]["vfocus_first_top1"],
  "automatic_live_promotion":False},sort_keys=True),flush=True)

if __name__=="__main__":
 main()
