#!/usr/bin/env python3
"""True original27 loss-directional finite-difference score + 2x2 empirical Fisher.

Each token's score is estimated by central NLL differences from real
27-layer Apple MLX runs at alpha= +/-0.05, *not* by squaring mean gradients.
Alpha changes a routed FFN output correction, not quantized weight parameters.
The 2x2 matrix is an approximate empirical Fisher in two *output-policy*
directions B and C. Does NOT estimate 48-coordinate weight-level Fisher or
NLL Hessian. 16 already seen TRAIN cases only; no heldout use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

ROOT=Path("/Volumes/D50/TailnetCommander/work/je01_postnll_alpha2_gold16_20261123_v1")
h=0.05
N=656
SHA={
 "native":"01a9a22649f0942084a4026910b6f348f424304cc6d2b6ac89590cfc59d0b0a3",
 "B":"fcf33feb58b87724f8be673c8457cf2d0b3e2ab0cf947ed6c3b0cb6e91f67fcf",
 "C":"9a61a792d7e529868109ee3e5c240fea545479ca89d681cbf0b0ea6543b17997",
 "original_output":"b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c",
 "on_B":"d164a4c26c2691b6bee7762235a3175bef37199fb3fc9cbc8efa69c92179ddcb",
 "on_C":"5ec420bebda8e56835bcc3803baa6a0a15641dc08cf9b5d4288426fbe4832881"
}
KNOWN={"B":2.2974258514952206,"C":2.2959651985430702}
BASELINE=2.291799702354307


def sha(s:bytes)->str:
 return hashlib.sha256(s).hexdigest()


def canonical(d)->str:
 return sha(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode())


def need(cond,code):
 if not cond:raise ValueError(code)


def get(arm,alpha):
 path=ROOT/arm/alpha/"JE01_ALPHA_REAL27_656GOLD_NLL.json"
 need(path.is_file() and not path.is_symlink(),"JE01_DF_NATIVE_NLL_INPUT_ABSENT_"+arm+"/"+alpha)
 raw=path.read_bytes()
 rec=json.loads(raw)
 seal=rec.pop("receipt_sha256",None)
 need(seal==canonical(rec),"JE01_DF_NATIVE_NLL_RECEIPT_SIGNATURE_DRIFT")
 rec["receipt_sha256"]=seal
 need(rec["arm"]==arm and
      rec["alpha"]=={"z000":"0.0","p100":"1.0","m005":"-0.05","p005":"+0.05"}[alpha] and
      rec["native_file_sha256"]==SHA["native"] and
      rec["physical_policy_sha256"]==SHA[arm] and
      rec["real_full_model_27_layers"] is True and
      rec["gold_tokens"]==N and rec["gold_train_cases"]==16 and
      rec["production_modified"] is False and
      rec["per_token_gradient"]=="NOT_MEASURED_THIS_RUN" and
      rec["parameter_weight_fisher"]=="NOT_MEASURED" and
      rec["independent_heldout"]=="NOT_ACCESSED" and rec["OOD"]=="NOT_ACCESSED",
      "JE01_DF_NATIVE_FISHER_SOURCE_SCOPE_VIOLATION")
 rows=rec["per_gold_target_NLL"]
 need(len(rows)==16 and
      sum(map(len,rows))==N and
      all(all(isinstance(x,(float,int)) and math.isfinite(x) and x>=0 for x in row) for row in rows) and
      len(rec["per_case"])==16 and
      abs(sum(map(sum,rows))/N-rec["mean_NLL"])<1e-10,
      "JE01_DF_REAL_656_TOKENS_OR_MEAN_NOT_REPRODUCIBLE")
 if alpha=="z000":
  need(rec["actual_native_gold_output_sha256"]==SHA["original_output"] and
       abs(rec["mean_NLL"]-BASELINE)<1e-12 and
       rec["status"]=="JE01_ORIGINAL27_ALPHA_ENDPOINT_PARITY_PASS",
       "JE01_DF_ALPHA0_NOT_REAL_ORIGINAL_AF4_BIT_PARITY")
 elif alpha=="p100":
  need(rec["actual_native_gold_output_sha256"]==SHA["on_"+arm] and
       abs(rec["mean_NLL"]-KNOWN[arm])<1e-12 and
       rec["status"]=="JE01_ORIGINAL27_ALPHA_ENDPOINT_PARITY_PASS",
       "JE01_DF_ALPHA1_NOT_EXACT_BC_MIXED_ON")
 else:
  need(rec["status"]=="JE01_ORIGINAL27_ALPHA_CENTRAL_DIFF_INPUT_NLL_MEASURED",
       "JE01_DF_NON_ENDPOINT_NOT_TRUE_NATIVE_NLL")
 return rec,sha(raw)


def percentile(a,p):
 a=sorted(a)
 idx=int(p*(len(a)-1))
 return a[idx]


def audit():
 recordings={}
 for arm in ("B","C"):
  for alpha in ("z000","p100","m005","p005"):
   recordings[(arm,alpha)]=get(arm,alpha)
 base_b=recordings["B","z000"][0]["per_gold_target_NLL"]
 base_c=recordings["C","z000"][0]["per_gold_target_NLL"]
 need(base_b==base_c,"JE01_DF_BC_ALPHA0_ORIGINAL_PER_TOKEN_NOT_EQUAL")
 per_case=[]
 sB=[];sC=[];dB=[];dC=[]
 hessB=[];hessC=[]
 for i in range(16):
  br=recordings["B","m005"][0]["per_gold_target_NLL"][i]
  bp=recordings["B","p005"][0]["per_gold_target_NLL"][i]
  cr=recordings["C","m005"][0]["per_gold_target_NLL"][i]
  cp=recordings["C","p005"][0]["per_gold_target_NLL"][i]
  z=base_b[i]
  need(len(z)==len(br)==len(bp)==len(cr)==len(cp)>0,
       "JE01_DF_SENTENCE_TOKENS_NOT_SAME")
  local=[]
  for y,nb,ab,nc,ac in zip(z,br,bp,cr,cp):
   # score = d log p/d alpha = -d NLL/d alpha
   gb=(ab-nb)/(2*h)
   gc=(ac-nc)/(2*h)
   local.append((gb,gc))
   dB.append(gb); dC.append(gc)
   sB.append(-gb);sC.append(-gc)
   hessB.append((ab-2*y+nb)/(h*h))
   hessC.append((ac-2*y+nc)/(h*h))
  per_case.append({"index":i,"tokens":len(z),
    "sum_dNLL_dalpha_B":math.fsum(v[0] for v in local),
    "sum_dNLL_dalpha_C":math.fsum(v[1] for v in local),
    "sum_score_B2":math.fsum(v[0]*v[0] for v in local),
    "sum_score_BC":math.fsum(v[0]*v[1] for v in local),
    "sum_score_C2":math.fsum(v[1]*v[1] for v in local)})
 need(len(sB)==len(sC)==N and all(math.isfinite(x) for x in sB+sC),
      "JE01_DF_NONFINITE_TRUE_TOKEN_SCORE")
 F00=math.fsum(x*x for x in sB)/N
 F11=math.fsum(x*x for x in sC)/N
 F01=math.fsum(x*y for x,y in zip(sB,sC))/N
 det=F00*F11-F01*F01
 need(F00>=0 and F11>=0 and det>=-1e-8*max(F00*F11,1e-8),
      "JE01_DF_EMPIRICAL_FISHER_PSD_FAILED")
 gB=math.fsum(dB)/N;gC=math.fsum(dC)/N
 hb=math.fsum(hessB)/N;hc=math.fsum(hessC)/N
 contrast={"B":{"gNLL_direction":gB,"F_direction":F00,"central_NLL_curvature":hb,
                "actual_endpoint_NLL_change":KNOWN["B"]-BASELINE,
                "taylor_F_proxy_deltaNLL":gB+0.5*F00,
                "taylor_curvature_proxy_deltaNLL":gB+0.5*hb},
           "C":{"gNLL_direction":gC,"F_direction":F11,"central_NLL_curvature":hc,
                "actual_endpoint_NLL_change":KNOWN["C"]-BASELINE,
                "taylor_F_proxy_deltaNLL":gC+0.5*F11,
                "taylor_curvature_proxy_deltaNLL":gC+0.5*hc}}
 rng=random.Random(20261123)
 g_b=[];g_c=[];fbb=[];fcc=[];fbc=[]
 for _ in range(5000):
  ids=[rng.randrange(16) for _ in range(16)]
  n=sum(per_case[i]["tokens"] for i in ids)
  g_b.append(math.fsum(per_case[i]["sum_dNLL_dalpha_B"] for i in ids)/n)
  g_c.append(math.fsum(per_case[i]["sum_dNLL_dalpha_C"] for i in ids)/n)
  fbb.append(math.fsum(per_case[i]["sum_score_B2"] for i in ids)/n)
  fcc.append(math.fsum(per_case[i]["sum_score_C2"] for i in ids)/n)
  fbc.append(math.fsum(per_case[i]["sum_score_BC"] for i in ids)/n)
 bs={"method":"SHARED_16_TRAIN_SENTENCE_CLUSTER_BOOTSTRAP_NOT_HELDOUT",
     "seed":20261123,"draws":5000,
     "gNLL_B_95":[percentile(g_b,.025),percentile(g_b,.975)],
     "gNLL_C_95":[percentile(g_c,.025),percentile(g_c,.975)],
     "F_BB_95":[percentile(fbb,.025),percentile(fbb,.975)],
     "F_CC_95":[percentile(fcc,.025),percentile(fcc,.975)],
     "F_BC_95":[percentile(fbc,.025),percentile(fbc,.975)]}
 r={"schema":"beglin-je01-real-native27-alpha-output-2dir-empirical-fisher-finite-difference-train16/1",
    "status":"JE01_REAL27_TWO_POLICY_OUTPUT_DIRECTIONAL_FISHER_NUMERIC_PASS",
    "source_original_fullgold_SHA":SHA["original_output"],
    "native_27layer_SHA":SHA["native"],
    "real_physical_43MB_policy_SHA":{"B":SHA["B"],"C":SHA["C"]},
    "alpha_step":h,
    "score_coordinate_system":"Two routed-expert-output interpolation scalars B,C around original AF4 alpha=0, NOT tensor weight coordinates. Central difference alpha +/-0.05 on true 27L token gold NLL.",
    "true_analytic_autodiff":"NOT_MEASURED",
    "actual_parameter_weight_fisher_48coords":"NOT_MEASURED",
    "input_gold_tokens":N,
    "input_sentence_clusters":16,
    "per_token_scores":"FROM_TWO_FINITE_DIFF_REAL_GOLD_NLL_END_TO_END_RUNS_PER_ARM",
    "grad_mean_NLL":[gB,gC],
    "empirical_Fisher_2x2_approx":[[F00,F01],[F01,F11]],
    "F_determinant":det,
    "curvature_NLL_diagonal_2dirs":[hb,hc],
    "Fisher_equals_true_Hessian":False,
    "contrast":contrast,
    "per_sentence_score_summaries":per_case,
    "bootstrap":bs,
    "input_receipts_sha256":{"%s_%s"%k:v[1] for k,v in recordings.items()},
    "training_scope":"PREVIOUSLY_OBSERVED_TRAIN16_ONLY",
    "OOD_gold_used":False,
    "heldout_validation_used":False,
    "production_modified":False}
 r["receipt_sha256"]=canonical(r)
 return r

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--out",type=Path,required=True)
 args=p.parse_args()
 r=audit()
 if args.out.exists() or args.out.is_symlink():
  raise FileExistsError("JE01_TRUE_DIR_FISHER_NO_OVERWRITE")
 args.out.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({k:v for k,v in r.items() if k not in ("per_sentence_score_summaries","input_receipts_sha256")},sort_keys=True),flush=True)
