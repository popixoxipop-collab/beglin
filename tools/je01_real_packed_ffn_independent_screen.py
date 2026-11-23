#!/usr/bin/env python3
"""Independent actual-BF16-derived n2/n6 FFN sensitivity on new SYNTHETIC activations.

Reference = all6 source re-quantization, NOT original AF 4bit or gold NLL.
"""
import argparse,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_real_72_to_equalbyte_bundle import verify_artifacts,plans
from je01_native_exact_byte_presolve import read_verified_ledger,validate_ledger,canonical
from je01_selected_bf16_native_artifacts import SHAPES,decode_native
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
ROLES=("gate_proj","up_proj","down_proj")
SEED=20261119
COUNT=12
SCHEMA="beglin-je01-real-original-BF16-n2-n6-independent-synthetic-FFN/1"

def matrix(root,rec):
    arrays=[]
    for kind in ("packed","scale"):
        s=rec[kind]
        p=root/s["filename"]
        if p.parent!=root or not p.is_file() or p.is_symlink() or p.stat().st_size!=s["actual_bytes"]:
            raise ValueError("JE01_SOURCE_PACKED_ROLE_SIZE_OR_PATH_CHANGED")
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=s["sha256"]:
            raise ValueError("JE01_SOURCE_PACKED_ROLE_SHA_CHANGED")
        arrays.append(np.frombuffer(raw,dtype=np.uint8 if kind=="packed" else "<f4"))
    return decode_native(arrays[0],arrays[1],SHAPES[rec["role"]],rec["tier"])

def ffn(X,roles):
    gate=X@roles["gate_proj"].T
    up=X@roles["up_proj"].T
    activated=gate/(1+np.exp(-np.clip(gate,-40,40)))
    return (activated*up)@roles["down_proj"].T

def measure(root):
    original=validate_ledger(read_verified_ledger(LEDGER))
    slots,records=verify_artifacts(root)
    if [(s["layer"],s["expert"]) for s in slots]!=[(s["layer"],s["expert"]) for s in original]:
        raise ValueError("JE01_FROZEN_8_SLOTS_DRIFT")
    plans3,formula=plans(slots,records)
    if len(formula)!=56 or len(plans3)!=3:
        raise ValueError("JE01_56_REAL_PLAN_SET_MISSING")
    rng=np.random.default_rng(SEED)
    per=[];losses={}
    for s in slots:
        layer,expert=s["layer"],s["expert"]
        roles={b:{r:matrix(root,records[(layer,expert,r,b)]) for r in ROLES} for b in (2,6)}
        X=rng.standard_normal((COUNT,2048)).astype(np.float32)
        X/=np.sqrt(np.mean(X*X,axis=1,keepdims=True)+1e-12)
        y6=ffn(X,roles[6]).astype(np.float64)
        y2=ffn(X,roles[2]).astype(np.float64)
        norm=np.mean(y6*y6,axis=1)
        if not np.isfinite(norm).all() or np.any(norm<=1e-24):
            raise ValueError("JE01_6BIT_FFN_REFERENCE_ZERO")
        relative=np.mean((y2-y6)**2,axis=1)/norm
        if not np.isfinite(relative).all():raise ValueError("JE01_REAL_WEIGHT_FFN_NONFINITE")
        loss=float(np.mean(relative))
        losses[(layer,expert)]=loss
        byrole={}
        for r in ROLES:
            ablated=dict(roles[6]);ablated[r]=roles[2][r]
            y=ffn(X,ablated).astype(np.float64)
            byrole[r]=float(np.mean(np.mean((y-y6)**2,axis=1)/norm))
        per.append({"layer":layer,"expert":expert,"n2_vs_n6_mean_relative_FFN_MSE":loss,
                    "real_router_last_token_count_used_only_for_prior_selection":s["calibration_last_token_routed_calls"],
                    "n2_one_role_at_a_time_relative_FFN_MSE":byrole,
                    "synthetic_case_relative_FFN_MSE":[float(x) for x in relative]})
        print(json.dumps({"L":layer,"E":expert,"n2_relative_FFN_MSE":loss,"roles":byrole}),flush=True)
    ordered=list(losses)
    combos=sorted((sum(losses[k] for k in comb)/8,tuple(sorted(comb)))
                  for comb in itertools.combinations(ordered,3))
    assert len(combos)==56
    result={}
    for name,twos in plans3.items():
        key=tuple(sorted(twos))
        matches=[v for v,c in combos if c==key]
        if len(matches)!=1:raise ValueError("JE01_SELECTED_POLICY_NOT_IN_REAL56")
        score=matches[0]
        result[name]={"two_bit_experts":[list(x) for x in key],
                      "mean8_relative_FFN_MSE_vs_all6":score,
                      "rank_among56_POSTHOC":1+sum(z<score-1e-15 for z,c in combos)}
    return {"schema":SCHEMA,"status":"REAL_BF16_SOURCED_PACKED_N2_N6_FFN_DIAGNOSTIC_PASS",
            "actual_8_expert_24_role_sources":True,
            "tested_locked_equalbyte_policies":result,
            "synthetic_activation_seed":SEED,"synthetic_cases_per_expert":COUNT,
            "reference_all6bit_bytes":56229888,
            "exact_matched_original_AF_budget_bytes":43253760,
            "posthoc_best_of56_relative_FFN_MSE":combos[0][0],
            "all56_mean_relative_FFN_MSE":sum(v for v,c in combos)/56,
            "per_expert":per,
            "reference_kind":"ALL6_NATIVE_REQUANTIZED_NOT_ORIGINAL_AF4",
            "full_model_real_hidden_states":"NOT_USED",
            "original_AF_q4_parity":"NOT_MEASURED",
            "full_model_gold_NLL":"NOT_MEASURED",
            "independent_128_128_labels_used":False,
            "production_changes":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    try:
        if a.out.exists():raise ValueError("JE01_OUTPUT_ALREADY_EXISTS")
        doc=measure(a.source)
        doc["receipt_sha256"]=canonical(doc)
        a.out.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"status":doc["status"],"policies":doc["tested_locked_equalbyte_policies"],
                          "receipt_sha256":doc["receipt_sha256"]},sort_keys=True),flush=True)
    except Exception as ex:
        print(json.dumps({"status":"BLOCKED","error":str(ex)[:260]}))
        raise SystemExit(2)
if __name__=="__main__":main()
