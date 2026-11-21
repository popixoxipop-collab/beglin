#!/usr/bin/env python3
"""JE01 real BF16-source FFN ROLE-level precision Bayesian *proxy* pilot.

24 roles, 6 n3 + 18 n5 at same original AF8 43,253,760B budget.
NIG risk is fit on synthetic FFN reconstruction losses, NEVER real gold NLL.
A new synthetic activation seed is used exclusively for policy validation.
"""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from je01_native_exact_byte_presolve import canonical,read_verified_ledger,validate_ledger
from je01_selected_bf16_native_artifacts import SHAPES
RECEIPT="04ddca2f84367f41098357b5fae905e76c1207ef023d648a9f388aaddb3bb0ca"
MODEL="36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
LEDGER=Path(__file__).resolve().parents[1]/"docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
ROLES=("gate_proj","up_proj","down_proj")
TRAIN_SEED=20261121
TEST_SEED=20261122
TRAIN_N=32
TEST_N=32
SCHEMA="beglin-je01-BF16-native-q3q5-role-NIG-synthetic-FFN-reconstruction/1"

def verify_source(root,slots):
    p=root/"JE01_ORIGINAL_BF16_E8_Q3Q5_PHYSICAL_RECEIPT.json"
    if not p.is_file() or p.is_symlink():raise ValueError("JE01_SOURCE_RECEIPT_MISSING")
    d=json.loads(p.read_text())
    sig=d.pop("receipt_sha256",None)
    if sig!=RECEIPT or canonical(d)!=RECEIPT or d["source_original_BF16_identity_SHA256"]!=MODEL:
        raise ValueError("JE01_REAL_Q35_SOURCE_IDENTITY_NOT_PINNED")
    if d["actual_full8_physical_tier_bytes"]!={"3":30277632,"5":47579136}:
        raise ValueError("JE01_Q35_PHYSICAL_BUDGET_CHANGED")
    expected={(s["layer"],s["expert"],r,b) for s in slots for r in ROLES for b in (3,5)}
    rows={}
    for v in d["records"]:
        key=(v["layer"],v["expert"],v["role"],v["bits"])
        if key not in expected or key in rows:raise ValueError("JE01_ROLE_TIER_DUPLICATE_OR_UNKNOWN")
        rows[key]=v
    if set(rows)!=expected:raise ValueError("JE01_Q35_ROLE_TIER_SET_MISSING")
    return rows

def packed_matrix(root,rec):
    files=rec["files"];bits=rec["bits"]
    shape=SHAPES[rec["role"]]
    out,ins=shape;ng=ins//64
    raws=[]
    for kind in ("packed","scale"):
        spec=files[kind];path=root/spec["filename"]
        if path.parent!=root or not path.is_file() or path.is_symlink() or path.stat().st_size!=spec["actual_bytes"]:
            raise ValueError("JE01_REAL_NATIVE_FILE_PATH_OR_SIZE_INVALID")
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=spec["sha256"]:
            raise ValueError("JE01_REAL_NATIVE_FILE_SHA_DRIFT")
        raws.append(raw)
    p=np.frombuffer(raws[0],dtype=np.uint8)
    s=np.frombuffer(raws[1],dtype="<f4")
    if p.size!=out*ins*bits//8 or s.size!=out*ng:
        raise ValueError("JE01_Q35_NATIVE_FILE_SHAPE_MISMATCH")
    planes=np.unpackbits(p.reshape(out,ng,bits,8),axis=-1,bitorder="little")
    codes=np.zeros((out,ng,64),np.int16)
    for b in range(bits):codes|=planes[:,:,b,:].astype(np.int16)<<b
    result=((codes-(1<<(bits-1)))*s.reshape(out,ng,1)).reshape(shape)
    if not np.isfinite(result).all():raise ValueError("JE01_ROLE_NATIVE_DECODE_NONFINITE")
    return result.astype(np.float32)

def ffn(X,mat):
    gate=X@mat["gate_proj"].T
    up=X@mat["up_proj"].T
    return (gate/(1+np.exp(-np.clip(gate,-40,40)))*up)@mat["down_proj"].T

def losses(X,mat3,mat5):
    base=ffn(X,mat5).astype(np.float64)
    denom=np.mean(base*base,axis=1)
    if np.any(denom<=1e-25) or not np.isfinite(denom).all():
        raise ValueError("JE01_Q5_REFERENCE_NORM_ZERO")
    def relative(m):
        y=ffn(X,m).astype(np.float64)
        delta=np.mean((y-base)**2,axis=1)/denom
        if not np.isfinite(delta).all():raise ValueError("JE01_Q35_ROLE_NONFINITE")
        return delta
    one_role={}
    for role in ROLES:
        m=dict(mat5);m[role]=mat3[role]
        one_role[role]=relative(m)
    return base,denom,one_role

def nig_upper(samples):
    x=np.asarray(samples,dtype=np.float64)
    n=x.size
    if n<32 or not np.isfinite(x).all():
        raise ValueError("JE01_ROLE_PROXY_NIG_OBSERVATIONS_MISSING")
    mean=float(np.mean(x));ss=float(np.sum((x-mean)**2))
    kappa=1+n;mu=n*mean/kappa
    alpha=2+n/2
    beta=1+ss/2+n*mean*mean/(2*kappa)
    scale=math.sqrt(beta*(kappa+1)/(alpha*kappa))
    return {"mean":mean,"posterior_location":mu,
            "predictive_scale":scale,"upper_risk":mu+2*scale,
            "sample_count":n}

def execute(root):
    slots=validate_ledger(read_verified_ledger(LEDGER))
    records=verify_source(root,slots)
    ttrain=np.random.default_rng(TRAIN_SEED)
    ttest=np.random.default_rng(TEST_SEED)
    state={};scores=[]
    for s in slots:
        e=(s["layer"],s["expert"])
        mat={bit:{role:packed_matrix(root,records[(*e,role,bit)])
                  for role in ROLES} for bit in (3,5)}
        def X(rng,count):
            value=rng.normal(size=(count,2048)).astype(np.float32)
            return value/np.sqrt(np.mean(value*value,axis=1,keepdims=True)+1e-12)
        _,_,single=losses(X(ttrain,TRAIN_N),mat[3],mat[5])
        train={role:nig_upper(single[role]) for role in ROLES}
        state[e]={"mat3":mat[3],"mat5":mat[5],
                  "testX":X(ttest,TEST_N),"train":train}
        for role in ROLES:
            scores.append((train[role]["upper_risk"],e,role))
        print(json.dumps({"trained_layer":e[0],"trained_expert":e[1],
                          "role_upper":{r:train[r]["upper_risk"] for r in ROLES}}),flush=True)
    scores.sort()
    bayes={(l,e,r) for _,(l,e),r in scores[:6]}
    if len(bayes)!=6:raise ValueError("JE01_BAYES_ROLE6_NOT_SELECTED")
    fixedfreq=sorted(slots,key=lambda s:(s["calibration_last_token_routed_calls"],s["layer"],s["expert"]))[:2]
    frequency={(s["layer"],s["expert"],r) for s in fixedfreq for r in ROLES}
    if len(frequency)!=6:raise ValueError("JE01_BASELINE_EXPERT2_NOT_6ROLES")
    def sha_sort(item):
        return hashlib.sha256(("JE01_ROLE6_RANDOM_20261118|"+str(item)).encode()).digest()
    random_roles=set(sorted((e[0],e[1],r) for e in state for r in ROLES))
    placebo=set(sorted(random_roles,key=sha_sort)[:6])
    # Explicit non-NLL weight-only MSE alternative, fixed from raw source.
    deltas=[]
    for e in state:
        for role in ROLES:
            n3=records[(*e,role,3)]["relative_weight_MSE_to_original_BF16"]
            n5=records[(*e,role,5)]["relative_weight_MSE_to_original_BF16"]
            deltas.append((n3-n5,e,role))
    deltas.sort()
    weight={(e[0],e[1],role) for _,e,role in deltas[:6]}
    policies={"NIG_proxy_upper_6roles":bayes,
              "weight_MSE_proxy_6roles":weight,
              "prelabel_frequency_2experts_x3":frequency,
              "fixed_role_permutation_6roles":placebo}
    for v in policies.values():
        if len(v)!=6:raise ValueError("JE01_ROLE6_POLICY_COUNT")
    real_cost=6*1261568+18*1982464
    if real_cost!=43253760:raise ValueError("JE01_ROLE_PRECISION_REAL_BYTE_MISMATCH")
    holdout={}
    for name,demotions in policies.items():
        errs=[];cases=[]
        for e,x in state.items():
            X=x["testX"]
            ref=ffn(X,x["mat5"]).astype(np.float64)
            denom=np.mean(ref*ref,axis=1)
            chosen={r:(x["mat3"][r] if (*e,r) in demotions else x["mat5"][r]) for r in ROLES}
            y=ffn(X,chosen).astype(np.float64)
            local=np.mean((y-ref)**2,axis=1)/denom
            if not np.isfinite(local).all():raise ValueError("JE01_TEST_SET_PROXY_NONFINITE")
            cases.extend(float(z) for z in local)
            errs.append(float(np.mean(local)))
        holdout[name]={"selected_n3_roles":[list(t) for t in sorted(demotions)],
                       "independent_heldout_synthetic_mean8_relative_FFN_MSE":float(np.mean(errs)),
                       "independent_case_count":len(cases),
                       "per_case_relative_MSE":cases}
        print(json.dumps({"tested_policy":name,
                          "independent_mean_FFN_MSE":holdout[name]["independent_heldout_synthetic_mean8_relative_FFN_MSE"]}),flush=True)
    result={"schema":SCHEMA,"status":"REAL_BF16_SOURCE_ROLE_LEVEL_NIG_PROXY_TRAIN32_TEST32_COMPLETE",
            "original_source_SHA256":MODEL,"source_native_role_receipt":RECEIPT,
            "baseline_original_AF8_byte_budget":43253760,
            "actual_role3_count":6,"actual_role5_count":18,
            "train_synthetic_seed":TRAIN_SEED,"test_synthetic_seed":TEST_SEED,
            "synthetic_samples_per_expert_per_split":32,
            "role_level_NIG_training_only_on_synthetic_FFN_MSE":True,
            "synthetic_test_policy_results":holdout,
            "source_router_hidden_states":"NOT_USED",
            "original_AF4_q4_gold_NLL":"NOT_MEASURED",
            "27layer_model_forward":"NOT_RUN","OOD_labels_used":False,
            "production_modified":False}
    result["receipt_sha256"]=canonical(result)
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    try:
        if a.output.exists():raise ValueError("JE01_OUTPUT_OVERWRITE_REFUSED")
        res=execute(a.source)
        a.output.write_text(json.dumps(res,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"status":res["status"],"receipt_sha256":res["receipt_sha256"],
                          "test_proxy_MSE":{k:v["independent_heldout_synthetic_mean8_relative_FFN_MSE"]
                                            for k,v in res["synthetic_test_policy_results"].items()}},sort_keys=True),flush=True)
    except Exception as e:
        print(json.dumps({"status":"BLOCKED","reason":str(e)[:270]}),flush=True)
        raise SystemExit(2)
