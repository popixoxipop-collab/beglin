#!/usr/bin/env python3
"""JE01 heldout proxy: measured BF16 expert weights, new synthetic activations.

External initial training receipt selected four Experts for exploratory 4bit
and four for 8bit at equal FORMULA byte budget to all Expert 6bit. The new
heldout activation seed was not used for Expert-bit assignment. Not BEGLIN
native codec or gold-token NLL; no pretrained full-model forward/promotion.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np

SCHEMA="beglin-je01-expert-ffn-proxy-independent-synthetic-heldout/1"
TRAIN_SHA="555e11d05ca3228a4cb8a828d6a2a3eef153ff5e4f21b4aa547f5820fdecaf2d"
SOURCE_BF16="36e7e3cafb5c01d3249d2d538ccb77fcb4c5ce2d1df213ea0881b2b85e6debfc"
HELDOUT_SEED=20261118
SAMPLES=16
BUDGET_BITS=48 # 8 experts at 6bits, OR four at 4 and four at 8


def load_module(path):
    spec=importlib.util.spec_from_file_location("real_pretrained_ffn_proxy",path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def avg(values):
    return sum(values)/len(values)


def run(base_path,training_path):
    m=load_module(base_path)
    with open(training_path,encoding="utf-8") as fd:
        training=json.load(fd)
    known=training.pop("receipt_sha256")
    if (known!=TRAIN_SHA or m.canonical(training)!=TRAIN_SHA or
        training["original_bf16_checkpoint_identity_sha256"]!=SOURCE_BF16 or
        training["actual_pretrained_weights_used"] is not True or
        training["experts"]!=list(m.EXPERTS)):
        raise ValueError("JE01_TRAINING_SOURCE_TAMPER")
    train={r["expert_id"]:r for r in training["results"]}
    # Rank solely by pre-heldout 4bit-vs8bit relative output MSE loss.
    sorted_experts=sorted(m.EXPERTS,key=lambda e:(
        train[e]["bit_proxy_loss"]["4"]["relative_output_mse"]-
        train[e]["bit_proxy_loss"]["8"]["relative_output_mse"],e))
    selected4=frozenset(sorted_experts[:4])
    selected8=frozenset(sorted_experts[4:])
    if len(selected4)!=4 or len(selected8)!=4 or selected4 & selected8:
        raise ValueError("JE01_INVALID_MIXED_BIT_PLAN")
    rng=np.random.default_rng(HELDOUT_SEED)
    X=rng.normal(size=(SAMPLES,2048)).astype(np.float32)
    X/=np.sqrt(np.mean(X*X,axis=1,keepdims=True)+1e-12)
    heldout_input_sha=hashlib.sha256(X.tobytes()).hexdigest()
    real=[]
    index_raw=(m.ROOT/"model.safetensors.index.json").read_bytes()
    if hashlib.sha256(index_raw).hexdigest()!=m.INDEX_SHA:
        raise ValueError("JE01_INDEX_SHA_DRIFT")
    names=json.loads(index_raw)["weight_map"]
    shard="model-00002-of-000004.safetensors"
    with (m.ROOT/shard).open("rb") as fh:
        headers,origin=m.load_header(fh)
        for expert in m.EXPERTS:
            parts={}
            for role,shape in (("gate",(1408,2048)),("up",(1408,2048)),
                               ("down",(2048,1408))):
                key=f"model.layers.11.mlp.experts.{expert}.{role}_proj.weight"
                if names.get(key)!=shard: raise ValueError("JE01_EXPERT_SHARD_DRIFT")
                weight,tensor_sha=m.native_weight(fh,headers,origin,key,shape)
                if tensor_sha!=train[expert]["consumed_bf16_tensor_sha256"][role]:
                    raise ValueError("JE01_BF16_EXPERT_CONTENT_DRIFT")
                parts[role]=weight
            baseline=m.ffn(X,parts).astype(np.float64)
            denom=np.mean(baseline*baseline)
            if not np.isfinite(denom) or denom<=0:
                raise ValueError("JE01_BASELINE_NONFINITE")
            costs={}
            for bits in (4,6,8):
                qparts={role:m.affine_group_roundtrip(t,bits) for role,t in parts.items()}
                y=m.ffn(X,qparts).astype(np.float64)
                loss=float(np.mean((y-baseline)**2)/denom)
                if not np.isfinite(loss) or loss<0:raise ValueError("JE01_PROXY_NONFINITE")
                costs[str(bits)]=loss
            real.append({"expert_id":expert,"heldout_relative_output_mse":costs})
    measured={x["expert_id"]:x["heldout_relative_output_mse"] for x in real}
    def cost(plan4):
        return avg([measured[e]["4" if e in plan4 else "8"]
                    for e in m.EXPERTS])
    combos=list(itertools.combinations(m.EXPERTS,4))
    mixtures=[cost(set(combo)) for combo in combos]
    selected_cost=cost(selected4)
    uniform6=avg([measured[e]["6"] for e in m.EXPERTS])
    formula_groups=3*1408*2048//64
    def bytes_for(n):
        return formula_groups*(n*8+8)
    mixed_bytes=4*bytes_for(4)+4*bytes_for(8)
    if mixed_bytes!=8*bytes_for(6):
        raise ValueError("JE01_PROXY_EQUAL_BIT_BUDGET_FAILED")
    result={
      "schema":SCHEMA,"status":"REAL_BF16_WEIGHTS_NEW_SYNTHETIC_INPUT_HOLDOUT_ONLY",
      "source_original_bf16_identity_sha256":SOURCE_BF16,
      "training_observation_receipt_sha256":TRAIN_SHA,
      "training_uses_only_seed":m.SEED,
      "heldout_uses_new_seed":HELDOUT_SEED,
      "heldout_input_sha256":heldout_input_sha,
      "heldout_inputs_per_expert":SAMPLES,
      "experts_bits4_selected_without_holdout":sorted(selected4),
      "experts_bits8_selected_without_holdout":sorted(selected8),
      "heldout_expert_results":real,
      "budget_byte_formula":"PROXY_64_GROUP_BITS_PLUS_2FP32_SCALE_BIAS_NOT_NATIVE_PACKED",
      "packed_proxy_bytes_per_expert":{"4":bytes_for(4),"6":bytes_for(6),"8":bytes_for(8)},
      "proxy_total_bytes_equal_for_mix4x4_4x8_and_uniform6":mixed_bytes,
      "observed_heldout_avg_relative_mse":{
         "selected_four4_four8":selected_cost,
         "uniform6":uniform6,
         "uniform4":avg([measured[e]["4"] for e in m.EXPERTS]),
         "uniform8":avg([measured[e]["8"] for e in m.EXPERTS]),
         "all70_random_equal_byte_mix_mean":avg(mixtures),
         "all70_random_equal_byte_mix_median":sorted(mixtures)[len(mixtures)//2],
         "heldout_posthoc_mix_oracle_min":min(mixtures),
      },
      "heldout_mix_rank_among_70":1+sum(x<selected_cost-1e-15 for x in mixtures),
      "selected_vs_random_mix_mse_reduction_fraction":1-selected_cost/avg(mixtures),
      "selected_vs_uniform6_mse_ratio":selected_cost/uniform6,
      "full_pretrained_model_forward":"NOT_RUN",
      "actual_model_router_hidden_states":False,
      "native_beglin_quantization_parity":"NOT_TESTED",
      "gold_token_nll":"NOT_MEASURED",
      "production_promotion_allowed":False
    }
    result["receipt_sha256"]=m.canonical(result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--training",required=True)
    a=p.parse_args()
    try:print(json.dumps(run(a.base,a.training),sort_keys=True,separators=(",",":"),allow_nan=False))
    except Exception as e:
        print(json.dumps({"schema":SCHEMA,"status":"BLOCKED","reason":str(e)[:240]}))
        raise SystemExit(2)
