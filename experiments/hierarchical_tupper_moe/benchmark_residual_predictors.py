#!/usr/bin/env python3
"""XOX CPU timing of learned micro-router score and residual predictor ONLY.

Fixed representative tensor shapes and synthetic activations; does not measure
pretrained task performance, native qNg64, or end-to-end serving latency.
"""
import argparse,json,hashlib,pathlib,time,statistics
import numpy as np
from trainable_micro_router import fit_kernel,predict,lowrank_residual_model,predict_lowrank_residual,mask_for

def nbytes_of(model):
    if model is None:return 0
    if isinstance(model,np.ndarray):return int(model.nbytes)
    if isinstance(model,dict):return sum(nbytes_of(x) for x in model.values())
    if isinstance(model,(tuple,list)):return sum(nbytes_of(x) for x in model)
    return 0

def benchmark(fn,repeat):
    for _ in range(15):fn()
    times=[]
    for _ in range(repeat):
        before=time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns()-before)/1e6)
    return {"median_ms":float(statistics.median(times)),
            "p95_ms":float(np.quantile(times,.95)),"repetitions":repeat}

def run(output,*,ntrain=80,hidden=2048,channels=88,rank=8,repeat=250):
    rng=np.random.default_rng(20261014)
    train=rng.normal(size=(ntrain,hidden)).astype(np.float64)
    test=rng.normal(size=(1,hidden)).astype(np.float64)
    labels=rng.normal(size=(ntrain,channels)).astype(np.float64)
    # Rank-8 signal plus small independent residual: model shape, not actual DeepSeek data.
    factors=rng.normal(size=(ntrain,rank))
    basis=rng.normal(size=(rank,hidden))
    targets=(factors@basis)/max(rank,1)**.5+0.05*rng.normal(size=(ntrain,hidden))
    gate=fit_kernel(train,labels,.25)
    full=fit_kernel(train,targets,2)
    compact=lowrank_residual_model(train,targets,rank=rank,lam=2)
    keep=max(1,channels//2)
    funcs={
       "gate_only":lambda:mask_for(predict(gate,test),keep),
       "gate_plus_mean":lambda:(mask_for(predict(gate,test),keep),compact["offset"]),
       "gate_plus_full_ridge":lambda:(mask_for(predict(gate,test),keep),predict(full,test)),
       "gate_plus_rank8":lambda:(mask_for(predict(gate,test),keep),predict_lowrank_residual(compact,test)),
    }
    sizes={"gate":nbytes_of(gate),"full_ridge_corrector":nbytes_of(full),
           "rank8_corrector":nbytes_of(compact),"mean_corrector":nbytes_of(compact["offset"])}
    results={name:benchmark(fn,repeat) for name,fn in funcs.items()}
    out={"schema":"beglin-micro-residual-cpu-cost-v1","status":"SHAPE_MATCHED_SYNTHETIC_INPUTS",
         "source_file_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
         "shape":{"train_examples":ntrain,"hidden":hidden,"micro_channels":channels,"rank":rank,"keep":keep},
         "model_resident_bytes_per_expert":sizes,
         "residual_bytes_ratio_rank8_over_ridge":sizes["rank8_corrector"]/sizes["full_ridge_corrector"],
         "cpu_batched_numpy_prediction_timing":results,"real_checkpoint_used":False,
         "micro_ffn_execution_in_timing":False,"native_qng64_used":False,"whole_model_token_accuracy":False,
         "production_touched":False}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    print("CPU_RESIDUAL_PREDICTION_BENCHMARK",json.dumps({"sizes":sizes,"median_ms":{k:round(v["median_ms"],6) for k,v in results.items()}},sort_keys=True))
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=pathlib.Path,required=True)
    p.add_argument("--repeats",type=int,default=250)
    a=p.parse_args()
    run(a.output,repeat=a.repeats)
