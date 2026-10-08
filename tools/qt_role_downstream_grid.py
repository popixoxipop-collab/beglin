#!/usr/bin/env python3
"""BEGLIN QT L0 role-budget/downstream-aware allocation diagnostic (fixed offline fixture).

Qwen2.5 0.5B: fixed checkpoint, patched engine and prompt registry.
No model promotion, no network, no production writes. Train/holdout disjoint.
"""
from __future__ import annotations
import array
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid

MODEL = Path("/tmp/qt-l0-fp32-cache/model.safetensors")
CONFIG = Path("/tmp/qt-l0-fp32-cache/config.json")
ENGINE = Path("/tmp/beglin-qt-l0-fp32-control/build-qt-l0-fp32/qwen_infer_gpu")
REPO = Path("/Users/xox/beglin")
OUTPUT_ROOT = Path("/Users/xox/mcp-sandbox/tailnet-commander/qt-role-grid-results")
ROLE_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj")
GROUP = 64
BUDGET = 128
STEP = 32
LAYER_COUNT = 24
HIDDEN = 896
SCHEMA = "beglin-qt-downstream-role-grid/1"
COMMIT = "4c072d8b827c0e0c4806feb5b02832e281a2bb87"
TRAIN = (
    ("train-a16-p7", tuple(range(16)), 7),
    ("train-a16-p15", tuple(range(16)), 15),
    ("train-b16-p7", tuple(range(1000, 1016)), 7),
    ("train-b16-p15", tuple(range(1000, 1016)), 15),
)
HOLDOUT = (
    ("holdout-c32-p15", tuple(range(2000, 2032)), 15),
    ("holdout-c32-p31", tuple(range(2000, 2032)), 31),
    ("holdout-d64-p31", tuple(range(3000, 3064)), 31),
    ("holdout-d64-p63", tuple(range(3000, 3064)), 63),
)

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def tensor(name: str):
    with MODEL.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n < 20_000_000:
            raise RuntimeError("INVALID_MODEL_HEADER")
        hdr = json.loads(f.read(n))
        item = hdr[name]
        start, end = item["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    if item["dtype"] == "BF16":
        buf = array.array("H")
        buf.frombytes(raw)
        if sys.byteorder != "little":
            buf.byteswap()
        as_int = array.array("I", (x << 16 for x in buf))
        val = array.array("f")
        val.frombytes(as_int.tobytes())
    elif item["dtype"] == "F32":
        val = array.array("f")
        val.frombytes(raw)
        if sys.byteorder != "little":
            val.byteswap()
    else:
        raise RuntimeError("UNEXPECTED_TENSOR_DTYPE_" + str(item["dtype"]))
    return val, item["shape"]

def qcell(source, bits: int):
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    mx = max(abs(v) for v in source)
    scale = mx / qmax if mx > 1e-12 else 1.
    inv = 1 / scale
    residual = 0.
    values = []
    for v in source:
        z = v + residual
        q = max(qmin, min(qmax, round(z * inv)))
        deq = q * scale
        values.append(deq)
        residual = z - deq
    return values

def relative_norm(reference, candidate):
    num = sum((b-a)**2 for a,b in zip(reference,candidate))
    den = sum(a*a for a in reference)
    return math.sqrt(num) / (math.sqrt(den) + 1e-12)

def cell_ranking():
    ranked = {role:[] for role in ROLE_ORDER}
    shape = {}
    for role in ROLE_ORDER:
        values, tensor_shape = tensor(f"model.layers.0.self_attn.{role}.weight")
        rows = 896 if role in ("q_proj", "o_proj") else 128
        if len(values) % rows:
            raise RuntimeError("TENSOR_SHAPE_MISMATCH_"+role)
        cols = len(values) // rows
        if cols % GROUP:
            raise RuntimeError("TENSOR_GROUP_MISMATCH_"+role)
        groups = cols // GROUP
        shape[role] = (rows, groups)
        for row in range(rows):
            for group in range(groups):
                base = row * cols + group * GROUP
                x = values[base:base+GROUP]
                gain = relative_norm(x, qcell(x,4)) - relative_norm(x, qcell(x,5))
                ranked[role].append((gain,row,group))
        ranked[role].sort(key=lambda x:(-x[0],x[1],x[2]))
    return ranked,shape

def original_global_counts(ranked):
    items = sorted(((gain,role,row,group) for role, values in ranked.items()
                   for gain,row,group in values), key=lambda x:(-x[0],ROLE_ORDER.index(x[1]),x[2],x[3]))
    counts = {role:0 for role in ROLE_ORDER}
    for _,role,_,_ in items[:BUDGET]:
        counts[role]+=1
    return counts

def validate_counts(counts):
    if set(counts)!=set(ROLE_ORDER) or sum(counts.values())!=BUDGET:
        raise RuntimeError("INVALID_Q5_BUDGET")
    if any(type(counts[role]) is not int or counts[role]<0 for role in ROLE_ORDER):
        raise RuntimeError("INVALID_ROLE_COUNTS")
    return counts

def write_map(work, counts, ranked, shape, cache):
    key=tuple(counts[role] for role in ROLE_ORDER)
    if key in cache:
        return cache[key]
    path = work / ("bits_"+"_".join(map(str,key)))
    path.mkdir()
    for role in ROLE_ORDER:
        rows,groups=shape[role]
        b=bytearray([4])*(rows*groups)
        for _,row,group in ranked[role][:counts[role]]:
            b[row*groups+group]=5
        if b.count(5)!=counts[role]:
            raise RuntimeError("COUNT_MISMATCH_"+role)
        name="model_layers_0_self_attn_"+role+"_weight.bits"
        (path/name).write_bytes(b)
    cache[key] = path
    return path

def read_dump(filename):
    raw=filename.read_bytes()
    if len(raw)!=4*HIDDEN*LAYER_COUNT:
        raise RuntimeError(f"INVALID_DUMP_SIZE:{filename}:{len(raw)}")
    vec=array.array("f")
    vec.frombytes(raw)
    if sys.byteorder!="little":
        vec.byteswap()
    if not all(math.isfinite(v) for v in vec):
        raise RuntimeError("NONFINITE_DUMP")
    return vec, digest(raw)

def compare(reference,candidate):
    if len(reference)!=HIDDEN*LAYER_COUNT or len(candidate)!=HIDDEN*LAYER_COUNT:
        raise RuntimeError("SHAPE_MISMATCH")
    rel=[]
    for k in range(LAYER_COUNT):
        i=k*HIDDEN
        rel.append(relative_norm(reference[i:i+HIDDEN],candidate[i:i+HIDDEN]))
    return rel

def aggregate(metrics):
    if not metrics:
        raise RuntimeError("EMPTY_METRICS")
    v=[float(x) for row in metrics for x in row]
    return {
        "worst":max(v),
        "mean":sum(v)/len(v),
        "per_case_max":[max(row) for row in metrics],
        "per_case_mean":[sum(row)/len(row) for row in metrics],
        "all_layer_relative_l2":metrics
    }

def check_no_overlap():
    train_ids={tuple(tokens) for _,tokens,_ in TRAIN}
    held_ids={tuple(tokens) for _,tokens,_ in HOLDOUT}
    assert train_ids.isdisjoint(held_ids),"LEAKED_HOLDOUT"
    for _,tokens,pos in TRAIN+HOLDOUT:
        assert len(tokens)>=16 and 0<=pos<len(tokens)
        assert max(tokens)<151936
    return True

def run_engine(work,tag,tokens,pos,manifest=None,fp32=False):
    target=work/"dumps"/tag
    if target.exists():
        raise RuntimeError("DUPLICATE_DUMP_ID_"+tag)
    target.parent.mkdir(parents=True,exist_ok=True)
    prompt=work/"prompts"/(tag+".i32")
    prompt.parent.mkdir(parents=True,exist_ok=True)
    raw=struct.pack(f"<{len(tokens)}i",*tokens)
    prompt.write_bytes(raw)
    env={k:v for k,v in os.environ.items() if not k.startswith("QWEN_")}
    env.update(QWEN_SAFETENSORS=str(MODEL),QWEN_HF_CONFIG=str(CONFIG),
               QWEN_PROMPT=str(prompt),QWEN_DEBUG_LAYERDUMP=str(target),
               QWEN_DEBUG_LAYERDUMP_POS=str(pos))
    if manifest is not None:
        env["QWEN_QT_MIXED_MANIFEST"]=str(manifest)
    if fp32:
        if manifest is None:
            raise RuntimeError("MISSING_FP32_MAP")
        env["QWEN_QT_FP32_PASSTHROUGH"]="1"
    t0=time.monotonic()
    p=subprocess.run([str(ENGINE),"greedy","1"],cwd=REPO,env=env,
                     capture_output=True,text=True,timeout=300)
    if p.returncode:
        raise RuntimeError("ENGINE_FAILED:"+tag+":stderr="+p.stderr[-1500:]+":stdout="+p.stdout[-500:])
    vec,sha=read_dump(target)
    markers=[line[:250] for line in p.stderr.splitlines() if "[qt mixed dense]" in line]
    if manifest is not None and len([x for x in markers if "target=model.layers.0.self_attn." in x])<4:
        raise RuntimeError("INCOMPLETE_MIXED_REGISTRATION_"+tag)
    return vec,{"dump_sha256":sha,"prompt_sha256":digest(raw),"position":pos,
                "input_tokens":len(tokens),"run_seconds":round(time.monotonic()-t0,4),
                "qt_markers":markers[:12]}

def candidate_evaluator(work,cases,reference,ranks,shapes,maps,cache,counts,label):
    key=(tuple(counts[r] for r in ROLE_ORDER),tuple(item[0] for item in cases))
    if key in cache:
        return cache[key]
    bmap=write_map(work,counts,ranks,shapes,maps)
    metrics=[];receipts=[]
    for name, tokens, pos in cases:
        tag=f"{label}-{name}"
        v,details=run_engine(work,tag,tokens,pos,manifest=bmap)
        metrics.append(compare(reference[name],v))
        receipts.append({"case":name,**details})
    out={"budget":dict(counts),"train_metrics":aggregate(metrics),"receipts":receipts}
    cache[key]=out
    return out

def clean_result(out):
    return {
        "budget":out["budget"],
        "worst":out["train_metrics"]["worst"],
        "mean":out["train_metrics"]["mean"]
    }

def main():
    check_no_overlap()
    for filename in (MODEL,CONFIG,ENGINE):
        if not filename.is_file() or not filename.stat().st_size:
            raise RuntimeError("MISSING_DEPENDENCY_"+str(filename))
    if not REPO.is_dir():
        raise RuntimeError("MISSING_BEGLIN_REPO")
    filename=Path(__file__).resolve()
    work=Path(tempfile.mkdtemp(prefix="qt-role-grid-",dir="/tmp"))
    ranks,shapes=cell_ranking()
    maps={}
    for case in TRAIN+HOLDOUT:
        name,tokens,pos=case
        (work/"cases").mkdir(parents=True,exist_ok=True)
        (work/"cases"/f"{name}.json").write_text(json.dumps({"ids":tokens,"position":pos}))
    allq4=write_map(work,{r:0 for r in ROLE_ORDER},ranks,shapes,maps)
    assert allq4.exists()
    ref={}
    reference_metadata={}
    baseline={}
    baseline_metadata={}
    contract={}
    for name,tokens,pos in TRAIN+HOLDOUT:
        # Same all-Q4 map and all L0 QKVO FP32 reference across all arms.
        a,rec=run_engine(work,"ref-"+name,tokens,pos,manifest=allq4,fp32=True)
        ref[name]=a;reference_metadata[name]=rec
        b,rec=run_engine(work,"base-"+name,tokens,pos)
        baseline[name]=b;baseline_metadata[name]=rec
        c,rec=run_engine(work,"q4only-"+name,tokens,pos,manifest=allq4)
        contract[name]=relative_norm(b,c)
        if contract[name]>1e-3:
            raise RuntimeError(f"Q4_CONTRACT_MISMATCH:{name}:{contract[name]}")
    def controls(cases):
        return aggregate([compare(ref[name],baseline[name]) for name,_,_ in cases])
    q4_train=controls(TRAIN)
    q4_holdout=controls(HOLDOUT)
    score_cache={}
    fixed={
        "legacy_global_top128":original_global_counts(ranks),
        "balanced_32_each":{r:32 for r in ROLE_ORDER},
        "q_focus_64_16_32_16":dict(zip(ROLE_ORDER,(64,16,32,16))),
        "v_focus_16_16_64_32":dict(zip(ROLE_ORDER,(16,16,64,32))),
    }
    for counts in fixed.values():
        validate_counts(counts)
    fixed_train={}
    for label,counts in fixed.items():
        fixed_train[label]=candidate_evaluator(work,TRAIN,ref,ranks,shapes,maps,score_cache,counts,label)
    greedy={r:0 for r in ROLE_ORDER}
    decisions=[]
    for k in range(BUDGET//STEP):
        offers=[]
        for role in ROLE_ORDER:
            proposal=dict(greedy)
            proposal[role]+=STEP
            label=f"greedy{k+1}-{role}"
            # Partial budgets are deliberately allowed only for internal proposals.
            rank=candidate_evaluator(work,TRAIN,ref,ranks,shapes,maps,score_cache,proposal,label)
            offers.append((rank["train_metrics"]["worst"],rank["train_metrics"]["mean"],ROLE_ORDER.index(role),role,rank))
        offers.sort(key=lambda x:x[:3])
        chosen=offers[0]
        greedy[chosen[3]]+=STEP
        decisions.append({"step":k+1,"picked_role":chosen[3],"counts":dict(greedy),
            "proposals":[{"role":item[3],**clean_result(item[4])} for item in offers]})
    validate_counts(greedy)
    greedy_train=candidate_evaluator(work,TRAIN,ref,ranks,shapes,maps,score_cache,greedy,"greedy_final")
    trains={**fixed_train,"greedy_downstream":greedy_train}
    winner_name=min(trains,key=lambda k:(trains[k]["train_metrics"]["worst"],trains[k]["train_metrics"]["mean"],k))
    chosen_counts=trains[winner_name]["budget"]
    holdouts={}
    for label,counts in {**fixed,"greedy_downstream":greedy}.items():
        holdouts[label]=candidate_evaluator(work,HOLDOUT,ref,ranks,shapes,maps,score_cache,counts,"validation-"+label)
    def summarize(out,ref_stat):
        metric=out["train_metrics"]
        return {"budget":out["budget"],"worst":metric["worst"],"mean":metric["mean"],
            "worst_improvement":ref_stat["worst"]-metric["worst"],
            "worst_relative_improvement":(ref_stat["worst"]-metric["worst"])/ref_stat["worst"] if ref_stat["worst"] else None,
            "mean_relative_improvement":(ref_stat["mean"]-metric["mean"])/ref_stat["mean"] if ref_stat["mean"] else None,
            "per_case_max":metric["per_case_max"],"per_case_mean":metric["per_case_mean"],
            "per_layer":metric["all_layer_relative_l2"],
            "receipts":out["receipts"]}
    summary={
        "schema":SCHEMA,
        "status":"SUCCEEDED",
        "production_write_allowed":False,
        "automatic_live_promotion":False,
        "scope":"L0 Q/K/V/O only; all-L0-QKVO original FP32 vs canonical Q4 and role-budget mixed Q5",
        "model":"Qwen2.5-0.5B-Instruct",
        "beglin_source_commit_expected":COMMIT,
        "fixture_sha256":digest(filename.read_bytes()),
        "engine_sha256":digest(ENGINE.read_bytes()),
        "model_sha256":digest(MODEL.read_bytes()),
        "config_sha256":digest(CONFIG.read_bytes()),
        "roles":ROLE_ORDER,"budget_cells":BUDGET,"group":GROUP,
        "rank_method":"per-role weight-local relative L2 Q4 minus Q5 gain; downstream greedy selects 32-cell role budget using training max error",
        "selection_rule":"minimize worst Relative-L2 across train prompts+positions+24 layers; tie-break mean",
        "train_cases":reference_metadata|{},
        "holdout_cases":{n:reference_metadata[n] for n,_,_ in HOLDOUT},
        "prompts":{"train":[{"case":n,"sha256":reference_metadata[n]["prompt_sha256"],"len":len(t),"position":pos} for n,t,pos in TRAIN],
                   "holdout":[{"case":n,"sha256":reference_metadata[n]["prompt_sha256"],"len":len(t),"position":pos} for n,t,pos in HOLDOUT]},
        "q4_control_direct_relative_l2":contract,
        "q4": {"train":q4_train,"holdout":q4_holdout},
        "greedy_decisions":decisions,
        "train_results":{label:summarize(v,q4_train) for label,v in trains.items()},
        "holdout_results":{label:summarize(v,q4_holdout) for label,v in holdouts.items()},
        "winner_selected_on_train_only":winner_name,
        "winner_budget":chosen_counts,
        "holdout_evaluated_after_selection":True,
    }
    raw=(json.dumps(summary,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    uid=uuid.uuid4().hex
    result=OUTPUT_ROOT/f"qt-role-grid-{uid}.json"
    with result.open("xb") as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    # Explicitly keep stdout concise; full evidence is in the sandbox.
    print(json.dumps({"schema":SCHEMA,"status":"SUCCEEDED","result_path":str(result),
        "result_sha256":digest(raw),"work_dir":str(work),
        "fixture_sha256":summary["fixture_sha256"],
        "model_sha256":summary["model_sha256"],
        "engine_sha256":summary["engine_sha256"],
        "q4_contract_worst":max(contract.values()),
        "train_q4_worst":q4_train["worst"],
        "holdout_q4_worst":q4_holdout["worst"],
        "winner":winner_name,"winner_budget":chosen_counts,
        "train_worst_by_arm":{key:v["train_metrics"]["worst"] for key,v in trains.items()},
        "holdout_worst_by_arm":{key:v["train_metrics"]["worst"] for key,v in holdouts.items()},
        "holdout_mean_by_arm":{key:v["train_metrics"]["mean"] for key,v in holdouts.items()},
        "automatic_live_promotion":False},sort_keys=True))

if __name__=="__main__":
    main()
