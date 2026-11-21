#!/usr/bin/env python3
"""Independent re-audit all 256 original DeepSeek AF last-token routed activations.

Decode original 64-way float32 route payloads from every XOX research chunk,
recompute all counts and top8, inspect fragile 6-vs-7 margins. Cross-fit:
choose 8 pilot route slots using first 128 unlabelled calibration prompts
and measure their *routing* binary association on disjoint last128 (Fisher
exact + Holm 28 pair adjustment). This does NOT measure precision-error NLL
covariance or show an Expert bit-allocation benefit.
"""
from collections import Counter
import hashlib,json,math,pathlib,struct,sys

ROOT=pathlib.Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original_router256_20261118_v1")
SCHEMA="beglin-je01-original-256-full64-router-independent-crossfit/1"
MAGIC=b"JE01_REAL_ORIGINAL_ROUTER_L26_T6_V1\n"
ORIGINAL_SHA="6674cefd09124f45599ba40a33a32ebf513968aa91f562f201f4e10e9f34cb3d"

def canon(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                           allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def ranks(rows):
    c=Counter()
    for sample in rows:
        for layer,ids in enumerate(sample,start=1):
            c.update((layer,e) for e in ids)
    return sorted([{"layer":l,"expert":e,"calls":c[(l,e)]}
                   for l in range(1,27) for e in range(64)],
                  key=lambda v:(-v["calls"],v["layer"],v["expert"])),c

def exact_fisher(N,a,b,k):
    lo=max(0,a+b-N);hi=min(a,b)
    if not lo<=k<=hi:raise ValueError("FISHER_INVALID_MARGIN")
    d=math.comb(N,b)
    pmf=lambda z:math.comb(a,z)*math.comb(N-a,b-z)/d
    ref=pmf(k)
    return min(1.0,math.fsum(pmf(z) for z in range(lo,hi+1) if pmf(z)<=ref*(1+1e-12)))

def coactivation(test_rows,slots):
    indicators=[[int(slot["expert"] in sample[slot["layer"]-1]) for sample in test_rows]
                for slot in slots]
    N=len(test_rows)
    pairs=[]
    for i in range(8):
        for j in range(i+1,8):
            a,b=indicators[i],indicators[j]
            x=sum(a);y=sum(b);xy=sum(int(k and v) for k,v in zip(a,b))
            n10=x-xy;n01=y-xy;n00=N-x-y+xy
            denominator=math.sqrt(x*(N-x)*y*(N-y))
            phi=(xy*n00-n10*n01)/denominator if denominator else None
            p=exact_fisher(N,x,y,xy) if denominator else 1.0
            pairs.append({"slot_a":{k:slots[i][k] for k in ("layer","expert")},
                          "slot_b":{k:slots[j][k] for k in ("layer","expert")},
                          "n":N,"a":x,"b":y,"joint":xy,
                          "expected_if_independent":x*y/N,
                          "phi":phi,"fisher_two_sided_p":p})
    order=sorted(range(len(pairs)),key=lambda k:(pairs[k]["fisher_two_sided_p"],k))
    m=len(order);previous=0.0
    for rank,i in enumerate(order):
        previous=max(previous,min(1.0,pairs[i]["fisher_two_sided_p"]*(m-rank)))
        pairs[i]["Holm_adjusted_p"]=previous
        pairs[i]["Holm_supported_at_0_05"]=previous<.05
    return pairs

def audit():
    p=ROOT/"JE01_ACTUAL_ORIGINAL_256_ROUTER_CAL_FINAL.json"
    d=json.loads(p.read_text());sig=d.pop("receipt_sha256")
    if canon(d)!=sig or d.get("status")!="PASS_256_REAL_ORIGINAL_AF_ROUTER_LAST_PROMPT_ONLY" or \
       d.get("calibration_prompt_count")!=256 or \
       d.get("source_calibration256_manifest_sha256")!=ORIGINAL_SHA:
        raise ValueError("JE01_256_NATIVE_FINAL_SOURCE_SHA_OR_SCOPE_DRIFT")
    if len(d["all_chunks"])!=16:raise ValueError("JE01_NOT_16_X16_CHUNKS")
    rows=[];margins=[];max_sumerr=0.0;chunk_sha=[]
    for n in range(16):
        root=ROOT/("chunk"+str(n).zfill(2))
        e=json.loads((root/"JE01_CHUNK_RECEIPT.json").read_text())
        claimed=e.pop("receipt_sha256")
        if claimed!=canon(e) or claimed!=d["all_chunks"][n]["receipt_sha256"]:
            raise ValueError("JE01_ORIGINAL_CHUNK_CANONICAL_SHA_BAD")
        raw=(root/"real64_route.bin").read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e["raw_original_router_sha256"]:
            raise ValueError("JE01_ORIGINAL_NATIVE_ROUTER_BLOB_SHA_BAD")
        if sha(root/"full102400_logits.bin")!=e["raw_original_logits_sha256"]:
            raise ValueError("JE01_NATIVE_FULL_LOGIT_BLOB_SHA_BAD")
        if not raw.startswith(MAGIC):raise ValueError("JE01_REAL_ROUTER_HEADER_BAD")
        pos=len(MAGIC)
        for local in range(16):
            req,index,nlayers=struct.unpack("<3I",raw[pos:pos+12]);pos+=12
            if req!=local or nlayers!=26 or index+1!=e["sample_records"][local]["positions"]:
                raise ValueError("JE01_REAL_ROUTER_SAMPLES_REORDERED")
            scores=struct.unpack("<1664f",raw[pos:pos+26*64*4]);pos+=26*64*4
            chosen=struct.unpack("<156i",raw[pos:pos+26*6*4]);pos+=26*6*4
            per=[]
            for layer in range(26):
                probs=scores[64*layer:64*(layer+1)]
                ids=chosen[6*layer:6*(layer+1)]
                if len(set(ids))!=6 or any(not 0<=x<64 for x in ids) or \
                   any(not math.isfinite(v) or v<0 for v in probs):
                    raise ValueError("JE01_REAL_ROUTER_VALUES_INVALID")
                err=abs(math.fsum(probs)-1)
                if err>1e-4:raise ValueError("JE01_REAL_ROUTER_NOT_NORMALIZED")
                max_sumerr=max(max_sumerr,err)
                gap=min(probs[k] for k in ids)-max(probs[k] for k in range(64) if k not in ids)
                if gap<0:raise ValueError("JE01_ORIGINAL_REAL_ROUTER_TOP6_MISMATCH")
                margins.append(gap);per.append(list(ids))
            if per!=e["sample_records"][local]["per_26_layer_real_top6"]:
                raise ValueError("JE01_SOURCE_PER_SAMPLE_NATIVE_TOP6_DRIFT")
            rows.append(per)
        if pos!=len(raw):raise ValueError("JE01_REAL_ROUTER_TRAILING_DATA")
        chunk_sha.append(claimed)
    if len(rows)!=256 or len(margins)!=6656:raise ValueError("JE01_REAL_ROUTER_256_RECORD_INCOMPLETE")
    full,c=ranks(rows)
    selected=d["exactly_eight_pilot_slots_frozen_by_unlabeled_calibration_last_token_routing"]
    expect=[{"layer":v["layer"],"expert":v["expert"],"calls":v["real_original_calibration_routed_calls"]}
            for v in selected]
    if full[:8]!=expect or canon([[c[(l,e)] for e in range(64)] for l in range(1,27)])!=d["real256_frequency_matrix_sha256"]:
        raise ValueError("JE01_REAL_256_TOP8_CANNOT_BE_REPRODUCED")
    first,_=ranks(rows[:128]);other,_=ranks(rows[128:])
    pairs=coactivation(rows[128:],first[:8])
    margins.sort()
    doc={"schema":SCHEMA,"status":"PASS_INDEPENDENT_ORIGINAL_256_ROUTER_SHAS_AND_CROSSFIT",
         "final_source_native256_receipt_sha256":sig,
         "source_original_calibration_manifest_sha256":ORIGINAL_SHA,
         "real_original_unlabeled_calibration_count":256,
         "actual_64way_model_router_events":6656,
         "actual_top6_expert_selection_events":39936,
         "original_64way_probabilities_verified":425984,
         "chunk_16_original_receipt_sha256":chunk_sha,
         "frozen_all256_recomputed_top8":full[:8],
         "crossfit_train128_top8":first[:8],
         "other128_top8_for_nonfrozen_sanity":other[:8],
         "crossfit_train128_selected_vs_all256_overlap":
             len({(v["layer"],v["expert"]) for v in first[:8]}&{(v["layer"],v["expert"]) for v in full[:8]}),
         "margin_min":margins[0],
         "margin_p01":margins[int(.01*len(margins))],
         "margin_p05":margins[int(.05*len(margins))],
         "margin_median":margins[len(margins)//2],
         "margin_below":{str(t):sum(v<t for v in margins) for t in (1e-8,1e-7,1e-6,1e-5,1e-4)},
         "full64_normalization_max_abs_error":max_sumerr,
         "crossfit_128_independent_last_half_routing_associations":pairs,
         "crossfit_Holm_28pairs_significant_count":sum(v["Holm_supported_at_0_05"] for v in pairs),
         "real_NLL_precision_effect_covariance":"NOT_MEASURED",
         "previously_disclosed_16_cal_gold_not_used_in_router_ranking":True,
         "entire_256_train_split_strictly_unseen_gold_before_router":False,
         "same_source_heldout_or_OOD_labels_accessed":False,
         "model_weights_modified":False,
         "production_promotion_allowed":False}
    doc["receipt_sha256"]=canon(doc)
    return doc

if __name__=="__main__":
    try:
        doc=audit()
        p=ROOT/"JE01_256_ROUTER_INDEPENDENT_CROSSFIT_AUDIT.json"
        with p.open("x") as f:f.write(json.dumps(doc,sort_keys=True,indent=2)+"\n")
        print(json.dumps({"status":doc["status"],"sha256":doc["receipt_sha256"],
                          "top8":doc["frozen_all256_recomputed_top8"],
                          "min_margin":doc["margin_min"],"near_ties":doc["margin_below"],
                          "Holm_28_significant":doc["crossfit_Holm_28pairs_significant_count"]},sort_keys=True))
    except BaseException as e:
        print(json.dumps({"status":"BLOCKED","error":type(e).__name__,"reason":str(e)[:400]},sort_keys=True))
        sys.exit(2)
