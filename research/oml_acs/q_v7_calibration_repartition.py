#!/usr/bin/env python3
"""Source-only Qv7 paired calibration seed reproducibility from real 128 BF16 data.

For each of 3 prelocked seeds, deterministically partition the SAME frozen
128 Qv6 ARC train cases into two disjoint halves of 64. Compute native Q4/n8
directional Fisher sensitivity and source-only, original kernel-cost HiGHS
MILP masks for each of six halves. No model or validation labels are called.
Across *different* seeds halves overlap: never call six independent datasets.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,sys
import numpy as np
from scipy.stats import spearmanr

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_async_activation_milp as v2

PROTOCOL=ROOT/"configs/research/BEGLIN_QV7_QWEN_SEED_REPARTITION_PRELOCK_20261022.json"
PRE_SHA="4fdb9cadfc63de605cb3d29310dd0a94cdb0a37c7394e9b938089b4b8da9474d"
ORIGINAL=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv6-fisher-20261022")
TRAIN=ORIGINAL/"lab_local/qv6_fisher_scaling_20261022/source_arc_128_frozen.json"
SCORES=ORIGINAL/"lab_local/qv6_fisher_scaling_20261022/fisher_24_64_128_source_private.json"
QWEN128_SHA="196376b30e6654aa10f985e5c258ead65b48afd1bfb6965fb329727aea67b2f1"
BASEMASK=ROOT/"configs/research/BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt"
OUTPUT=ROOT/"docs/research/BEGLIN_QV7_QWEN64_SEED_REPARTITION_EVIDENCE_20261022.json"
ROLES=v2.ROLES

def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def sealed(p:pathlib.Path):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("source original Qv6 raw BF16 training array no longer private "+str(p))
    return p.read_bytes()

def evaluate():
    if OUTPUT.exists():raise FileExistsError("source seed repartition already sealed")
    if sha(PROTOCOL.read_bytes())!=PRE_SHA or sha(sealed(SCORES))!=QWEN128_SHA:
        raise RuntimeError("Qv7 preregistration or real original Qwen BF16 source changed")
    c=json.loads(PROTOCOL.read_text())
    if c["partition_random_seeds"]!=[17,41,73] or c["sample_size_each"]!=64 or c["no_auto_promotion"] is not True:
        raise RuntimeError("seed contract changed")
    s=json.loads(sealed(SCORES));rows=s["score_rows_196"]
    train=json.loads(sealed(TRAIN))["all_source_questions"]
    if len(train)!=128 or len(rows)!=196:
        raise RuntimeError("original nested source record incomplete")
    cols4=np.asarray([rows[f"{role}:{i}"]["q4_squared"] for role in ROLES for i in range(28)],dtype=np.float64)
    cols8=np.asarray([rows[f"{role}:{i}"]["n8_squared"] for role in ROLES for i in range(28)],dtype=np.float64)
    if cols4.shape!=(196,128) or cols8.shape!=(196,128) or not np.isfinite(cols4).all() or not np.isfinite(cols8).all():
        raise RuntimeError("original Q4/n8 empirical directional Fisher source missing")
    deltas=cols4-cols8
    _,_,stages,serial,memory=v2.verified_sources()
    results={}
    for seed in c["partition_random_seeds"]:
        rng=np.random.default_rng(seed)
        indices=rng.permutation(128).tolist()
        if sorted(indices)!=list(range(128)):
            raise RuntimeError("random source partition drift")
        folds={}
        for ix,name in enumerate(("A","B")):
            subset=indices[ix*64:(ix+1)*64]
            mean=deltas[:,subset].mean(axis=1)
            scale=float(np.max(np.abs(mean)))
            if scale<=0 or not np.isfinite(mean).all():
                raise RuntimeError("zero or nonfinite source selection score")
            quality={"quality_upgrade_by_role_layer":{role:[float(v/scale) for v in mean[j*28:(j+1)*28]]
                for j,role in enumerate(ROLES)}}
            sol=v2.solve(quality,stages,serial,memory)
            if sol["n8_tensors"]!=28 or sol["mip_gap"]>1e-7:
                raise RuntimeError("source exact MILP not optimal "+name)
            key={f"{role}:{layer}" for role,layer in sol["selected_ids"]}
            if len(key)!=28:raise RuntimeError("missing original projection role unique identities")
            folds[name]={
               "source_selected_indices_SHA256":sha(json.dumps(subset).encode()),
               "actual_gradients_count":64,
               "source_gain_vector_SHA256":sha(np.asarray(mean,dtype="<f8").tobytes()),
               "source_derived_mask_sha256":sha("".join(f"{role} {layer}\n" for role,layer in sol["selected_ids"]).encode()),
               "n8_mask":sorted(key),
               "role_counts":sol["selected_role_counts"],
               "projected_bytes":sol["estimated_projection_weight_bytes"],
               "serial_model_cost_proxy_ms":sol["original_fullmodel_serial_estimated_ms"],
               "stage_concurrent_proxy_ms":sol["isolated_async_CPU_stage_total_ms"],
               "mip_gap":sol["mip_gap"],
               "no_output_dataset_loaded":True}
        ra=deltas[:,indices[:64]].mean(axis=1)
        rb=deltas[:,indices[64:]].mean(axis=1)
        A=set(folds["A"]["n8_mask"]);B=set(folds["B"]["n8_mask"])
        results[str(seed)]={"source_128_partition_indices_sha256":sha(json.dumps(indices).encode()),
           "fold_A_B_disjoint_source_64_each":len(set(indices[:64])&set(indices[64:]))==0,
           "rank_spearman_fold_A_B":float(spearmanr(ra,rb).statistic),
           "selected_mask_intersection_size":len(A&B),
           "selected_mask_jaccard":len(A&B)/len(A|B),
           "folds":folds}
    rhos=[v["rank_spearman_fold_A_B"] for v in results.values()]
    overlap=[v["selected_mask_intersection_size"] for v in results.values()]
    doc={"schema":"BEGLIN-Qv7-three-source-partitions-six-independent-within-seed-half64/1",
      "status":"SOURCE_ONLY_QWEN_BF16_3SEED_2DISJOINT_HALVES_EACH_6_MILP_OPTIMAL",
      "preregistered_protocol_sha256":PRE_SHA,"original_128_train_Fisher_receipt_sha256":QWEN128_SHA,
      "original_128_Qv6_policy_sha256":sha(BASEMASK.read_bytes()),
      "source_gradient_question_count_total":128,"per_fold_calibration_count":64,
      "independence_warning":c["independence_warning"],
      "source_24_64_128_Qv6_evaluation_NOT_used":True,
      "cross_family_TinyLlama_48_evaluation_NOT_used":True,
      "3_seed_pairwise":results,
      "mean_spearman_across_three_seed_pair_partitions":float(np.mean(rhos)),
      "min_max_spearman":[float(min(rhos)),float(max(rhos))],
      "mean_selected_28_overlap":float(np.mean(overlap)),
      "min_max_overlap":[int(min(overlap)),int(max(overlap))],
      "exact_optimizer_cost_constraints_copied_from_Qv6":True,
      "model_quality_superiority_not_assessed":True,
      "runtime_promotion_forbidden":True}
    data=(json.dumps(doc,sort_keys=True,indent=2)+"\n").encode()
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    print("QV7_THREE_SEED_PAIRED64_REAL_SOURCE_MILP_PASS",json.dumps({
       "seed_rhos":{k:round(v["rank_spearman_fold_A_B"],4) for k,v in results.items()},
       "mask_overlaps":{k:v["selected_mask_intersection_size"] for k,v in results.items()},
       "receipt_sha256":sha(data)},sort_keys=True),flush=True)

if __name__=="__main__":evaluate()
