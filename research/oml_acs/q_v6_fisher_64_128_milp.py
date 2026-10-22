#!/usr/bin/env python3
"""Source-only Qv6 nested 64/128 directional Fisher + real Apple CPU MILP.

Original Qv5 24-bit mask and source estimates are preserved as baseline.
Data-split SHA was frozen before all new original BF16 gradients. No cross
validation logits/answers enter the allocator. HiGHS proves exact constrained
optimum on original measured Q4/n8 Apple NEON kernel costs.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,sys
import numpy as np
from scipy.stats import spearmanr
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_async_activation_milp as v2

PRE=ROOT/"configs/research/BEGLIN_QV6_SAMPLE_SCALING_PRELOCK_20261022.json"
SRC=ROOT/"docs/research/BEGLIN_QV6_FISHER_SCALING_196_SOURCE_AGGREGATE_20261022.json"
FULL=ROOT/"lab_local/qv6_fisher_scaling_20261022/fisher_24_64_128_source_private.json"
CROSS=ROOT/"lab_local/qv6_fisher_scaling_20261022/openbookqa_val48_frozen.json"
OLD=ROOT/"docs/research/BEGLIN_QV5_FISHER_HIGHS_SOURCE_MILP_20261022.json"
OLDMASK=ROOT/"configs/research/BEGLIN_QV5_FISHER_28N8_SOURCE_POLICY_20261022.txt"
OUT=ROOT/"docs/research/BEGLIN_QV6_NESTED_64_128_SOURCE_POLICY_20261022.json"
MASKS={
  "64":ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",
  "128":ROOT/"configs/research/BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt"
}
PRE_SHA="2e35764878191717ef7b84ed546e72e2a1c513523ecb17d24b54489bd29f319b"
OLDMASK_SHA="88fdc38e6f55a16b0bdd4f7385e70f6de053db577a26d334a3ef3cb831113533"
CROSS_SHA="c09831bf86c6eedebed433dc2a0398a0992afb19fddc9f86750c2d137d71d1b0"

def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def make_policy(ids):
    return "".join(f"{role} {layer}\n" for role,layer in ids).encode()
def verified():
    if OUT.exists() or any(p.exists() for p in MASKS.values()):
        raise FileExistsError("no post-cross-validation Fisher fitting / file overwrite")
    if sha(PRE.read_bytes())!=PRE_SHA or sha(CROSS.read_bytes())!=CROSS_SHA:
        raise RuntimeError("Qv6 source or held cohort drift")
    if not SRC.is_file() or not FULL.is_file() or not OLD.is_file():
        raise RuntimeError("missing original Fisher/BF16 full 128 source evidence")
    src=json.loads(SRC.read_text()); full=json.loads(FULL.read_text())
    if (src["status"]!="SOURCE_ONLY_24_64_128_GRADIENT_SCALING_DONE" or
        src["source_private_full_receipt_SHA256"]!=sha(FULL.read_bytes()) or
        not src["no_validation_label_or_logits_accessed"] or
        src["source_gradients"]!=128 or src["actually_executed_new_BF16_backward_items"]!=104):
        raise RuntimeError("not original source-only 128 gradients")
    if src["nested_split_stats"]!=full["nested_split_stats"] or src["nested_pairwise_rank_stability"]!=full["nested_pairwise_rank_stability"]:
        raise RuntimeError("source aggregate differs from sealed private")
    old=json.loads(OLD.read_text())
    if old["mask_sha256"]!=OLDMASK_SHA or sha(OLDMASK.read_bytes())!=OLDMASK_SHA:
        raise RuntimeError("prior original Qv5 24 Fisher 8bit mask drift")
    gains=src["quality_upgrade_by_role_layer_per_size"]
    if set(gains)!={"24","64","128"}:
        raise RuntimeError("missing exact planned 24/64/128 nested source")
    for n in gains:
        if set(gains[n])!=set(v2.ROLES) or any(len(gains[n][role])!=28 for role in v2.ROLES):
            raise RuntimeError("incomplete 196 gradients at "+n)
    original24=json.loads((ROOT/"docs/research/BEGLIN_QV5_FISHER_SOURCE_AGGREGATE_20261022.json").read_text())
    for r in v2.ROLES:
        for idx in range(28):
            if abs(gains["24"][r][idx]-original24["quality_upgrade_by_role_layer"][r][idx])>1e-8:
                raise RuntimeError("original Qv5 24 source Fisher changed")
    return src,old,gains

def run():
    src,old,gains=verified()
    _,_,stages,serial,memory=v2.verified_sources()
    results={}
    for n in ("64","128"):
        vals=gains[n]
        vector=np.asarray([vals[r][idx] for r in v2.ROLES for idx in range(28)],dtype=np.float64)
        mx=float(np.max(np.abs(vector)))
        if not math.isfinite(mx) or mx<=0:raise RuntimeError("invalid Fisher source score magnitude")
        quality={"quality_upgrade_by_role_layer":{r:[float(g/mx) for g in vals[r]] for r in v2.ROLES}}
        result=v2.solve(quality,stages,serial,memory)
        if (result["source_only_solver_status"]!=0 or result["mip_gap"]>1e-7
          or len(result["selected_ids"])!=28 or result["n4_tensors"]!=168):
            raise RuntimeError("Qv6 64/128 native CPU constrained optimum not proven")
        content=make_policy(result["selected_ids"])
        with MASKS[n].open("xb") as f:
            f.write(content);f.flush();os.fsync(f.fileno())
        results[n]={
            "estimated_native_projection_bytes":result["estimated_projection_weight_bytes"],
            "predicted_single_token_serial_ms":result["original_fullmodel_serial_estimated_ms"],
            "isolated_async_stage_total_cost_ms":result["isolated_async_CPU_stage_total_ms"],
            "source_only_gap":result["mip_gap"],
            "mask_sha256":sha(content),
            "role_counts":result["selected_role_counts"],
            "native_full_layer_stage_plan":result["layer_stage_options"],
            "normalization_max_abs_source_Fisher":mx,
            "new_source_candidate_set":result["selected_ids"],
            "original_no_current_crossdataset_results_seen":True}
        print("QV6_SOURCE_ONLY_HIGHS_EXACT",n,"mask_SHA",sha(content),
              "rolecounts",result["selected_role_counts"],flush=True)
    oldset={tuple(x.split()) for x in OLDMASK.read_text().splitlines()}
    nestedsets={"24":oldset}
    for n in ("64","128"):
        nestedsets[n]={tuple(x.split()) for x in MASKS[n].read_text().splitlines()}
    overlap={}
    for a,b in ((24,64),(24,128),(64,128)):
        x=nestedsets[str(a)];y=nestedsets[str(b)]
        overlap[f"{a}_{b}"]={
           "policy_28_overlap":len(x&y),"policy_jaccard":len(x&y)/len(x|y),
           "source_gradient_rank_spearman":src["nested_pairwise_rank_stability"][f"{a}_{b}"]["full_196_spearman"]}
    output={
       "schema":"BEGLIN-Qv6-source-only-nested-24-64-128-HiGHS-Apple-NEON-MILP/1",
       "status":"FROZEN_64_128_POLICIES_BEFORE_NEW_OPENBOOKQA_VALIDATION",
       "Qv6_prelock_SHA256":PRE_SHA,
       "original_source_128_aggregate_SHA256":sha(SRC.read_bytes()),
       "original_source_128_private_SHA256":src["source_private_full_receipt_SHA256"],
       "cross_corpus_48_frozen_private_SHA256":CROSS_SHA,
       "original_Qv5_24_mask_sha256":OLDMASK_SHA,
       "original_Qv5_24_source_milp_evidence_sha256":sha(OLD.read_bytes()),
       "source_24_64_128_rank_stability":src["nested_split_stats"],
       "source_nested_pairwise_rank":src["nested_pairwise_rank_stability"],
       "selected_policies":results,
       "constrained_selected_overlap":overlap,
       "all_24_64_128_have_exactly_28_n8":True,
       "production_write_allowed":False,"auto_precision_promotion":False,
       "evaluation_48_OpenBookQA_logits_not_opened":True,
       "scientific_adoption_status":"NO_VERDICT_REQUIRE_CROSS_DATASET_PROOF"}
    data=(json.dumps(output,sort_keys=True,indent=2)+"\n").encode()
    with OUT.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    print("QV6_SOURCE_POLICIES_FROZEN",json.dumps({"SHA":sha(data),
      "rank_stability":src["nested_pairwise_rank_stability"],
      "policy_overlap":overlap,
      "policy_sha":{n:results[n]["mask_sha256"] for n in results}},sort_keys=True),flush=True)
if __name__=="__main__":run()
