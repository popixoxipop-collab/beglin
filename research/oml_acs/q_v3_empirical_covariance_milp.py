#!/usr/bin/env python3
"""Source-only exact MILP for BEGLIN Qv3 *empirical full-input-covariance*
local projection reconstruction error and real Apple async stage costs.

Reuses the independently audited Qv2 native HiGHS 448 binary stage MILP,
with ONLY source quality gains replaced by original BF16 empirical
cross-channel local-output error scores. Never consumes next-token logits,
NLL, final task labels, or the previously blocked final-model interventions.
Returns a research-only SHA-pinned 28 n8 / 168 Q4 mask, not a serving switch.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,sys,time
import numpy as np
from scipy.stats import spearmanr

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_async_activation_milp as v2

PROTO=ROOT/"configs/research/BEGLIN_Q_V3_EMPIRICAL_COVARIANCE_PRELOCK_20261022.json"
GAIN=ROOT/"docs/research/BEGLIN_Q_V3_EMPIRICAL_COV_196_AGGREGATE_20261022.json"
OUT=pathlib.Path.home()/".local/share/beglin/oml_acs/20261022_qv3_empirical_covariance_policy"
EXPECTED_V3_PROTOCOL_SHA="5e8824f74b7d80dac191bf944a48d418f87212ca5ed53835f7ab7edf8eb0c267"
EXPECTED_MODEL_REV="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private(p:pathlib.Path,blob:bytes):
    with p.open("xb") as f:
        f.write(blob);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def main():
    if OUT.exists():raise FileExistsError("previous source-only Qv3 policy exists; refuse post-eval refit")
    if sha(PROTO.read_bytes())!=EXPECTED_V3_PROTOCOL_SHA:
        raise RuntimeError("precommitted covariance/new evaluation contract changed")
    contract=json.loads(PROTO.read_text())
    if len(contract["fresh_evaluation_prompts"])!=10:
        raise RuntimeError("prelocked evaluation prompts changed")
    if not GAIN.is_file():
        raise RuntimeError("original BF16 empirical-covariance local error source not attested")
    srcbytes=GAIN.read_bytes()
    gains=json.loads(srcbytes)
    if gains["source_v3_protocol_sha256"]!=EXPECTED_V3_PROTOCOL_SHA:
        raise RuntimeError("empirical covariance quality uses a different source calibration contract")
    if gains["model_source_rev"]!=EXPECTED_MODEL_REV or gains["calibration_prompts"]!=12 or gains["original_source_tokens"]!=461:
        raise RuntimeError("source original BF16 or calibration data drift")
    if gains["no_downstream_final_output_interventions"] is not True or gains["no_eval_prompts_loaded"] is not True:
        raise RuntimeError("source empirical quality probe contaminated by model target test")
    if not gains["off_diagonal_covariance_retained_in_x_delta_projection"]:
        raise RuntimeError("source only diagonal, cross-channel covariance wasn't measured")
    if not gains["FP16_residual_cache_FP32_activation_matmul"]:
        raise RuntimeError("source local-output numeric contract unknown")
    new=gains["quality_upgrade_by_role_layer"]
    if set(new)!=set(ROLES) or any(len(new[r])!=28 for r in ROLES):
        raise RuntimeError("incomplete 196 input projection source scores")
    original,baseline,async_cost,serial,memory=v2.verified_sources()
    newvec=np.asarray([new[r][i] for r in ROLES for i in range(28)],dtype=np.float64)
    baselinevec=np.asarray([baseline["quality_upgrade_by_role_layer"][r][i] for r in ROLES for i in range(28)],dtype=np.float64)
    if not np.isfinite(newvec).all() or np.min(newvec)<=0:
        raise RuntimeError("source local full-covariance benefit not positive/finite")
    corr=float(spearmanr(newvec,baselinevec).statistic)
    if not np.isfinite(corr):raise RuntimeError("source quality ranking numerical error")
    # Constraint matrix and underlying measured QKV/gate-up stage encodings
    # are deliberately IDENTICAL to the previously prelocked v2 allocator.
    result=v2.solve({"quality_upgrade_by_role_layer":new},async_cost,serial,memory)
    if len(result["selected_ids"])!=28 or result["mip_gap"]>1e-7:
        raise RuntimeError("native source-only mixed precision MILP not solved")
    os.umask(0o077);OUT.mkdir(parents=True,mode=0o700)
    ids=OUT/"qv3_frozen_n8_role_layers.txt"
    mask="".join(f"{r} {layer}\n" for r,layer in result["selected_ids"]).encode()
    private(ids,mask)
    receipt={
      "schema":"BEGLIN-Qv3-source-only-empirical-covariance-async-MILP/1",
      "status":"SOURCE_ONLY_OPTIMAL_MILP_POLICY_FROZEN_BEFORE_NEW_EVAL",
      "model_original_BF16_revision":EXPECTED_MODEL_REV,
      "source_score_aggregate_sha256":sha(srcbytes),
      "source_score_private_SHA256":gains["private_full_source_SHA256"],
      "source_v3_preregistration_SHA256":EXPECTED_V3_PROTOCOL_SHA,
      "baseline_original_source_activation_v2_sha256":baseline["source_SHA"],
      "source_serial_and_async_real_kernel_profile_sha256":async_cost["source_SHA"],
      "quality_spearman_vs_diagonal_source_ranking":corr,
      "n8_policy_mask_sha256":sha(mask),
      "frozen_optimizer":result,
      "quality_is_local_projection_error_not_global_NLL":True,
      "full_model_native_8_public_eval_not_opened":True,
      "fresh_10_prelocked_public_evaluation_inputs_not_opened":True,
      "old_V6_24_holdout_unknown_and_not_opened":True,
      "runtime_code_and_main_unchanged":True,
      "scientific_Q_verdict":"NO_VERDICT"
    }
    dest=OUT/"qv3_frozen_empirical_covariance_policy.json"
    private(dest,(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV3_EMPIRICAL_COV_POLICY_FROZEN",json.dumps({
      "receipt_sha256":sha(dest.read_bytes()),
      "policy_sha256":sha(mask),
      "source_rank_correlation_to_diagonal":corr,
      "role_counts":{r:sum(rr==r for rr,layer in result["selected_ids"]) for r in ROLES},
      "memory_bytes":result["estimated_projection_weight_bytes"],
      "serial_cost_ms":result["original_fullmodel_serial_estimated_ms"],
      "async_stage_sum_ms":result["isolated_async_CPU_stage_total_ms"],
      "new_source_proxy_gain":result["source_diag_activation_quality_gain"],
      "mip_gap":result["mip_gap"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
