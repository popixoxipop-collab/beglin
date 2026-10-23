#!/usr/bin/env python3
"""BEGLIN Qv20 low-capacity deterministic prompt-only precision router.
Training may read ONLY 32 ARC TRAIN calibration scores from original Qwen28 C;
the two heldout splits are NOT opened, used for feature fitting, or inspected.
Both candidate masks have exactly 765493248 projection packed weight bytes and
matched n8 role counts. The policy is frozen BEFORE any heldout model forward.
Scientific gate: leave-one-out question CV must improve best calibration static
gold NLL by >=0.03 AND both masks selected >=4; otherwise fallback fixed best
calibration mask. All thresholds/hyperparameters hardcoded before reading scores.
"""
import hashlib,json,math,os,pathlib
import numpy as np
BASE=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024")
ROOT=BASE/"native_original_C_static/cal_arc_train"
COHORT=BASE/"cal_arc_train/frozen_inputs_private.json"
OUT=BASE/"policy"
PINS={
 "cal":"6bedf97854f140939b248001a491b6f7ae38a078b502a8af9a85d3a47cc99f28",
 "qv8":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
 "random":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde",
 "orig_exec":"f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221"}
N=32
ALPHA=6.0
WINSOR_NLL=2.0
SWITCH_MARGIN=0.10
MIN_LOOCV_IMPROVEMENT=0.03
MIN_USAGE_EACH=4
FEATURES=["ln_prefix_over85","unique_fraction","high_vocab_id_fraction","mean_tokenid_fraction","adjacent_repeat_fraction"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError("QV20_TRAIN_ONLY_GATE_FAILED_"+msg)
def obj(o):return (json.dumps(o,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
def put(p,bytes_):
 with p.open("xb") as f:f.write(bytes_);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def feat(ids):
 n=len(ids)
 require(18<=n<=208,"source token count")
 return [math.log(n/85.0),len(set(ids))/n,
         sum(int(i>=100000) for i in ids)/n,
         sum(ids)/(151936.0*n),
         sum(ids[i]==ids[i-1] for i in range(1,n))/(n-1)]
def ridge(X,y):
 mu=np.mean(X,axis=0);sigma=np.maximum(np.std(X,axis=0),0.05)
 Z=np.column_stack([np.ones(len(X)),(X-mu)/sigma])
 reg=np.eye(Z.shape[1])*ALPHA;reg[0,0]=0.0
 beta=np.linalg.solve(Z.T@Z+reg,Z.T@y)
 return mu,sigma,beta
def predict(v,model):
 mu,sigma,beta=model
 return float(beta[0]+np.dot((v-mu)/sigma,beta[1:]))
def choose(pred,fallback):
 if pred>SWITCH_MARGIN:return 0
 if pred<-SWITCH_MARGIN:return 1
 return fallback
def run():
 if OUT.exists():raise FileExistsError("policy previously frozen; no retrain")
 require(sha(COHORT)==PINS["cal"],"calibration lock")
 for mode,mask in (("static_source","qv8"),("static_random","random")):
  require(sha(BASE/"masks"/(mode+".txt"))==PINS[mask],"static mask "+mode)
 lock=json.loads(COHORT.read_text());rows=lock["selected_rows"]
 require(len(rows)==N,"calibration 32 exact original rows")
 arms={}
 for mode in ("static_source","static_random"):
  p=ROOT/mode/"PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json"
  require(p.is_file(),"actual original Qwen28 C calibration scores unavailable: "+mode)
  doc=json.loads(p.read_text())
  require(doc["state"]=="ACTUAL_ORIGINAL_QWEN28_C_STATIC_"+mode.upper()+"_PASS" and
   doc["cohort"]=="cal_arc_train" and doc["n"]==32 and
   doc["original_Qwen28_C_executable_sha256"]==PINS["orig_exec"] and
   doc["source_frozen_private_SHA256"]==PINS["cal"],"native case receipt "+mode)
  require(len(doc["original_full_vocab_gold_nexttoken_NLL"])==N,"all 32 source results")
  arms[mode]=np.array(doc["original_full_vocab_gold_nexttoken_NLL"],dtype=float)
  require(np.all(np.isfinite(arms[mode])) and np.all(arms[mode]>=0),"nonfinite or negative losses")
 vals=np.asarray([feat(r["prefix_ids"]) for r in rows],dtype=float)
 y=np.clip(arms["static_random"]-arms["static_source"],-WINSOR_NLL,WINSOR_NLL)
 means=[float(np.mean(arms[z])) for z in ("static_source","static_random")]
 best_index=int(np.argmin(means))
 loo=[];selected=[];fold_gains=[]
 for i in range(N):
  mask=np.arange(N)!=i
  model=ridge(vals[mask],y[mask])
  fallback=int(np.argmin([np.mean(arms["static_source"][mask]),np.mean(arms["static_random"][mask])]))
  d=choose(predict(vals[i],model),fallback)
  selected.append(d)
  loo.append(float(arms["static_source"][i] if d==0 else arms["static_random"][i]))
 cvmean=float(np.mean(loo))
 bestmean=min(means)
 cv_improvement=bestmean-cvmean
 support_both=min(sum(v==0 for v in selected),sum(v==1 for v in selected))>=MIN_USAGE_EACH
 accept=bool(cv_improvement>=MIN_LOOCV_IMPROVEMENT and support_both)
 globalmodel=ridge(vals,y)
 fitted_prediction=[predict(x,globalmodel) for x in vals]
 pred_masks=[choose(d,best_index) for d in fitted_prediction]
 policy={
 "schema":"BEGLIN-Qv20-prompt-token-feature-ridge-two-masks-calibration-only/1",
 "training_split":"ARC-Challenge train 32; heldout ARC validation24 and OpenBookQA validation24 NOT opened by this fit",
 "model_type":"fixed-feature robust winsorized ridge, LOOCV pass or static baseline fallback",
 "model_choice":"prompt_features_only",
 "calibration_input_SHA256":PINS["cal"],
 "source_mask_sha256":PINS["qv8"],"random_mask_sha256":PINS["random"],
 "candidate_projection_packed_weight_bytes":765493248,
 "features":FEATURES,
 "feature_means":[float(x) for x in globalmodel[0]],
 "feature_scales":[float(x) for x in globalmodel[1]],
 "coefficients_intercept_and_slopes":[float(x) for x in globalmodel[2]],
 "regularization_alpha":ALPHA,
 "winsorize_training_NLL_difference_to_abs":WINSOR_NLL,
 "profile_Qv8_if_predicted_delta_greater_than":SWITCH_MARGIN,
 "profile_random73_if_predicted_delta_less_than":-SWITCH_MARGIN,
 "otherwise_calibration_best_static_index":best_index,
 "loocv_required_gain_over_best_calibration_static_NLL":MIN_LOOCV_IMPROVEMENT,
 "loocv_required_min_cases_each_profile":MIN_USAGE_EACH,
 "loocv_observed_gain_vs_best_static_NLL":cv_improvement,
 "loocv_profile_counts":[selected.count(0),selected.count(1)],
 "calibration_mean_NLL_source":means[0],
 "calibration_mean_NLL_random73":means[1],
 "calibration_LOOCV_router_mean_NLL":cvmean,
 "dynamic_policy_admitted_from_calibration":accept,
 "selected_active_mode":"CALIBRATED_PROMPT_ROUTER" if accept else "CALIBRATION_BEST_STATIC",
 "static_fallback_index":best_index,
 "all_user_request_policy_features_derived_from_prefix_token_ids_only":True,
 "original_C_eval_heldout_scores_opened_at_fit":False,
 "cross_model_family_or_untouched_heldout_accuracy_gain_NOT_YET_PROVEN":True,
 "metal_GPU_quality_NOT_MEASURED":True,
 "extra_inactive_cache_needed_for_dynamic_two_mask_switches":True}
 os.umask(0o077);OUT.mkdir(parents=True,mode=0o700)
 private=OUT/"calibration_policy_private.json";put(private,obj(policy))
 public={k:v for k,v in policy.items() if k not in ("calibration_mean_NLL_source","calibration_mean_NLL_random73","calibration_LOOCV_router_mean_NLL")}
 public["calibration_policy_private_sha256"]=sha(private)
 public["heldout_results_not_available_when_frozen"]=True
 report=OUT/"Qv20_FROZEN_POLICY_PUBLIC.json";put(report,obj(public))
 print("QV20_CALIBRATION_ONLY_POLICY_FROZEN",json.dumps({
 "training_cases":N,"admitted":accept,"selected_mode":policy["selected_active_mode"],
 "LOOCV_mean_NLL_gain":round(cv_improvement,7),
 "LOOCV_profile_counts":policy["loocv_profile_counts"],
 "full_model_predicted_mask_counts":[pred_masks.count(0),pred_masks.count(1)],
 "private_SHA256":sha(private),"public_SHA256":sha(report)
 },sort_keys=True),flush=True)
if __name__=="__main__":run()
