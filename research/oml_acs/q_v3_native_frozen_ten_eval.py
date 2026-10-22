#!/usr/bin/env python3
"""SHA-locked V3 original BF16 source-covariance kernel/bit policy on native Qwen.

Pre-registered ten fresh public prompts and source-only original BF16 full
covariance local projection quality. On EOE Apple CPU native BEGLIN, run:
source BF16-widened F32, all Q4G64, old frozen RNG73, Qv2 diagonal async-DAG
mask, Qv3 full-covariance mask. For ALL modes, same 10 exact input token
windows, same checkpoint and native Qwen code, serial full-model execution.

Does not invoke blocked downstream per-tensor intervention, old V6 test,
any real user data or production promotion. Original checkpoint read-only.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,subprocess,time,struct
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
SOURCES=ROOT/"lab_local/q_v3_fresh_10dev_20261022"
OUT=ROOT/"lab_local/q_v3_native_10dev_20261022"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
MANIFEST=SOURCES/"pre_eval_input_receipt.json"
PROTO=ROOT/"configs/research/BEGLIN_Q_V3_EMPIRICAL_COVARIANCE_PRELOCK_20261022.json"
V3=ROOT/"docs/research/BEGLIN_Q_V3_FULL_COV_MILP_SOURCE_EVIDENCE_20261022.json"
OLD_BASE=ROOT/"lab_local/q_full_model_20261021_v2/beglin_q_lab"
MIX_BIN=ROOT/"lab_local/q_kernel_aware_native_20261022/beglin_kernel_aware_native_Qwen"
MASKS={
 "role_random":ROOT/"lab_local/qng64_n8_repacked_full_20261022/policy_random.txt",
 "v2_diag":ROOT/"configs/research/BEGLIN_QV2_ASYNC_DAG_N8_POLICY_20261022.txt",
 "v3_cov":ROOT/"configs/research/BEGLIN_Q_V3_FROZEN_28N8_POLICY_20261022.txt",
}
EXPECTED_SHA={
 "original_input_receipt":"c2b33c5b15a5bcbba02df53d4122a1c1807f4595894c257af40f988c97c5d2cc",
 "preregistration":"5e8824f74b7d80dac191bf944a48d418f87212ca5ed53835f7ab7edf8eb0c267",
 "frozen_source_policy":"f8346e975777a261cd703daafc66d8b68479ddf6508cb6a3850fc707837c5f33",
 "base_binary":"d59f463ba5a13d6559c00229f047802c0ebbb74362b00257fd2aaa4b06fc1e9d",
 "mixed_binary":"172613dc69f79f38c8553ff7087e653ec9c0031ca80aec8efd2f76f948599f14",
 "role_random":"72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5",
 "v2_diag":"e044675ea5b47a5aca0ca75980a55070b932d1a4dc330815bbb097acbf75df7b",
 "v3_cov":"82ae93e06afa7380175afa57535c1f0d1f7fd4320b8d28f981cdc8ee0cf24ef3",
}
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
MODES=("f32","q4","role_random","v2_diag","v3_cov")
ROLE_LIMITS={"f32":(196,0,0),"q4":(0,0,196),"role_random":(0,28,168),"v2_diag":(0,28,168),"v3_cov":(0,28,168)}
V3_COUNTS=[3,1,17,6,0,1,0]
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def secure(path:pathlib.Path)->bytes:
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private original source missing/unsecured: "+str(path))
    return path.read_bytes()
def private_write(path:pathlib.Path,data:bytes):
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def check_original():
    if OUT.exists():raise FileExistsError("frozen V3 native ten-window development run exists; refuse overwrite")
    for key,p in (("original_input_receipt",MANIFEST),("preregistration",PROTO),
                  ("frozen_source_policy",V3),("base_binary",OLD_BASE),("mixed_binary",MIX_BIN)):
        if not p.is_file() or sha(p.read_bytes())!=EXPECTED_SHA[key]:
            raise RuntimeError("original pinned source changed "+key)
    rec=json.loads(secure(MANIFEST))
    v3=json.loads(V3.read_text())
    if rec["status"]!="PASS_LOCKED_PUBLIC_INPUTS_BEFORE_V3_EVALUATION" or rec["prompt_count"]!=10:
        raise RuntimeError("input not preregistered before evaluation")
    if rec["teacher_forced_prediction_positions_expected_sum_input_lengths_minus_10"]!=187:
        raise RuntimeError("expected 187 public synthetic source token positions")
    if v3["status"]!="SOURCE_ONLY_OPTIMAL_MILP_POLICY_FROZEN_BEFORE_NEW_EVAL" or \
       v3["full_model_native_8_public_eval_not_opened"]!=True or \
       v3["fresh_10_prelocked_public_evaluation_inputs_not_opened"]!=True:
        raise RuntimeError("frozen source-only cov MILP provenance failed")
    if not (SNAP/"model.safetensors").is_file():
        raise RuntimeError("original source BF16 Qwen cache absent")
    for name,p in MASKS.items():
        if not p.is_file() or sha(p.read_bytes())!=EXPECTED_SHA[name]:
            raise RuntimeError("masked source changed "+name)
        data=[ln.split() for ln in p.read_text().splitlines()]
        if len(data)!=28 or len(set(tuple(x) for x in data))!=28 or \
           any(len(x)!=2 or x[0] not in ROLES or not x[1].isdigit() or int(x[1])>=28 for x in data):
            raise RuntimeError("not exactly 28 permitted original tensor upgrades "+name)
        if name=="v3_cov" and [sum(x[0]==r for x in data) for r in ROLES]!=V3_COUNTS:
            raise RuntimeError("v3 source frozen role counts changed")
    return rec,v3
def setup():
    rec,v3=check_original()
    os.umask(0o077)
    OUT.mkdir(parents=True,mode=0o700)
    assignments={}
    expected=rec["exact_official_HF_token_ids_sha256"]
    assert len(expected)==10
    for name in MODES:
        work=OUT/name
        work.mkdir(mode=0o700)
        for sub in ("results","eval","eval/ppl_work"):
            (work/sub).mkdir(exist_ok=True,mode=0o700)
        shas=[]
        for i in range(10):
            src=SOURCES/f"eval/ppl_work/w{i}.i32"
            data=secure(src)
            if sha(data)!=expected[i]:
                raise RuntimeError("source prelocked eval tokens changed")
            private_write(work/f"eval/ppl_work/w{i}.i32",data)
            shas.append(sha(data))
        private_write(work/"prompt_ids.i32",secure(SOURCES/"prompt_ids.i32"))
        if sha((work/"prompt_ids.i32").read_bytes())!=shas[0]:
            raise RuntimeError("first prompt not equal to held windows")
        binary=OLD_BASE if name in ("f32","q4") else MIX_BIN
        policy=MASKS.get(name)
        assignments[name]={
           "source_precision":"f32" if name=="f32" else "q4" if name=="q4" else "mixed",
           "binary":str(binary.resolve()),
           "binary_sha256":sha(binary.read_bytes()),
           "mask_file":str(policy.resolve()) if policy else None,
           "mask_sha256":sha(policy.read_bytes()) if policy else None,
           "input_shas":shas}
    frozen={
      "schema":"BEGLIN-QV3-source-fullcov-fixed-ten-public-native-eval/1",
      "status":"FROZEN_V3_POLICY_AND_10_PUBLIC_WINDOWS_BEFORE_NATIVE_EVALUATION",
      "source_V3_policy_document_SHA256":EXPECTED_SHA["frozen_source_policy"],
      "source_V3_28_n8_mask_SHA256":EXPECTED_SHA["v3_cov"],
      "source_before_model_eval_rec_SHA256":EXPECTED_SHA["original_input_receipt"],
      "preregistration_sha256":EXPECTED_SHA["preregistration"],
      "same_native_source_BF16_Qwen":SNAP.name,
      "held_public_windows":10,"expected_prediction_positions":187,
      "policy_names":list(MODES),
      "model_workloads":assignments,
      "Qv3_source_scores_never_used_eval_inputs":True,
      "old_V6_24_holdout_untouched":True,
      "no_output_tensor_interventions":True,
      "production_release_or_main_unchanged":True}
    private_write(OUT/"frozen_10public_manifest.json",
                   (json.dumps(frozen,sort_keys=True,indent=2)+"\n").encode())
    print("QV3_10_PUBLIC_NATIVE_FIXTURE_SHA",sha(secure(OUT/"frozen_10public_manifest.json")),flush=True)
    return frozen
def native_run(name,phase,manifest):
    folder=OUT/name
    cfg=manifest["model_workloads"][name]
    env=os.environ.copy()
    for k in ("QWEN_GGUF","QWEN_INT4_BIN","QWEN_WT_PROMOTION_FILE",
              "QWEN_DENSE_QNG64_OVERRIDE","BEGLIN_Q_LAB_N8_ROLES_FILE"):
        if k in env:raise RuntimeError("unapproved inherited quantization environment "+k)
    env.update({
       "VECLIB_MAXIMUM_THREADS":"1",
       "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
       "QWEN_HF_CONFIG":str(SNAP/"config.json"),
       "QWEN_SAFETENSORS_MAXSEQ":"256",
       "QWEN_BASE":str(folder.resolve()),
       "QWEN_PROMPT":str((folder/"prompt_ids.i32").resolve()),
       "BEGLIN_Q_LAB_PRECISION":cfg["source_precision"]})
    if cfg["mask_file"]:
        p=pathlib.Path(cfg["mask_file"])
        if sha(p.read_bytes())!=cfg["mask_sha256"]:
            raise RuntimeError("role upgrade policy not immutable")
        env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(p)
    binary=pathlib.Path(cfg["binary"])
    if sha(binary.read_bytes())!=cfg["binary_sha256"]:
        raise RuntimeError("native model compiled binary drift")
    args=[str(binary),phase]
    if phase=="ppl":args=["/usr/bin/time","-l",str(binary),"ppl","10"]
    if phase=="bench":args.append("1")
    ts=time.monotonic()
    done=subprocess.run(args,env=env,cwd=str(ROOT),capture_output=True,text=True,timeout=300)
    elapsed=time.monotonic()-ts
    log=(done.stdout+"\n"+done.stderr).encode()
    private_write(folder/f"QV3_{phase}_native_log.txt",log)
    if done.returncode!=0:
        raise RuntimeError(name+"/"+phase+" failed "+str(done.returncode)+done.stderr[-3000:])
    matched=re.search(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)",done.stdout+"\n"+done.stderr)
    if matched is None or tuple(int(x) for x in matched.groups())!=ROLE_LIMITS[name]:
        raise RuntimeError("incorrect model weight precision tier "+name)
    row={"elapsed_wall_seconds":round(elapsed,2),"exit":0,"native_log_sha256":sha(log),
         "projection_counts":[int(x) for x in matched.groups()],
         "model_total_decoder_layers":28}
    if phase=="dump":
        p=folder/"results/c_logits_last.f32"
        a=secure(p)
        if len(a)!=151936*4:
            raise RuntimeError("full-vocab logits wrong length")
        logits=np.frombuffer(a,dtype="<f4")
        if not np.isfinite(logits).all():
            raise RuntimeError("model produced nonfinite output")
        row.update(full_logits_sha256=sha(a),argmax_id=int(logits.argmax()))
    if phase=="ppl":
        text=done.stdout+"\n"+done.stderr
        val=re.search(r"\[ppl\] fp32 ppl ([\d.]+) over (\d+) tokens",text)
        rss=re.search(r"(\d+)\s+maximum resident set size",text)
        windows=re.findall(r"\[ppl\] window (\d+) done \((\d+) toks\)",text)
        if not val or not rss or len(windows)!=10:
            raise RuntimeError("native evaluation missing ten public PPL windows")
        if int(val.group(2))!=187 or not math.isfinite(float(val.group(1))):
            raise RuntimeError("wrong fixed evaluation token count")
        row.update(ppl=float(val.group(1)),NLL=math.log(float(val.group(1))),
                   teacher_forced_prediction_positions=187,
                   peak_RSS_bytes=int(rss.group(1)),windows=len(windows))
    if phase=="bench":
        b=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",done.stdout+"\n"+done.stderr)
        if not b:raise RuntimeError("real model speed receipt absent")
        row.update(native_single_token_best_of_3_tps=float(b.group(1)),
                   native_single_token_best_of_3_ms=float(b.group(2)))
    print("QV3_10PUBLIC_NATIVE",name,phase,json.dumps(row,sort_keys=True),flush=True)
    return row
def main():
    manifest=setup()
    results={}
    for name in MODES:
        results[name]={phase:native_run(name,phase,manifest) for phase in ("dump","ppl","bench")}
    teacher=np.frombuffer(secure(OUT/"f32/results/c_logits_last.f32"),dtype="<f4").astype(np.float64)
    p=np.exp(teacher-teacher.max());p/=p.sum()
    comp={}
    for name in MODES:
        logit=np.frombuffer(secure(OUT/name/"results/c_logits_last.f32"),dtype="<f4").astype(np.float64)
        q=np.exp(logit-logit.max());q/=q.sum()
        comp[name]={
          "KL_F32_to_policy_nats":float(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))))),
          "relative_L2_F32_to_policy":float(np.linalg.norm(logit-teacher)/np.linalg.norm(teacher)),
          "top1_agreement_F32":bool(logit.argmax()==teacher.argmax()),
          "top5_overlap_F32":len(set(np.argsort(logit)[-5:])&set(np.argsort(teacher)[-5:])),
          "PPL":results[name]["ppl"]["ppl"],
          "NLL":results[name]["ppl"]["NLL"],
          "native_bench_best_ms":results[name]["bench"]["native_single_token_best_of_3_ms"],
          "native_bench_tps":results[name]["bench"]["native_single_token_best_of_3_tps"],
          "native_PPL_peak_RSS_bytes":results[name]["ppl"]["peak_RSS_bytes"],
          "full_logits_sha256":results[name]["dump"]["full_logits_sha256"]}
    receipt={
       "schema":"BEGLIN-QV3-frozen-source-empirical-covariance-ten-public-native-eval/1",
       "status":"ALL_NATIVE_EVAL_RUNS_COMPLETED__QUALITY_NEEDS_INDEPENDENT_AUDIT",
       "manifest_sha256":sha(secure(OUT/"frozen_10public_manifest.json")),
       "original_source_BF16_rev":SNAP.name,
       "source_frozen_policy_sha256":EXPECTED_SHA["v3_cov"],
       "window_count":10,"prediction_positions":187,
       "all_native_mode_results":results,
       "same_source_10_public_test_windows":True,
       "quality_speed_memory":comp,
       "original_bf16_source_model_unchanged":True,
       "source_only_covariance_policy_not_tuned_on_eval":True,
       "real_fullmodel_async_scheduling_used":False,
       "previous_v2_actual_parallel_failclosed_preserved":True,
       "scientific_Q_verdict":"NO_VERDICT",
       "production_or_main_modified":False}
    private_write(OUT/"v3_native_10public_receipt.json",
                  (json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV3_10PUBLIC_NATIVE_FINISHED",json.dumps({
       "private_receipt_sha256":sha(secure(OUT/"v3_native_10public_receipt.json")),
       "native_modes":list(results),
       "PPL":{k:v["PPL"] for k,v in comp.items()},
       "tps":{k:v["native_bench_tps"] for k,v in comp.items()},
       "v3_model_quality_not_assumed":True},sort_keys=True),flush=True)
if __name__=="__main__":main()
