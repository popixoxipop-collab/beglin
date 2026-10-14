#!/usr/bin/env python3
"""P16 independent, original BF16/FP32 full-teacher vs native P9 A/B logit audit.

All four P15 original full DeepSeek full-vocabulary .npy arrays are compared to
eight real, C-origin native first-prefill float32 buffers, byte-SHA verified.
No use of answer labels, no noninferiority or production-promotion claim.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,pathlib,re,sys
import numpy as np
import precision_context as pc
from hcp_p4_native_policies_ab import A,B

FROZEN_SHA="0ea208ca6b6bf9fa404fd291679e164f63217c91b1d6e15756d331261998c36e"
RUN_SHA="902452b285faf48927445d8b8df95f20666fc6d0367de7aa3a96b6ffe858efc1"
COPY_SHA="4b1555e02a09a69529b426edf5cef2ed8f906fe9f7a6120e702d23db075d0534"
STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
BINARY="3d0f19b64125dbffb3d80e26ea07eff8d5b2ab80d3aa1d83bc9f59dd729ef9c5"
SOURCE_C="aedb3f915c532b7e454515598b6d9836a4962e23391e2e304980cde64bc7c1af"
SOURCE_MLX="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
TEACHER_BASE="fb8e4bea935f7f0b3511e1fac3b87c1bf54f604d"
TRACE=re.compile(r"^P16_NATIVE_FULL_LOGITS rid=([0-9a-f]{32}) inputsha=([0-9a-f]{64}) plen=([0-9]+) pos=([0-9]+) next=([0-9]+) vocab=([0-9]+) bytes=([0-9]+) floatsha=([0-9a-f]{64})$")

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def distribution(z):
    z=np.asarray(z,dtype=np.float64).ravel()
    e=np.exp(z-z.max())
    if e.sum()<=0 or not np.isfinite(e.sum()):raise ValueError("Invalid logits normalization")
    return e/e.sum()
def stable_kl(p,q):
    m=(p>0)&(q>0)
    if np.any((p>0)&(q==0)):
        raise ValueError("Unexpected exact-zero softmax probability in reference comparison")
    return float(np.sum(p[m]*np.log(p[m]/q[m])))
def compact_float(z):return round(float(z),10)
def compare(native,teacher):
    n=np.asarray(native,dtype=np.float64).ravel()
    t=np.asarray(teacher,dtype=np.float64).ravel()
    if n.shape!=(102400,) or t.shape!=(102400,) or not np.isfinite(n).all() or not np.isfinite(t).all():
        raise ValueError("P16 native/teacher full-vocab logits malformed")
    pn=distribution(n);pt=distribution(t);m=(pn+pt)/2
    ntop=np.argsort(-n)[:10];ttop=np.argsort(-t)[:10]
    d=n-t
    argn=int(np.argmax(n));argt=int(np.argmax(t))
    return {
      "teacher_top1_token":argt,"native_top1_token":argn,
      "top1_equal":argn==argt,"top5_overlap":len(set(ntop[:5])&set(ttop[:5])),
      "top10_overlap":len(set(ntop)&set(ttop)),
      "logits_relative_L2":float(np.linalg.norm(d)/np.linalg.norm(t)),
      "logits_RMSE":float(np.sqrt(np.mean(d*d))),
      "logits_mean_absolute_difference":float(np.mean(np.abs(d))),
      "logits_max_absolute_difference":float(np.max(np.abs(d))),
      "JS_divergence_nats":0.5*stable_kl(pt,m)+0.5*stable_kl(pn,m),
      "KL_teacher_to_native_nats":stable_kl(pt,pn),
      "teacher_entropy_nats":float(-(pt[pt>0]*np.log(pt[pt>0])).sum()),
      "teacher_NLL_of_native_selected_token":float(-np.log(pt[argn])),
      "native_NLL_of_teacher_selected_token":float(-np.log(pn[argt])),
      "teacher_prob_of_native_selected_token":float(pt[argn]),
      "native_prob_of_teacher_selected_token":float(pn[argt]),
      "P16_goodness_computed_from_unlabeled_engineering_prompt_only":True
    }

def audit(repo,private,teachers):
    repo,private,teachers=map(pathlib.Path,(repo,private,teachers))
    source=repo/"experiments/hcp_p16/FROZEN_TEACHER_VS_NATIVE_V1.json"
    if sha(source)!=FROZEN_SHA:raise ValueError("P16 pre-inference reference SHA drifted")
    frozen=json.loads(source.read_text())
    original=private/"P16_NATIVE_8_CAPTURE.json"
    manifest=private/"P16_ORIGINAL_CAPTURE_COPIES_MANIFEST.json"
    if sha(original)!=RUN_SHA or sha(manifest)!=COPY_SHA:
        raise ValueError("P16 actual native original/C-copy evidence changed")
    run=json.loads(original.read_text());copy=json.loads(manifest.read_text())
    if (run.get("status")!="ALL_NATIVE_FIRST_LOGITS_CAPTURED_STABLE" or
        run.get("source_prereg_commit")!="f2ed4b49541db2cdedc2103fcf1aade09a9997c7" or
        run.get("native_requests_total")!=8 or
        run.get("source_C_p16_observer_SHA256")!=SOURCE_C or
        run.get("source_P9_MLX_SHA256")!=SOURCE_MLX or
        run.get("instrumented_native_binary_sha256")!=BINARY or
        run.get("exact_frozen_reference_SHA256")!=FROZEN_SHA or
        run.get("quality_noninferiority_certified") is not False or
        run.get("production_touched") is not False):
        raise ValueError("P16 native run source/quality scope not legitimate")
    live={"binary_sha256":STAGE,"route_sha256":ROUTE,"PIDs":[35481,35493]}
    if run["original_stage_before"]!=live or run["original_stage_after"]!=live:
        raise ValueError("P16 original live stage route or PIDs changed")
    if copy["original_signed_run_report_sha256"]!=RUN_SHA or len(copy["original_raw_native_logits"])!=8:
        raise ValueError("P16 XOX raw logits copy manifest incomplete")
    lines=[]
    for z in (private/"P16_ORIGINAL_WORKER.log").read_text(errors="replace").splitlines():
        if "P16_NATIVE_FULL_LOGITS" in z:
            match=TRACE.fullmatch(z.strip())
            if not match:raise ValueError("P16 native raw C witness not valid")
            lines.append(match.groups())
    if len(lines)!=8:raise ValueError("P16 native source log did not witness all eight captures")
    trace={x[0]:x for x in lines}
    if len(trace)!=8:raise ValueError("P16 C trace duplicate request_id")
    teacher_vectors={}
    for index,row in enumerate(frozen["full_teacher_refs"]):
        for arithmetic in ("bf16","fp32"):
            label=f"prompt{index}_{arithmetic}"
            p=teachers/(label+".logits.npy")
            meta=teachers/(label+".json")
            if not p.is_file() or p.is_symlink() or sha(p)!=row["teacher_ref_full_logit_NPY_SHA"][arithmetic] or sha(meta)!=row["teacher_ref_logit_report_SHA"][arithmetic]:
                raise ValueError("P16 P15 original full-vocab teacher source not SHA pinned")
            t=np.load(p,allow_pickle=False)
            if t.dtype!=np.float32 or t.shape!=(1,102400) or not np.isfinite(t).all():
                raise ValueError("P16 original full logits array wrong shape/dtype")
            teacher_vectors[(index,arithmetic)]=t.ravel()
    results=[];duplicate=collections.defaultdict(list)
    last_epoch=None
    last_policy=None
    for n,row in enumerate(run["capture_rows"]):
        item=frozen["native_order"][n]
        index=item["prompt_index"];policy=item["policy"]
        token=frozen["full_teacher_refs"][index]
        rid=row["native_capture_request_id"]
        if (row["request_index"]!=n or row["input_name"]!=token["prompt_id"] or
            row["policy"]!=policy or
            row["prompt_tokens_sha256"]!=token["token_sha256"] or
            row["prompt_token_count"]!=token["length"] or
            row["actual_native_policy_sha256"]!=pc.policy_hash({"A":A,"B":B}[policy]) or
            row["native_logits_finite"] is not True or
            row["native_logits_argmax_match_greedy_token"] is not True or
            row["actual_native_precision_epoch"]<1 or
            not 0<row["max_worker_rss_observed_bytes"]<14*1024**3 or
            (row["native_transitioned"] and row["terminal_ACK_status"]!="REBIND_SET_APPLIED")):
            raise ValueError("P16 actual input/policy/first-token decision differs from preregistration")
        t=trace.get(rid)
        if not t or t[1]!=token["token_sha256"] or int(t[2])!=token["length"] or int(t[3])!=token["length"]-1 or int(t[4])!=row["native_real_first_generated_token_id"] or int(t[5])!=102400 or int(t[6])!=409600 or t[7]!=row["native_full_logits_sha256"]:
            raise ValueError("P16 C first-prefill actual token input provenance differs from Python witness")
        file=private/(rid+".f32")
        if (not file.is_file() or file.is_symlink() or
            file.name!=row["native_full_logits_binary_filename"] or
            file.stat().st_size!=409600 or sha(file)!=row["native_full_logits_sha256"] or
            copy["original_raw_native_logits"][file.name]["sha256"]!=sha(file)):
            raise ValueError("P16 native binary output is missing or modified")
        raw=np.fromfile(file,dtype="<f4")
        if raw.shape!=(102400,) or not np.isfinite(raw).all() or int(np.argmax(raw))!=row["native_real_first_generated_token_id"]:
            raise ValueError("P16 actual native raw logits do not produce native greedy token")
        if (row["native_transitioned"]!=(last_policy is not None and last_policy!=policy) or
            (last_epoch is not None and
                row["actual_native_precision_epoch"] != last_epoch+int(row["native_transitioned"]))):
            raise ValueError("P16 native real policy transition / epoch monotonicity violated")
        last_epoch=row["actual_native_precision_epoch"]
        last_policy=policy
        duplicate[(index,policy)].append(row["native_full_logits_sha256"])
        pairs={arithmetic:compare(raw,teacher_vectors[(index,arithmetic)]) for arithmetic in ("bf16","fp32")}
        result={"id":row["request_index"],"prompt_index":index,"input_name":row["input_name"],
                "policy":policy,"native_full_logits_SHA256":row["native_full_logits_sha256"],
                "native_first_generated_token":row["native_real_first_generated_token_id"],
                "real_runtime_precision_epoch":row["actual_native_precision_epoch"],
                "real_runtime_ACK_status":row["terminal_ACK_status"],
                "native_vs_teacher":pairs}
        results.append(result)
    if len(duplicate)!=4 or any(len(v)!=2 or v[0]!=v[1] for v in duplicate.values()):
        raise ValueError("P16 duplicate ON/OFF sequence first-logit array not bit stable")
    unique=[]
    for (index,policy),digests in sorted(duplicate.items()):
        row=next(x for x in results if x["prompt_index"]==index and x["policy"]==policy)
        unique.append({"prompt_index":index,"policy":policy,"full_logits_sha256":digests[0],
                       "first_generated_token":row["native_first_generated_token"],
                       "native_vs_BF16":row["native_vs_teacher"]["bf16"],
                       "native_vs_FP32":row["native_vs_teacher"]["fp32"]})
    paired=[]
    for index in (0,1):
        a=next(x for x in unique if x["prompt_index"]==index and x["policy"]=="A")
        b=next(x for x in unique if x["prompt_index"]==index and x["policy"]=="B")
        paired.append({"prompt_index":index,
            "policy_A_vs_B_full_logit_arrays_equal":a["full_logits_sha256"]==b["full_logits_sha256"],
            "policy_A_vs_B_first_token_same":a["first_generated_token"]==b["first_generated_token"],
            "B_minus_A_JS_against_BF16_nats":b["native_vs_BF16"]["JS_divergence_nats"]-a["native_vs_BF16"]["JS_divergence_nats"],
            "B_minus_A_KL_teacher_to_native_BF16_nats":b["native_vs_BF16"]["KL_teacher_to_native_nats"]-a["native_vs_BF16"]["KL_teacher_to_native_nats"],
            "B_minus_A_JS_against_FP32_nats":b["native_vs_FP32"]["JS_divergence_nats"]-a["native_vs_FP32"]["JS_divergence_nats"],
            "B_minus_A_KL_teacher_to_native_FP32_nats":b["native_vs_FP32"]["KL_teacher_to_native_nats"]-a["native_vs_FP32"]["KL_teacher_to_native_nats"],
            "B_minus_A_logit_relative_L2_to_BF16":b["native_vs_BF16"]["logits_relative_L2"]-a["native_vs_BF16"]["logits_relative_L2"]})
    summary={"schema":"beglin-hcp-p16-original-full-vocab-teacher-vs-native-P9-A-B-independent-audit-v1",
             "status":"REAL_FIRST_PREFILL_102400_LOGITS_NATIVE_A_B_VS_ORIGINAL_BF16_FP32_TEACHERS_MEASURED",
             "original_native_capture_report_sha256":RUN_SHA,
             "original_XOX_log_file_sha256":sha(private/"P16_ORIGINAL_WORKER.log"),
             "all_native_full_arrays_sha_validated":True,
             "all_P15_BF16_FP32_teacher_arrays_sha_validated":True,
             "native_8_requests":8,"teacher_inputs":2,"precision_policies":["A","B"],
             "independent_native_full_logits_each":102400,
             "all_actual_C_prefill_token_SHA_and_position_verified":True,
             "original_P15_model_27_layers_source_tensor_count":5291,
             "teacher_reference_weight_storage_precision":"BF16",
             "teacher_reference_computation_precisions":["BF16","FP32"],
             "all_duplicate_input_policy_cells_exactly_byte_equal":True,
             "all_original_production_stage_and_PIDs_unchanged":True,
             "unique_native_policy_prompt_metrics":unique,
             "paired_policy_B_minus_A_per_prompt":paired,
             "full_logits_native_outcome_not_task_label_accuracy":True,
             "two_prompts_not_enough_for_noninferiority_CI":True,
             "full_native_multitoken_quality_tested":False,
             "quality_noninferiority_certified":False,
             "main_modified":False,"automatic_live_promotion":False}
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True);p.add_argument("--native",required=True)
    p.add_argument("--teachers",required=True);p.add_argument("--output",required=True)
    a=p.parse_args()
    result=audit(a.repo,a.native,a.teachers)
    out=pathlib.Path(a.output)
    with out.open("x") as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    print("P16_REAL_NATIVE_FULL_LOGITS_VS_TWO_FULL_TEACHERS",json.dumps({
      "status":result["status"],"native_requests":8,"unique_cells":4,
      "original_FULL_BF16_reference_measured":True,
      "quality_noninferiority_certified":False,
      "summary_sha256":sha(out)},sort_keys=True))
