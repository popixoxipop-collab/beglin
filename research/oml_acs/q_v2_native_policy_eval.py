#!/usr/bin/env python3
"""Native BEGLIN Qwen full-model Qv2 source-frozen policy evaluation.

Evaluation-only after SHA-pinned BF16 original activation quality and Apple
asynchronous stage MILP were frozen. Serial full-model BEGLIN CPU C remains
unmodified. Six independent precision policies run the EXACT SAME 8 public
pre-registered evaluation prompts. Process execution sequential (avoids
thread oversubscription and memory interference). Does NOT implement actual
async full-model scheduling nor retry blocked per-tensor interventions.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,re,struct,subprocess,sys,time
import numpy as np
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
CONFIG=ROOT/"configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json"
FROZEN=ROOT/"configs/research/BEGLIN_QV2_ASYNC_ACTIVATION_PRE_EVAL_FROZEN_POLICY_20261022.json"
TARGET=ROOT/"lab_local/q_v2_activation_async_native_eval_20261022"
PRIOR=ROOT/"lab_local"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
CASES=("v2_async_activation","role_balanced_random","v1_kernel_weight","all_q4","role_balanced_proxy","source_f32")
ORIG={
 "original_lab_binary":PRIOR/"q_full_model_20261021_v2/beglin_q_lab",
 "repacked_mixed_binary":PRIOR/"qng64_n8_repacked_full_20261022/beglin_mixed_repacked",
 "variable_28_binary":PRIOR/"q_kernel_aware_native_20261022/beglin_kernel_aware_native_Qwen",
 "random_mask":PRIOR/"qng64_n8_repacked_full_20261022/policy_random.txt",
 "proxy_mask":PRIOR/"qng64_n8_repacked_full_20261022/policy_proxy.txt",
 "v1_mask":PRIOR/"q_kernel_aware_native_20261022/chosen_n8_policy.txt",
 "public_prompt":PRIOR/"q_full_model_20261021_v2/q4/prompt_ids.i32"
}
SOURCE_BIN_SHAS={
 "original_lab_binary":"d59f463ba5a13d6559c00229f047802c0ebbb74362b00257fd2aaa4b06fc1e9d",
 "repacked_mixed_binary":"8c16013a343b924b8117f66edcbfd7fb11a86f991127b9cd97fb6ec5552b6664",
 "variable_28_binary":"172613dc69f79f38c8553ff7087e653ec9c0031ca80aec8efd2f76f948599f14"
}
SHAS={
 "random_mask":"72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5",
 "proxy_mask":"f65f50421cd9e9bac32bc780805942d68c3d503075ce962ce25315fdb5914cb9",
 "v1_mask":"148ae7dd773b11e2834339397657248f014f16718920f384394aeb724739f0d1",
 "v2_mask":"cb9268c9177d47705e548c2a90a6b22772569b3f2bb54dfef2767ca53ffa3e76"
}
PRELOCK="0d9177b152286ccd413028e74838529e2a5aad57b42ec793df3095f99208d4af"
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def readobj(p):
    return json.loads(p.read_text())
def private_write(p:pathlib.Path,b:bytes):
    if p.exists():raise FileExistsError("do not overwrite prior model evaluation "+str(p))
    with p.open("xb") as f:
        f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def check_private_file(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("private file absent or unsafe "+str(p))
    return p.read_bytes()
def prerequisites():
    if sha(CONFIG.read_bytes())!=PRELOCK:raise RuntimeError("held evaluation source prelock changed")
    source=readobj(CONFIG)
    frozen=readobj(FROZEN)
    if frozen.get("status")!="PRE_EVALUATION_FROZEN_SOURCE_ONLY" or frozen["selected_policy_sha256"]!=SHAS["v2_mask"]:
        raise RuntimeError("Qv2 tensor mask was not pre-evaluation frozen")
    if frozen["evaluation_prompts_sha256"]!=sha(json.dumps(source["evaluation_public_only"],sort_keys=True).encode()):
        raise RuntimeError("Qv2 fixed evaluation source changed since source-only selection")
    if frozen["n8_tensors"]!=28 or frozen["n4_tensors"]!=168:raise RuntimeError("wrong 28/196 budget")
    if frozen["original_BF16_revision"]!=source["model"]["revision"]:
        raise RuntimeError("official BF16 pin mismatch")
    if not SNAP.is_dir() or not (SNAP/"model.safetensors").is_file() or not (SNAP/"tokenizer.json").is_file():
        raise RuntimeError("missing Apple Qwen source BF16 model")
    for key,digest in SOURCE_BIN_SHAS.items():
        p=ORIG[key]
        if not p.is_file() or sha(p.read_bytes())!=digest:
            raise RuntimeError("research-only BEGLIN native executable changed: "+key)
    for key in ("random_mask","proxy_mask","v1_mask"):
        if sha(ORIG[key].read_bytes())!=SHAS[key]:raise RuntimeError("old frozen source control changed "+key)
    return source,frozen

def mask_lines(ids):
    chosen={(role,int(layer)) for role,layer in ids}
    if len(chosen)!=28 or any(role not in ROLES or l<0 or l>=28 for role,l in chosen):
        raise RuntimeError("not 28 unique architecture projection upgrades")
    return "".join(f"{r} {l}\n" for r in ROLES for l in sorted(x for role,x in chosen if role==r)).encode()

def prepare():
    config,frozen=prerequisites()
    if TARGET.exists():raise FileExistsError("target Qv2 native test already prepared")
    os.umask(0o077);TARGET.mkdir(parents=True,mode=0o700)
    data=mask_lines(frozen["selected_n8_role_layer_ids"])
    if sha(data)!=SHAS["v2_mask"]:raise RuntimeError("frozen actual role layer IDs not matching SHA")
    private_write(TARGET/"v2_mask.txt",data)
    for key,name in (("random_mask","random_mask.txt"),("proxy_mask","proxy_mask.txt"),("v1_mask","v1_mask.txt")):
        private_write(TARGET/name,ORIG[key].read_bytes())
    tokenizer=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    prompts=config["evaluation_public_only"]
    if len(prompts)!=8 or len(set(prompts))!=8:raise RuntimeError("not 8 prelocked public evaluation prompts")
    windows=[]
    for text in prompts:
        # Same Qwen2.5 Instruct user/assistant framing as source BF16 teacher.
        framed="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
        ids=tokenizer.encode(framed,add_special_tokens=False).ids
        if not (10<=len(ids)<128) or any(t<0 or t>=151936 for t in ids):
            raise RuntimeError("unexpected Qwen fixed eval framing or token IDs")
        windows.append(struct.pack("<"+"i"*len(ids),*ids))
    control=ORIG["public_prompt"].read_bytes()
    if len(control)!=40:raise RuntimeError("original 10-token prompt changed")
    all_digests=[]
    for case in CASES:
        d=TARGET/case
        d.mkdir(mode=0o700)
        (d/"results").mkdir(mode=0o700)
        (d/"eval").mkdir(mode=0o700)
        (d/"eval/ppl_work").mkdir(mode=0o700)
        private_write(d/"prompt_ids.i32",control)
        for i,window in enumerate(windows):
            private_write(d/f"eval/ppl_work/w{i}.i32",window)
            all_digests.append((case,i,sha(window)))
    for i in range(8):
        d={digest for case,num,digest in all_digests if num==i}
        if len(d)!=1:raise RuntimeError("evaluation source input mismatch across cases")
    public_meta=dict(schema="BEGLIN-Qv2-native-public-fixed-eight-window-inputs/1",
        status="PREPARED_AFTER_SOURCE_FROZEN_BEFORE_ANY_NATIVE_EVALUATION",
        frozen_source_SHA256=sha(FROZEN.read_bytes()),
        prelock_SHA256=PRELOCK,
        frozen_selected_policy_sha256=SHAS["v2_mask"],
        source_original_model_rev=SNAP.name,
        evaluation_public_window_count=8,
        eval_window_token_lengths=[len(b)//4 for b in windows],
        eval_window_sha256=[sha(b) for b in windows],
        total_predicted_tokens=sum(len(b)//4-1 for b in windows),
        all_six_conditions_exact_same_inputs=True,
        original_baseline_prompt_sha256=sha(control),
        user_private_or_prior_C1D_or_V6_inputs_used=False,
        full_async_runtime_not_implemented=True,
        production_untouched=True)
    private_write(TARGET/"fixed_input_receipt.json",(json.dumps(public_meta,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_NATIVE_EVAL_INPUTS_FROZEN",json.dumps(public_meta,sort_keys=True),flush=True)

def case_config(case):
    if case not in CASES:raise ValueError("not an allowed pinned study condition")
    if case in ("source_f32","all_q4"):
        return ORIG["original_lab_binary"],"f32" if case=="source_f32" else "q4",None
    if case in ("role_balanced_random","role_balanced_proxy"):
        return ORIG["repacked_mixed_binary"],"mixed",TARGET/("random_mask.txt" if case=="role_balanced_random" else "proxy_mask.txt")
    if case=="v1_kernel_weight":
        return ORIG["variable_28_binary"],"mixed",TARGET/"v1_mask.txt"
    return ORIG["variable_28_binary"],"mixed",TARGET/"v2_mask.txt"

def run_case(case):
    prerequisites()
    metadata=readobj(TARGET/"fixed_input_receipt.json")
    if not metadata["all_six_conditions_exact_same_inputs"] or metadata["frozen_selected_policy_sha256"]!=SHAS["v2_mask"]:
        raise RuntimeError("native eval attempted without frozen inputs")
    root=TARGET/case
    output=TARGET/f"{case}_observed_native.json"
    if output.exists():raise FileExistsError("actual native case already measured")
    for i,dig in enumerate(metadata["eval_window_sha256"]):
        if sha(check_private_file(root/f"eval/ppl_work/w{i}.i32"))!=dig:
            raise RuntimeError("unexpected per-case public evaluation bytes")
    exe,mode,mask=case_config(case)
    env=os.environ.copy()
    for forbidden in ("QWEN_GGUF","QWEN_INT4_BIN","QWEN_DENSE_QNG64_OVERRIDE","QWEN_WT_PROMOTION_FILE"):
        if forbidden in env:raise RuntimeError("unexpected pretrained weight override "+forbidden)
    env.update({
      "VECLIB_MAXIMUM_THREADS":"1",
      "QWEN_SAFETENSORS":str((SNAP/"model.safetensors").resolve()),
      "QWEN_HF_CONFIG":str((SNAP/"config.json").resolve()),
      "QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(root.resolve()),
      "QWEN_PROMPT":str((root/"prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":mode})
    if mask is not None:
        mb=check_private_file(mask)
        expected=SHAS["v2_mask"] if case=="v2_async_activation" else SHAS["v1_mask"] if case=="v1_kernel_weight" else SHAS["random_mask"] if case=="role_balanced_random" else SHAS["proxy_mask"]
        if sha(mb)!=expected:raise RuntimeError("role-layer map changed")
        env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(mask.resolve())
    os.umask(0o077)
    phases=["ppl","dump","bench"] if case=="v2_async_activation" else ["ppl"]
    results={}
    for phase in phases:
        args=[str(exe.resolve()),phase]
        if phase=="ppl":args=["/usr/bin/time","-l"]+args+["8"]
        if phase=="bench":args+=["1"]
        if phase=="dump" and (root/"results/c_logits_last.f32").exists():
            raise FileExistsError("v2 new full-vocab logits already generated")
        start=time.monotonic()
        completed=subprocess.run(args,env=env,cwd=str(ROOT),capture_output=True,text=True,timeout=480)
        elapsed=time.monotonic()-start
        log=completed.stdout+"\n"+completed.stderr
        private_write(TARGET/f"{case}_{phase}_log.txt",log.encode())
        if completed.returncode!=0:raise RuntimeError("native failed "+case+"/"+phase+" "+log[-1900:])
        entry=dict(exit_code=0,wall_seconds=round(elapsed,3),log_SHA256=sha(log.encode()),phase=phase)
        if phase=="ppl":
            m=re.search(r"\bppl\s+([0-9.]+)\s+tokens\s+(\d+)\b",completed.stdout)
            peak=re.search(r"(\d+)\s+maximum resident set size",completed.stderr)
            if not m or not peak:raise RuntimeError("native BEGLIN lacks original PPL/RSS receipt")
            val=float(m.group(1));ntokens=int(m.group(2))
            if not math.isfinite(val) or ntokens!=metadata["total_predicted_tokens"]:
                raise RuntimeError("actual NLL token count or finite PPL invalid")
            entry.update(PPL=val,NLL_nats_per_token=math.log(val),
                        prediction_positions=ntokens,peak_RSS_bytes=int(peak.group(1)))
        elif phase=="dump":
            arr=root/"results/c_logits_last.f32"
            b=check_private_file(arr)
            if len(b)!=151936*4:raise RuntimeError("full-vocab Qwen output length changed")
            z=np.frombuffer(b,dtype="<f4")
            if not np.isfinite(z).all():raise RuntimeError("nonfinite full-vocab native output")
            entry.update(logits_sha256=sha(b),top1_id=int(np.argmax(z)))
        elif phase=="bench":
            m=re.search(r"\btok_s\s+([0-9.]+)\s+ms_tok\s+([0-9.]+)",completed.stdout)
            if not m:raise RuntimeError("no native bench")
            entry.update(best_of_3_one_decode_ms=float(m.group(2)),
                        best_of_3_one_decode_tok_s=float(m.group(1)))
        if mode=="mixed":
            cm=re.search(r"projection_role_tensors:\s*f32=0 n8=(\d+) q4=(\d+)",log)
            if not cm or int(cm.group(1))!=28 or int(cm.group(2))!=168:
                raise RuntimeError("new precision mask not applied by native BEGLIN")
            fast=re.search(r"fast_gemv=(\d+) fast_batch_rows=(\d+) other_bit_gemv=(\d+) other_bit_batches=(\d+)",log)
            if not fast or int(fast.group(1))<=0 or any(int(t) for t in fast.groups()[1:]):
                raise RuntimeError("native packed n8 dispatch not observed")
            entry["native_repacked_n8_fast_kernel_calls"]=int(fast.group(1))
        results[phase]=entry
        print("Q_V2_NATIVE_CASE_PHASE_PASS",json.dumps({"case":case,**entry},sort_keys=True),flush=True)
    record=dict(schema="BEGLIN-Qv2-native-fixed-public-window-model-case/1",
        status="PASS_REAL_QWEN_NATIVE_28_LAYER",
        case=case,quality_measurement="public fixed 8 evaluation windows; NLL is 8-window aggregate only",
        benchmark_worker_has_sequential_dependency=True,
        original_BF16_revision=SNAP.name,
        pre_eval_frozen_quality_mask_sha256=SHAS["v2_mask"],
        evaluated_native_BEGLIN_not_Metal=True,
        policy_mode=mode,native_binary_sha256=sha(exe.read_bytes()),
        phases=results,production_touched=False,main_touched=False)
    private_write(output,(json.dumps(record,sort_keys=True,indent=2)+"\n").encode())
    print("Q_V2_NATIVE_CASE_FINISHED",json.dumps({"case":case,"case_receipt_sha256":sha(output.read_bytes()),
            "PPL":results["ppl"]["PPL"],
            "NLL_nats":results["ppl"]["NLL_nats_per_token"]},sort_keys=True),flush=True)

def summarize():
    vals={}
    for case in CASES:
        p=TARGET/f"{case}_observed_native.json"
        j=json.loads(check_private_file(p))
        if j["case"]!=case or j["status"]!="PASS_REAL_QWEN_NATIVE_28_LAYER":
            raise RuntimeError("missing source-frozen native evidence "+case)
        vals[case]=j
    models={c:dict(PPL=v["phases"]["ppl"]["PPL"],NLL=v["phases"]["ppl"]["NLL_nats_per_token"],
             tokens=v["phases"]["ppl"]["prediction_positions"],RSS=v["phases"]["ppl"]["peak_RSS_bytes"],
             receipt_sha256=sha(check_private_file(TARGET/f"{c}_observed_native.json")),
             wall_time_s=v["phases"]["ppl"]["wall_seconds"],
             code_sha256=v["native_binary_sha256"]) for c,v in vals.items()}
    a=models["v2_async_activation"];old=models["role_balanced_random"]
    initial=readobj(TARGET/"fixed_input_receipt.json")
    summary=dict(schema="BEGLIN-Qv2-activation-and-async-aware-native-eval/1",
        status="ENGINEERING_SOURCE_ONLY_POLICY_TESTED__SCIENTIFIC_Q_NO_VERDICT",
        pinned_source_model_rev=SNAP.name,
        original_calibration_code_unchanged=True,
        source_eval_prompts_prelocked_SHA256=initial["evaluation_public_window_sha256"] if "evaluation_public_window_sha256" in initial else initial["eval_window_sha256"],
        frozen_policy_sha256=SHAS["v2_mask"],
        all_six_policies_same_public_8_windows=True,
        total_prediction_positions=initial["total_predicted_tokens"],
        cases=models,
        v2_beats_prior_random_NLL=bool(a["NLL"]<old["NLL"]),
        v2_beats_allQ4_NLL=bool(a["NLL"]<models["all_q4"]["NLL"]),
        isolated_parallel_stage_speed_measured_but_not_endtoend_async=True,
        final_task_AUROC_or_independent_OOD_test_not_performed=True,
        blocked_per_tensor_output_sensitivity_not_retried=True,
        production_and_main_unchanged=True,
        limitations=["8 public synthetic prompts are prospective within research, not final IID/OOD",
          "This run measures sequential full-model BEGLIN only, not an async decoder",
          "BF16 checkpoint widened-F32 reference is not true BF16 math",
          "NLL and one-token warmup best-of-three are separate metrics; no statistical CI",
          "Existing V6 24 consumed row identities remain unknown, prior test rows sealed",
          "No deployment, energy or real workload request batching measured"])
    p=TARGET/"qv2_6case_native_evaluation_summary_private.json"
    private_write(p,(json.dumps(summary,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_ALL_SIX_NATIVE_EVAL_SUMMARY",json.dumps({
        "sha256":sha(p.read_bytes()),
        "tokens":summary["total_prediction_positions"],
        "PPL":{c:v["PPL"] for c,v in models.items()},
        "v2_quality_win_vs_random":summary["v2_beats_prior_random_NLL"],
        "v2_quality_win_vs_Q4":summary["v2_beats_allQ4_NLL"]},sort_keys=True),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--run-case",choices=CASES)
    group.add_argument("--summarize",action="store_true")
    a=parser.parse_args()
    if a.prepare:prepare()
    elif a.run_case:run_case(a.run_case)
    else:summarize()
