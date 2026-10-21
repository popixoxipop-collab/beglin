#!/usr/bin/env python3
"""Evaluate a source-frozen Qv2 Qwen bit policy on fresh PUBLIC test prompts.

Five existing source/engine conditions run one at a time on EOE to prevent
CPU and RAM contention. The Qv2 MILP selection was already finalized on
Alienware before any of these eight evaluation windows were tokenized.
No original V6, old 300/600/200 evaluation, production runtime, or HF model
is modified. This is *development* NLL, not original independent final test.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,re,struct,subprocess,sys,time
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
SOURCE=ROOT/"configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json"
FROZEN=ROOT/"docs/research/BEGLIN_Q_V2_MILP_FROZEN_ACTIVATION_ASYNC_POLICY_20261022.json"
LAB=ROOT/"lab_local/q_v2_native_eval_20261022"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
OLD=ROOT/"lab_local"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
EXPECTED_POLICY="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
EXPECTED_PROTOCOL="0d9177b152286ccd413028e74838529e2a5aad57b42ec793df3095f99208d4af"
EXPECTED_V1_RANDOM="72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5"
EXPECTED_V1_KERNEL="148ae7dd773b11e2834339397657248f014f16718920f384394ae b724739f0d1".replace(" ","")
CONDITIONS=("f32","all_q4","v1_random","v1_kernel","v2_activation_async")
def sha(b:bytes):return hashlib.sha256(b).hexdigest()
def put(path:pathlib.Path,data:bytes):
    if path.exists():raise FileExistsError("will not overwrite previous evaluation: "+str(path))
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def prepare():
    if LAB.exists():raise FileExistsError("Qv2 development eval already exists")
    cfgraw=SOURCE.read_bytes()
    if sha(cfgraw)!=EXPECTED_PROTOCOL:raise RuntimeError("source test prompts not originally preregistered")
    prereg=json.loads(cfgraw)
    ev=prereg["evaluation_public_only"]
    assert len(ev)==8 and len(set(ev))==8
    frozen=json.loads(FROZEN.read_text())
    if frozen["status"]!="OPTIMAL_HIGHS_SOURCE_ONLY_BEFORE_TEST" or frozen["selected_n8_policy_sha256"]!=EXPECTED_POLICY or frozen["selected_28_n8"]!=28:
        raise RuntimeError("source-only Qv2 MILP policy not SHA-frozen")
    if not SNAP.is_dir() or not (SNAP/"model.safetensors").exists():raise RuntimeError("original BF16 source checkpoint missing")
    choices=frozen["selected_upgraded_roles"]
    if set(choices)!=set(ROLES):raise RuntimeError("not 7 original tensor roles")
    policyblob="".join(f"{role} {layer}\n" for role in ROLES for layer in sorted(choices[role])).encode()
    if len(policyblob.splitlines())!=28 or sha(policyblob)!=EXPECTED_POLICY:
        raise RuntimeError("public aggregate not enough to reproduce locked policy")
    v1random=OLD/"qng64_n8_repacked_full_20261022/policy_random.txt"
    v1kernel=OLD/"q_kernel_aware_native_20261022/chosen_n8_policy.txt"
    if sha(v1random.read_bytes())!=EXPECTED_V1_RANDOM or sha(v1kernel.read_bytes())!=EXPECTED_V1_KERNEL:
        raise RuntimeError("existing comparator policies drift")
    native={
      "f32":OLD/"q_full_model_20261021_v2/beglin_q_lab",
      "all_q4":OLD/"q_full_model_20261021_v2/beglin_q_lab",
      "v1_random":OLD/"qng64_n8_repacked_full_20261022/beglin_mixed_repacked",
      "v1_kernel":OLD/"q_kernel_aware_native_20261022/beglin_kernel_aware_native_Qwen",
      "v2_activation_async":OLD/"q_kernel_aware_native_20261022/beglin_kernel_aware_native_Qwen"
    }
    if any(not p.is_file() for p in native.values()):raise RuntimeError("missing native Qwen research binary")
    os.umask(0o077)
    LAB.mkdir(parents=True,mode=0o700)
    put(LAB/"selected_v2_policy.txt",policyblob)
    tokenizer=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    strings=["<|im_start|>user\n"+t+"<|im_end|>\n<|im_start|>assistant\n" for t in ev]
    tokenarrays=[tokenizer.encode(t,add_special_tokens=False).ids for t in strings]
    if any(not 12<=len(seq)<=90 or any(i<0 or i>=151936 for i in seq) for seq in tokenarrays):
        raise RuntimeError("public synthetic eval tokenization was invalid")
    old_prompt=OLD/"q_full_model_20261021_v2/q4/prompt_ids.i32"
    if old_prompt.stat().st_mode&0o777 !=0o600:raise RuntimeError("old public synthetic prompt not private")
    for condition in CONDITIONS:
        work=LAB/condition;work.mkdir(mode=0o700)
        (work/"results").mkdir(mode=0o700)
        (work/"eval").mkdir(mode=0o700)
        (work/"eval/ppl_work").mkdir(mode=0o700)
        put(work/"prompt_ids.i32",old_prompt.read_bytes())
        for idx,seq in enumerate(tokenarrays):
            put(work/f"eval/ppl_work/w{idx}.i32",struct.pack("<"+"i"*len(seq),*seq))
    manifest={
      "schema":"beglin-Qv2-independent-native-eval-public-inputs/1",
      "source_model_revision":SNAP.name,"frozen_policy_SHA256":EXPECTED_POLICY,
      "protocol_public_eval_SHA256":EXPECTED_PROTOCOL,
      "n_eval_windows":len(ev),
      "eval_token_lengths":[len(x) for x in tokenarrays],
      "expected_teacher_forcing_token_positions":sum(len(x)-1 for x in tokenarrays),
      "token_ids_hashes":[sha(struct.pack("<"+"i"*len(seq),*seq)) for seq in tokenarrays],
      "no_old_V6_or_task_labels_accessed":True,
      "native_model_conditions":list(CONDITIONS),
      "max_concurrent_full_model_evaluations":1,
      "source_only_selection_frozen_prior_to_tokenization":True
    }
    put(LAB/"public_eval_source_receipt.json",(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode())
    print("V2_EVAL_PUBLIC_WINDOWS_FROZEN",json.dumps(manifest,sort_keys=True),flush=True)
    return manifest,native

def run_condition(name,native,manifest):
    work=LAB/name
    env=os.environ.copy()
    forbidden=("QWEN_GGUF","QWEN_WT_PROMOTION_FILE","QWEN_DENSE_QNG64_OVERRIDE","QWEN_INT4_BIN")
    if any(env.get(k) for k in forbidden):raise RuntimeError("preexisting precision override")
    env.update({
      "QWEN_SAFETENSORS":str((SNAP/"model.safetensors").resolve()),
      "QWEN_HF_CONFIG":str((SNAP/"config.json").resolve()),
      "QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(work.resolve()),
      "QWEN_PROMPT":str((work/"prompt_ids.i32").resolve()),
      "VECLIB_MAXIMUM_THREADS":"1",
      "BEGLIN_Q_LAB_PRECISION":"f32" if name=="f32" else ("q4" if name=="all_q4" else "mixed")
    })
    if name in ("v1_random","v1_kernel","v2_activation_async"):
        policypath={
          "v1_random":OLD/"qng64_n8_repacked_full_20261022/policy_random.txt",
          "v1_kernel":OLD/"q_kernel_aware_native_20261022/chosen_n8_policy.txt",
          "v2_activation_async":LAB/"selected_v2_policy.txt"
        }[name]
        env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(policypath.resolve())
    else:env.pop("BEGLIN_Q_LAB_N8_ROLES_FILE",None)
    cmd=["/usr/bin/time","-l",str(native[name].resolve()),"ppl",str(manifest["n_eval_windows"])]
    begin=time.perf_counter()
    result=subprocess.run(cmd,env=env,cwd=str(ROOT),capture_output=True,text=True,timeout=900)
    wall=time.perf_counter()-begin
    logs=(result.stdout+"\n"+result.stderr).encode("utf-8")
    put(LAB/f"native_{name}_ppl_log.txt",logs)
    if result.returncode!=0:raise RuntimeError(name+" native BEGLIN failed "+result.stderr[-1800:])
    log=logs.decode()
    expected="f32=0 n8=28 q4=168"
    if name in ("v1_random","v1_kernel","v2_activation_async") and expected not in log:
        raise RuntimeError("native mixed 28/196 projection role count wrong "+name)
    p=re.search(r"\[ppl\] fp32 ppl ([\d.]+) over (\d+) tokens",log)
    rss=re.search(r"(\d+)\s+maximum resident set size",log)
    if not p or not rss:raise RuntimeError("native public PPL missing "+name)
    npos=int(p.group(2))
    if npos!=manifest["expected_teacher_forcing_token_positions"]:
        raise RuntimeError("native loss token positions not identical")
    ppl=float(p.group(1))
    if not 1<ppl<1e8:raise RuntimeError("invalid PPL")
    return dict(mode=name,ppl=ppl,nll=float(__import__("math").log(ppl)),
       native_ppl_tokens=npos,process_exit=0,run_wall_seconds=round(wall,2),
       maximum_resident_bytes=int(rss.group(1)),
       native_source_bin_sha256=sha(native[name].read_bytes()),
       log_sha256=sha(logs),
       native_Qwen_28_layers=True,
       inference_kernel_runtime_async_parallel_enabled=False,
       selected_QKV_async_schedule_only_predictive=True)
def main():
    manifest,native=prepare()
    outputs={}
    for name in CONDITIONS:
        outputs[name]=run_condition(name,native,manifest)
        print("V2_EVAL_NATIVE_CONDITION",json.dumps(outputs[name],sort_keys=True),flush=True)
    receipt={
      "schema":"BEGLIN-Qv2-public-development-native-ppl-compared/1",
      "status":"ALL_FIVE_NATIVE_CONDITIONS_EVALUATED_AND_SOURCE_POLICY_FROZEN",
      "policy_sha256":EXPECTED_POLICY,
      "public_eval_source_receipt_sha256":sha((LAB/"public_eval_source_receipt.json").read_bytes()),
      "native_observations":outputs,
      "num_public_eval_windows":manifest["n_eval_windows"],
      "num_teacher_forced_token_positions":manifest["expected_teacher_forcing_token_positions"],
      "quality_proxy_improved_NLL_vs_original_random":outputs["v2_activation_async"]["nll"]<outputs["v1_random"]["nll"],
      "source_calibration_not_eval":True,
      "new_policy_not_posthoc_modified":True,
      "model_weights_untouched":True,"main_or_release_untouched":True,
      "full_model_runtime_async_dispatch_not_included":True,
      "scientific_Q_final_result":"NO_VERDICT",
      "scope_limitations":["Only eight public synthetic development windows: may not generalize",
          "BF16 checkpoint F32-widened native engine is not BF16 arithmetic",
          "QKV and gate/up async stage timings are CPU microbenchmarks, NOT full model async serving",
          "No old V6 24 used source identity, independent IID/OOD or task AUROC",
          "No causal intervention from each separate tensor was performed"]
    }
    out=LAB/"q_v2_public_eval_private_receipt.json"
    put(out,(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
    print("V2_EVAL_NATIVE_FINAL_RECEIPT",json.dumps({"sha256":sha(out.read_bytes()),
       "relative_nll_policy_vs_random":outputs["v2_activation_async"]["nll"]/outputs["v1_random"]["nll"]-1,
       "PPL":{k:v["ppl"] for k,v in outputs.items()}},sort_keys=True),flush=True)
if __name__=="__main__":main()
