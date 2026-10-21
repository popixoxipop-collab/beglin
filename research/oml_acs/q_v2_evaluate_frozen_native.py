#!/usr/bin/env python3
"""Frozen post-selection native BEGLIN Qv2 development evaluation.

Creates 8 public synthetic prospective PPL windows prelisted in Qv2 JSON
before activation capture. Same byte-identical windows for all conditions.
Reads pinned source-only policy manifest 0600, NEVER re-optimizes from scores
or uses eval prompts to change quant selection. Runs original Qwen C
F32(BF16 source widened), Q4, old frozen random and two new source-only masks.
All benchmark output / prompt tokens are private 0600; model unmodified.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,struct,subprocess,time
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
EXPR=ROOT/"lab_local/q_v2_joint_activation_async_20261022"
PRE=ROOT/"configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
FIX=ROOT/"lab_local/q_full_model_20261021_v2"
MIX=ROOT/"lab_local/qng64_n8_repacked_full_20261022"
ALL=ROOT/"lab_local/q_kernel_aware_native_20261022"
OUT=ROOT/"lab_local/q_v2_new_public_development_20261022"
MODELS={
 "widened_f32":(FIX/"beglin_q_lab","f32",None),
 "full_q4":(FIX/"beglin_q_lab","q4",None),
 "prior_random_mixed":(MIX/"beglin_mixed_repacked","mixed",MIX/"policy_random.txt"),
 "activation_serial100":(ALL/"beglin_kernel_aware_native_Qwen","mixed",EXPR/"serial100_source_only_roles.txt"),
 "activation_async80_candidate":(ALL/"beglin_kernel_aware_native_Qwen","mixed",EXPR/"parallel80_source_only_roles.txt"),
}
EXPECTED_FREEZE_SHA="8dd48956820d47073f7f41" # prefix must check exact sha from original capture
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private(path:pathlib.Path)->bytes:
    if not path.is_file() or path.is_symlink() or (path.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("missing private input "+str(path))
    return path.read_bytes()
def put(path:pathlib.Path,data:bytes):
    if path.exists():raise FileExistsError("refuse to overwrite previous evaluation")
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def prepare():
    if OUT.exists():raise FileExistsError("no retuning after new evaluation")
    freeze=private(EXPR/"frozen_v2_source_only_policies.json")
    if not sha(freeze).startswith(EXPECTED_FREEZE_SHA):
        raise RuntimeError("Qv2 source-only policy freeze changed after selection")
    j=json.loads(freeze)
    if j["status"]!="FROZEN_BEFORE_NATIVE_CANDIDATE_EVALUATION":
        raise RuntimeError("candidate source-only not frozen")
    p=json.loads(PRE.read_text())
    tests=p["evaluation_public_only"]
    if len(tests)!=8 or len(set(tests))!=8:raise RuntimeError("pre-registered public prompts changed")
    for name,(_,_,policy) in MODELS.items():
        if policy:
            if name in ("activation_serial100","activation_async80_candidate"):
                key="serial100" if name=="activation_serial100" else "parallel80"
                if sha(private(policy))!=j["selected_policies"][key]["policy_sha256"]:
                    raise RuntimeError("selected source-only mask mismatch")
            elif sha(private(policy))!="72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5":
                raise RuntimeError("previous random policy mismatch")
    if not SNAP.is_dir() or not (SNAP/"model.safetensors").exists():
        raise RuntimeError("original source BF16 model unavailable")
    os.umask(0o077)
    OUT.mkdir(parents=True,mode=0o700)
    tokenizer=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    ids=[]
    for sentence in tests:
        text="<|im_start|>user\n"+sentence+"<|im_end|>\n<|im_start|>assistant\n"
        encoding=tokenizer.encode(text,add_special_tokens=False).ids
        if not 8<=len(encoding)<=128 or any(not isinstance(x,int) or x<0 or x>=151936 for x in encoding):
            raise RuntimeError("invalid tokenizer result")
        ids.append(encoding)
    sha_prompts=sha(json.dumps(tests,sort_keys=True).encode())
    for name in MODELS:
        work=OUT/name
        work.mkdir(mode=0o700)
        (work/"eval").mkdir(mode=0o700)
        (work/"eval/ppl_work").mkdir(mode=0o700)
        (work/"results").mkdir(mode=0o700)
        for i,seq in enumerate(ids):
            put(work/f"eval/ppl_work/w{i}.i32",struct.pack("<"+"i"*len(seq),*seq))
        # Same exact pre-registered first new eval prompt for one-token logits.
        put(work/"prompt_ids.i32",struct.pack("<"+"i"*len(ids[0]),*ids[0]))
    lengths=[len(x) for x in ids]
    metadata=dict(schema="BEGLIN-Qv2-fixed-new-public-eval/1",
       status="FROZEN_BEFORE_ANY_EVALUATION_SCORE",model="Qwen2.5-1.5B-Instruct",
       model_original_BF16_rev=SNAP.name,public_eval_prompt_sha256=sha_prompts,
       frozen_source_policy_sha256=sha(freeze),
       n_public_eval_prompts=len(tests),window_lengths=lengths,
       teacher_forcing_positions=sum(x-1 for x in lengths),
       all_conditions_same_window_bytes=True,
       former_calibration_prompts_not_used=True,
       V6_old_final_not_accessed=True,prod_main_unchanged=True)
    put(OUT/"eval_prelock_receipt.json",(json.dumps(metadata,sort_keys=True,indent=2)+"\n").encode())
    print("QV2_SAME_EVAL_WINDOWS_PREPARED",json.dumps(metadata,sort_keys=True),flush=True)
    return metadata

def run_case(name,binary,mode,policy,info):
    base=OUT/name
    env=os.environ.copy()
    if any(x in env for x in ("QWEN_GGUF","QWEN_DENSE_QNG64_OVERRIDE","QWEN_WT_PROMOTION_FILE")):
        raise RuntimeError("unexpected precision/weight override in parent environment")
    env.update({
      "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
      "QWEN_HF_CONFIG":str(SNAP/"config.json"),
      "QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(base.resolve()),
      "QWEN_PROMPT":str((base/"prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":mode,
      "VECLIB_MAXIMUM_THREADS":"1",
    })
    if policy:env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(policy.resolve())
    elif "BEGLIN_Q_LAB_N8_ROLES_FILE" in env:raise RuntimeError("inherited role override")
    logit_job=[str(binary.resolve()),"dump"]
    ppl_job=["/usr/bin/time","-l",str(binary.resolve()),"ppl","8"]
    t=time.monotonic()
    result={"name":name,"weight_precision":mode,"main_or_production_touched":False}
    for action,argv in (("dump",logit_job),("ppl",ppl_job)):
        p=subprocess.run(argv,env=env,cwd=str(ROOT),text=True,
              capture_output=True,timeout=600)
        blob=(p.stdout+"\n"+p.stderr).encode()
        put(OUT/f"{name}_{action}_log.txt",blob)
        if p.returncode!=0:
            raise RuntimeError(f"Qv2 {name} native {action} failed: {p.returncode}: "+p.stderr[-900:])
        result[action+"_log_SHA256"]=sha(blob)
        text=blob.decode()
        if action=="dump":
            file=base/"results/c_logits_last.f32"
            if not file.exists() or file.stat().st_size!=151936*4:
                raise RuntimeError("native final logits invalid")
            if file.stat().st_mode&0o777!=0o600:
                raise RuntimeError("native logits not private (umask)")
            result["logits_SHA256"]=sha(file.read_bytes())
            top=re.search(r"\[engine\] dumped, argmax=(\d+)",text)
            if not top:raise RuntimeError("native dump missing actual result")
            result["argmax"]=int(top.group(1))
        else:
            ppl=re.search(r"\[ppl\] .*ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",text)
            rss=re.search(r"(\d+)\s+maximum resident set size",text)
            if not ppl or not rss or int(ppl.group(2))!=info["teacher_forcing_positions"]:
                raise RuntimeError("model PPL evaluation not matching frozen public windows")
            result["PPL"]=float(ppl.group(1))
            result["NLL_nats_per_token"]=math.log(result["PPL"])
            result["tokens"]=int(ppl.group(2))
            result["RSS_bytes"]=int(rss.group(1))
        print("BEGLIN_QV2_EVAL_CASE_PROGRESS",name,action,
          "elapsed",round(time.monotonic()-t,2),flush=True)
    result["elapsed_seconds"]=round(time.monotonic()-t,2)
    return result

def main():
    meta=prepare()
    cases={}
    for name,(binary,mode,policy) in MODELS.items():
        if not binary.is_file():raise RuntimeError("missing prior native BEGLIN C executable")
        cases[name]=run_case(name,binary,mode,policy,meta)
    result=dict(schema="BEGLIN-Qv2-native-new-public-development-eval/1",
        status="PASS_NATIVE_EVALUATION_DEVELOPMENT_ONLY",
        freeze_receipt_sha256=sha(private(OUT/"eval_prelock_receipt.json")),
        source_policy_sha256=meta["frozen_source_policy_sha256"],
        actual_new_public_eval_tokens=meta["teacher_forcing_positions"],
        conditions=cases,
        no_original_V6_24_or_C1D_label_data=True,
        PPL_is_development_not_final=True,
        whole_model_ASYNC_dispatch_not_implemented=True,
        production_unchanged=True)
    output=OUT/"native_results_private.json"
    put(output,(json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_NATIVE_NEW_EVAL_ALL_PASS",json.dumps({
      "receipt_SHA256":sha(output.read_bytes()),
      "eval_positions":meta["teacher_forcing_positions"],
      "PPL":{k:v["PPL"] for k,v in cases.items()},
      "RSS":{k:v["RSS_bytes"] for k,v in cases.items()}},sort_keys=True),flush=True)
if __name__=="__main__":main()
