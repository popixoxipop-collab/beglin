#!/usr/bin/env python3
"""BEGLIN Q v2 frozen-policy prospective public native Qwen evaluation.

This is an *execution driver*, not the optimizer. It will NOT recalibrate
or change the 28 tensor assignments. It checks the source-only MILP policy
SHA committed BEFORE native output validation and uses eight independent
public prompts sealed in the original Q v2 preregistration. No old V6 final,
C1-D labels, private texts or user records are accessed.

Run only in the separated EOE scratch checkout, not live operator:
  python3 research/oml_acs/q_v2_8window_native_evaluator.py --prepare
  python3 research/oml_acs/q_v2_8window_native_evaluator.py --ppl
  python3 research/oml_acs/q_v2_8window_native_evaluator.py --bench
  python3 research/oml_acs/q_v2_8window_native_evaluator.py --dump

Each phase refuses to overwrite a prior per-condition record. The native
C binary is original BEGLIN source-derived, research-only; output head same
FP32 tied embedding all modes; on-disk BF16 original unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, pathlib, re, struct, subprocess, time
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
PRELOCK=ROOT/"configs/research/BEGLIN_Q_ACTIVATION_ASYNC_PARALLEL_V2_PRELOCK_20261022.json"
V2_MILP=ROOT/"docs/research/BEGLIN_Q_V2_SOURCE_ONLY_ACTIVATION_ASYNC_MILP_POLICY_20261022.json"
SELECTED=ROOT/"configs/research/BEGLIN_Q_V2_SELECTED_28_N8_ROLE_LAYER_POLICY_20261022.txt"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
OLD_F32=ROOT/"lab_local/q_full_model_20261021_v2"
OLD_MIX=ROOT/"lab_local/qng64_n8_repacked_full_20261022"
OLD_KERNEL=ROOT/"lab_local/q_kernel_aware_native_20261022"
LAB=ROOT/"lab_local/q_v2_8public_evaluation_20261022"
EXPECTED_CONFIG_SHA="0d9177b152286ccd413028e74838529e2a5aad57b42ec793df3095f99208d4af"
EXPECTED_V2_POLICY_SHA="9324c5ec891d638f2713d3bbd4187399e1aae8a9a74df129dfabc4e8ea870212"
EXPECTED_V1_RANDOM_SHA="72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5"
EXPECTED_V1_KERNEL_SHA="148ae7dd773b11e2834339397657248f014f16718920f384394aeb724739f0d1"
EXPECTED_V1_KERNEL_BINARY_SHA="172613dc69f79f38c8553ff7087e653ec9c0031ca80aec8efd2f76f948599f14"
EXPECTED_ORIG_MIX_BINARY_SHA="8c16013a343b924b8117f66edcbfd7fb11a86f991127b9cd97fb6ec5552b6664"
N_WINDOWS=8
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
CONDITIONS={
 "original_BF16_widened_F32":(OLD_F32/"beglin_q_lab","f32",None),
 "all_Q4G64":(OLD_F32/"beglin_q_lab","q4",None),
 "frozen_seed73_random":(OLD_MIX/"beglin_mixed_repacked","mixed",OLD_MIX/"policy_random.txt"),
 "former_weight_kernel_aware":(OLD_KERNEL/"beglin_kernel_aware_native_Qwen","mixed",OLD_KERNEL/"chosen_n8_policy.txt"),
 "v2_activation_async_MILP":(OLD_KERNEL/"beglin_kernel_aware_native_Qwen","mixed",SELECTED),
}

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private(path:pathlib.Path,data:bytes):
    if path.exists():raise FileExistsError("refuse to overwrite actual native eval file "+str(path))
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def validated_sources():
    if sha(PRELOCK.read_bytes())!=EXPECTED_CONFIG_SHA:raise RuntimeError("prospective eight evaluation texts changed")
    cfg=json.loads(PRELOCK.read_text())
    texts=cfg["evaluation_public_only"]
    if len(texts)!=N_WINDOWS or len(set(texts))!=N_WINDOWS:
        raise RuntimeError("prospective evaluation prompts must be eight distinct public strings")
    if not V2_MILP.is_file():raise RuntimeError("source-only MILP GitHub freeze evidence missing")
    lock=json.loads(V2_MILP.read_text())
    if lock["source_policy_tensor_mask_sha256"]!=EXPECTED_V2_POLICY_SHA or lock["status"]!="OPTIMAL_HIGHS_SOURCE_ONLY_BEFORE_TEST":
        raise RuntimeError("MILP must have been frozen BEFORE the evaluation")
    for path,dig in ((SELECTED,EXPECTED_V2_POLICY_SHA),(OLD_MIX/"policy_random.txt",EXPECTED_V1_RANDOM_SHA),(OLD_KERNEL/"chosen_n8_policy.txt",EXPECTED_V1_KERNEL_SHA)):
        if not path.is_file() or sha(path.read_bytes())!=dig:
            raise RuntimeError("32-bit tensor policy identity missing/drift: "+str(path))
    rows=[line.split() for line in SELECTED.read_text().splitlines()]
    if len(rows)!=28 or len({tuple(x) for x in rows})!=28:
        raise RuntimeError("not exactly 28 v2 selected n8 tensors")
    for name,_,mask in CONDITIONS.values():
        if not name.is_file():raise RuntimeError("native BEGLIN binary missing "+str(name))
    if sha((OLD_KERNEL/"beglin_kernel_aware_native_Qwen").read_bytes())!=EXPECTED_V1_KERNEL_BINARY_SHA:
        raise RuntimeError("variable-n8 original BEGLIN C experiment binary drift")
    if sha((OLD_MIX/"beglin_mixed_repacked").read_bytes())!=EXPECTED_ORIG_MIX_BINARY_SHA:
        raise RuntimeError("old source-only random BEGLIN mixed binary drift")
    if not (SNAP/"model.safetensors").is_file() or not (SNAP/"config.json").is_file() or not (SNAP/"tokenizer.json").is_file():
        raise RuntimeError("original BF16 source and tokenizer not in read-only snapshot")
    return cfg,texts,lock

def prepare():
    cfg,texts,lock=validated_sources()
    if LAB.exists():raise FileExistsError("prospective native Q v2 eval already prepared")
    os.umask(0o077);LAB.mkdir(parents=True,mode=0o700)
    tok=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    allids=[]
    for text in texts:
        chat="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
        ids=tok.encode(chat,add_special_tokens=False).ids
        if not 10<=len(ids)<90 or any(not (0<=x<151936) for x in ids):
            raise RuntimeError("bad pinned Qwen eval ID/window length")
        allids.append(ids)
    common=[sha(struct.pack("<"+"i"*len(ids),*ids)) for ids in allids]
    for name in CONDITIONS:
        where=LAB/name
        where.mkdir(mode=0o700)
        for folder in ("results","eval","eval/ppl_work","logs"):
            (where/folder).mkdir(exist_ok=True,mode=0o700)
        for i,ids in enumerate(allids):
            data=struct.pack("<"+"i"*len(ids),*ids)
            private(where/f"eval/ppl_work/w{i}.i32",data)
        data=struct.pack("<"+"i"*len(allids[0]),*allids[0])
        private(where/"prompt_ids.i32",data)
    receipt=dict(schema="BEGLIN-Qv2-eight-public-prospective-eval-input-lock/1",
        status="INPUTS_SHA_FROZEN_NO_MODEL_OUTPUT_READ",
        original_BF16_rev=SNAP.name,
        model_name="Qwen2.5-1.5B-Instruct",
        prelock_file_sha256=EXPECTED_CONFIG_SHA,
        source_only_policy_sha256=EXPECTED_V2_POLICY_SHA,
        original_optimizer_lock_status=lock["status"],
        windows=N_WINDOWS,token_lengths=[len(x) for x in allids],
        evaluation_public_text_sha256=sha(json.dumps(texts,sort_keys=True).encode()),
        window_ids_sha256=common,
        all_five_conditions_have_byte_identical_inputs=True,
        original_V6_used_holdout_not_opened=True,
        main_and_production_unchanged=True,
        native_parallelism_not_claimed=True,
        note="BEGLIN C ppl over all chat-template token positions, not task-specific assistant-only NLL")
    private(LAB/"input_lock_receipt.json",(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
    print("BEGLIN_QV2_NATIVE_EVAL_PREPARED",json.dumps(receipt,sort_keys=True),flush=True)

def common_env(name):
    if not (LAB/"input_lock_receipt.json").is_file():raise RuntimeError("evaluation inputs not prepared")
    if name not in CONDITIONS:raise ValueError("unknown candidate")
    cfg,texts,lock=validated_sources()
    binfile,precision,mask=CONDITIONS[name]
    folder=(LAB/name).resolve()
    copied=[]
    for i in range(N_WINDOWS):
        p=folder/f"eval/ppl_work/w{i}.i32"
        if not p.is_file() or (p.stat().st_mode&0o777)!=0o600:raise RuntimeError("missing private eval")
        copied.append(sha(p.read_bytes()))
    manifest=json.loads((LAB/"input_lock_receipt.json").read_text())
    if copied!=manifest["window_ids_sha256"]:raise RuntimeError("prospective evaluation order/file bytes changed")
    env=os.environ.copy()
    for k in list(env):
        if k.startswith("QWEN_") or k.startswith("BEGLIN_Q_LAB_"):
            env.pop(k)
    env.update({
       "VECLIB_MAXIMUM_THREADS":"1","QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
       "QWEN_HF_CONFIG":str(SNAP/"config.json"),
       "QWEN_SAFETENSORS_MAXSEQ":"256",
       "QWEN_BASE":str(folder),"QWEN_PROMPT":str(folder/"prompt_ids.i32"),
       "BEGLIN_Q_LAB_PRECISION":precision,
    })
    if mask is not None:env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(mask.resolve())
    return [str(binfile.resolve())],env,folder

def run_phase(phase):
    if phase not in ("ppl","bench","dump"):raise ValueError("unsupported measurement")
    os.umask(0o077)
    outputs={}
    for name in CONDITIONS:
        argv,env,where=common_env(name)
        target=where/f"logs/{phase}.txt"
        if target.exists():raise FileExistsError("native measurement already exists; don't rerun "+str(target))
        args=argv+([phase,str(N_WINDOWS)] if phase=="ppl" else ([phase,"1"] if phase=="bench" else [phase]))
        if phase=="ppl":args=["/usr/bin/time","-l"]+args
        begin=time.monotonic()
        p=subprocess.run(args,cwd=str(ROOT),env=env,text=True,capture_output=True,timeout=600)
        log=p.stdout+"\n"+p.stderr
        private(target,log.encode())
        if p.returncode!=0:raise RuntimeError("native BEGLIN "+name+" "+phase+" failed (log private): "+str(p.returncode))
        if "projection_role_tensors:" not in log:
            raise RuntimeError("BEGLIN not using SHA-verified real source model")
        policy=(28 if env["BEGLIN_Q_LAB_PRECISION"]=="mixed" else (196 if env["BEGLIN_Q_LAB_PRECISION"]=="n8" else 0))
        if policy==28 and "n8=28 q4=168" not in log:raise RuntimeError("unexpected precision role execution")
        receipt=dict(model=name,phase=phase,exit=0,
                     native_binary_sha256=sha(pathlib.Path(argv[0]).read_bytes()),
                     log_sha256=sha(log.encode()),
                     elapsed_seconds=round(time.monotonic()-begin,2),
                     source_revision=SNAP.name,source_only_policy_sha256=sha(pathlib.Path(env["BEGLIN_Q_LAB_N8_ROLES_FILE"]).read_bytes()) if "BEGLIN_Q_LAB_N8_ROLES_FILE" in env else None)
        if phase=="ppl":
            match=re.search(r"ppl\s+([0-9.]+)\s+tokens\s+(\d+)",log)
            rss=re.search(r"(\d+)\s+maximum resident set size",log)
            if not match or not rss:
                raise RuntimeError("missing real native perplexity / max RSS")
            receipt.update(perplexity=float(match.group(1)),nll_nats_per_token=math.log(float(match.group(1))),
                           scored_tokens=int(match.group(2)),max_rss_bytes=int(rss.group(1)))
            expected_tokens=sum(n-1 for n in json.loads((LAB/"input_lock_receipt.json").read_text())["token_lengths"])
            if receipt["scored_tokens"]!=expected_tokens:raise RuntimeError("wrong held eval token count")
        if phase=="bench":
            match=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",log)
            if not match:raise RuntimeError("no actual BEGLIN native decode timing")
            receipt.update(single_decode_best_of_three_ms=float(match.group(2)),
                           native_report_tps=float(match.group(1)))
        if phase=="dump":
            logits=where/"results/c_logits_last.f32"
            if not logits.is_file() or logits.stat().st_size!=151936*4:
                raise RuntimeError("model full-vocab final logits absent")
            if (logits.stat().st_mode&0o777)!=0o600:
                raise RuntimeError("model outputs not mode0600; refuse publish")
            receipt["full_vocab_logits_sha256"]=sha(logits.read_bytes())
        private(where/f"logs/{phase}_receipt.json",(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
        outputs[name]=receipt
        print("BEGLIN_QV2_NATIVE_EVAL",phase,json.dumps(receipt,sort_keys=True),flush=True)
    final=LAB/f"all_{phase}_public_eval_receipt.json"
    private(final,(json.dumps(dict(schema="BEGLIN-Qv2-native-prospective-eval/1",
             stage=phase,status="PASS_ALL_FIVE_PUBLIC_EVALUATIONS",
             locked_public_eval_inputs=sha((LAB/"input_lock_receipt.json").read_bytes()),
             models=outputs,original_main_or_release_modified=False,
             full_model_async_execution_not_implemented=True),sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_QV2_EVAL_FINISHED",phase,sha(final.read_bytes()),flush=True)

def main():
    parser=argparse.ArgumentParser()
    one=parser.add_mutually_exclusive_group(required=True)
    for flag in ("prepare","ppl","bench","dump"):
        one.add_argument("--"+flag,action="store_true")
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.ppl:run_phase("ppl")
    elif args.bench:run_phase("bench")
    else:run_phase("dump")
if __name__=="__main__":main()
