#!/usr/bin/env python3
"""Prepare sealed-after-source-policy 8 public prompt native BEGLIN Qv2 test.

All 5 model variants see exact same token IDs from pinned Qwen tokenizer.
Does not run any model, or inspect any quality outputs, and requires the
source-only 3584-choice MILP manifest/28-bit policy to have been committed.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,struct
from tokenizers import Tokenizer

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/q_v2_native_8public_eval_20261022"
PRELOCK=ROOT/"configs/research/BEGLIN_QV2_ASYNC_DAG_MILP_PRELOCK_20261022.json"
MILP=ROOT/"docs/research/BEGLIN_QV2_ASYNC_DAG_MILP_SOURCE_POLICIES_20261022.json"
N8_ASYNC=ROOT/"configs/research/BEGLIN_QV2_ASYNC_DAG_N8_POLICY_20261022.txt"
N8_SERIAL=ROOT/"configs/research/BEGLIN_QV2_SERIAL_N8_CONTROL_20261022.txt"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
OLD_ROOT=ROOT/"lab_local"
OLD_RANDOM=OLD_ROOT/"qng64_n8_repacked_full_20261022/policy_random.txt"
BIN_BASE=OLD_ROOT/"q_full_model_20261021_v2/beglin_q_lab"
BIN_RANDOM=OLD_ROOT/"qng64_n8_repacked_full_20261022/beglin_mixed_repacked"
BIN_NEW=OLD_ROOT/"q_kernel_aware_native_20261022/beglin_kernel_aware_native_Qwen"
ROLE_ORDER=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
EXPECTED_ASYNC="e044675ea5b47a5aca0ca75980a55070b932d1a4dc330815bbb097acbf75df7b"
EXPECTED_SERIAL="06f07fcfab7fdba22528c6c80e09582f18cfa77d0760a26e4ec35bc6a80dd72c"
EXPECTED_OLD="72065ac7a4ae5d68b1c7944dc4a67e40ac9c3c825db54a1d441ca3002ec015c5"
MODES=(
 ("f32","f32",BIN_BASE,None),
 ("q4","q4",BIN_BASE,None),
 ("old_random","mixed",BIN_RANDOM,OLD_RANDOM),
 ("async_DAG","mixed",BIN_NEW,N8_ASYNC),
 ("serial_only","mixed",BIN_NEW,N8_SERIAL),
)
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private_file(path:pathlib.Path,buf:bytes):
    if path.exists():raise FileExistsError("will not overwrite original source or prior run")
    with path.open("xb") as f:
        f.write(buf);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def prepare():
    if LAB.exists():raise FileExistsError("private fixed evaluation already prepared")
    if not (SNAP/"model.safetensors").is_file() or not (SNAP/"tokenizer.json").is_file():
        raise RuntimeError("revision-pinned BF16 Qwen source absent")
    p=json.loads(PRELOCK.read_text())
    s=json.loads(MILP.read_text())
    if p["status"]!="FROZEN_BEFORE_POLICY_AND_HELD_EVAL" or s["status"]!="OPTIMAL_SOURCE_ONLY_POLICIES_FROZEN_BEFORE_EVAL":
        raise RuntimeError("preregistered source-only freeze not authoritative")
    if s["actual_original_BF16_source_revision"]!=p["revision"] or p["revision"]!="989aa7980e4cf806f80c7fef2b1adb7bc71aa306":
        raise RuntimeError("native evaluation mismatched model revision")
    for path,expect,tag in [(N8_ASYNC,EXPECTED_ASYNC,"async_DAG"),
                             (N8_SERIAL,EXPECTED_SERIAL,"serial_only"),
                             (OLD_RANDOM,EXPECTED_OLD,None)]:
        text=path.read_text()
        if sha(text.encode())!=expect:raise RuntimeError("pre-eval precision policy changed "+str(path))
        pairs=[tuple(line.split()) for line in text.splitlines()]
        if len(pairs)!=28 or len(set(pairs))!=28 or any(r not in ROLE_ORDER or not l.isdigit() or int(l)>27 for r,l in pairs):
            raise RuntimeError("wrong 28-layer n8 policy")
        if tag and expect!=next(x["frozen_policy_sha256"] for x in s["policies"] if x["name"]==tag):
            raise RuntimeError("stored optimizer mask was changed")
    texts=p["eval_public_prompts"]
    assert len(texts)==8 and len(set(texts))==8
    tokenizer=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    windows=[]
    for text in texts:
        formatted="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
        ids=tokenizer.encode(formatted,add_special_tokens=False).ids
        if not 8<=len(ids)<=120 or max(ids)>=151936 or min(ids)<0:
            raise RuntimeError("invalid public Qwen wordpiece window")
        windows.append(tuple(ids))
    os.umask(0o077);LAB.mkdir(parents=True,mode=0o700)
    manifest_modes={}
    for name,mode,binary,policy in MODES:
        if not binary.is_file():raise RuntimeError("verified original native BEGLIN executable missing "+str(binary))
        work=LAB/name
        work.mkdir(mode=0o700)
        for sub in ("results","eval","eval/ppl_work"):
            (work/sub).mkdir(exist_ok=True,mode=0o700)
        for i,ids in enumerate(windows):
            rel=work/f"eval/ppl_work/w{i}.i32"
            private_file(rel,struct.pack("<"+"i"*len(ids),*ids))
        private_file(work/"prompt_ids.i32",struct.pack("<"+"i"*len(windows[0]),*windows[0]))
        manifest_modes[name]={
            "mode":mode,
            "binary":str(binary.resolve()),
            "binary_sha256":sha(binary.read_bytes()),
            "precision_policy_file":str(policy.resolve()) if policy else None,
            "precision_policy_SHA256":sha(policy.read_bytes()) if policy else None,
            "public_eval_window_bytes_sha256":[sha((work/f"eval/ppl_work/w{i}.i32").read_bytes()) for i in range(len(windows))]
        }
    manifest=dict(schema="BEGLIN-Qv2-8-fixed-public-native-eval-fixture/1",
       status="PREPARED_AFTER_SOURCE_POLICY_FROZEN_BEFORE_NATIVE_EVAL",
       prereg_sha256=sha(PRELOCK.read_bytes()),
       source_optimizer_manifest_sha256=sha(MILP.read_bytes()),
       source_eval_texts_sha256=sha(json.dumps(texts,sort_keys=True).encode()),
       source_original_Qwen="Qwen/Qwen2.5-1.5B-Instruct",
       original_BF16_model_snapshot=SNAP.name,
       tokenizer_sha256=sha((SNAP/"tokenizer.json").read_bytes()),
       eval_public_prompt_count=len(windows),
       token_counts=[len(x) for x in windows],
       input_window_shas=manifest_modes["q4"]["public_eval_window_bytes_sha256"],
       independent_input_files_match_for_all_modes=True,
       policy_sources_verified_before_eval=True,
       model_workloads=manifest_modes,
       production_main_release_unchanged=True,
       no_previous_V6_24_used_test_access=True,
       quality_selection_never_saw_eval_prompts=True)
    private_file(LAB/"public_native_eval_fixture_manifest.json",
                 (json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode())
    print("QV2_FIXED_PUBLIC_NATIVE_EVAL_PREPARED",json.dumps({
       "source_eval_sha256":manifest["source_eval_texts_sha256"],
       "manifest_sha256":sha((LAB/"public_native_eval_fixture_manifest.json").read_bytes()),
       "mode_count":len(MODES),"eval_token_counts":manifest["token_counts"],
       "same_across_all_modes":True},sort_keys=True),flush=True)
if __name__=="__main__":prepare()
