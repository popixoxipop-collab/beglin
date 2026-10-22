#!/usr/bin/env python3
"""Native BEGLIN CPU evaluation of source-frozen strict async-stage Qv2 policy.

The bit mask was globally optimized with 448 stage-choice HiGHS binaries
BEFORE reading the same prelocked 8 synthetic evaluation windows. This
driver reuses a SHA-pinned existing research-only 28-layer Qwen C executable
that can accept any 28n8/168Q4 role-layer policy. Important: executable
runs SEQUENTIALLY; its async DAG constraint was an offline cost surrogate.
Native prospective development output is never used to retune the policy.
"""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,re,shutil,subprocess,time

ROOT=pathlib.Path(__file__).resolve().parents[2]
FROZEN=ROOT/"configs/research/BEGLIN_Q_V2_STRICT_50MS_SOURCE_POLICY_FROZEN_20261022.json"
PRELOCK=ROOT/"configs/research/BEGLIN_Q_V2_ASYNC_DAG_MILP_PRELOCK_20261022.json"
SOURCE_INPUT=ROOT/"lab_local/q_v2_8public_evaluation_20261022"
NATIVE=ROOT/"lab_local/q_kernel_aware_native_20261022/beglin_kernel_aware_native_Qwen"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
LAB=ROOT/"lab_local/q_v2_strict_50ms_native_20261022"
ROLES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
EXPECTED_BINARY_SHA="172613dc69f79f38c8553ff7087e653ec9c0031ca80aec8efd2f76f948599f14"
EXPECTED_FROZEN_PRIVATE_SHA="7d19591c22846f8d001256f4c4addacfa7babeebf49eae9b3783128e9c464ea4"
EXPECTED_POLICY_SHA="cb9268c9177d47705e548c2a90a6b22772569b3f2bb54dfef2767ca53ffa3e76"
EXPECTED_INPUT_LOCK_SHA="352d2f1b2161ce47233dd25ffe377971b54f321ef3e6ea833c379619d4daa23b"

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def write_private(path:pathlib.Path,data:bytes):
    with path.open("xb") as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(0o600)
def checked():
    if not FROZEN.is_file() or not PRELOCK.is_file():raise RuntimeError("original source-only optimizer prelock missing")
    j=json.loads(FROZEN.read_text())
    if j["private_receipt_sha256"]!=EXPECTED_FROZEN_PRIVATE_SHA or j["policy_text_sha256"]!=EXPECTED_POLICY_SHA:
        raise RuntimeError("source-only strict 50ms policy SHA drift")
    if j["new_output_or_evaluation_used"] or not j["no_fullmodel_parallel_speed_claim"]:
        raise RuntimeError("source test contamination or false whole-model async claim")
    pre=json.loads(PRELOCK.read_text())
    if pre["solver"]["constraints"]["exact_n8_tensors"]!=28 or pre["solver"]["constraints"]["native_parallel_sibling_stage_sum_28layers_ms_at_most"]!=50:
        raise RuntimeError("prelocked constraints drift")
    if not NATIVE.is_file() or sha(NATIVE.read_bytes())!=EXPECTED_BINARY_SHA:
        raise RuntimeError("native experimental Qwen binary SHA drift")
    root=SOURCE_INPUT/"input_lock_receipt.json"
    if not root.is_file() or sha(root.read_bytes())!=EXPECTED_INPUT_LOCK_SHA:
        raise RuntimeError("previously prelocked eight public prompts absent")
    inp=json.loads(root.read_text())
    if inp["windows"]!=8 or not inp["all_five_conditions_have_byte_identical_inputs"]:
        raise RuntimeError("8 public windows not originally locked")
    if not (SNAP/"model.safetensors").is_file():
        raise RuntimeError("unmodified BF16 Qwen snapshot missing")
    selected=[(str(r),int(i)) for r,i in j["selected_n8_role_layer_pairs"]]
    if len(selected)!=28 or len(set(selected))!=28 or any(r not in ROLES or not 0<=i<28 for r,i in selected):
        raise RuntimeError("not exactly 28 unique source-only n8 projections")
    sortedpair=sorted(selected,key=lambda x:(ROLES.index(x[0]),x[1]))
    text="".join(f"{r} {l}\n" for r,l in sortedpair).encode()
    if sha(text)!=EXPECTED_POLICY_SHA:
        raise RuntimeError("role sorted n8 policy bytes not original HiGHS frozen bytes")
    return j,inp,text

def prepare():
    if LAB.exists():raise FileExistsError("strict Qv2 native experiment directory already exists")
    j,inp,mask=checked()
    os.umask(0o077)
    LAB.mkdir(parents=True,mode=0o700)
    where=LAB/"chosen"
    where.mkdir(mode=0o700)
    for folder in ("results","eval","eval/ppl_work","logs"):
        (where/folder).mkdir(mode=0o700,exist_ok=True)
    write_private(LAB/"chosen_n8_roles.txt",mask)
    old=SOURCE_INPUT/"v2_activation_async_MILP"
    for name in ("prompt_ids.i32",)+tuple(f"eval/ppl_work/w{i}.i32" for i in range(8)):
        f=old/name
        if not f.is_file() or (f.stat().st_mode&0o777)!=0o600:
            raise RuntimeError("original frozen input not privately accessible")
        content=f.read_bytes()
        if name.startswith("eval/ppl_work/w"):
            idx=int(f.name[1:-4])
            if sha(content)!=inp["window_ids_sha256"][idx]:
                raise RuntimeError("public evaluation input bytes SHA drift")
        write_private(where/name,content)
    r=dict(schema="BEGLIN-Q-v2-strict50-actual-native-eval-initializer/1",
      status="PREPARED_BUT_NO_MODEL_OUTPUT_OBSERVED",
      existing_native_C_binary_sha256=EXPECTED_BINARY_SHA,
      source_only_private_policy_receipt_sha256=EXPECTED_FROZEN_PRIVATE_SHA,
      source_only_policy_mask_sha256=EXPECTED_POLICY_SHA,
      originally_frozen_public_eight_window_lock_sha256=EXPECTED_INPUT_LOCK_SHA,
      windows=8,window_lengths=inp["token_lengths"],
      preselected_n8=28,preselected_q4=168,
      fixed_projection_bytes=j["packed_projection_bytes"],
      exact_solved_stage_time_ms=j["measured_stage_async_DAG_ms"],
      note="same eight public development inputs already used by related research branches; not an independent final holdout",
      native_full_model_exec_will_be_sequential=True,
      production_main_and_immutable_release_unchanged=True)
    write_private(LAB/"source_receipt.json",(json.dumps(r,sort_keys=True,indent=2)+"\n").encode())
    print("Q2_STRICT_NATIVE_PREPARED",json.dumps(r,sort_keys=True),flush=True)

def evaluate(phase:str):
    if phase not in ("dump","bench","ppl8"):raise ValueError("unknown native Qwen phase")
    config,inp,mask=checked()
    if not LAB.exists():raise RuntimeError("run --prepare before any eval")
    if sha((LAB/"chosen_n8_roles.txt").read_bytes())!=EXPECTED_POLICY_SHA:
        raise RuntimeError("candidate n8 mask was changed")
    outfile=LAB/"chosen/logs"/f"{phase}.log"
    if outfile.exists():raise FileExistsError("preserve original phase result")
    if phase=="dump" and (LAB/"chosen/results/c_logits_last.f32").exists():
        raise FileExistsError("do not overwrite first logits")
    env=os.environ.copy()
    for key in ("QWEN_GGUF","QWEN_DENSE_QNG64_OVERRIDE","QWEN_WT_PROMOTION_FILE"):
        if key in env:raise RuntimeError("unexpected precision override "+key)
    env.update({
     "VECLIB_MAXIMUM_THREADS":"1",
     "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
     "QWEN_HF_CONFIG":str(SNAP/"config.json"),
     "QWEN_SAFETENSORS_MAXSEQ":"256",
     "QWEN_BASE":str((LAB/"chosen").resolve()),
     "QWEN_PROMPT":str((LAB/"chosen/prompt_ids.i32").resolve()),
     "BEGLIN_Q_LAB_PRECISION":"mixed",
     "BEGLIN_Q_LAB_N8_ROLES_FILE":str((LAB/"chosen_n8_roles.txt").resolve())
    })
    if phase=="bench":cmd=[str(NATIVE.resolve()),"bench","1"]
    elif phase=="ppl8":cmd=["/usr/bin/time","-l",str(NATIVE.resolve()),"ppl","8"]
    else:cmd=[str(NATIVE.resolve()),"dump"]
    os.umask(0o077)
    now=time.monotonic()
    cp=subprocess.run(cmd,env=env,cwd=str(ROOT),capture_output=True,text=True,timeout=260)
    log=(cp.stdout+"\n"+cp.stderr).encode()
    write_private(outfile,log)
    if cp.returncode!=0:raise RuntimeError("native experiment failed "+phase+": "+cp.stderr[-1400:])
    text=log.decode()
    if not re.search(r"projection_role_tensors: f32=0 n8=28 q4=168",text):
        raise RuntimeError("BEGLIN execution policy did not actually have 28 n8 +168 Q4")
    fast=re.search(r"\[BEGLIN-N8-REPACKED-LAB\] fast_gemv=(\d+)\s+fast_batch_rows=(\d+)\s+other_bit_gemv=(\d+)\s+other_bit_batches=(\d+)",text)
    if not fast or int(fast.group(1))<=0 or any(int(x)>0 for x in fast.groups()[1:]):
        raise RuntimeError("native 8-bit repack kernel missing / fallback unexpectedly active")
    result=dict(phase=phase,status="PASS_SEQUENTIAL_REAL_NATIVE_MODEL",
        expected_serial_process=True,
        source_only_policy_sha256=EXPECTED_POLICY_SHA,
        n8_repacked_native_GEMV_calls=int(fast.group(1)),
        native_log_SHA256=sha(log),
        elapsed_wall_seconds=round(time.monotonic()-now,3),
        permanent_source_PPL_original_not_modified=True)
    if phase=="bench":
        m=re.search(r"tok_s\s+([\d.]+)\s+ms_tok\s+([\d.]+)",text)
        if not m:raise RuntimeError("native bench result absent")
        result.update(best_of_three_one_token_ms=float(m.group(2)),
            native_tok_per_s_reported=float(m.group(1)))
    if phase=="dump":
        f=LAB/"chosen/results/c_logits_last.f32"
        if not f.is_file() or f.stat().st_size!=151936*4:
            raise RuntimeError("full native Qwen vocab logits missing")
        f.chmod(0o600)
        result.update(first_full_vocab_logits_SHA256=sha(f.read_bytes()),logits_vocab=151936)
    if phase=="ppl8":
        m=re.search(r"\[ppl\][^\n]*ppl\s+([\d.]+)\s+over\s+(\d+)\s+tokens",text)
        rss=re.search(r"(\d+)\s+maximum resident set size",text)
        if not m or not rss or int(m.group(2))!=131:
            raise RuntimeError("native PPL is not same eight public 131 predicted tokens")
        result.update(public_development_PPL=float(m.group(1)),public_development_token_positions=131,
            maximum_resident_bytes=int(rss.group(1)))
    write_private(LAB/f"{phase}_receipt.json",(json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
    print("Q2_STRICT_NATIVE_PHASE_COMPLETED",json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    grp=cli.add_mutually_exclusive_group(required=True)
    grp.add_argument("--prepare",action="store_true")
    grp.add_argument("--phase",choices=("dump","bench","ppl8"))
    args=cli.parse_args()
    prepare() if args.prepare else evaluate(args.phase)
