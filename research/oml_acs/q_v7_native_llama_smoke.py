#!/usr/bin/env python3
"""Read-only pinned Llama-family model native BEGLIN Q4 smoke.

Uses the exact original Qv5 Apple Silicon C research Q4 backend under pinned
non-gated TinyLlama weights. The Qv7 source+eval cohorts must be frozen first.
Only creates a private scratch directory; production/main remain untouched.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,re,struct,subprocess,time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
P=ROOT/"configs/research/BEGLIN_QV7_CROSSFAMILY_CALIBRATION_PRELOCK_20261022.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV7_CROSSFAMILY_TINYLLAMA_INPUT_FREEZE_20261022.json"
TEST=ROOT/"lab_local/qv7_tinyllama_20261022/eval_openbookqa_test48_tinyllama_frozen.json"
MODEL=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv7-tinyllama-model-20261022")
BIN=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022/lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base")
BIN_SHA="408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"
OUT=ROOT/"lab_local/qv7_tinyllama_20261022/native_full_llama_q4_smoke"
PRE_SHA="1b5fa904cf11f25292720efc75f658fc10cedc8923236b4d5b0ae7e70d3e88c6"
FREEZE_SHA="21725c2c412d5fbc3d50619ee03d99e422945ecb582c0eb0a2cdb9ed2355cd39"
TEST_SHA="8807002e79afadcb2d150b32dffdefa204afb0d70678375a608b7900f18d0ab0"

def sha(b):return hashlib.sha256(b).hexdigest()
def safe(p):
    if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o600:
        raise RuntimeError("Qv7 sealed eval 48 unavailable")
    return p.read_bytes()
def write(p,b):
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)

def main():
    if OUT.exists():raise FileExistsError("native smoke already attempted; preserve raw output")
    if sha(P.read_bytes())!=PRE_SHA or sha(PUBLIC.read_bytes())!=FREEZE_SHA or sha(safe(TEST))!=TEST_SHA:
        raise RuntimeError("model/dataset source identity changed")
    if not MODEL.joinpath("model.safetensors").is_file() or sha(BIN.read_bytes())!=BIN_SHA:
        raise RuntimeError("TinyLlama source weights missing or original audited C binary drift")
    test=json.loads(safe(TEST))["test48"]
    first=test[0];tokens=first["prefix_ids"]
    if not 5<len(tokens)<256:raise RuntimeError("TinyLlama original evaluation prompt shape unexpected")
    os.umask(0o077)
    OUT.mkdir(parents=True,mode=0o700)
    (OUT/"results").mkdir(mode=0o700)
    (OUT/"eval/ppl_work").mkdir(parents=True,mode=0o700)
    write(OUT/"prompt_ids.i32",struct.pack("<"+"i"*len(tokens),*tokens))
    env={k:v for k,v in os.environ.items() if not k.startswith(("QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV4_","BEGLIN_QV5_","BEGLIN_QV6_"))}
    env.update({
      "VECLIB_MAXIMUM_THREADS":"1","QWEN_SAFETENSORS":str(MODEL/"model.safetensors"),
      "QWEN_HF_CONFIG":str(MODEL/"config.json"),"QWEN_SAFETENSORS_MAXSEQ":"256",
      "QWEN_BASE":str(OUT.resolve()),"QWEN_PROMPT":str((OUT/"prompt_ids.i32").resolve()),
      "BEGLIN_Q_LAB_PRECISION":"q4",
      "BEGLIN_QV2_NATIVE_ASYNC":"0"})
    t=time.monotonic()
    done=subprocess.run([str(BIN),"dump"],env=env,cwd=str(ROOT),
       capture_output=True,text=True,timeout=450)
    log=(done.stdout+"\n"+done.stderr).encode()
    write(OUT/"native_dump_log.txt",log)
    print("QV7_LLAMA_NATIVE_SMOKE_PROCESS_EXIT",done.returncode,"log_sha",sha(log),flush=True)
    if done.returncode:
        raise RuntimeError("existing BEGLIN CPU backend cannot load pinned Llama family, isolate fix required: "+done.stderr[-1600:])
    proj=re.search(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)",log.decode())
    if proj is None or tuple(map(int,proj.groups()))!=(0,0,154):
        raise RuntimeError("real TinyLlama 22layers x7 not represented")
    raw=OUT/"results/c_logits_last.f32"
    if not raw.is_file() or raw.stat().st_size!=32000*4:
        raise RuntimeError("original TinyLlama vocab logits missing")
    logits=np.frombuffer(raw.read_bytes(),dtype="<f4")
    if not np.isfinite(logits).all():
        raise RuntimeError("TinyLlama C NaN logits")
    receipt={
      "schema":"BEGLIN-Qv7-actual-C-native-22-layer-Llama-safetensors-Q4-smoke/1",
      "status":"PASS_154_REAL_Q4G64_PROJECTION_ROLES_FIRST_FULLVOCAB_LOGITS",
      "architecture":"llama","model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "model_revision":"fe8a4ea1ffedaf415f4da2f062534de366a451e6",
      "native_executable_sha256":BIN_SHA,"native_log_sha256":sha(log),
      "native_Q4_projection_count":154,"native_N8_projection_count":0,
      "first_prompt_sha256":sha((OUT/"prompt_ids.i32").read_bytes()),
      "model_vocab_size":32000,
      "first_native_fullvocab_logits_sha256":sha(raw.read_bytes()),
      "predicted_token_id":int(np.argmax(logits)),
      "elapsed_wall_s":round(time.monotonic()-t,2),
      "production_changed":False,"model_weight_mutated":False}
    b=(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode()
    write(OUT/"real_llama_q4_smoke_receipt.json",b)
    print("QV7_TINYLLAMA_DIFFERENT_FAMILY_BEG_NATIVE_Q4_PASS",json.dumps({
       "sha":sha(b),"count":154,"top1":receipt["predicted_token_id"],
       "time":receipt["elapsed_wall_s"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
