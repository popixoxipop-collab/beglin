#!/usr/bin/env python3
"""4-arm BEGLIN 28-layer real native CPU HellaSwag (48) answer-token test.

One source-only frozen Qv8 global signed-gradient mask versus all-Q4 and prior
Qv6 Fisher64/128. All arms use the same native Qwen original checkpoint,
identical frozen single-letter A/B/C/D correct-token inputs and original
immutable C research binaries. Only private scratch files; no runtime writes.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,re,struct,subprocess,time
from scipy.stats import binomtest
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
OLDER=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv5-fisher-20261022")
BASE=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-q-apple-20261021")
LAB=ROOT/"lab_local/qv8_joint_signed_20261022"
HELD=LAB/"pinned_HellaSwag_val48.json"
PRE=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_SOURCE_PRELOCK_20261022.json"
POL=ROOT/"docs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_OPTIMIZER_20261022.json"
NEW=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
RUN=LAB/"native_hellaswag48_four_modes"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRE_SHA="9b833f5fc2729d13320c87e368baaff952c890ed068586622badfc7da63b4db1"
HELD_SHA="45c95170a2eb01b4c79ad44719ab9a1877e568f2583a90d84340d7448e0d1a33"
MODES=("q4","qv6_diag64","qv6_diag128","qv8_signed96")
BINS={
 "base":(OLDER/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_base",
  "408e74ae8f268b350ec2fc39c4843b048983e23a84802dbd9c42d067a79fcf7c"),
 "mixed":(OLDER/"lab_local/qv5_fisher_20261022/native_40val_6mode/bin/beglin_qv5_mixed",
  "f64bae90250d83c051b55fc4907837ab1482a991dc27f35aee637860e40c1221")}
MASKS={
 "qv6_diag64":(ROOT/"configs/research/BEGLIN_QV6_FISHER_64N8_SOURCE_POLICY_20261022.txt",
  "74fb949e45d13f63a26d412effc39377492ea71bda20ae3690f56fddf6303e08"),
 "qv6_diag128":(ROOT/"configs/research/BEGLIN_QV6_FISHER_128N8_SOURCE_POLICY_20261022.txt",
  "8b9e2404f9a38d7b57784b7d663efa19661553ccaa7dc79bfcbf6e6180fc7997"),
 "qv8_signed96":(NEW,None)}
REGEX_ROW=re.compile(r"^\[qv5-choice\] window (\d+) NLL ([0-9eE.+-]+) expected (\d+) predicted (\d+) prefix (\d+)$",re.M)
REGEX_SUM=re.compile(r"^qv5_choice NLL ([0-9eE.+-]+) items (\d+) correct (\d+) mean ([0-9eE.+-]+) ppl ([0-9eE.+-]+)$",re.M)
REGEX_ROLE=re.compile(r"projection_role_tensors: f32=(\d+) n8=(\d+) q4=(\d+)")
def sha(b):return hashlib.sha256(b).hexdigest()
def sealed(p):
    if not p.is_file() or p.is_symlink() or p.stat().st_mode&0o777!=0o600:
        raise RuntimeError("original held or evaluation sealed file missing")
    return p.read_bytes()
def put(p,b):
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def verified():
    if sha(PRE.read_bytes())!=PRE_SHA or sha(sealed(HELD))!=HELD_SHA:
        raise RuntimeError("Qv8 dataset or source prelock changed")
    if not POL.is_file() or not NEW.is_file():
        raise RuntimeError("source-only objective not frozen before held eval")
    p=json.loads(POL.read_text())
    if p["status"]!="SOURCE_ONLY_28N8_STRICT_FEASIBLE_LOCAL_SWAP_DONE_NO_EVAL_OPENED" or \
       p["mask_sha256"]!=sha(NEW.read_bytes()) or p["no_held_data_labels_or_logits_loaded"] is not True:
        raise RuntimeError("joint signed model source policy refitted after held inputs")
    h=json.loads(sealed(HELD))
    if h["held_count"]!=48 or len(h["held_validation48"])!=48:
        raise RuntimeError("new domain held evaluation not 48 exact locked questions")
    for mode,(file,digest) in BINS.items():
        if not file.is_file() or sha(file.read_bytes())!=digest:
            raise RuntimeError("original audited Qwen native C executable changed "+mode)
    for name,(file,digest) in MASKS.items():
        expected=digest if digest else p["mask_sha256"]
        if not file.is_file() or sha(file.read_bytes())!=expected:
            raise RuntimeError("baseline/new mixed-mask original SHA drift "+name)
        ids=[tuple(row.split()) for row in file.read_text().splitlines()]
        if len(ids)!=28 or len(set(ids))!=28:raise RuntimeError("not source-only unique 28n8 "+name)
    return h,p
def prepare():
    held,policy=verified()
    if RUN.exists():raise FileExistsError("Qv8 evaluation data already prepared; no reselect")
    os.umask(0o077);RUN.mkdir(parents=True,mode=0o700)
    manifest=[]
    for i,row in enumerate(held["held_validation48"]):
        ids=row["prefix_ids"]; correct=row["correct"];choices=row["suffix_ids"]
        if not 18<=len(ids)<=208 or set(choices)!={"A","B","C","D"} or correct not in choices:
            raise RuntimeError("Qwen held narrative token boundary invalid")
        data=struct.pack("<"+"i"*len(ids),*ids)
        suffix=(" ".join(str(choices[letter]) for letter in "ABCD")+" "+str("ABCD".index(correct))+"\n").encode()
        manifest.append({"item_id_SHA":sha(row["id"].encode()),
          "prompt_i32_SHA":sha(data),"choices_SHA":sha(suffix),
          "prefix_length":len(ids)})
        for mode in MODES:
            dest=RUN/mode
            (dest/"eval/ppl_work").mkdir(parents=True,mode=0o700,exist_ok=True)
            (dest/"results").mkdir(mode=0o700,exist_ok=True)
            put(dest/f"eval/ppl_work/w{i}.i32",data)
            put(dest/f"eval/ppl_work/w{i}.choices",suffix)
            if i==0:put(dest/"prompt_ids.i32",data)
    record={
      "schema":"BEGLIN-Qv8-native-Qwen28-HellaSwag48-four-frozen-policies/1",
      "status":"FROZEN_SOURCE_ONLY_MASKS_AND_NEW_HELD_INPUT_BYTES_PRE_NATIVE",
      "prelock_SHA":PRE_SHA,"held_file_SHA":HELD_SHA,
      "joint_source_policy_document_sha256":sha(POL.read_bytes()),
      "joint_selected_mask_sha256":policy["mask_sha256"],
      "all_candidate_baseline_masks_SHA":{key:(dig or policy["mask_sha256"]) for key,(p,dig) in MASKS.items()},
      "same_native_q4_or_mixed_Qwen_binary_SHA":{key:digest for key,(p,digest) in BINS.items()},
      "all_48_input_sha":manifest,"evaluator_modes":list(MODES),
      "original_model_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "not_trained_on_held_NLL":True,"main_or_operational_release_modified":False}
    raw=(json.dumps(record,sort_keys=True,indent=2)+"\n").encode()
    put(RUN/"locked_model_48x4_manifest.json",raw)
    print("QV8_NATIVE_NEW_NARRATIVE48_ALL_FOUR_POLICY_INPUTS_FROZEN",
      json.dumps({"manifest_SHA":sha(raw),"cases":48,"modes":len(MODES),"policySHA":policy["mask_sha256"]},sort_keys=True),flush=True)

def parse(log,mode,frozen):
    expected=(0,0,196) if mode=="q4" else (0,28,168)
    tier=REGEX_ROLE.findall(log)
    if len(tier)!=1 or tuple(map(int,tier[0]))!=expected:
        raise RuntimeError("source BEGLIN CPU role actual tier different "+mode)
    cases=REGEX_ROW.findall(log)
    if len(cases)!=48:raise RuntimeError("not 48 real Qwen28 new held narrative losses")
    values=[];yes=[];guess=[]
    for i,(idx,nll,correct,pred,n) in enumerate(cases):
        value=float(nll)
        if not (int(idx)==i and math.isfinite(value) and value>=0 and
                int(n)==frozen["all_48_input_sha"][i]["prefix_length"] and
                0<=int(correct)<=3 and 0<=int(pred)<=3):
            raise RuntimeError("Qwen original native C per-question bound inconsistent")
        values.append(value);yes.append(int(correct==pred));guess.append(int(pred))
    totals=REGEX_SUM.findall(log)
    if len(totals)!=1:raise RuntimeError("native held aggregate missing")
    tot,count,correct,mean,ppl=map(float,totals[0])
    if count!=48 or int(correct)!=sum(yes) or abs(tot-sum(values))>1e-7 or \
       abs(mean-tot/48)>1e-7 or abs(math.log(ppl)-mean)>1e-7:
        raise RuntimeError("source NLL normalization or four-option accuracy mismatch")
    return {"correct":int(correct),"total":48,"accuracy":int(correct)/48,
      "mean_correct_answer_NLL":mean,"per_case_NLL":values,
      "per_case_right":yes,"per_case_prediction_indices":guess}
def evaluate(mode):
    held,policy=verified()
    if mode not in MODES:raise ValueError("unfrozen experiment mode")
    frozen=json.loads(sealed(RUN/"locked_model_48x4_manifest.json"))
    if frozen["status"]!="FROZEN_SOURCE_ONLY_MASKS_AND_NEW_HELD_INPUT_BYTES_PRE_NATIVE" or \
       frozen["joint_source_policy_document_sha256"]!=sha(POL.read_bytes()) or \
       frozen["joint_selected_mask_sha256"]!=policy["mask_sha256"]:
        raise RuntimeError("source-only Qv8 score admission failed")
    out=RUN/mode; logfile=out/"actual_native_hellaswag48_log.txt"
    if logfile.exists():raise FileExistsError("cannot rerun original held comparison")
    for i,q in enumerate(frozen["all_48_input_sha"]):
        if sha(sealed(out/f"eval/ppl_work/w{i}.i32"))!=q["prompt_i32_SHA"] or \
           sha(sealed(out/f"eval/ppl_work/w{i}.choices"))!=q["choices_SHA"]:
            raise RuntimeError("frozen HellaSwag question/label SHA altered")
    which="base" if mode=="q4" else "mixed"
    executable=BINS[which][0]
    env={k:v for k,v in os.environ.items() if not k.startswith((
        "QWEN_","BEGLIN_Q_LAB_","BEGLIN_QV2_","BEGLIN_QV4_","BEGLIN_QV5_","BEGLIN_QV6_","BEGLIN_QV7_","BEGLIN_QV8_"))}
    env.update({"VECLIB_MAXIMUM_THREADS":"1",
       "QWEN_SAFETENSORS":str(SNAP/"model.safetensors"),
       "QWEN_HF_CONFIG":str(SNAP/"config.json"),
       "QWEN_SAFETENSORS_MAXSEQ":"256",
       "QWEN_BASE":str(out.resolve()),"QWEN_PROMPT":str((out/"prompt_ids.i32").resolve()),
       "BEGLIN_Q_LAB_PRECISION":"q4" if mode=="q4" else "mixed",
       "BEGLIN_QV5_FROZEN_ARC_CHALLENGE":"1",
       "BEGLIN_QV2_NATIVE_ASYNC":"0"})
    if mode!="q4":env["BEGLIN_Q_LAB_N8_ROLES_FILE"]=str(MASKS[mode][0])
    tic=time.monotonic()
    r=subprocess.run([str(executable),"masked_choice_qv5","48"],
         cwd=str(BASE),env=env,text=True,capture_output=True,timeout=1500)
    log=(r.stdout+"\n"+r.stderr).encode()
    os.umask(0o077);put(logfile,log)
    if r.returncode!=0:raise RuntimeError("real Qwen native Q4/n8 HellaSwag eval failure "+str(r.returncode)+" "+r.stderr[-800:])
    score=parse(log.decode(),mode,frozen)
    rec={"schema":"BEGLIN-Qv8-real-original-Qwen28-native-48-HellaSwag-source-frozen/1",
      "status":"NATIVE_REAL_28_LAYER_FULL_VOCAB_Q4_N8_48_HELLASWAG_PASS",
      "mode":mode,"original_native_executable_SHA256":sha(executable.read_bytes()),
      "original_native_log_SHA256":sha(log),"frozen_eval_manifest_SHA":sha(sealed(RUN/"locked_model_48x4_manifest.json")),
      "source_only_28n8_mask_SHA":None if mode=="q4" else sha(MASKS[mode][0].read_bytes()),
      "scores":score,"model_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "cpu_wall_s":round(time.monotonic()-tic,2),
      "no_production_activation":True,"no_global_quality_claim":True}
    raw=(json.dumps(rec,sort_keys=True,indent=2)+"\n").encode()
    put(out/"native_hellaswag48_receipt.json",raw)
    print("QV8_ACTUAL_NEW_NARRATIVE_NATIVE",mode,
       json.dumps({"correct":score["correct"],"mean_NLL":score["mean_correct_answer_NLL"],
        "receipt_SHA":sha(raw),"seconds":rec["cpu_wall_s"]},sort_keys=True),flush=True)
def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("prepare","eval"))
    p.add_argument("--mode",choices=MODES)
    a=p.parse_args()
    if a.stage=="prepare":prepare()
    elif a.mode:evaluate(a.mode)
    else:p.error("--mode required for eval")
if __name__=="__main__":main()
