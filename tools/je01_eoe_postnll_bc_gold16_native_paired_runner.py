#!/usr/bin/env python3
"""JE01 actual original27 BC frozen post-NLL arms B/C (TRAIN16 ONLY).

Runs exact compiled native Apple MLX C++ original27 baseline OFF with the
original 656-gold output SHA parity gate, then true 27-layer B/C ON with
separately native Q4/5 physically packed 43,253,760 byte role bundles.
No validation/OOD access, no parameter Fisher, no promotion, no re-optimization.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path("/Volumes/D50/TailnetCommander/work")
BUILD=ROOT/"je01_postnll_bc_native_compile_20261123_v1"
OUTPUT=ROOT/"je01_postnll_bc_original27_native_gold16_20261123_v1"
OLD=ROOT/"je01_eoe_realaf_nll_pilot16_20261122_v1"
PARENT=BUILD/"checkout/tools/je01_eoe_cal240_rolewise_gold16_paired_runner.py"
PARENT_BLOB="0ee9f3c9a45392ba13ee453056b3421fec28a080"
EXE=BUILD/"bin/je01_eoe_original384_cal240_bc_rolewise_fullgold"
EXE_SHA="d53db80fd7326d5c247fc8d9544b70b60b4601abd1fbe84715dd180b297737ac"
BUILD_RECEIPT_SHA="5fc038e5ac644fac006d5f55caa532f551b22f21d3d885ee6b17d34f1482f43d"
BASELINE_SHA="b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c"
BASELINE_NLL=2.291799702354307
SOURCE_GIT_HEAD="d187697bf9ed730f05042f89355e1281e79d1d4f"
PRE=BUILD/"checkout/docs/research/JE01_REAL27_NLL_POST_FAILURE_EARLY_EXPERT_SWAP_PREREG_20261122.json"
PRE_BLOB="08c493215078b83481c871203d1e08232f2646e3"
MARKER="JE01_CAL240_POSTNLL_BC_SEVEN_EXPERTS_REGISTERED"
RESULT="RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16"
ARMS={
  "B":{"policy":"B_PROTECT_EARLY_GATE",
       "dir":"je01_postnll_B_protect_early_gate_20261123_v1",
       "binary_sha":"fcf33feb58b87724f8be673c8457cf2d0b3e2ab0cf947ed6c3b0cb6e91f67fcf",
       "manifest_sha":"3a1d5871e80e8b361d1173a26aca8bc900a8fb88bdbe5f36d90f35e4dd5f0ddd",
       "receipt_sha":"c58b1a3bdae9f5d913d31b3f96f592a88a1b761a2c2852f4f9109fc2c190a3f4"},
  "C":{"policy":"C_PROTECT_EARLY_UP",
       "dir":"je01_postnll_C_protect_early_up_20261123_v1",
       "binary_sha":"9a61a792d7e529868109ee3e5c240fea545479ca89d681cbf0b0ea6543b17997",
       "manifest_sha":"416ed0f206d885404e2ce4ce5731235c7e6505e50b1071f3defcbaebe1ca5f5a",
       "receipt_sha":"ab81e8180f2cbf8facd45fd9ec5739514b2730f012a88a59efdca81cced47d2b"}
}


def filehash(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(2<<20),b""):h.update(block)
    return h.hexdigest()


def blob(p):
    b=p.read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()


def canon(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def fail(x,what):
    if not x:raise ValueError(what)


def read_signed(p,expected):
    fail(p.is_file() and not p.is_symlink(),"JE01_BC_SIGNED_FILE_MISSING_OR_LINK")
    d=json.loads(p.read_text())
    fail(d.get("receipt_sha256")==expected and
         canon({k:v for k,v in d.items() if k!="receipt_sha256"})==expected,
         "JE01_BC_SIGNED_SOURCE_RECEIPT_DRIFT:"+p.name)
    return d


def source_preflight(mode):
    fail(mode in ("off","B","C"),"JE01_BC_ONLY_OFF_B_C")
    fail(OLD.is_dir() and BUILD.is_dir() and not BUILD.is_symlink() and
         PARENT.is_file() and not PARENT.is_symlink() and blob(PARENT)==PARENT_BLOB,
         "JE01_BC_ORIGINAL_27L_AUDITOR_SOURCE_NOT_PINNED")
    spec=importlib.util.spec_from_file_location("je01_original27_gold16_unmodified_pilot",PARENT)
    orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
    # Strict A original AF4 9.8GB SHA, 16 prompts, 656-token gold and
    # OFF parity evidence are rechecked from original signed source.
    _,_,_,manifest,goldmeta,lib,ref=orig.preflight("off")
    fail(PRE.is_file() and not PRE.is_symlink() and blob(PRE)==PRE_BLOB,
         "JE01_BC_FIXED_ROLE_PREREG_SOURCE_DRIFT")
    prereg=json.loads(PRE.read_text())
    fail(prereg["stage"]=="PRE_REGISTERED_CONDITIONAL_UNRUN" and
         prereg["budget"]["total_exact_bytes"]==43253760,
         "JE01_BC_PREREG_NOT_FROZEN")
    build=BUILD/"JE01_POSTNLL_BC_LOCAL_NATIVE_BUILD_RECEIPT.json"
    fail(build.is_file() and not build.is_symlink() and filehash(build)==BUILD_RECEIPT_SHA,
         "JE01_BC_NEW_NATIVE_BUILD_PROOF_FILE_CHANGED")
    native=json.loads(build.read_text())
    fail(native["status"]=="JE01_POSTNLL_BC_7EXPERT_ORIGINAL27_NATIVE_COMPILE_PASS" and
         native["base_git_commit"]==SOURCE_GIT_HEAD and
         native["native_executable_sha256"]==EXE_SHA and
         native["source_patch"]["production_modified"] is False and
         native["production_modified"] is False,
         "JE01_BC_NATIVE_BUILD_RECEIPT_INVALID")
    fail(EXE.is_file() and not EXE.is_symlink() and filehash(EXE)==EXE_SHA,
         "JE01_BC_ACTUAL_APPLE_NATIVE_EXE_SHA_DRIFT")
    if mode=="off":
        return orig,manifest,goldmeta,lib,ref,None,None,None
    old=OUTPUT/"off"/"JE01_BC_REAL27_GOLD16_NATIVE_NLL.json"
    off=read_signed(old,json.loads(old.read_text()).get("receipt_sha256")) if old.is_file() else None
    fail(off is not None and off["mode"]=="off" and
         off["status"]=="JE01_BC_TRUE_ORIGINAL27_OFF_EXACT_SHA_PASS" and
         off["native_output_sha256"]==BASELINE_SHA and
         off["binary_sha256"]==EXE_SHA and off["gold_tokens"]==656 and
         off["production_modified"] is False,
         "JE01_BC_NEW_BINARY_OFF_NOT_EXACT_ORIGINAL")
    arm=ARMS[mode]
    folder=ROOT/arm["dir"]
    receipts=folder/("JE01_POSTNLL_"+mode+"_SOURCE_RECEIPT.json")
    d=read_signed(receipts,arm["receipt_sha"])
    binary=folder/("JE01_POSTNLL_"+mode+"_EXACT43MB.bin")
    tsv=folder/("JE01_POSTNLL_"+mode+"_24ROLES.tsv")
    ass=next(a for a in prereg["arms"] if a["id"]==arm["policy"])
    fail(d["status"]=="JE01_PREREG_BC_ACTUAL_NATIVE_Q4Q5_PHYSICAL_43MB_BUILT" and
         d["arm"]==arm["policy"] and
         d["prereg_git_blob_sha1"]==PRE_BLOB and
         d["actual_original_AF4_sha256"]==orig.AF_SHA and
         d["counts"]=={"AF4":12,"Q4":6,"Q5":6} and
         d["physical_file_size_bytes"]==43253760 and
         d["physical_file_sha256"]==arm["binary_sha"] and
         d["role_manifest_sha256"]==arm["manifest_sha"] and
         d["q4_roles"]==sorted(ass["q4"]) and
         d["q5_roles"]==sorted(prereg["budget"]["Q5_FROZEN"]) and
         d["production_checkpoint_modified"] is False and
         d["heldout_validation_OOD_used"] is False,
         "JE01_BC_SOURCE_ARM_ROLE_MISMATCH")
    fail(binary.is_file() and not binary.is_symlink() and
         binary.stat().st_size==43253760 and filehash(binary)==arm["binary_sha"] and
         tsv.is_file() and not tsv.is_symlink() and filehash(tsv)==arm["manifest_sha"],
         "JE01_BC_ACTUAL_43MB_BYTE_OR_MANIFEST_SHA_MISMATCH")
    return orig,manifest,goldmeta,lib,ref,binary,tsv,d


def run(mode):
    old,manifest,goldmeta,lib,ref,bundle,tsv,receipt=source_preflight(mode)
    if mode=="off":
        fail(not OUTPUT.exists() and not OUTPUT.is_symlink(),"JE01_BC_EVAL_ROOT_ALREADY_CREATED")
        OUTPUT.mkdir(mode=0o700)
    folder=OUTPUT/mode
    fail(not folder.exists() and not folder.is_symlink(),"JE01_BC_MODE_ALREADY_EXECUTED")
    folder.mkdir(mode=0o700)
    tmp=folder/"tmp"
    tmp.mkdir(mode=0o700)
    target=folder/"gold_native.bin"
    stdout=folder/"stdout.txt"
    stderr=folder/"stderr.txt"
    env={
      "PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
      "HOME":"/Users/eoe","TMPDIR":str(tmp),
      "DYLD_LIBRARY_PATH":str(lib),
      "QWEN_MOE_BASE":str(old.CHECKPOINT),
      "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
      "QWEN_MOE_CB_SLOTS":"1","QWEN_MOE_CB_REQS":"16",
      "QWEN_MOE_CB_PREFILL_BUDGET":"1",
      "QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
      "QWEN_MOE_GPU_SORT_THRESHOLD":"64",
      "QWEN_MOE_GPU_CBATCH_BUDGET":"0",
      "JE01_ORIGINAL_GOLD_META":str(goldmeta),
      "JE01_ORIGINAL_GOLD_OUTPUT":str(target)}
    if mode!="off":
        env["JE01_CAL240_POSTNLL_ARM"]=ARMS[mode]["policy"]
        env["JE01_REAL_Q35_BUNDLE"]=str(bundle)
        env["JE01_REAL_Q35_TSV"]=str(tsv)
    started=time.monotonic()
    minimum_free=old.memfree()
    fail(minimum_free>=55,"JE01_BC_MEMORY_FREE_BELOW55")
    print(json.dumps({
       "event":"JE01_BC_REAL_ORIGINAL27_FULLGOLD16_START",
       "mode":mode,"native_exe_sha256":EXE_SHA,
       "physical_bundle_sha256":receipt["physical_file_sha256"] if receipt else "ORIGINAL_AF4_OFF",
       "memory_free_pct":minimum_free,"gold_tokens":656},sort_keys=True),flush=True)
    stop=None
    with stdout.open("xb") as out,stderr.open("xb") as err:
        proc=subprocess.Popen([str(EXE)],cwd=str(folder),env=env,
            stdout=out,stderr=err)
        while proc.poll() is None:
            try:
                free=old.memfree()
                minimum_free=min(minimum_free,free)
                if free<35:stop="JE01_BC_MEMORY_BELOW35"
                if time.monotonic()-started>420:stop="JE01_BC_FULL27_TIMEOUT420"
                rss=subprocess.run(["/bin/ps","-o","rss=","-p",str(proc.pid)],
                       capture_output=True,text=True,timeout=5).stdout.strip()
                if rss.isdigit() and int(rss)>16*1024*1024:stop="JE01_BC_NATIVE_RSS_OVER16GB"
            except Exception as e:stop="JE01_BC_RESOURCE_PROBE_"+type(e).__name__
            if stop:
                proc.terminate()
                break
            time.sleep(.8)
        if stop and proc.poll() is None:
            try:proc.wait(timeout=6)
            except subprocess.TimeoutExpired:proc.kill()
        code=proc.wait()
    stderr_text=stderr.read_text(errors="replace")
    if (code!=0 or stop or "FATAL:" in stderr_text or RESULT not in stderr_text or
        ((MARKER in stderr_text)!=(mode!="off"))):
        raise RuntimeError("JE01_BC_REAL27_NATIVE_FAILED_"+
                           mode+"/"+str(stop)+"/"+str(code)+"/"+stderr_text[-1300:])
    measured,_,_=old.original_656(target.read_bytes(),goldmeta.read_bytes())
    sums=[math.fsum(row) for row in measured]
    base=[math.fsum(row) for row in ref]
    deltas=[a-b for a,b in zip(sums,base)]
    mean=math.fsum(sums)/656
    reference=OLD/"gold_original_nll.bin"
    if mode=="off":
        fail(max(map(abs,deltas))<1e-10 and
             abs(mean-BASELINE_NLL)<1e-12 and
             filehash(target)==BASELINE_SHA and
             filehash(target)==filehash(reference),
             "JE01_BC_NEW_BINARY_OFF_ORIGINAL27_OUTPUT_NOT_BIT_IDENTICAL")
    rec={
       "schema":"beglin-je01-original27-postnll-bc-full656-native-loss/1",
       "status":"JE01_BC_TRUE_ORIGINAL27_OFF_EXACT_SHA_PASS" if mode=="off" else "JE01_BC_REAL_ORIGINAL27_ON_NLL_MEASURED",
       "mode":mode,"policy_arm":ARMS[mode]["policy"] if mode!="off" else "ORIGINAL_AF4",
       "native_full27_model":True,"native_executable_sha256":EXE_SHA,
       "binary_sha256":EXE_SHA,
       "real_original_AF4_sha256":old.AF_SHA,
       "real_physical_bundle_sha256":receipt["physical_file_sha256"] if receipt else None,
       "real_physical_manifest_sha256":receipt["role_manifest_sha256"] if receipt else None,
       "real_physical_budget_bytes":43253760 if receipt else None,
       "real_tier_counts":{"AF4":12,"Q4":6,"Q5":6} if receipt else None,
       "original_AF4_output_sha256":BASELINE_SHA,
       "native_output_sha256":filehash(target),
       "mean_gold_NLL":mean,
       "same_source_original_NLL":BASELINE_NLL,
       "NLL_candidate_minus_original":mean-BASELINE_NLL,
       "gold_train_cases":16,"gold_tokens":656,
       "per_case":[{"row":i,"tokens":len(ref[i]),
          "original_NLL_sum":base[i],"measured_mode_NLL_sum":sums[i],
          "candidate_minus_original_NLL_sum":deltas[i]} for i in range(16)],
       "per_gold_target_actual_NLL":measured,
       "initial_input_source_training":"ALREADY_OBSERVED_WIKITEXT_TRAIN16_EXPLORATORY",
       "heldout_validation128":"NOT_ACCESSED",
       "heldout_OOD128":"NOT_ACCESSED",
       "empirical_fisher_weights":"NOT_MEASURED",
       "min_memory_free_pct":minimum_free,
       "elapsed_seconds":round(time.monotonic()-started,3),
       "production_modified":False,"original_model_weights_modified":False,
       "scientific_adoption_gate":"NOT_EVALUATED"}
    rec["receipt_sha256"]=canon(rec)
    with (folder/"JE01_BC_REAL27_GOLD16_NATIVE_NLL.json").open("x") as f:
        f.write(json.dumps(rec,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in rec.items() if k not in ("per_gold_target_actual_NLL","per_case")},sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("off","B","C"),required=True)
    args=p.parse_args()
    try:run(args.mode)
    except BaseException as e:
        print(json.dumps({"status":"JE01_POSTNLL_BC_REAL27_FAIL_CLOSED",
             "mode":args.mode,"error_type":type(e).__name__,"error":str(e)[:1200]},sort_keys=True),flush=True)
        raise SystemExit(2)
