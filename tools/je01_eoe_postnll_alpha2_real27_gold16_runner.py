#!/usr/bin/env python3
"""JE01 original27 MLX output-policy alpha 0,+/-0.05,1 native gold16.

Scientific scope: true 27-layer per-token NLL under *routed-output* linear
interpolation between original AF4 and one frozen mixed 43MB policy.
Finite differences estimate a directional log-probability score; they do
NOT identify the 48-role weight-level Fisher/Hessian of a Bayesian optimizer.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time

ROOT=Path("/Volumes/D50/TailnetCommander/work")
CODE=ROOT/"je01_postnll_bc_native_compile_20261123_v1"/"je01_postnll_bc_gold16_native_runner_v1.py"
CODE_BLOB="5454f1064a86db4bdbc0eef9f1940d9086eeaf40"
NATIVE=ROOT/"je01_postnll_alpha2_native_20261123_v1"
NATIVE_SHA="01a9a22649f0942084a4026910b6f348f424304cc6d2b6ac89590cfc59d0b0a3"
RECEIPT_SHA="6cd54f0fde97008b13d4b2a75a47107c3b809864060695f16bbce63f9708c3ae"
OUTPUT=ROOT/"je01_postnll_alpha2_gold16_20261123_v1"
MARKER="JE01_CAL240_POSTNLL_BC_SEVEN_EXPERTS_REGISTERED"
RESULT="RESULT: MoE GPU V5h online cbatch gate complete, B=1 R=16"
ALPHA={"0.0":"z000","+0.05":"p005","-0.05":"m005","1.0":"p100"}
KNOWN_ON={"B":"d164a4c26c2691b6bee7762235a3175bef37199fb3fc9cbc8efa69c92179ddcb",
          "C":"5ec420bebda8e56835bcc3803baa6a0a15641dc08cf9b5d4288426fbe4832881"}
SOURCE_GOLD="b23715f0dec13de881f9ad948bacf3f8e8885068ff39aef25733fa236070493c"


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def blob(p):
    b=p.read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()


def canon(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def gate(ok,what):
    if not ok:raise ValueError(what)


def preflight(arm,alpha):
    gate(arm in KNOWN_ON and alpha in ALPHA,"JE01_ALPHA_UNPREREGISTERED_ARM_OR_STEP")
    gate(CODE.is_file() and not CODE.is_symlink() and blob(CODE)==CODE_BLOB,
         "JE01_ALPHA_STRICT_PARENT_NATIVE_SOURCE_BLOB_MISMATCH")
    spec=importlib.util.spec_from_file_location("je01_postnll_frozen_original27_BC",CODE)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    m,manifest,goldmeta,lib,ref,bundle,tsv,phys=original.source_preflight(arm)
    gate(NATIVE.is_dir() and not NATIVE.is_symlink(),"JE01_ALPHA_MODEL_SCRATCH_NOT_ROOT")
    native_file=NATIVE/"bin/je01_postnll_alpha2_original27_fullgold"
    native_receipt=NATIVE/"JE01_ALPHA2_NATIVE_BUILD_RECEIPT.json"
    gate(native_file.is_file() and not native_file.is_symlink() and
         sha(native_file)==NATIVE_SHA and native_receipt.is_file() and
         not native_receipt.is_symlink() and sha(native_receipt)==RECEIPT_SHA,
         "JE01_ALPHA_SOURCE_NATIVE_COMPILED_BINARY_SHA_DRIFT")
    proof=json.loads(native_receipt.read_text())
    gate(proof["status"]=="JE01_ACTUAL_ORIGINAL27_ALPHA2_RESEARCH_NATIVE_BUILD_PASS" and
         proof["native_sha256"]==NATIVE_SHA and proof["production_modified"] is False and
         proof["source_patch_receipt"]["original_cpp_sha256"]==
         "640da6e8afcace5e5440177ef9953732bee1cf83927680992f809a7af3f6950a" and
         proof["source_patch_receipt"]["unchanged_c_sha256"]==
         "fe4b82b70a68ced39fa59357e3dba1f117ff831c503d528b5777602f6f5cd7ae",
         "JE01_ALPHA_ORIGINAL27_DISPOSABLE_COMPILE_PROOF_INVALID")
    gate(phys["physical_file_sha256"]==original.ARMS[arm]["binary_sha"] and
         phys["physical_file_size_bytes"]==43253760 and
         phys["heldout_validation_OOD_used"] is False and
         phys["production_checkpoint_modified"] is False,
         "JE01_ALPHA_REAL_BC_NATIVE_43MB_FILE_SOURCE_SCOPE")
    return original,m,manifest,goldmeta,lib,ref,bundle,tsv,native_file,phys


def run(arm,alpha):
    original,m,manifest,goldmeta,lib,ref,bundle,tsv,exe,phys=preflight(arm,alpha)
    gate(not OUTPUT.is_symlink(),"JE01_ALPHA_OUTPUT_SYMLINK_DENIED")
    OUTPUT.mkdir(parents=True,mode=0o700,exist_ok=True)
    folder=OUTPUT/arm/ALPHA[alpha]
    gate(not folder.exists() and not folder.is_symlink(),"JE01_ALPHA_THIS_EXACT_RUN_ALREADY_EXISTS")
    folder.mkdir(parents=True,mode=0o700)
    (folder/"tmp").mkdir(mode=0o700)
    target=folder/"gold_native.bin"
    stderr=folder/"stderr.txt"
    stdout=folder/"stdout.txt"
    env={"PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
         "HOME":"/Users/eoe",
         "TMPDIR":str(folder/"tmp"),
         "DYLD_LIBRARY_PATH":str(lib),
         "QWEN_MOE_BASE":str(m.CHECKPOINT),
         "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
         "QWEN_MOE_CB_SLOTS":"1",
         "QWEN_MOE_CB_REQS":"16",
         "QWEN_MOE_CB_PREFILL_BUDGET":"1",
         "QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
         "QWEN_MOE_GPU_SORT_THRESHOLD":"64",
         "QWEN_MOE_GPU_CBATCH_BUDGET":"0",
         "JE01_ORIGINAL_GOLD_META":str(goldmeta),
         "JE01_ORIGINAL_GOLD_OUTPUT":str(target),
         "JE01_REAL_Q35_BUNDLE":str(bundle),
         "JE01_REAL_Q35_TSV":str(tsv),
         "JE01_CAL240_POSTNLL_ARM":original.ARMS[arm]["policy"],
         "JE01_POSTNLL_ALPHA":alpha}
    initial=m.memfree()
    gate(initial>=55,"JE01_ALPHA_PREMODEL_MEMORY_BELOW55")
    started=time.monotonic()
    minimum=initial
    print("JE01_ALPHA_REAL_FULL27_START",json.dumps({
       "arm":arm,"alpha":alpha,"gold_tokens":656,
       "native_sha256":NATIVE_SHA,"actual_physical_sha256":phys["physical_file_sha256"],
       "initial_mem_free":initial},sort_keys=True),flush=True)
    stop=None
    with stdout.open("xb") as out,stderr.open("xb") as err:
        proc=subprocess.Popen([str(exe)],cwd=folder,env=env,stdout=out,stderr=err)
        while proc.poll() is None:
            try:
                free=m.memfree()
                minimum=min(minimum,free)
                if free<35:stop="JE01_ALPHA_NATIVE_MEMORY_FREE_UNDER35"
                if time.monotonic()-started>420:stop="JE01_ALPHA_NATIVE_TIMEOUT420"
                rss=subprocess.run(["/bin/ps","-o","rss=","-p",str(proc.pid)],
                    capture_output=True,text=True,timeout=5).stdout.strip()
                if rss.isdigit() and int(rss)>16*1024*1024:
                    stop="JE01_ALPHA_NATIVE_RSS_OVER16GB"
            except Exception as e:stop="JE01_ALPHA_RESOURCE_SENSOR_"+type(e).__name__
            if stop:
                proc.terminate()
                break
            time.sleep(.8)
        if stop and proc.poll() is None:
            try:proc.wait(timeout=6)
            except subprocess.TimeoutExpired:proc.kill()
        code=proc.wait()
    error=stderr.read_text(errors="replace")
    gate(code==0 and not stop and "FATAL:" not in error and MARKER in error and RESULT in error,
         "JE01_ALPHA_NATIVE_GPU_EXECUTION_FAIL_CLOSED:"+str(stop)+"/"+str(code)+"/"+error[-900:])
    actual,_,_=m.original_656(target.read_bytes(),goldmeta.read_bytes())
    controls=[math.fsum(r) for r in ref]
    sums=[math.fsum(r) for r in actual]
    mean=math.fsum(sums)/656
    goldsha=sha(target)
    if alpha=="0.0":
        gate(goldsha==SOURCE_GOLD and abs(mean-original.BASELINE_NLL)<1e-12,
             "JE01_ALPHA0_ORIGINAL_AF4_NOT_BIT_IDENTICAL")
    if alpha=="1.0":
        gate(goldsha==KNOWN_ON[arm] and
             abs(mean - (2.2974258514952206 if arm=="B" else 2.2959651985430702))<1e-12,
             "JE01_ALPHA1_PREVIOUS_BC_NATIVE_ON_NOT_BIT_IDENTICAL")
    per_case=[{"index":i,"tokens":len(ref[i]),"base_NLL_sum":controls[i],
               "actual_NLL_sum":sums[i],
               "delta_NLL_sum":sums[i]-controls[i]} for i in range(16)]
    receipt={"schema":"beglin-je01-original27-preregistered-output-policy-alpha-gold16/1",
       "status":("JE01_ORIGINAL27_ALPHA_ENDPOINT_PARITY_PASS" if alpha in ("0.0","1.0")
                 else "JE01_ORIGINAL27_ALPHA_CENTRAL_DIFF_INPUT_NLL_MEASURED"),
       "arm":arm,"alpha":alpha,"native_file_sha256":NATIVE_SHA,
       "physical_policy_sha256":phys["physical_file_sha256"],
       "original_AF4_model_SHA256":m.AF_SHA,
       "mean_NLL":mean,
       "baseline_NLL":original.BASELINE_NLL,
       "delta_NLL":mean-original.BASELINE_NLL,
       "actual_native_gold_output_sha256":goldsha,
       "real_full_model_27_layers":True,
       "gold_tokens":656,"gold_train_cases":16,
       "per_gold_target_NLL":actual,
       "per_case":per_case,
       "min_memory_free_pct":minimum,
       "elapsed_seconds":round(time.monotonic()-started,3),
       "per_token_gradient":"NOT_MEASURED_THIS_RUN",
       "directional_fisher":"NOT_MEASURED_THIS_RUN",
       "parameter_weight_fisher":"NOT_MEASURED",
       "independent_heldout":"NOT_ACCESSED",
       "OOD":"NOT_ACCESSED",
       "production_modified":False}
    receipt["receipt_sha256"]=canon(receipt)
    (folder/"JE01_ALPHA_REAL27_656GOLD_NLL.json").write_text(json.dumps(
        receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("JE01_ALPHA_REAL_FULL27_FINISHED",json.dumps({
        k:v for k,v in receipt.items() if k not in ("per_gold_target_NLL","per_case")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=("B","C"),required=True)
    p.add_argument("--alpha",choices=tuple(ALPHA),required=True)
    args=p.parse_args()
    try:run(args.arm,args.alpha)
    except BaseException as e:
        print("JE01_ALPHA_ORIGINAL27_FAIL_CLOSED",json.dumps({
            "type":type(e).__name__,"error":str(e)[:1000],
            "arm":args.arm,"alpha":args.alpha},sort_keys=True),flush=True)
        raise SystemExit(2)
