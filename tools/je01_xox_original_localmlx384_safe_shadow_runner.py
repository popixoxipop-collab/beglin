#!/usr/bin/env python3
"""Bounded isolated JE01 original BEGLIN native BF16-source-derived AF inference.

Never touch serving route, supervisor, production worker PIDs or original
weights. Enforce artifact binary SHA, original-token 512 verified receipt,
production health identity and >=35% memory pressure before, during and after.
Any gate failure stops only this scratch subprocess. No user-visible quality
claim made from greedy tokens or logits without paired NLL.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path("/Users/xox/vdsp_shadow_runs/precision_allocator_bench/je01_original384_20261117")
BIN=ROOT/"build_xox_mlxabi384/qwen_infer_gpu_je01_context384_xoxabi"
BIN_SHA="60a0b559496180d29b65d5a9cb86bba4d16a12238a140cdfdd157602fcc90437"
MLXLIB="/Users/xox/Library/Python/3.14/lib/python/site-packages/mlx/lib"
WEIGHTS="/Users/xox/vdsp_local_data/moe_base_deepseek"
ROUTE="beglin-candidate-shared-up-l3-n6"
MIN_PRESSURE=35
MODE=sys.argv[1] if len(sys.argv)>1 else ""
if MODE not in ("short","long","stress330","stress383","calibration16"):
    raise SystemExit("usage: runner.py short|long|stress330")
DEST=ROOT/("run384_localabi_"+MODE)

def sha(file):
    h=hashlib.sha256()
    with open(file,"rb") as fd:
        for b in iter(lambda:fd.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def health():
    with urlopen("http://127.0.0.1:18765/healthz",timeout=5) as response:
        h=json.load(response)
    if h.get("status")!="ok" or h.get("route_generation")!=6 or h.get("active_route",{}).get("route_id")!=ROUTE:
        raise ValueError("JE01_PRODUCTION_HEALTH_OR_ROUTE_CHANGED")
    if len(h.get("workers",{}))!=2 or any(not v.get("alive") for v in h["workers"].values()):
        raise ValueError("JE01_PRODUCTION_WORKERS_NOT_TWO_ALIVE")
    return {"generation":6,"route_id":ROUTE,
            "workers":{k:v["pid"] for k,v in h["workers"].items()}}

def pressure():
    raw=subprocess.check_output(["/usr/bin/memory_pressure","-Q"],
                                text=True,timeout=8)
    found=re.search(r"System-wide memory free percentage:\s*(\d+)%",raw)
    if not found:raise ValueError("JE01_PRESSURE_SENSOR_UNAVAILABLE")
    return int(found.group(1))

def log(event,**extra):
    print(json.dumps({"event":event,"mode":MODE,**extra},sort_keys=True),flush=True)

if DEST.exists() or DEST.is_symlink():raise SystemExit("JE01_RUN_DIR_EXISTS_STOP_NO_OVERWRITE")
if sha(BIN)!=BIN_SHA:raise SystemExit("JE01_BINARY_SHA_DRIFT")
if not (Path(MLXLIB)/"libmlx.dylib").is_file():raise SystemExit("JE01_NATIVE_MLX_DLL_MISSING")
receipt=json.loads((ROOT/"manifests_xox/receipt.json").read_text())
if receipt.get("status")!="PASS_512_REAL_SHA_AND_DESTINATION_LOCAL_MANIFEST" or \
   receipt.get("source_receipt_sha256")!="2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34":
    raise SystemExit("JE01_XOX_NATIVE_INPUT_NOT_VERIFIED")
if MODE in ("short","long"):
    manifest=ROOT/"manifests_xox"/(MODE+".manifest")
else:
    manifest=ROOT/"manifests_xox"/(MODE+".manifest")
if not manifest.is_file() or manifest.is_symlink():raise SystemExit("JE01_INPUT_MANIFEST_UNAVAILABLE")
if len(manifest.read_text().splitlines())!=(16 if MODE=="calibration16" else 1):raise SystemExit("JE01_EXPECT_EXACTLY_ONE_CASE")
for name in ("arch_config_moe.txt","layout_af.txt","layout_f32.txt","deepseek_moe_af.bin","deepseek_moe_f32.bin"):
    if not (Path(WEIGHTS)/"weights_moe"/name).is_file():raise SystemExit("JE01_ORIGINAL_AF_SOURCE_MISSING:"+name)
baseline=health()
free=pressure()
if free<45:raise SystemExit("JE01_BEFORE_FREE_MEM_PRESSURE_TOO_LOW")
DEST.mkdir(mode=0o700)
log("PREFLIGHT_PASS",free_percent=free,route=baseline["route_id"],pids=baseline["workers"],
    native_binary_sha256=BIN_SHA,manifest_sha256=sha(manifest))
local_bin=BIN
local_bin.chmod(local_bin.stat().st_mode|0o100)
env={
  "PATH":"/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
  "HOME":"/Users/xox",
  "TMPDIR":str(DEST),
  "DYLD_LIBRARY_PATH":MLXLIB,
  "QWEN_MOE_BASE":WEIGHTS,
  "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
  "QWEN_MOE_CB_SLOTS":"1",
  "QWEN_MOE_CB_REQS":"16" if MODE=="calibration16" else "1",
  "QWEN_MOE_CB_PREFILL_BUDGET":"1",
  "QWEN_MOE_CB_PROMPT_MANIFEST":str(manifest),
  "QWEN_MOE_GPU_SORT_THRESHOLD":"64",
  "QWEN_MOE_GPU_CBATCH_BUDGET":"0",
}
started=time.monotonic()
killed=None
peak_rss_kib=0
lowest_pressure=free
log("RUN_LAUNCHING",args=["original-pinned-native384",MODE])
with (DEST/"stdout.txt").open("x") as stdout,(DEST/"stderr.txt").open("x") as stderr:
    p=subprocess.Popen([str(local_bin)],cwd=str(DEST),env=env,stdout=stdout,stderr=stderr)
    while p.poll() is None:
        if time.monotonic()-started>175:
            killed="native_timeout";p.terminate();break
        try:
            mem=pressure()
            lowest_pressure=min(lowest_pressure,mem)
            if mem<MIN_PRESSURE:
                killed="low_system_memory";p.terminate();break
            h=health()
            if h!=baseline:
                killed="production_identity_drift";p.terminate();break
            rss=subprocess.run(["/bin/ps","-o","rss=","-p",str(p.pid)],
                               capture_output=True,text=True,timeout=4).stdout.strip()
            if rss.isdigit():
                peak_rss_kib=max(peak_rss_kib,int(rss))
                if peak_rss_kib>16*1024*1024:
                    killed="native_peak_rss_guard";p.terminate();break
        except BaseException as err:
            killed="monitor_error_"+type(err).__name__;p.terminate();break
        time.sleep(1.0)
    if killed:
        try:p.wait(timeout=8)
        except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
    rc=p.wait()
finished=health()
last_pressure=pressure()
elapsed=time.monotonic()-started
stderr=(DEST/"stderr.txt").read_text(errors="replace")
stdout=(DEST/"stdout.txt").read_text(errors="replace")
status="NATIVE_REAL_MODEL_FORWARD_PASS" if rc==0 and not killed and "FATAL:" not in stderr else "BLOCKED"
report={
    "schema":"JE01-XOX-ISOLATED-ORIGINAL-AF-NATIVE-384-FORWARD/1",
    "status":status,"mode":MODE,"exit_code":rc,"forced_stop_reason":killed,
    "seconds":round(elapsed,3),"before_production":baseline,
    "after_production":finished,"production_pids_same":baseline==finished,
    "mem_pressure_free_before":free,"mem_pressure_free_min":lowest_pressure,
    "mem_pressure_free_after":last_pressure,"peak_sampled_rss_kib":peak_rss_kib,
    "original_model_Af_readonly":True,"binary_SHA256":BIN_SHA,
    "native_manifest_sha256":sha(manifest),
    "stderr_sha256":sha(DEST/"stderr.txt"),"stdout_sha256":sha(DEST/"stdout.txt"),
    "stderr_tail":stderr[-2300:],"stdout_tail":stdout[-1200:],
    "finite_logits":("finite_logits=1" in stdout or "finite_logits=1" in stderr),
    "teacher_forced_NLL":"NOT_MEASURED",
    "production_write_allowed":False
}
report["receipt_sha256"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest()
with (DEST/"report.json").open("x") as f:json.dump(report,f,sort_keys=True,indent=2)
log("FINAL_RESULT",**{k:report[k] for k in ("status","exit_code","forced_stop_reason","seconds",
  "production_pids_same","mem_pressure_free_min","peak_sampled_rss_kib","stderr_tail","stdout_tail","receipt_sha256")})
if status!="NATIVE_REAL_MODEL_FORWARD_PASS":raise SystemExit(2)
