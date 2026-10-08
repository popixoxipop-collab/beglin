#!/usr/bin/env python3
"""Create isolated instrumentation from *exact* certified QT control source.

Only changes a temporary GitHub Actions checkout's qwen_infer.c, never the
pre-existing XOX control worktree, engine, model or EOE release.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path.cwd().resolve()
ORIGINAL = Path("/tmp/beglin-qt-l0-fp32-control/qwen_infer.c")
ORIGINAL_SHA = "7dc4f3dc41ea5a338bf231ecb4ef661482649264854c29641071c996af73b9c4"
ORIGINAL_BINARY = Path("/tmp/beglin-qt-l0-fp32-control/build-qt-l0-fp32/qwen_infer_gpu")
ORIGINAL_BINARY_SHA = "a023fe2e15da39eddbf7b2a483d65a25e3897eb3c3d9825db211b4ec2adce96e"
FROZEN_BASE = "4c072d8b827c0e0c4806feb5b02832e281a2bb87"
BUILD = ROOT / "build-qt-logit-isolation"
DEST = ROOT / "qwen_infer.c"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def check():
    if sha(ORIGINAL) != ORIGINAL_SHA or sha(ORIGINAL_BINARY) != ORIGINAL_BINARY_SHA:
        raise SystemExit("PINNED_ORIGINAL_SOURCE_OR_ENGINE_CHANGED")
    if not ROOT.is_relative_to(Path("/Users/xox/actions-runner-beglin-xox").resolve()):
        raise SystemExit("REFUSE_NON_EPHEMERAL_RUNNER_WORKSPACE")
    p=subprocess.run(["git","merge-base","--is-ancestor",FROZEN_BASE,"HEAD"],cwd=ROOT,capture_output=True)
    if p.returncode:
        raise SystemExit("SOURCE_COMMIT_NOT_DESCENDANT_OF_FROZEN_BASE")
    if BUILD.exists():
        raise SystemExit("BUILD_PATH_ALREADY_EXISTS")
    raw=ORIGINAL.read_text()
    if "QWEN_QT_DIAG_LOGITS" in raw:
        raise SystemExit("LOGIT_HOOK_ALREADY_PRESENT")
    return raw

def patch(raw):
    a='        int pos=np-1; printf("greedy:");\n        for(int g=0;g<n_gen;g++){ int am=argmax_v(logits);'
    b='''        const char *qt_logit_path = getenv("QWEN_QT_DIAG_LOGITS");
        FILE *qt_logit_file = NULL;
        if (qt_logit_path && qt_logit_path[0]) {
            if (n_gen < 1 || n_gen > 32) {
                fprintf(stderr,"FATAL: QT diagnostic token count must be 1..32\\\\n"); exit(1);
            }
            qt_logit_file = fopen(qt_logit_path, "wb");
            if (!qt_logit_file) {
                fprintf(stderr,"FATAL: QT diagnostic logit file open failed\\\\n"); exit(1);
            }
        }
        int pos=np-1; printf("greedy:");
        for(int g=0;g<n_gen;g++){
            if (qt_logit_file) {
                if (fwrite(logits,sizeof(float),(size_t)g_cfg.vocab,qt_logit_file)!=(size_t)g_cfg.vocab) {
                    fprintf(stderr,"FATAL: QT diagnostic logit write failed\\\\n"); exit(1);
                }
            }
            int am=argmax_v(logits);'''
    if raw.count(a)!=1:raise SystemExit("UNEXPECTED_GREEDY_ANCHOR")
    raw=raw.replace(a,b,1)
    a='        printf("\\n");\n    } else if (!strcmp(mode,"bench")) {'
    b='''        if (qt_logit_file && fclose(qt_logit_file)!=0) {
            fprintf(stderr,"FATAL: QT diagnostic logit close failed\\\\n"); exit(1);
        }
        printf("\\n");
    } else if (!strcmp(mode,"bench")) {'''
    if raw.count(a)!=1:raise SystemExit("UNEXPECTED_GREEDY_TAIL")
    raw=raw.replace(a,b,1)
    return raw

def main():
    raw=check()
    new=patch(raw)
    DEST.write_text(new)
    subprocess.run(["cmake","-S",".","-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],cwd=ROOT,check=True,timeout=300)
    subprocess.run(["cmake","--build",str(BUILD),"--target","qwen_infer_gpu","-j4"],cwd=ROOT,check=True,timeout=900)
    binary=BUILD/"qwen_infer_gpu"
    if not binary.is_file() or binary.stat().st_size < 200000:
        raise SystemExit("DIAGNOSTIC_BINARY_MISSING_OR_TOO_SMALL")
    result={
        "schema":"qt-logit-diag-build/1",
        "frozen_source_commit":FROZEN_BASE,
        "source_from_sha256":ORIGINAL_SHA,
        "original_binary_sha256":ORIGINAL_BINARY_SHA,
        "instrumented_c_sha256":sha(DEST),
        "diagnostic_binary_sha256":sha(binary),
        "instrumentation":"greedy-only logging of each pre-argmax fp32 logit vector to caller-provided file, no added quantization or generation logic",
        "binary_path":str(binary),
        "promotable":False
    }
    dest=ROOT/"QT_LOGIT_DIAG_BUILD.json"
    dest.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("QT_LOGIT_INSTRUMENTED_BUILD="+json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":main()
