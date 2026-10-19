#!/usr/bin/env python3
"""Synthetic adversarial C++ safetensors manifest admission regression.

All fixture artifacts are temporary, synthetic (< 10KB), and never touch D50
or any operational model. Native C++ verifier reads source metadata only.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent
VERIFIER=ROOT/"verify_full27_source_bounds_cpp_v1"
VERIFIER_CPP=ROOT/"verify_full27_source_bounds_cpp_v1.cpp"
CPP_SHA="35eeb6671a6462595bbb874cfc249f4cdfa8650e6430935d07b05f9c843babc5"
SHARD="model-00001-of-000001.safetensors"

def check(ok,msg):
    if not ok: raise AssertionError("BEGLIN_NATIVE_CPP_BOUNDS_NEGATIVE_FAIL: "+msg)
def digest(raw: bytes)->str:
    return hashlib.sha256(raw).hexdigest()
def create(root:Path):
    data=root/"data"
    data.mkdir()
    manifest=root/"boundary.tsv"
    quant="model.layers.0.self_attn.q_proj.weight"
    other="model.embed_tokens.weight"
    records={
        quant: {"dtype":"BF16","shape":[64,64],"data_offsets":[0,8192]},
        other: {"dtype":"BF16","shape":[32],"data_offsets":[8192,8256]},
    }
    header=json.dumps(records,sort_keys=True,separators=(",",":")).encode()
    header_length=len(header)
    source=struct.pack("<Q",header_length)+header+b"\0"*8256
    (data/SHARD).write_bytes(source)
    header_line="BEGLIN-SOURCE-MANIFEST/1\t"+ "a"*64+ "\t2\t1\t1\t1\t3840\t8256"
    shard_line="\t".join(["S",SHARD,str(len(source)),str(len(header)),digest(header),"2"])
    tensor1="\t".join(["T",quant,SHARD,"BF16","64,64","0","8192","7","3840"])
    tensor2="\t".join(["T",other,SHARD,"BF16","32","8192","8256","0","0"])
    lines=[header_line,shard_line,tensor1,tensor2]
    manifest.write_text("\n".join(lines)+"\n")
    return data,manifest,lines

def run(binary:Path,dir:Path,manifest:Path,mode="--fixture"):
    result=subprocess.run([str(binary),mode,str(dir),str(manifest)],
                          capture_output=True,text=True,timeout=6)
    return result.returncode,result.stdout.strip(),result.stderr.strip()

def tamper(binary,idx,case,modify):
    with tempfile.TemporaryDirectory(prefix="beglin-cpp-manifest-neg-") as t:
        root=Path(t)
        data,manifest,lines=create(root)
        modify(data,lines)
        manifest.write_text("\n".join(lines)+"\n")
        exit_code,stdout,stderr=run(binary,data,manifest)
        check(exit_code==2 and "FAIL_CLOSED" in stderr,
              case+": must fail closed, got "+repr((exit_code,stdout[:110],stderr[:150])))
        return case,stderr.split(": ",1)[-1][:95]

def main():
    check(digest(VERIFIER_CPP.read_bytes())==CPP_SHA,"verifier source pin")
    check(VERIFIER.is_file() and os.access(VERIFIER,os.X_OK),"compiled verifier absent")
    with tempfile.TemporaryDirectory(prefix="beglin-cpp-manifest-positive-") as t:
        data,manifest,lines=create(Path(t))
        rc,out,err=run(VERIFIER,data,manifest)
        check(rc==0 and json.loads(out)["qng64_matrices"]==1 and
              json.loads(out)["tensors"]==2,"positive test")
        print(json.dumps({"stage":"synthetic_positive","status":"PASS",
                          "tensors":2,"compressed":1}),flush=True)
        rc,out,err=run(VERIFIER,data,manifest,"--real")
        check(rc==2 and "FAIL_CLOSED" in err,"fixture never admitted as real")
    cases=[
      ("NONDECIMAL_UNSIGNED",lambda d,l: l[2].replace("8192","n8192")),
    ]
    tests=[
        ("SHARD_FILENAME_TRAVERSAL",lambda d,l:l.__setitem__(1,l[1].replace(SHARD,"../secret.safetensors"))),
        ("HEADER_SHA_MISMATCH",lambda d,l:l.__setitem__(1,l[1].replace(digest((d/SHARD).read_bytes()[8:8+struct.unpack('<Q',(d/SHARD).read_bytes()[:8])[0]]),"f"*64))),
        ("HEADER_LENGTH_WRONG",lambda d,l:l.__setitem__(1,l[1].replace("\t"+str(struct.unpack('<Q',(d/SHARD).read_bytes()[:8])[0])+"\t","\t999999\t"))),
        ("SHARD_SIZE_MISMATCH",lambda d,l:l.__setitem__(1,l[1].replace(str((d/SHARD).stat().st_size),str((d/SHARD).stat().st_size+1)))),
        ("SHARD_SYMLINK",lambda d,l:((d/SHARD).rename(d/"target.safetensors"),(d/SHARD).symlink_to("target.safetensors"))),
        ("NONDECIMAL_OFFSET",lambda d,l:l.__setitem__(2,l[2].replace("\t8192\t7\t","\tn8192\t7\t"))),
        ("OVERLAPPING_SPAN",lambda d,l:l.__setitem__(3,l[3].replace("\t8192\t8256\t","\t8000\t8064\t"))),
        ("GAP_IN_SPAN",lambda d,l:l.__setitem__(3,l[3].replace("\t8192\t8256\t","\t8200\t8264\t"))),
        ("OUT_OF_BOUNDS",lambda d,l:l.__setitem__(3,l[3].replace("\t8192\t8256\t","\t8192\t8257\t"))),
        ("DUPLICATE_TENSOR",lambda d,l:l.__setitem__(3,l[3].replace("model.embed_tokens.weight","model.layers.0.self_attn.q_proj.weight"))),
        ("TENSOR_DTYPE_MISMATCH",lambda d,l:l.__setitem__(2,l[2].replace("\tBF16\t","\tF32\t"))),
        ("UNSUPPORTED_QUANT_BITS",lambda d,l:l.__setitem__(2,l[2].rsplit("\t",2)[0]+"\t6\t3840")),
        ("INCORRECT_PACKED_BYTES",lambda d,l:l.__setitem__(2,l[2].rsplit("\t",1)[0]+"\t3841")),
        ("OUT_OF_BOUNDS_EXPERT",lambda d,l:l.__setitem__(2,l[2].replace("model.layers.0.self_attn.q_proj.weight","model.layers.1.mlp.experts.64.gate_proj.weight"))),
        ("NONQUANT_ROLE_WITH_BITS",lambda d,l:l.__setitem__(3,l[3].rsplit("\t",2)[0]+"\t7\t0")),
        ("NONZERO_UNQUANTIZED_BYTES",lambda d,l:l.__setitem__(3,l[3].rsplit("\t",1)[0]+"\t50")),
        ("INVALID_GROUP64_SHAPE",lambda d,l:l.__setitem__(2,l[2].replace("\t64,64\t","\t64,63\t"))),
        ("SHARD_TENSOR_COUNT_MISMATCH",lambda d,l:l.__setitem__(1,l[1].rsplit("\t",1)[0]+"\t3")),
        ("HEADER_QUANT_COUNT_MISMATCH",lambda d,l:l.__setitem__(0,l[0].replace("\t2\t1\t1\t1","\t2\t2\t1\t1"))),
        ("ILLEGAL_LINE_KIND",lambda d,l:l.__setitem__(2,"X"+l[2][1:])),
        ("EXTRA_TENSOR_SEGMENT",lambda d,l:l.append(l[3].replace("model.embed_tokens.weight","other.weight"))),
        ("TRUNCATED_SHARD",lambda d,l:(d/SHARD).write_bytes((d/SHARD).read_bytes()[:-1])),
        ("NO_FINAL_NEWLINE",lambda d,l:l[-1].__class__),
    ]
    results=[]
    for idx,(name,mutator) in enumerate(tests):
        if name=="NO_FINAL_NEWLINE":
            with tempfile.TemporaryDirectory(prefix="beglin-cpp-manifest-newline-") as temp:
                data,manifest,lines=create(Path(temp))
                manifest.write_text("\n".join(lines))
                rc,out,err=run(VERIFIER,data,manifest)
                check(rc==2 and "FAIL_CLOSED" in err,"missing final newline must reject")
                results.append((name,err.split(": ",1)[-1][:95]))
            continue
        results.append(tamper(VERIFIER,idx,name,mutator))
    print(json.dumps({
        "status":"PASS_CPP_MANIFEST_ADVERSARIAL_NEGATIVE",
        "negative_rejections":len(results)+1,
        "fixture_positive_pass":True,
        "real_source_not_mutated":True,
        "tests":[x[0] for x in results]
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
