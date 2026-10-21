#!/usr/bin/env python3
"""Build a Qwen 1.5B isolated BEGLIN native precision comparison.

Original qwen_infer.c is never changed. A SHA-pinned COPY in private lab gets
one opt-in load-time control: f32 original BF16 -> FP32 projections, n8 native
BEGLIN qNg64, or default Q4G64. Untied lm_head stays common Q8 in ALL runs,
avoiding a head-precision confound. This is NOT a BF16 arithmetic runtime,
and not valid final Q hypothesis acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"qwen_infer.c"
EXPECT_SOURCE_SHA="52c8f5fd683b5ef0726e4738ab4b441624f54c807a3e71fb9d932cb79938d112"
LAB=ROOT/"lab_local/q_full_model_20261021"
SNAP=Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
MODEL=SNAP/"model.safetensors"
MODEL_REV="4ae77cb209f06199b8df1c94e21ff341332a3a89"
OFFICIAL_REV="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_QPROJ_SHA="527bf1760462c181be3ab71e3c394cdf50283a4df6b5d80d0bd2a5ab0b7eded5"
TIERS=("f32","n8","q4")

def sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()

ANCHOR="""static void load_safetensors_weights(void) {
    int n_q4 = 0, n_f32 = 0;"""
NEW_ANCHOR="""static void load_safetensors_weights(void) {
    const char *q_lab = getenv("BEGLIN_Q_LAB_PRECISION");
    if (q_lab) {
        if (strcmp(q_lab,"f32") && strcmp(q_lab,"n8") && strcmp(q_lab,"q4")) {
            fprintf(stderr,"FATAL: BEGLIN_Q_LAB_PRECISION must be f32, n8, or q4\\n");
            exit(1);
        }
        if (!getenv("QWEN_SAFETENSORS") || getenv("QWEN_GGUF") ||
            getenv("QWEN_INT4_BIN") || getenv("QWEN_DENSE_QNG64_OVERRIDE") ||
            getenv("QWEN_WT_PROMOTION_FILE")) {
            fprintf(stderr,"FATAL: research Q tiers require isolated safetensors without other precision overrides\\n");
            exit(1);
        }
        fprintf(stderr,"[Q-NATIVE-RESEARCH] precision=%s, source-original-BF16, common untied head Q8\\n",q_lab);
    }
    int n_lab_n8=0;
    int n_q4 = 0, n_f32 = 0;"""
ROLE_OLD="""            if (is_norm) { st_register_f32_as(name); n_f32++; }
            else {
                int qng_n=dense_qng64_override_n(name);
                if(qng_n) st_register_qng64_as(name,qng_n);
                else { st_register_q4g64_as(name); n_q4++; }
            }"""
ROLE_NEW="""            if (is_norm) { st_register_f32_as(name); n_f32++; }
            else if (q_lab && !strcmp(q_lab,"f32")) {
                st_register_f32_as(name); n_f32++;
            } else if (q_lab && !strcmp(q_lab,"n8")) {
                st_register_qng64_as(name,8); n_lab_n8++;
            } else {
                int qng_n=dense_qng64_override_n(name);
                if(qng_n) st_register_qng64_as(name,qng_n);
                else { st_register_q4g64_as(name); n_q4++; }
            }"""
# Keep original head Q8 for every tier so only the 196 projection roles vary.
LOG_OLD='''    fprintf(stderr, "[engine] safetensors: registered %d tensors (%d K_Q4G64, %d K_Q8G64, %d K_F32)\\\\n",
            g_nwt, n_q4, n_q8, n_f32);'''
LOG_NEW='''    fprintf(stderr, "[engine] safetensors: registered %d tensors (%d K_Q4G64, %d K_Q8G64, %d K_F32)\\\\n",
            g_nwt, n_q4, n_q8, n_f32);
    if (q_lab) fprintf(stderr, "[Q-NATIVE-RESEARCH] projection_role_tensors: f32=%d n8=%d q4=%d common_head_q8=%d\\\\n",
        n_f32-2*g_cfg.nl-2, n_lab_n8, n_q4, n_q8);'''

def check_source_and_patch()->str:
    if platform.system()!="Darwin" or platform.machine()!="arm64":
        raise RuntimeError("BEGLIN original CPU native probe requires macOS ARM64")
    if not MODEL.is_file() or not (SNAP/"config.json").is_file() or not (SNAP/"tokenizer.json").is_file():
        raise RuntimeError("pinned BF16 Qwen cache missing")
    original=SOURCE.read_bytes()
    if sha(original)!=EXPECT_SOURCE_SHA:
        raise RuntimeError("original BEGLIN source SHA mismatch, do not patch unknown source")
    s=original.decode("utf-8")
    for needle in (ANCHOR,ROLE_OLD,LOG_OLD):
        if s.count(needle)!=1:
            raise RuntimeError("expected unique original source patch signature not found:"+needle[:55])
    s=s.replace(ANCHOR,NEW_ANCHOR,1).replace(ROLE_OLD,ROLE_NEW,1).replace(LOG_OLD,LOG_NEW,1)
    assert s.count("BEGLIN_Q_LAB_PRECISION")==2
    return s

def write_private(p:Path,bytestr:bytes):
    if p.exists():
        raise FileExistsError("do not replace prior experiment "+str(p))
    with p.open("xb") as f:
        f.write(bytestr);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)

def prepare():
    source=check_source_and_patch()
    if LAB.exists():
        raise FileExistsError("private Q lab already exists; refuse overwrite")
    os.umask(0o077)
    LAB.mkdir(mode=0o700,parents=True)
    source_file=LAB/"qwen_infer_q_lab.c"
    write_private(source_file,source.encode("utf-8"))
    obj=LAB/"qwen_infer_q_lab.o"
    other=sorted(x for x in (ROOT/".build").glob("*.o") if x.name!="qwen_infer.o")
    if len(other)!=21:
        raise RuntimeError("missing original BEGLIN dependencies: "+str(len(other)))
    command=["clang","-O3","-w","-I",str(ROOT),"-c",str(source_file),"-o",str(obj)]
    subprocess.run(command,check=True)
    binary=LAB/"beglin_q_lab"
    subprocess.run(["clang","-O3",str(obj)]+[str(x) for x in other]+
                   ["-o",str(binary),"-framework","Accelerate","-lpthread"],check=True)
    binary.chmod(0o700)
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(SNAP/"tokenizer.json"))
    # Only public synthetic input; no 300 anchors / 600 labeled / 200 C1-D dev.
    texts=(
       "<|im_start|>user\nHi.<|im_end|>\n<|im_start|>assistant\n",
       "<|im_start|>user\nGive a simple greeting.<|im_end|>\n<|im_start|>assistant\n",
    )
    ids=[]
    for text in texts:
        tokens=tok.encode(text,add_special_tokens=False).ids
        if not 3<=len(tokens)<=32 or any(x<0 or x>=151936 for x in tokens):
            raise RuntimeError("invalid public synthetic prompt encoding")
        ids.append(tokens)
    import struct
    for tier in TIERS:
        p=LAB/tier
        p.mkdir(mode=0o700)
        (p/"results").mkdir(mode=0o700)
        (p/"eval").mkdir(mode=0o700)
        (p/"eval/ppl_work").mkdir(mode=0o700)
        write_private(p/"prompt_ids.i32",struct.pack("<"+"i"*len(ids[0]),*ids[0]))
        for idx,seq in enumerate(ids):
            write_private(p/f"eval/ppl_work/w{idx}.i32",struct.pack("<"+"i"*len(seq),*seq))
    metadata={
      "schema":"oml-acs-beqlin-Q-fullmodel-lab-source/1",
      "model":"Qwen/Qwen2.5-1.5B-Instruct",
      "official_model_revision":OFFICIAL_REV,
      "local_BF16_revision":MODEL_REV,
      "original_BEGILN_source_sha256":EXPECT_SOURCE_SHA,
      "lab_modified_source_sha256":sha(source.encode()),
      "binary_sha256":sha(binary.read_bytes()),
      "source_q_proj_bf16_sha256":MODEL_QPROJ_SHA,
      "all_precision_tiers_use_same_BF16_model_and_Q8_head":True,
      "tiers":list(TIERS),
      "baseline":"Original BF16-sourced projection weights widened to F32 BLAS; common lm_head Q8",
      "n8":"Original BF16-sourced per projection qNg64 bitplane n8; common lm_head Q8",
      "q4":"Original BF16-sourced per projection Q4G64 error-feedback nibble; common lm_head Q8",
      "synthetic_prompt_lengths":[len(x) for x in ids],
      "synthetic_prompt_id_shas":[sha(struct.pack("<"+"i"*len(x),*x)) for x in ids],
      "model_checkpoint_not_modified":True,
      "main_not_modified":True,"production_touched":False,
      "caveats":["NOT true BF16 arithmetic baseline: F32 projections widened from pinned BF16 checkpoint", "Cannot infer task AUROC, energy or final hypothesis Q from two synthetic prompts", "Common Q8 head is held constant across all tiers"]
    }
    write_private(LAB/"lab_metadata.json",(json.dumps(metadata,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_Q_FULLMODEL_LAB_PREPARED",json.dumps(metadata,sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prepare",action="store_true",required=True)
    args=p.parse_args()
    prepare()
