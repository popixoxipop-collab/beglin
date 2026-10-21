#!/usr/bin/env python3
"""Real Qwen2.5-1.5B BF16 L0 q_proj fixture for native BEGLIN Q benchmark.

Extracts ORIGINAL BF16 q_proj, input_layernorm, and token embedding vectors
from an already-cached, SHA-pinned MLX-community BF16 safetensors copy.
The selected tensors are byte-identical to the official Qwen revision used
in earlier 300-anchor BF16 research. This is a single-tensor experiment, not
full-LLM behavior AUROC. No checkpoint, main or production writes.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, struct
import numpy as np
from tokenizers import Tokenizer

BF16_SNAPSHOT=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
OFFICIAL_REV="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LOCAL_REV="4ae77cb209f06199b8df1c94e21ff341332a3a89"
TARGET="model.layers.0.self_attn.q_proj.weight"
NORM="model.layers.0.input_layernorm.weight"
EMB="model.embed_tokens.weight"
EXPECTED_W_SHA="527bf1760462c181be3ab71e3c394cdf50283a4df6b5d80d0bd2a5ab0b7eded5"
EXPECTED_NORM_SHA="9150388359dd3b4221e9c863ed8d5d54dd8284ad496766d698d4310a13a2ea3a"
PROMPTS=[
    "The movie was engaging and thoughtful, with beautiful cinematography.",
    "The movie was boring and disappointing, with confusing scenes.",
    "A lovely bakery made our morning much brighter than expected.",
    "I found the presentation clear, engaging, and easy to follow.",
    "The trip was uncomfortable because of delayed trains and rain.",
    "A scientist studies quantum physics and writes careful experiments.",
    "Machine learning systems depend on reproducible model evaluations.",
    "Input dependent bit allocation requires uncertainty measurements.",
    "We measured numerical reconstruction error across hundreds of inputs.",
    "An efficient packed kernel should avoid expanding an entire matrix.",
    "This restaurant serves a colorful salad and a delicious soup.",
    "The customer was unhappy with the defective headphones and delays.",
    "Why do different neural network layers require different precision?",
    "A calm conversation about history, literature, and mathematics.",
    "Several students are learning to program and analyze data.",
    "The engineer compared an 8 bit decoder with a packed 4 bit kernel.",
    "Optimization can improve memory traffic but degrade model accuracy.",
    "A reliable evaluation should preserve failures and negative results.",
    "The city council announced plans for a new public library.",
    "A small dog chased a red ball through a quiet park.",
    "Please summarize the article in one sentence without speculation.",
    "How much energy does a fast large language model inference use?",
    "The latest model checkpoint was pinned to a specific Git revision.",
    "Every layer's activations should be independently audited for drift.",
]
N_INPUTS=96

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def convert_bf16(data:bytes)->np.ndarray:
    bits=np.frombuffer(data,dtype="<u2")
    return (bits.astype("<u4")<<16).view("<f4").copy()

def save_new(path:pathlib.Path,data:bytes):
    with path.open("xb") as fd:
        fd.write(data)
        fd.flush()
        os.fsync(fd.fileno())
    path.chmod(0o600)

def main():
    os.umask(0o077)
    out=pathlib.Path(__file__).resolve().parents[2]/"lab_local"/"q_apple_cpu_20261021"
    if out.exists():
        raise RuntimeError("fixture directory exists; do not overwrite")
    out.mkdir(parents=True,mode=0o700)
    src=BF16_SNAPSHOT/"model.safetensors"
    with src.open("rb") as fp:
        hsize=struct.unpack("<Q",fp.read(8))[0]
        if not 1000<hsize<5_000_000:raise RuntimeError("safetensors header invalid")
        h=json.loads(fp.read(hsize))
        offset=8+hsize
        for name in (TARGET,NORM,EMB):
            if name not in h or h[name]["dtype"]!="BF16":raise RuntimeError("missing/non-BF16 tensor "+name)
        def read_tensor(name):
            start,end=h[name]["data_offsets"]
            fp.seek(offset+start)
            b=fp.read(end-start)
            if len(b)!=end-start:raise RuntimeError("tensor bytes truncated "+name)
            return b
        raw_w=read_tensor(TARGET)
        raw_norm=read_tensor(NORM)
        if sha(raw_w)!=EXPECTED_W_SHA or sha(raw_norm)!=EXPECTED_NORM_SHA:
            raise RuntimeError("official reference BF16 tensor hash mismatch")
        w=convert_bf16(raw_w).reshape(1536,1536)
        gamma=convert_bf16(raw_norm)
        assert gamma.shape==(1536,)
        config=json.loads((BF16_SNAPSHOT/"config.json").read_text())
        eps=float(config["rms_norm_eps"])
        assert config["hidden_size"]==1536 and config["num_hidden_layers"]==28
        tokenizer=Tokenizer.from_file(str(BF16_SNAPSHOT/"tokenizer.json"))
        ids=[]
        for text in PROMPTS:
            for i in tokenizer.encode(text,add_special_tokens=False).ids:
                if 0<=i<h[EMB]["shape"][0] and i not in ids:
                    ids.append(i)
        if len(ids)<N_INPUTS:raise RuntimeError("insufficient distinct real prompt token ids")
        ids=ids[:N_INPUTS]
        emb_start,emb_end=h[EMB]["data_offsets"]
        X=[]
        for token_id in ids:
            fp.seek(offset+emb_start+token_id*1536*2)
            bb=fp.read(1536*2)
            if len(bb)!=1536*2:raise RuntimeError("embedding row truncated")
            vector=convert_bf16(bb)
            x=(vector / np.sqrt(np.mean(vector*vector,dtype=np.float32)+eps))*gamma
            X.append(x)
        X=np.stack(X).astype(np.float32)
    assert X.shape==(N_INPUTS,1536) and w.shape==(1536,1536)
    assert np.isfinite(X).all() and np.isfinite(w).all()
    save_new(out/"q_proj_weight.f32",w.tobytes(order="C"))
    save_new(out/"layer0_token_norm_inputs.f32",X.tobytes(order="C"))
    metadata={
        "schema":"oml-acs-q-native-real-tensor-fixture/1",
        "model":"Qwen/Qwen2.5-1.5B-Instruct",
        "official_revision":OFFICIAL_REV,"local_mlxbfloat16_snapshot":LOCAL_REV,
        "source_BF16_qproj_raw_sha256":EXPECTED_W_SHA,"source_BF16_norm_raw_sha256":EXPECTED_NORM_SHA,
        "tensor_name":TARGET,"shape":[1536,1536],"dataset":"synthetic public text/tokenization only",
        "n_token_activation_vectors":N_INPUTS,"input_shape":[N_INPUTS,1536],
        "input_interpretation":"exact layer0 RMSNorm(embedding[token]) float32 formula, not multi-layer context",
        "token_ids_sha256":sha(np.asarray(ids,dtype=np.int32).tobytes()),
        "inputs_f32_sha256":sha(X.tobytes(order="C")),"weight_f32_sha256":sha(w.tobytes(order="C")),
        "rmsnorm_eps":eps,"group_size":64,
        "reference":"original BF16 weights widened to float32; not full-BF16 model forward",
        "production_touched":False,"main_touched":False,
        "source_prompts_are_not_private_c1d_data":True,
    }
    save_new(out/"fixture_public_metadata.json",(json.dumps(metadata,indent=2,sort_keys=True)+"\n").encode())
    print("Q_REAL_FIXTURE_PASS",json.dumps(metadata,sort_keys=True),flush=True)
if __name__=="__main__":main()
