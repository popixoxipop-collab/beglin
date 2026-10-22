#!/usr/bin/env python3
"""Freeze exact 10 NEW public evaluation prompt token windows (no model calls).

The public synthetic source strings were locked before Qv3 calibration in
BEGLIN_Q_V3_EMPIRICAL_COVARIANCE_PRELOCK_20261022.json. Original official
Qwen tokenizer and EOE local BF16 cache tokenizer were independently verified
to produce exactly these 10 token-array SHA-256 hashes, despite different
raw tokenizer file hashes. Do not open new model logits/NLL until Qv3 quality
policy is frozen.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,struct
import numpy as np
from tokenizers import Tokenizer
ROOT=pathlib.Path(__file__).resolve().parents[2]
PRELOCK=ROOT/"configs/research/BEGLIN_Q_V3_EMPIRICAL_COVARIANCE_PRELOCK_20261022.json"
TOK=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/tokenizer.json")
LAB=ROOT/"lab_local/q_v3_fresh_10dev_20261022"
EXPECTED_PROTO="5e8824f74b7d80dac191bf944a48d418f87212ca5ed53835f7ab7edf8eb0c267"
SOURCE_VERIFIED_IDS_SHA=[
"15d28832247c34c030e685b1e6beb5c8f24a41ccca63f280357c3cbbc0af59b9",
"d6c0340416cae2a2615a5558e0180d7c71e0768bcd28e531d80263967b8a8a31",
"b4a32753ba53b97583ef7ee2b1a80746f0a2617b01ab16a7c2125758c7203ef8",
"08713194b9ff32cc4e6645de9557170c9d806599586e1989f6b5847b25278bd9",
"464d06095db719a29917fea74f7b57b8e223ce4ca0fc8db238f48d63ff870e8d",
"db4223812ae69fa1115c3df58f506411fc5ecf04c719bd5dea96ed6356f53864",
"ab2b11aea26939028e4a677dacf3ad55d4444d66037d3fee919c260c63f759e6",
"b8a8566ad80a1bfbafa45fa7442d099b6febc0df3ced5c7281a1cee72e0ee019",
"ad4ff5dd88281ecc35fd8419686ede6e22f9426045c88582d0eab308e5bf7b8a",
"2ab67a7f08de6c0dde5b37ab6bdaa60dca2648a2a0fd604109317b67f4648be5"
]
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def private(p:pathlib.Path,b:bytes):
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def main():
    if LAB.exists():raise FileExistsError("never overwrite prelocked development input windows")
    if sha(PRELOCK.read_bytes())!=EXPECTED_PROTO:
        raise RuntimeError("preregistered fresh evaluation data changed")
    cfg=json.loads(PRELOCK.read_text())
    items=cfg["fresh_evaluation_prompts"]
    if len(items)!=10 or len(set(items))!=10 or not TOK.is_file():
        raise RuntimeError("missing source prelocked public evaluation or tokenizer")
    t=Tokenizer.from_file(str(TOK))
    os.umask(0o077)
    LAB.mkdir(parents=True,mode=0o700)
    directory=LAB/"eval/ppl_work"
    directory.mkdir(parents=True,mode=0o700)
    lengths=[]
    for i,text in enumerate(items):
        msg=f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        seq=t.encode(msg,add_special_tokens=False).ids
        blob=np.asarray(seq,dtype="<i4").tobytes()
        if sha(blob)!=SOURCE_VERIFIED_IDS_SHA[i] or not 8<=len(seq)<=80:
            raise RuntimeError("official / local cache token sequence not equivalent for prelocked entry "+str(i))
        private(directory/f"w{i}.i32",blob)
        lengths.append(len(seq))
    private(LAB/"prompt_ids.i32",(directory/"w0.i32").read_bytes())
    if len(set(SOURCE_VERIFIED_IDS_SHA))!=10:
        raise RuntimeError("duplicate prelocked evaluation input")
    receipt=dict(schema="BEGLIN-Q-v3-public-development-10-token-windows/1",
       status="PASS_LOCKED_PUBLIC_INPUTS_BEFORE_V3_EVALUATION",
       source_protocol_sha256=EXPECTED_PROTO,local_tokenizer_sha256=sha(TOK.read_bytes()),
       exact_official_HF_token_ids_sha256=SOURCE_VERIFIED_IDS_SHA,
       prompt_count=10,prompt_token_lengths=lengths,
       total_source_input_token_count=sum(lengths),
       teacher_forced_prediction_positions_expected_sum_input_lengths_minus_10=sum(lengths)-10,
       same_native_windows_required_for_all_precision_policies=True,
       private_file_modes="0600",
       model_original_BF16_and_main_untouched=True,
       source_old_V6_test_not_opened=True)
    private(LAB/"pre_eval_input_receipt.json",(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
    print("BEGLIN_Q3_PUBLIC_10_INPUTS_FROZEN",json.dumps({
      "sha256":sha((LAB/"pre_eval_input_receipt.json").read_bytes()),
      "source_prompt_count":10,"expected_prediction_positions":receipt["teacher_forced_prediction_positions_expected_sum_input_lengths_minus_10"],
      "token_lengths":lengths
    },sort_keys=True),flush=True)
if __name__=="__main__":main()
