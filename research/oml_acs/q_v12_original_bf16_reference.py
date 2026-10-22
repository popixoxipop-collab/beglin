#!/usr/bin/env python3
"""Unmodified original BF16 Qwen MPS reference for prospective new RACE16.

Same frozen distinct full articles and one-letter answer token IDs as genuine
BEGLIN C Q4/1n8/2n8. Different backend arithmetic; not used as direct bit
intervention baseline and no exact HF-vs-C numerical parity claim.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv12_twotensor_20261023"
SRC=LAB/"new_RACE_middle_test16_disjoint_Qv11_frozen.json"
PUB=ROOT/"docs/research/BEGLIN_QV12_ORIGINAL_BF16_NEW_RACE16_REFERENCE_PUBLIC_20261023.json"
OUT=LAB/"original_HF_BF16_MPS_Qwen_new_RACEtest16_private.json"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PIN="0650f791937f0d924706db6462b7f5e370fc58c1819b1228e91402d4937ca63e"
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
 if PUB.exists() or OUT.exists():raise FileExistsError("Qv12 original model already evaluated; no override")
 if not SRC.is_file() or sha(SRC.read_bytes())!=PIN:
  raise RuntimeError("Qv12 frozen new RACE16 data altered")
 samples=json.loads(SRC.read_text())["full_frozen_questions"]
 if len(samples)!=16 or len(set(q["article_original_id"] for q in samples))!=16:
  raise RuntimeError("not 16 independent original RACE articles")
 if not torch.backends.mps.is_available():raise RuntimeError("original Qwen BF16 Apple MPS unavailable")
 torch.set_num_threads(2)
 start=time.monotonic()
 model=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,
   local_files_only=True,trust_remote_code=False,attn_implementation="eager").eval().to("mps")
 rows=[]
 with torch.inference_mode():
  for i,q in enumerate(samples):
   ids=torch.tensor([q["prefix_ids"]],dtype=torch.long,device="mps")
   logits=model(input_ids=ids,use_cache=False).logits[0,-1,:].float()
   label=torch.tensor([q["suffix_ids"][q["correct"]]],dtype=torch.long,device="mps")
   loss=float(F.cross_entropy(logits.unsqueeze(0),label).cpu())
   predicted="ABCD"[int(logits[[q["suffix_ids"][k] for k in "ABCD"]].argmax().cpu())]
   if not math.isfinite(loss) or loss<0:raise RuntimeError("original BF16 RACE case finite output missing")
   rows.append({"article_SHA256":sha(q["article_original_id"].encode()),
       "correct":q["correct"],"predicted":predicted,"right":int(predicted==q["correct"]),
       "full_vocab_correct_label_NLL":loss})
   if (i+1)%4==0:print("QV12_ORIGINAL_BF16_REF_NEW_RACE",i+1,"/16",flush=True)
 right=sum(x["right"] for x in rows);nll=sum(x["full_vocab_correct_label_NLL"] for x in rows)/16
 info={"schema":"BEGLIN-Qv12-unmodified-HF-BF16-MPS-model-Qwen-new-RACE16/1",
  "status":"REFERENCE_ORIGINAL_BF16_QWEN_NEW_DISJOINT_RACE16_MODEL_COMPLETE",
  "original_checkpoint_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  "frozen_new_RACEtest16_input_SHA256":PIN,
  "n":16,"correct":right,"accuracy":right/16,"mean_fullvocab_correct_answer_NLL":nll,
  "model_original_BF16_per_case_outputs":rows,
  "different_arithmetic_from_original_BEGLIN_C_Q4_widened_F32":True,
  "not_sustained_C_CPU_speed_measurement":True,"original_model_unmodified":True,
  "operator_release_unchanged":True,
  "elapsed_wall_s":round(time.monotonic()-start,2)}
 os.umask(0o077)
 raw=(json.dumps(info,sort_keys=True,indent=2)+"\n").encode()
 with OUT.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 OUT.chmod(0o600)
 public={k:v for k,v in info.items() if k!="model_original_BF16_per_case_outputs"}
 public["private_original_reference_receipt_SHA256"]=sha(raw)
 blob=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
 with PUB.open("xb") as f:f.write(blob);f.flush();os.fsync(f.fileno())
 print("QV12_HF_BF16_NEW_TEST16_REF_SUCCESS",json.dumps({
    "correct":right,"NLL":nll,"public_sha":sha(blob),"private_sha":sha(raw)},sort_keys=True),flush=True)
if __name__=="__main__":main()
