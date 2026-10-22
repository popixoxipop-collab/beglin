#!/usr/bin/env python3
"""Original unmodified BF16 Qwen Apple MPS baseline on SHA-frozen RACE48.

Separate software arithmetic from native C Q4/n8; do not claim byte-level
numeric parity or quantify quantization's causal effect solely from this ref.
"""
from __future__ import annotations
import hashlib,json,math,os,pathlib,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv10_topk_trust_20261023"
DATA=LAB/"RACE_middle_validation_48_locked.json"
POL=ROOT/"docs/research/BEGLIN_QV10_TOPK_OTHER_TRUST_SOURCE_POLICY_20261023.json"
MODEL=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRIVATE=LAB/"original_Qwen_BF16_MPS_RACE48_reference_private.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV10_ORIGINAL_BF16_RACE48_REFERENCE_PUBLIC_20261023.json"
DATA_SHA="acdfd36477df4f440eeb9d03ad68342a34ce2435028eff1b8cef0de4c9f859de"
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 if PRIVATE.exists() or PUBLIC.exists():raise FileExistsError("BF16 original RACE48 reference already sealed")
 if sha(DATA.read_bytes())!=DATA_SHA or not POL.exists():
  raise RuntimeError("new RACE48 original model reference before source-only policy freeze")
 pol=json.loads(POL.read_text())
 if pol["status"]!="SOURCE_ONLY_REAL_TOPK_OTHER_GROUPED_KL_28N8_MASK_FROZEN_BEFORE_DIAG_OR_RACE":
  raise RuntimeError("source topK+OTHER precision mask was not frozen")
 questions=json.loads(DATA.read_text())["questions"]
 if len(questions)!=48 or len({x["passage_original_id"] for x in questions})!=48:
  raise RuntimeError("RACE48 one passage each source lock changed")
 if not torch.backends.mps.is_available():raise RuntimeError("original Qwen BF16 Apple MPS required")
 torch.set_num_threads(2)
 t=time.monotonic()
 model=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,
   local_files_only=True,trust_remote_code=False,attn_implementation="eager").eval().to("mps")
 scores=[]
 with torch.inference_mode():
  for i,q in enumerate(questions):
   x=torch.tensor([q["prompt_token_ids"]],dtype=torch.long,device="mps")
   final=model(input_ids=x,use_cache=False).logits[0,-1,:].float()
   y=torch.tensor([q["option_ids"][q["correct"]]],dtype=torch.long,device="mps")
   nll=float(F.cross_entropy(final.unsqueeze(0),y).cpu())
   candidate_ids=[q["option_ids"][letter] for letter in "ABCD"]
   guess="ABCD"[int(final[candidate_ids].argmax().cpu())]
   if not math.isfinite(nll) or nll<0:raise RuntimeError("original Qwen RACE48 BF16 nonfinite")
   scores.append({"case_id_SHA256":sha(q["id"].encode()),"article_id_SHA256":sha(q["passage_original_id"].encode()),
      "correct_letter":q["correct"],"predicted_letter":guess,
      "right":guess==q["correct"],"correct_letter_fullvocab_NLL":nll})
   if (i+1)%12==0:print("QV10_QWEN_BF16_RACE48_REFERENCE",i+1,"/48",flush=True)
 right=sum(r["right"] for r in scores)
 mean=sum(r["correct_letter_fullvocab_NLL"] for r in scores)/48
 receipt={"schema":"BEGLIN-Qv10-unchanged-BF16-Qwen-RACE48-original-model-reference/1",
  "status":"REAL_ORIGINAL_BF16_MODEL_RACE48_REFERENCE_FINISHED",
  "source_model_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  "source_RACE48_SHA256":DATA_SHA,"source_only_Top2OTHER_mask_SHA256":pol["mask_sha256"],
  "total":48,"correct":right,"accuracy":right/48,
  "full_vocab_correct_letter_mean_NLL":mean,
  "per_question_model_outputs":scores,"Apple_MPS_original_BF16_model_unchanged":True,
  "native_C_widened_FP32_ne_not_identical":True,
  "no_operational_promotion":True,"elapsed_s":round(time.monotonic()-t,2)}
 os.umask(0o077)
 priv=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode()
 with PRIVATE.open("xb") as f:f.write(priv);f.flush();os.fsync(f.fileno())
 PRIVATE.chmod(0o600)
 public={k:v for k,v in receipt.items() if k!="per_question_model_outputs"}
 public["original_private_ref_SHA256"]=sha(priv)
 raw=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
 with PUBLIC.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 print("QV10_ORIGINAL_BF16_RACE48_REFERENCE_PASS",
       json.dumps({"correct":right,"NLL":mean,"public_sha":sha(raw),"private_sha":sha(priv)},sort_keys=True),flush=True)
if __name__=="__main__":main()
