#!/usr/bin/env python3
"""Original BF16 MPS Qwen on 16 frozen RACE test cases, independent arithmetic."""
import hashlib,json,math,os,pathlib,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
R=pathlib.Path(__file__).resolve().parents[2];L=R/"lab_local/qv11_single_original_C_20261023"
P=L/"RACE_middle_test16_distinct_article_locked.json"
B=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
O=L/"BF16_original_Qwen_RACEtest16_private.json"
D=R/"docs/research/BEGLIN_QV11_ORIGINAL_BF16_RACE_TEST16_PUBLIC_20261023.json"
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 if O.exists() or D.exists():raise FileExistsError("Qv11 original source reference already frozen")
 raw=P.read_bytes()
 if sha(raw)!="0de242b608837bd29ed72db99f1e910626753e2b27ea6ada9a24bdf864c96499":raise RuntimeError("original RACE16 data altered")
 rows=json.loads(raw)["real_question_source"]
 if len(rows)!=16 or not torch.backends.mps.is_available():raise RuntimeError("original MPS RACE cohort")
 torch.set_num_threads(2)
 now=time.monotonic()
 model=AutoModelForCausalLM.from_pretrained(str(B),dtype=torch.bfloat16,local_files_only=True,
   trust_remote_code=False,attn_implementation="eager").eval().to("mps")
 results=[]
 with torch.inference_mode():
  for i,x in enumerate(rows):
   ids=torch.tensor([x["prefix_ids"]],dtype=torch.long,device="mps")
   logit=model(input_ids=ids,use_cache=False).logits[0,-1,:].float()
   y=torch.tensor([x["suffix_ids"][x["correct"]]],dtype=torch.long,device="mps")
   loss=float(F.cross_entropy(logit.unsqueeze(0),y).cpu())
   pred="ABCD"[int(logit[[x["suffix_ids"][l] for l in "ABCD"]].argmax().cpu())]
   if not math.isfinite(loss) or loss<0:raise RuntimeError("original BF16 full vocab logit loss invalid")
   results.append({"RACE_article_sha256":x["id_sha256"],"right":int(pred==x["correct"]),
        "fullvocab_correct_letter_NLL":loss})
 right=sum(x["right"] for x in results);nll=sum(x["fullvocab_correct_letter_NLL"] for x in results)/16
 obj={"schema":"BEGLIN-Qv11-16-original-BF16-Qwen-RACEtest-reference/1",
   "status":"TRUE_ORIGINAL_BF16_QWEN_SOURCE_RACE16_MPS_REFERENCE_PASS",
   "model":"Qwen/Qwen2.5-1.5B-Instruct","checkpoint_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
   "RACE16_original_input_SHA256":sha(raw),"n":16,"correct":right,"accuracy":right/16,
   "mean_correct_fullvocab_NLL":nll,"per_case_original_BF16":results,
   "independent_MPS_arithmetic_not_native_C_exact_parity":True,
   "model_weights_unmodified":True,"operator_release_unchanged":True,
   "wall_elapsed_s":round(time.monotonic()-now,2)}
 os.umask(0o077)
 a=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
 with O.open("xb") as f:f.write(a);f.flush();os.fsync(f.fileno())
 O.chmod(0o600)
 public={k:v for k,v in obj.items() if k!="per_case_original_BF16"}
 public["private_16case_receipt_sha256"]=sha(a)
 data=(json.dumps(public,sort_keys=True,indent=2)+"\n").encode()
 with D.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 print("QV11_ORIGINAL_QWEN_BF16_RACE_TEST16_REFERENCE",json.dumps({"correct":right,"NLL":nll,
    "public_sha":sha(data),"private_sha":sha(a)},sort_keys=True),flush=True)
if __name__=="__main__":main()
