#!/usr/bin/env python3
"""Independent original BF16 Transformers MPS reference, frozen OpenBookQA48.

Only runs after Qv6 64/128 policies have been source-only finalized, cannot
affect bit selection. Same exact frozen original Qwen A/B/C/D prefix-token
strings as actual BEGLIN C. Never mutates base BF16 model weights.
"""
import hashlib,json,math,os,pathlib,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv6_fisher_scaling_20261022"
DATA=LAB/"openbookqa_val48_frozen.json"
POL=ROOT/"docs/research/BEGLIN_QV6_NESTED_64_128_SOURCE_POLICY_20261022.json"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
PRIVATE=LAB/"reference_original_BF16_MPS_OpenBookQA48.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV6_ORIGINAL_BF16_OPENBOOKQA48_REFERENCE_20261022.json"
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
    if PRIVATE.exists() or PUBLIC.exists():raise FileExistsError("reference already sealed")
    if sha(DATA.read_bytes())!="c09831bf86c6eedebed433dc2a0398a0992afb19fddc9f86750c2d137d71d1b0":
        raise RuntimeError("frozen cross-corpus inputs tampered")
    if not POL.is_file() or json.loads(POL.read_text())["status"]!="FROZEN_64_128_POLICIES_BEFORE_NEW_OPENBOOKQA_VALIDATION":
        raise RuntimeError("no BF16 held scoring before source-only policies frozen")
    frozen=json.loads(DATA.read_text())["unseen_questions"]
    if len(frozen)!=48:raise RuntimeError("not fixed 48 OpenBookQA inputs")
    if not torch.backends.mps.is_available():raise RuntimeError("true BF16 MPS reference unavailable")
    torch.set_num_threads(2)
    t=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(str(SNAP),dtype=torch.bfloat16,
        local_files_only=True,trust_remote_code=False,attn_implementation="eager").eval().to("mps")
    results=[]
    with torch.inference_mode():
        for i,c in enumerate(frozen):
            x=torch.tensor([c["prefix_ids"]],dtype=torch.long,device="mps")
            logits=model(input_ids=x,use_cache=False).logits[0,-1,:].float()
            target=torch.tensor([c["suffix_ids"][c["correct"]]],dtype=torch.long,device="mps")
            val=float(F.cross_entropy(logits.unsqueeze(0),target).cpu())
            predicted="ABCD"[int(logits[[c["suffix_ids"][ch] for ch in "ABCD"]].argmax().cpu())]
            if not math.isfinite(val) or val<0:raise RuntimeError("original BF16 model output nonfinite")
            results.append({"question_id_sha":sha(c["id"].encode()),
                "correct":c["correct"],"predicted":predicted,"NLL":val,"correct_prediction":predicted==c["correct"]})
            if (i+1)%12==0:
                print("QV6_BF16_REFERENCE_PROGRESS",i+1,"/48",flush=True)
    acc=sum(x["correct_prediction"] for x in results);nll=sum(x["NLL"] for x in results)/48
    obj={"schema":"BEGLIN-Qv6-independent-original-BF16-MPS-OpenBookQA48/1",
      "status":"PASS_UNMODIFIED_ORIGINAL_BF16_MODEL_OODISH_SCIENCE_CROSSCORPUS",
      "model_HF_source_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "dataset_revision":"388097ea7776314e93a529163e0fea805b8a6454",
      "frozen_crosscorpus_data_SHA256":sha(DATA.read_bytes()),
      "frozen_source_only_MILP_SHA256":sha(POL.read_bytes()),
      "correct":acc,"total":48,"accuracy":acc/48,
      "original_BF16_correct_oneletter_NLL":nll,"per_case_results":results,
      "framework":"Transformers BF16 on MPS",
      "different_arithmetic_than_BEGLIN_widened_F32":True,
      "weights_mutated":False,"production_mutated":False,
      "elapsed_s":round(time.monotonic()-t,2)}
    b=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
    os.umask(0o077)
    with PRIVATE.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    PRIVATE.chmod(0o600)
    pub={k:v for k,v in obj.items() if k!="per_case_results"}
    pub["private_receipt_SHA256"]=sha(b)
    pb=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(pb);f.flush();os.fsync(f.fileno())
    print("QV6_INDEPENDENT_BF16_OPENBOOKQA48_PASS",json.dumps({"correct":acc,"NLL":nll,"public_sha":sha(pb),"private_sha":sha(b)},sort_keys=True),flush=True)
if __name__=="__main__":run()
