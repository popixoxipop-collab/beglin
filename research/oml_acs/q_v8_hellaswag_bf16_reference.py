#!/usr/bin/env python3
"""Independent original BF16 Qwen Transformers reference on new HellaSwag48.

After Qv8 source-only signed joint mask freeze, computes original unchanged
BF16 model correctness and full-vocabulary correct-letter NLL on identical
48 frozen candidate inputs. BF16 vs native widened-F32 may differ numerically.
"""
import hashlib,json,math,os,pathlib,time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
ROOT=pathlib.Path(__file__).resolve().parents[2]
LAB=ROOT/"lab_local/qv8_joint_signed_20261022"
HELD=LAB/"pinned_HellaSwag_val48.json"
POL=ROOT/"docs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_OPTIMIZER_20261022.json"
MASK=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
SNAP=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89")
OUT=LAB/"original_BF16_Qwen_HellaSwag48_oracle_private.json"
PUBLIC=ROOT/"docs/research/BEGLIN_QV8_ORIGINAL_BF16_HELLASWAG48_REFERENCE_20261022.json"
HELD_SHA="45c95170a2eb01b4c79ad44719ab9a1877e568f2583a90d84340d7448e0d1a33"
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
    if OUT.exists() or PUBLIC.exists():raise FileExistsError("original BF16 oracle results already sealed")
    if sha(HELD.read_bytes())!=HELD_SHA or not POL.exists():
        raise RuntimeError("HellaSwag original input identity or source-frozen policy absent")
    pol=json.loads(POL.read_text())
    if pol["status"]!="SOURCE_ONLY_28N8_STRICT_FEASIBLE_LOCAL_SWAP_DONE_NO_EVAL_OPENED" or \
       pol["mask_sha256"]!=sha(MASK.read_bytes()):
        raise RuntimeError("not source-only Qv8 policy before answer-label scoring")
    val=json.loads(HELD.read_text())["held_validation48"]
    if len(val)!=48 or not torch.backends.mps.is_available():
        raise RuntimeError("not 48 official original Qwen BF16 MPS evaluations")
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    torch.set_num_threads(2);t=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(str(SNAP),dtype=torch.bfloat16,local_files_only=True,
       trust_remote_code=False,attn_implementation="eager").eval().to("mps")
    rows=[]
    with torch.inference_mode():
        for ix,q in enumerate(val):
            ids=torch.tensor([q["prefix_ids"]],dtype=torch.long,device="mps")
            logits=model(input_ids=ids,use_cache=False).logits[0,-1,:].float()
            target=torch.tensor([q["suffix_ids"][q["correct"]]],dtype=torch.long,device="mps")
            nll=float(F.cross_entropy(logits.unsqueeze(0),target).cpu())
            predicted="ABCD"[int(logits[[q["suffix_ids"][k] for k in "ABCD"]].argmax().cpu())]
            if not math.isfinite(nll) or nll<0:
                raise RuntimeError("BF16 original HellaSwag answer logits nonfinite")
            rows.append({"id_sha":sha(q["id"].encode()),"correct":q["correct"],
               "guess":predicted,"was_right":q["correct"]==predicted,"NLL":nll})
            if (ix+1)%12==0:print("QV8_ORIGINAL_BF16_HELLASWAG_PROGRESS",ix+1,"/48",flush=True)
    correct=sum(x["was_right"] for x in rows);mean=sum(x["NLL"] for x in rows)/48.
    rec={"schema":"BEGLIN-Qv8-original-BF16-Qwen-48-HellaSwag-standalone/1",
        "status":"REFERENCE_ORIGINAL_BF16_48_NARRATIVE_ANSWERS",
        "source_revision":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "dataset_revision":"218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "frozen_source_only_new_mask_SHA256":pol["mask_sha256"],
        "held_input_SHA256":HELD_SHA,"case_count":48,
        "correct_count":correct,"accuracy":correct/48,"mean_correct_answer_NLL":mean,
        "per_case":rows,"actual_model_forward_dtype":"BF16 MPS from unchanged original HF source",
        "no_original_model_weight_mutation":True,
        "no_native_CPU_numerical_parity_assertion":True,
        "no_live_engine_changed":True,"elapsed_seconds":round(time.monotonic()-t,2)}
    os.umask(0o077)
    b=(json.dumps(rec,sort_keys=True,indent=2)+"\n").encode()
    with OUT.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    OUT.chmod(0o600)
    p={k:v for k,v in rec.items() if k!="per_case"}
    p["private_original_BF16_receipt_SHA256"]=sha(b)
    pb=(json.dumps(p,sort_keys=True,indent=2)+"\n").encode()
    with PUBLIC.open("xb") as f:f.write(pb);f.flush();os.fsync(f.fileno())
    print("QV8_ORIGINAL_BF16_NEW_HELLASWAG48_PASS",json.dumps({"accuracy":correct/48,"mean_nll":mean,
         "public_sha":sha(pb),"private_sha":sha(b)},sort_keys=True),flush=True)
if __name__=="__main__":run()
