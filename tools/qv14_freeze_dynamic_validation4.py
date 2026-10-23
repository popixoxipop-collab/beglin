#!/usr/bin/env python3
"""BEGLIN Qv14 immutable ORIGINAL-ARC-validation mechanistic 4-prompt freeze.
No model forward, label-based selection, tuning, or operational release changes.
"""
import hashlib,json,os,pathlib,struct
from datasets import Dataset
from tokenizers import Tokenizer
ROOT=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023")
OUT=LAB/"mechanistic_arc_validation4"
ARROW=pathlib.Path("/Users/eoe/.cache/huggingface/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-validation.arrow")
TOKEN=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/tokenizer.json")
SOURCE=ROOT/"configs/research/BEGLIN_QV8_JOINT_SIGNED_28N8_SOURCE_POLICY_20261022.txt"
RANDOM73=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023/masks/random73.txt")
SHAS={"arrow":"866e13e71ac78fbb940387242c7142dc0e1b6c353c792e566494aca96779a904",
"token":"a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
"source":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
"random":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
THRESHOLD=85
def sha(b):return hashlib.sha256(b).hexdigest()
def shaf(p):return sha(p.read_bytes())
def sealed(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def obj(o):return (json.dumps(o,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
def row(r,tok):
 labels=r["choices"]["label"];texts=r["choices"]["text"]
 if not len(labels)==len(texts)==4 or len(set(labels))!=4 or r["answerKey"] not in labels:return None
 stem=r["question"]
 text=("Choose the correct answer.\nQuestion: "+stem.strip()+"\n"+"\n".join(f"{k}. {v.strip()}" for k,v in zip("ABCD",texts))+"\nAnswer using one letter A, B, C, or D.")
 prefix="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
 ids=tok.encode(prefix,add_special_tokens=False).ids
 if not 18<=len(ids)<=208:return None
 suff={c:tok.encode(prefix+c,add_special_tokens=False).ids for c in "ABCD"}
 if any(len(t)!=len(ids)+1 or t[:len(ids)]!=ids for t in suff.values()):return None
 if len(set(s[-1] for s in suff.values()))!=4:return None
 return {"original_source_id":r["id"],"question_sha256":sha(stem.encode()),
  "prompt_sha256":sha(text.encode()),"selection_key":sha(("ARC-validation/Qv14/"+r["id"]+"/"+sha(stem.encode())).encode()),
  "prefix_ids":ids,"suffix_ids":{c:s[-1] for c,s in suff.items()},
  "correct":"ABCD"[labels.index(r["answerKey"])],"prefix_len":len(ids)}
def freeze():
 if OUT.exists():raise FileExistsError("Qv14 original validation4 freeze already exists")
 for key,p in [("arrow",ARROW),("token",TOKEN),("source",SOURCE),("random",RANDOM73)]:
  if shaf(p)!=SHAS[key]:raise RuntimeError("Qv14 source identity mismatch "+key)
 os.umask(0o077)
 LAB.mkdir(parents=True,exist_ok=True,mode=0o700)
 OUT.mkdir(mode=0o700)
 tok=Tokenizer.from_file(str(TOKEN))
 allrows=Dataset.from_file(str(ARROW))
 eligible=[]
 for r in allrows:
  m=row(r,tok)
  if m is not None:eligible.append(m)
 short=sorted((m for m in eligible if m["prefix_len"]<THRESHOLD),key=lambda m:m["selection_key"])[:2]
 long=sorted((m for m in eligible if m["prefix_len"]>=THRESHOLD),key=lambda m:m["selection_key"])[:2]
 if len(short)!=2 or len(long)!=2:raise RuntimeError("insufficient disjoint short long inputs")
 selected=[short[0],long[0],short[1],long[1]]
 if len(set(x["original_source_id"] for x in selected))!=4:raise RuntimeError("not four unique inputs")
 selections=["source_qv8","random73","source_qv8","random73"]
 for mode in ("dynamic","static_source","static_random"):
  for i,x in enumerate(selected):
   base=OUT/mode
   prefix=struct.pack("<"+"i"*len(x["prefix_ids"]),*x["prefix_ids"])
   answer=(" ".join(str(x["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(x["correct"]))+"\n").encode()
   sealed(base/"eval/ppl_work"/f"w{i}.i32",prefix)
   sealed(base/"eval/ppl_work"/f"w{i}.choices",answer)
   if i==0:sealed(base/"prompt_ids.i32",prefix)
 for k,p in (("source_qv8",SOURCE),("random73",RANDOM73)):
  sealed(OUT/"masks"/f"{k}.txt",p.read_bytes())
 private={"schema":"BEGLIN-Qv14-4-frozen-ARC-validation-requests-1",
 "status":"FROZEN_BEFORE_QV14_C_NATIVE_FORWARD",
 "original_arrow_sha256":SHAS["arrow"],"original_tokenizer_sha256":SHAS["token"],
 "mask_sha256":{"source_qv8":SHAS["source"],"random73":SHAS["random"]},
 "threshold_tokens":THRESHOLD,"selection_rule":"n_prefix_tokens < 85 -> source_qv8; >=85 -> random73",
 "selection_hypothesis":"ENGINE_MECHANISM_ONLY, NO_ACCURACY_CLAIM, NOT_OUTPUT_OPTIMIZED",
 "rows":selected,"assigned_masks":selections,
 "calibration_or_prior_test_labels_not_used_for_mask_selection":True}
 privbytes=obj(private)
 sealed(OUT/"frozen_inputs_private.json",privbytes)
 manifest={"schema":"BEGLIN-Qv14-native-in-process-dynamic-ARC-val4-preexec-PUBLIC-1",
 "status":"SEALED_4_REQUESTS_AND_TWO_FIXED_MASKS_BEFORE_NATIVE_EXECUTION",
 "dataset":"ARC-Challenge validation (not Qv13 test)",
 "original_dataset_rows":len(allrows),"eligible_rows":len(eligible),"n":4,
 "request_selection":"two <85 and two >=85, SHA-sorted by source-only question/id; alternating short/long",
 "input_length_tokens":[x["prefix_len"] for x in selected],
 "input_id_order_sha256":sha(obj([x["original_source_id"] for x in selected])),
 "private_input_sha256":sha(privbytes),
 "mask_sha256":private["mask_sha256"],"source_Qv8_and_random73_same_projected_bytes":765493248,
 "short_mask":"source_qv8","long_mask":"random73","threshold":THRESHOLD,
 "actual_inprocess_switch_not_run_yet":True,
 "not_a_model_accuracy_generalization_experiment":True,
 "source_model_and_operator_release_untouched":True}
 pubbytes=obj(manifest)
 sealed(OUT/"frozen_manifest_PUBLIC.json",pubbytes)
 print("QV14_PRELOCKED_INPUTS",json.dumps({"private_sha256":sha(privbytes),"public_sha256":sha(pubbytes),
 "input_lengths":manifest["input_length_tokens"],"expected_masks":selections,
 "eligible":len(eligible),"path":str(OUT)},sort_keys=True),flush=True)
if __name__=="__main__":freeze()
