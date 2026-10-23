#!/usr/bin/env python3
"""Qv20 research freeze: distinct source-only calibration and no-peek evaluation.
Calibration ARC-Challenge TRAIN: 16 short and 16 long.
Evaluation: ARC-Challenge VALIDATION 12+12, excluding Qv14/Qv16's previous 76;
             OpenBookQA VALIDATION 12+12, not Qv13 test.
Only prior original Qv14/16 case identity exclusions are read; NO previous model
scores/logits or heldout evaluation answers are used for precision allocation.
"""
import hashlib,json,os,pathlib,struct
from datasets import Dataset
from tokenizers import Tokenizer
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv20-prompt-aware-20261024")
CACHE=pathlib.Path("/Users/eoe/.cache/huggingface/datasets")
TOKENIZER=pathlib.Path("/Users/eoe/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-bf16/snapshots/4ae77cb209f06199b8df1c94e21ff341332a3a89/tokenizer.json")
SRC=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023/mechanistic_arc_validation4/masks/source_qv8.txt")
RANDOM=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023/mechanistic_arc_validation4/masks/random73.txt")
Q14=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023")
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
ARROW={
 "cal_arc_train":CACHE/"allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-train.arrow",
 "test_arc_validation":CACHE/"allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-validation.arrow",
 "test_ob_validation":CACHE/"allenai___openbookqa/main/0.0.0/388097ea7776314e93a529163e0fea805b8a6454/openbookqa-validation.arrow",
}
KNOWN={
 "cal_arc_train":"2bb1e5e8875ff5944d3a6bf469ef6ea811217be8bc083abd01722ac62cdaa873",
 "test_arc_validation":"866e13e71ac78fbb940387242c7142dc0e1b6c353c792e566494aca96779a904",
 "test_ob_validation":"af0369b50bd3e28d2cb0260868df0ae981cfbd47390bf8f803a690da23e46b4d",
 "qv14_old4":"e12ba4ae07dba6476c4452357ce77b32f2a4c7ba64c5c1bb013e74cf1245e032",
 "qv14_old24":"e836a4bb6ea86c43f5a9c28ecbbc235c4c9d7fd306dfc006511027e3bddf3dc6",
 "qv16_old48":"a95f678f656b8d04f74f0bb643542cdd12bde8e317c8d7fa013795c4c4b3c296",
 "mask_source":"9db954db6ba690d92dd96ee1638b76afe53de2d637aab462b4e83de90cc35922",
 "mask_random":"5086bc9e1764edbfbcbeb7089d3a9aabba7978d773e904d600b2bfdc899b2fde"}
SEED="BEGLIN_QV20_20261024_NO_PEEK_FIXED_STRATA_SEED_v1"
MAX_PREFIX=208
def sha(b):return hashlib.sha256(b).hexdigest()
def filehash(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for z in iter(lambda:f.read(262144),b""):h.update(z)
 return h.hexdigest()
def check(ok,msg):
 if not ok:raise RuntimeError("QV20_FREEZE_FAIL_CLOSED_"+msg)
def enc(o):return (json.dumps(o,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
def atomic_new(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def eligible(ds,raw,tok):
 stem=raw["question_stem"] if ds=="test_ob_validation" else raw["question"]
 ch=raw["choices"];texts=ch["text"];labels=ch["label"]
 if len(texts)!=4 or len(labels)!=4 or len(set(labels))!=4 or raw["answerKey"] not in labels: return None
 if not stem.strip() or any(not t.strip() for t in texts):return None
 text="Choose the correct answer.\nQuestion: "+stem.strip()+"\n"+"\n".join(f"{k}. {v.strip()}" for k,v in zip("ABCD",texts))+"\nAnswer using one letter A, B, C, or D."
 prompt="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
 ids=tok.encode(prompt,add_special_tokens=False).ids
 if not 18<=len(ids)<=MAX_PREFIX:return None
 suff={}
 for k in "ABCD":
  z=tok.encode(prompt+k,add_special_tokens=False).ids
  if len(z)!=len(ids)+1 or z[:-1]!=ids:return None
  suff[k]=z[-1]
 if len(set(suff.values()))!=4:return None
 questionhash=sha(stem.encode())
 key=sha((SEED+"/"+ds+"/"+raw["id"]+"/"+questionhash).encode())
 return {"source_id":raw["id"],"question_sha256":questionhash,
 "full_prompt_sha256":sha(text.encode()),"selection_key":key,
 "prefix_ids":ids,"suffix_ids":suff,
 "correct":"ABCD"[labels.index(raw["answerKey"])],
 "prefix_len":len(ids)}
def freeze():
 if ROOT.exists():raise FileExistsError("Qv20 immutable freeze already exists; do not overwrite")
 old4=Q14/"mechanistic_arc_validation4/frozen_inputs_private.json"
 old24=Q14/"replication_arc_validation24/frozen_inputs_private.json"
 old48=Q16/"independent_arc_validation48/frozen_inputs_private.json"
 for n,p in [("qv14_old4",old4),("qv14_old24",old24),("qv16_old48",old48),
             ("mask_source",SRC),("mask_random",RANDOM)]:
  check(p.is_file() and filehash(p)==KNOWN[n],"source_id_"+n)
 old_ids={v["original_source_id"] for v in json.loads(old4.read_text())["rows"]+
          json.loads(old24.read_text())["exact_input_rows"]+json.loads(old48.read_text())["requests"]}
 check(len(old_ids)==76,"prior_76_heldout_val_ids")
 for ds,p in ARROW.items():check(filehash(p)==KNOWN[ds],"dataset_arrow_"+ds)
 os.umask(0o077);ROOT.mkdir(parents=True,exist_ok=False,mode=0o700)
 tok=Tokenizer.from_file(str(TOKENIZER))
 output={}
 seen_hashes=set();seen_ids=set()
 for ds,p in ARROW.items():
  d=Dataset.from_file(str(p))
  vals=[v for row in d if (v:=eligible(ds,row,tok)) is not None]
  if ds=="test_arc_validation":vals=[v for v in vals if v["source_id"] not in old_ids]
  short=sorted((x for x in vals if x["prefix_len"]<85),key=lambda x:x["selection_key"])
  long=sorted((x for x in vals if x["prefix_len"]>=85),key=lambda x:x["selection_key"])
  quota=16 if ds=="cal_arc_train" else 12
  check(len(short)>=quota and len(long)>=quota,"sufficient_short_long_"+ds)
  chosen=[v for i in range(quota) for v in (short[i],long[i])]
  check(len({v["source_id"] for v in chosen})==len(chosen),"unique_questions_"+ds)
  check(not any(v["question_sha256"] in seen_hashes for v in chosen),"across_domains_question_reuse")
  seen_hashes.update(v["question_sha256"] for v in chosen)
  output[ds]=chosen
  for mode in ("static_source","static_random","dynamic"):
   for i,v in enumerate(chosen):
    ids=struct.pack("<"+"i"*len(v["prefix_ids"]),*v["prefix_ids"])
    answers=(" ".join(str(v["suffix_ids"][x]) for x in "ABCD")+" "+str("ABCD".index(v["correct"]))+"\n").encode()
    base=ROOT/ds/mode
    atomic_new(base/"eval/ppl_work"/f"w{i}.i32",ids)
    atomic_new(base/"eval/ppl_work"/f"w{i}.choices",answers)
    if i==0:atomic_new(base/"prompt_ids.i32",ids)
  manifest={"schema":"BEGLIN-Qv20-frozen-actual-original-C-inputs-private/1",
   "dataset_split":ds,"dataset_SHA256":KNOWN[ds],"source_tokenizer_SHA256":filehash(TOKENIZER),
   "selection_seed":SEED,"quota_per_length":quota,"role_n8_mask_sha256":{"static_source":KNOWN["mask_source"],"static_random":KNOWN["mask_random"]},
   "selected_rows":chosen,"profile_length_threshold_reference_only":85,
   "no_model_scoring_performed_at_freeze":True}
  raw=enc(manifest)
  atomic_new(ROOT/ds/"frozen_inputs_private.json",raw)
  output[ds+"_private_sha"]=sha(raw)
 for mode,src in (("static_source",SRC),("static_random",RANDOM)):
  atomic_new(ROOT/"masks"/(mode+".txt"),src.read_bytes())
 public={
  "schema":"BEGLIN-Qv20-ARCtrain-calibration-ARCvalidation-OBvalidation-external-preexecution-public-freeze/1",
  "state":"SEALED_SEPARATE_CALIBRATION_AND_EVALUATION_BEFORE_NEW_C_MODEL_FORWARD",
  "selection_seed":SEED,
  "calibration":"ARC-Challenge train 16 short+16 long, original source labels for supervised calibration only",
  "evaluation":["ARC-Challenge validation 12 short+12 long excluding Qv14/Qv16's previous 76 cases",
                "OpenBookQA validation 12 short+12 long distinct from prior Qv13 OpenBookQA test"],
  "three_domain_and_split_source_SHA256":KNOWN,
  "tokenizer_sha256":filehash(TOKENIZER),
  "separate_heldout_input_SHA256":{k:output[k+"_private_sha"] for k in ARROW},
  "sample_count":{k:len(output[k]) for k in ARROW},
  "exact_prefix_length_public":{k:[v["prefix_len"] for v in output[k]] for k in ARROW},
  "source_ids_public":False,"gold_labels_public":False,"individual_tokens_public":False,
  "selected_source_Qv8_projection_weight_packed_bytes":765493248,
  "matched_random73_projection_weight_packed_bytes":765493248,
  "static_Q4_projection_weight_packed_bytes":736985088,
  "model_accuracy_and_conditional_NLL_not_yet_evaluated":True,
  "no_look_ahead_from_Qv20_heldout_log_or_answers_at_policy_training":True,
  "no_late_changes_to_calibration_or_heldout_selection_allowed":True}
 content=enc(public);atomic_new(ROOT/"Qv20_PREEXEC_PUBLIC.json",content)
 print("QV20_SEALED_BEFORE_ANY_NEW_NATIVE_C_INFERENCE",json.dumps({
  "public_SHA256":sha(content),
  "frozen_sha":{k:output[k+"_private_sha"] for k in ARROW},
  "n":{k:len(output[k]) for k in ARROW},
  "ob_eligible_long_total":sum(1 for row in Dataset.from_file(str(ARROW["test_ob_validation"]))
    if (v:=eligible("test_ob_validation",row,tok)) is not None and v["prefix_len"]>=85),
  "old_ARC_validation_previous_76_excluded":True,
  "path":str(ROOT)},sort_keys=True),flush=True)
if __name__=="__main__":freeze()
