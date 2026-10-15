#!/usr/bin/env python3
"""Hash-frozen, source-disjoint 480-item 0/3/5-shot P18 *input* corpus.

This script does no inference. It refuses to replace prior files and binds exact
P17 tokenizer, public source files, and fixed A/B evaluation question identities.
It does not certify native 4096-token support or quality noninferiority.
"""
from __future__ import annotations
import collections
import hashlib
import json
import pathlib
import sys
from tokenizers import Tokenizer

ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"vendor"))
from hcp_p18_sample_public import normalize,hashobj
import duckdb

SEED=20261015
TEST_SHA="ea7cb77649fcdc285393234e019d56db0d7178fb7787cf23268a7c52c072bb5a"
DEV_MANIFEST_SHA="2d9f6e63fe49cfe646493f0e3e79870ff2eca34adac5081b3bb8fb305b28d836"
TOK_SHA="41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
P17_PARITY_SHA="3b7f99193e26c7bc6bca6c9cf8252256b880ed97f66e15d94836fa06ee0ad5f9"
CAPS=(8,32,128)
SHOTS=(0,3,5)
MAX_SEQ=4096

def sha(path):
 path=pathlib.Path(path)
 if not path.is_file() or path.is_symlink():raise ValueError("Refuse nonexistent/symlink source: "+str(path))
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def hashtext(value):
 return hashlib.sha256(value.encode("utf-8")).hexdigest()
def hashids(ids):
 return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def question_norm(text):
 return " ".join(str(text).lower().split())
def tokenids(tok,text):
 ids=tok.encode(text,add_special_tokens=True).ids
 if not ids or any(type(i)!=int or not 0<=i<102400 for i in ids):
  raise ValueError("Invalid full-model tokenization")
 return ids
def rows(path):
 con=duckdb.connect()
 q=con.execute("select * from read_parquet(?)",[str(path)])
 names=[t[0] for t in q.description]
 for ordinal,values in enumerate(q.fetchall()):
  yield ordinal,dict(zip(names,values))
 con.close()
def demo_body(prompt,gold):
 marker="\nQuestion: "
 if prompt.count(marker)!=1 or not prompt.endswith("\nAnswer:\n"):
  raise ValueError("Unexpected normalized demonstration format")
 return "Question: "+prompt.split(marker,1)[1]+gold+"\n\n"
def target_with_examples(target_prompt,examples):
 if not examples:return target_prompt
 marker="\nQuestion: "
 if target_prompt.count(marker)!=1 or not target_prompt.endswith("\nAnswer:\n"):
  raise ValueError("Unexpected test question format")
 instruction,target=target_prompt.split(marker,1)
 return instruction+"\n\n"+ "".join(demo_body(e["prompt"],e["answer"]) for e in examples)+ "Question: "+target
def main():
 if not ROOT.is_dir() or ROOT.is_symlink():raise ValueError("Wrong protected EOE D50 root")
 if sha(ROOT/"candidate_bank_PRIVATE.json")!=TEST_SHA or sha(ROOT/"dev_source_manifest.json")!=DEV_MANIFEST_SHA:
  raise ValueError("P18 public test/dev source identity changed")
 if sha(ROOT/"tokenizer_exact_XOX/tokenizer.json")!=TOK_SHA or sha(ROOT/"p17_tokenizer_parity.json")!=P17_PARITY_SHA:
  raise ValueError("P17 tokenizer parity not passed or source mutated")
 parity=json.loads((ROOT/"p17_tokenizer_parity.json").read_text())
 if parity["exact_prompt_id"]!=12 or parity["exact_full_context_id"]!=12 or parity["exact_gold_id"]!=12:
  raise ValueError("P17 original token ID parity not certified")
 tok=Tokenizer.from_file(str(ROOT/"tokenizer_exact_XOX/tokenizer.json"))
 bank=json.loads((ROOT/"candidate_bank_PRIVATE.json").read_text())
 if len(bank["items"])!=480:raise ValueError("Expected 480 P18 source-selected items")
 dev=json.loads((ROOT/"dev_source_manifest.json").read_text())
 bydataset=collections.defaultdict(list)
 for i in bank["items"]:
  bydataset[i["dataset"]].append(i)
 if set(bydataset)!= {"mmlu","arc_challenge","gsm8k"} or any(len(x)!=160 for x in bydataset.values()):
  raise ValueError("P18 evaluation source balance drifted")
 exemplar_bank=[]
 for src in sorted(dev["source_pins"],key=lambda x:x["dataset"]):
  dataset=src["dataset"];filename=pathlib.Path(src["local_path"])
  if sha(filename)!=src["sha256"]:raise ValueError("Dev source artifact changed")
  test_questions={hashtext(question_norm(x["source_text"]["question"])) for x in bydataset[dataset]}
  pool={}
  rejects=collections.Counter()
  for index,raw in rows(filename):
   try:
    prompt,gold=normalize(dataset,raw)
    if not gold:raise ValueError("empty answer")
    ids=tokenids(tok,prompt);both=tokenids(tok,prompt+gold)
    if both[:len(ids)]!=ids or not (1<=len(both)-len(ids)<=8):
     rejects["boundary_or_gold"]=1+rejects["boundary_or_gold"];continue
    if len(ids)>160:
     rejects["too_long"]=1+rejects["too_long"];continue
    qhash=hashtext(question_norm(raw["question"]))
    if qhash in test_questions:
     rejects["overlap_with_test"]=1+rejects["overlap_with_test"];continue
    h=hashobj(raw)
    score=hashtext(f"{SEED}|fewshot|{dataset}|{h}")
    row={"dataset":dataset,"source_split":src["split"],"source_row_ordinal":index,
         "source_sha256":src["sha256"],"question_key_sha256":qhash,
         "record_sha256":h,"rank_sha256":score,"prompt":prompt,"answer":gold,
         "prompt_token_sha256":hashids(ids),"gold_token_sha256":hashids(both[len(ids):])}
    previous=pool.get(qhash)
    if previous is None or (score,h,index)<(previous["rank_sha256"],previous["record_sha256"],previous["source_row_ordinal"]):
     pool[qhash]=row
   except (ValueError,KeyError,TypeError) as e:
    rejects["malformed_or_unusable"]+=1
  candidates=sorted(pool.values(),key=lambda r:(r["rank_sha256"],r["record_sha256"],r["source_row_ordinal"]))
  if len(candidates)<5:raise ValueError("Fewer than five disjoint dev examples for "+dataset)
  exemplar_bank.append({"dataset":dataset,"source_sha256":src["sha256"],
                         "eligible_unique":len(candidates),"rejected":dict(rejects),
                         "exemplars":candidates[:5]})
 dev_map={x["dataset"]:x["exemplars"] for x in exemplar_bank}
 frozen=[];count=collections.defaultdict(collections.Counter)
 for dataset in ("mmlu","arc_challenge","gsm8k"):
  items=bydataset[dataset]
  for n,item in enumerate(items):
   shot=SHOTS[(n//3)%3]
   cap=CAPS[n%3]
   examples=dev_map[dataset][:shot]
   prompt=target_with_examples(item["prompt"],examples)
   prompt_ids=tokenids(tok,prompt);full=tokenids(tok,prompt+item["reference_answer"])
   if full[:len(prompt_ids)]!=prompt_ids:raise ValueError("BPE boundary unstable, abort before freeze: "+item["item_id"])
   gold=full[len(prompt_ids):]
   if not (1<=len(gold)<=cap):raise ValueError("Target gold beyond assigned fixed cap: "+item["item_id"])
   if len(prompt_ids)+cap>MAX_SEQ:raise ValueError("Planned maximum context outside 4096, abort: "+item["item_id"])
   cell={"item_id":item["item_id"],"dataset":dataset,
     "dataset_revision":item["dataset_revision"],"split":"test","record_sha256":item["record_sha256"],
     "source_prompt_sha256":hashtext(item["prompt"]),
     "prompt_token_sha256":hashids(prompt_ids),"gold_token_sha256":hashids(gold),
     "prompt_token_count":len(prompt_ids),"gold_token_count":len(gold),
     "gold_text_sha256":hashtext(item["reference_answer"]),
     "shot_count":shot,"dev_example_record_sha256":[e["record_sha256"] for e in examples],
     "max_new_tokens":cap,"decode_steps":cap,
     "prefix_boundary_stable":True,
     "prompt_ids":prompt_ids,"gold_ids":gold,"task_scoring":item["task_scoring"]}
   frozen.append(cell)
   count[dataset][f"shots_{shot}"]+=1
   count[dataset][f"cap_{cap}"]+=1
   count[dataset][f"cross_{shot}_{cap}"]+=1
 out={
   "schema":"beglin-hcp-p18-token-exact-frozen-480-external-multishot-v1",
   "state":"P18_COHORT_TOKEN_FROZEN_NATIVE_CAPABILITY_PENDING",
   "selection_seed":SEED,"source_selection_sha256":TEST_SHA,
   "dev_source_manifest_sha256":DEV_MANIFEST_SHA,"original_tokenizer_sha256":TOK_SHA,
   "original_P17_token_ID_parity_receipt_sha256":P17_PARITY_SHA,
   "question_count":len(frozen),"max_planned_context":MAX_SEQ,
   "exemplars":exemplar_bank,"question_count_matrix":{k:dict(v) for k,v in count.items()},
   "items":frozen,"production_touched":False,"automatic_live_promotion":False,
   "external_test_unknown_pretraining_contamination":True,
   "native_4096_context_verified":False,"real_BF16_forward_count":0,"native_AB_forward_count":0}
 path=ROOT/"p18_multishot_frozen_PRIVATE.json"
 receiptpath=ROOT/"p18_multishot_freeze_receipt.json"
 if path.exists() or receiptpath.exists():raise FileExistsError("P18 frozen cohort already exists; no overwrite allowed")
 with path.open("x") as f:json.dump(out,f,indent=2,sort_keys=True,ensure_ascii=False);f.write("\n")
 lens=[r["prompt_token_count"] for r in frozen];golds=[r["gold_token_count"] for r in frozen]
 summary={"schema":"beglin-hcp-p18-token-freeze-receipt-v1","status":out["state"],
   "private_frozen_cohort_sha256":sha(path),"selection_source_sha256":TEST_SHA,
   "dev_source_manifest_sha256":DEV_MANIFEST_SHA,
   "original_tokenizer_sha256":TOK_SHA,"p17_token_parity_sha256":P17_PARITY_SHA,
   "question_count":len(frozen),"source_counts":{k:sum(v.get(f"shots_{s}",0) for s in SHOTS) for k,v in count.items()},
   "shot_cap_matrix":out["question_count_matrix"],
   "prompt_tokens_range":[min(lens),max(lens)],
   "gold_tokens_range":[min(golds),max(golds)],
   "count_within_128_and_assigned_cap":sum(r["prompt_token_count"]+r["max_new_tokens"]<=128 for r in frozen),
   "count_within_4096_and_assigned_cap":sum(r["prompt_token_count"]+r["max_new_tokens"]<=4096 for r in frozen),
   "actual_BF16_forward_count":0,"actual_AB_forward_count":0,"quality_certified":False,
   "production_touched":False,"automatic_live_promotion":False}
 with receiptpath.open("x") as f:json.dump(summary,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps({"status":summary["status"],"frozen_sha256":summary["private_frozen_cohort_sha256"],
   "receipt_sha256":sha(receiptpath),"question_count":480,
   "prompt_tokens_range":summary["prompt_tokens_range"],"gold_tokens_range":summary["gold_tokens_range"],
   "count_within_128":summary["count_within_128_and_assigned_cap"],
   "count_within_4096":summary["count_within_4096_and_assigned_cap"],
   "matrix":summary["shot_cap_matrix"]},sort_keys=True))
if __name__=="__main__":main()
