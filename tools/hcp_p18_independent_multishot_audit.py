#!/usr/bin/env python3
"""Independent P18 multishot frozen-input source/token audit; no model loads."""
from __future__ import annotations
import collections,hashlib,json,pathlib,sys
from tokenizers import Tokenizer
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"vendor"))
from hcp_p18_sample_public import normalize,hashobj
import duckdb
PIN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
TOK="41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"
TEST="ea7cb77649fcdc285393234e019d56db0d7178fb7787cf23268a7c52c072bb5a"
DEV="2d9f6e63fe49cfe646493f0e3e79870ff2eca34adac5081b3bb8fb305b28d836"
SCHEMA="beglin-hcp-p18-token-exact-frozen-480-external-multishot-v1"
CAPS=[8,32,128]; SHOTS=[0,3,5]
def sha(p):
 p=pathlib.Path(p)
 if p.is_symlink() or not p.is_file():raise ValueError("missing/unsafe source")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def hashid(ids):
 return hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
def hashstr(s):return hashlib.sha256(s.encode("utf-8")).hexdigest()
def normq(q):return " ".join(str(q).lower().split())
def check():
 srcpath=ROOT/"p18_multishot_frozen_PRIVATE.json"
 if sha(srcpath)!=PIN or sha(ROOT/"tokenizer_exact_XOX/tokenizer.json")!=TOK:
  raise ValueError("unexpected frozen token data/tokenizer SHA")
 if sha(ROOT/"candidate_bank_PRIVATE.json")!=TEST or sha(ROOT/"dev_source_manifest.json")!=DEV:
  raise ValueError("source test/dev changed")
 if sha(ROOT/"candidate_independent_audit.json")!="06b337558c0c60935de9e8839edf17c913cf0b4b3aacbaef5dbf2147a0df30ee":
  raise ValueError("external source independent test audit changed")
 frozen=json.loads(srcpath.read_text()); source=json.loads((ROOT/"candidate_bank_PRIVATE.json").read_text())
 dev=json.loads((ROOT/"dev_source_manifest.json").read_text())
 if frozen.get("schema")!=SCHEMA or frozen.get("question_count")!=480 or frozen.get("production_touched") is not False:
  raise ValueError("frozen corpus state or count invalid")
 if len(frozen.get("items",[]))!=480:raise ValueError("frozen records absent")
 if len(frozen["exemplars"])!=3 or {x["dataset"] for x in frozen["exemplars"]}!={"mmlu","arc_challenge","gsm8k"}:
  raise ValueError("dev exemplar sources not complete")
 tokenizer=Tokenizer.from_file(str(ROOT/"tokenizer_exact_XOX/tokenizer.json"))
 pins={d["dataset"]:d for d in dev["source_pins"]}
 bydev={}
 for cell in frozen["exemplars"]:
  name=cell["dataset"]; pin=pins[name]
  if sha(pin["local_path"])!=cell["source_sha256"] or cell["source_sha256"]!=pin["sha256"]:
   raise ValueError("dev SHA mismatch")
  q=duckdb.connect().execute("SELECT * FROM read_parquet(?)",[pin["local_path"]])
  keys=[v[0] for v in q.description]; values=q.fetchall()
  test_qhashes={hashstr(normq(q["source_text"]["question"])) for q in source["items"] if q["dataset"]==name}
  if len(cell["exemplars"])!=5:raise ValueError("fewshot five exemplars absent")
  records=[]
  for e in cell["exemplars"]:
   idx=e["source_row_ordinal"]
   if not isinstance(idx,int) or not 0<=idx<len(values):raise ValueError("dev ordinal invalid")
   raw=dict(zip(keys,values[idx]))
   expected_prompt,expected_answer=normalize(name,raw)
   if hashobj(raw)!=e["record_sha256"] or e["prompt"]!=expected_prompt or e["answer"]!=expected_answer:
    raise ValueError("incorrect chosen raw dev answer/record")
   if hashstr(normq(raw["question"])) in test_qhashes:
    raise ValueError("source disjointness failure")
   ids=tokenizer.encode(expected_prompt,add_special_tokens=True).ids
   entire=tokenizer.encode(expected_prompt+expected_answer,add_special_tokens=True).ids
   if (entire[:len(ids)]!=ids or hashid(ids)!=e["prompt_token_sha256"]
        or hashid(entire[len(ids):])!=e["gold_token_sha256"] or len(ids)>160):
    raise ValueError("invalid dev tokenizer boundary or hash")
   records.append(e)
  if len({r["question_key_sha256"] for r in records})!=5:
   raise ValueError("duplicate dev demonstration")
  bydev[name]=records
 frozen_items=frozen["items"]; grouped=collections.defaultdict(list)
 for f in frozen_items:grouped[f["dataset"]].append(f)
 if {x:len(y) for x,y in grouped.items()}!={"mmlu":160,"arc_challenge":160,"gsm8k":160}:
  raise ValueError("unbalanced P18 questions")
 count=collections.Counter();seen=set();prompt_max=0
 for dataset in ("mmlu","arc_challenge","gsm8k"):
  originals=[r for r in source["items"] if r["dataset"]==dataset]
  for pos,(record,original) in enumerate(zip(grouped[dataset],originals)):
   shot=SHOTS[(pos//3)%3];cap=CAPS[pos%3]
   if record["shot_count"]!=shot or record["max_new_tokens"]!=cap or record["decode_steps"]!=cap:
    raise ValueError("posthoc cap or shot changed")
   if record["item_id"]!=original["item_id"] or record["record_sha256"]!=original["record_sha256"]:
    raise ValueError("source test item identity changed")
   if record["item_id"] in seen:raise ValueError("duplicate evaluation question")
   seen.add(record["item_id"])
   original_prompt=original["prompt"]
   if shot:
    split=original_prompt.split("\nQuestion: ")
    if len(split)!=2:raise ValueError("unrecognized original test prompt layout")
    examples=bydev[dataset][:shot]
    blocks=[]
    for ex in examples:
     demo_parts=ex["prompt"].split("\nQuestion: ")
     if len(demo_parts)!=2:raise ValueError("unrecognized dev prompt layout")
     blocks.append("Question: "+demo_parts[1]+ex["answer"]+"\n\n")
    exact_prompt=split[0]+"\n\n"+"".join(blocks)+"Question: "+split[1]
   else:exact_prompt=original_prompt
   prompt=tokenizer.encode(exact_prompt,add_special_tokens=True).ids
   both=tokenizer.encode(exact_prompt+original["reference_answer"],add_special_tokens=True).ids
   if both[:len(prompt)]!=prompt:
    raise ValueError("BPE boundary instability in frozen question")
   gold=both[len(prompt):]
   if record["prompt_ids"]!=prompt or record["gold_ids"]!=gold:
    raise ValueError("actual token array differs")
   if record["prompt_token_sha256"]!=hashid(prompt) or record["gold_token_sha256"]!=hashid(gold):
    raise ValueError("token hash incorrect")
   if record["gold_token_count"]!=len(gold) or record["prompt_token_count"]!=len(prompt):
    raise ValueError("count mismatch")
   if len(prompt)+cap>4096 or not 1<=len(gold)<=cap:
    raise ValueError("invalid target/prompt size")
   if record["dev_example_record_sha256"]!=[e["record_sha256"] for e in bydev[dataset][:shot]]:
    raise ValueError("wrong exemplar lineage")
   count[dataset]+=1;prompt_max=max(prompt_max,len(prompt))
 if len(seen)!=480:raise ValueError("not all 480 tests independently audited")
 out={"schema":"beglin-hcp-p18-independent-multishot-input-audit-v1",
      "status":"P18_480_TOKEN_AND_SOURCE_INDEPENDENT_AUDIT_PASS",
      "source_cohort_SHA256":TEST,"dev_source_manifest_sha256":DEV,
      "frozen_cohort_sha256":PIN,"tokenizer_sha256":TOK,
      "audited_question_count":len(seen),"audited_exemplar_count":15,
      "verified_shot_cap_design":"orthogonal 3x3 fixed ordinal cycle",
      "observed_prompt_token_max":prompt_max,"dataset_counts":dict(count),
      "actual_model_forward_count":0,"native_AB_forward_count":0,
      "native_4096_context_verified":False,
      "automatic_live_promotion":False,"production_touched":False}
 path=ROOT/"p18_multishot_independent_audit.json"
 if path.exists():raise FileExistsError("independent audit already exists")
 with path.open("x") as f:json.dump(out,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":out["status"],"audited":len(seen),"dev_examples":15,
       "audit_sha256":sha(path),"prompt_max":prompt_max},sort_keys=True))
if __name__=="__main__":check()
