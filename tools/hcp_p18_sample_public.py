#!/usr/bin/env python3
"""Select 480 first-use public benchmark candidates without model access.
Raw question/answer text remains on D50 only; GitHub receives SHA receipts.
"""
import hashlib,json,pathlib,re,sys
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
sys.path.insert(0,str(ROOT/"vendor"))
import duckdb
SEED=20261015
COUNT=160
def canon(o):
 return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def hashbytes(data):return hashlib.sha256(data).hexdigest()
def hashobj(o):return hashbytes(canon(o).encode("utf-8"))
def filesha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def rows(path):
 con=duckdb.connect(database=":memory:",read_only=False)
 try:
  result=con.execute("SELECT * FROM read_parquet(?)",[str(path)])
  keys=[v[0] for v in result.description]
  for values in result.fetchall():yield dict(zip(keys,values))
 finally:con.close()
def arc_options(raw):
 x=raw.get("choices")
 if isinstance(x,dict):
  labels=x.get("label",[]);texts=x.get("text",[])
  return list(zip(labels,texts))
 if isinstance(x,list):
  return [(t.get("label"),t.get("text")) for t in x if isinstance(t,dict)]
 raise ValueError("unsupported ARC choices layout")
def normalize(dataset,r):
 if dataset=="gsm8k":
  q=str(r["question"]).strip()
  ans=str(r["answer"])
  if "####" not in ans:raise ValueError("GSM8K lacks final answer tag")
  gold=ans.rsplit("####",1)[1].strip().replace(",","")
  if not re.fullmatch(r"-?\d+(?:\.\d+)?",gold):
   raise ValueError("GSM8K final answer unsupported; no cherry-pick")
  prompt="Solve the word problem and output only the final number.\nQuestion: "+q+"\nAnswer:\n"
 elif dataset=="mmlu":
  q=str(r["question"]).strip()
  choices=r["choices"]
  if not isinstance(choices,(list,tuple)) or len(choices)!=4:raise ValueError("bad MMLU choices")
  answer=int(r["answer"])
  if not 0<=answer<4:raise ValueError("bad MMLU answer")
  options="\n".join(chr(65+i)+") "+str(x) for i,x in enumerate(choices))
  gold=chr(65+answer)
  prompt="Choose the correct option. Output its letter only.\nQuestion: "+q+"\n"+options+"\nAnswer:\n"
 elif dataset=="arc_challenge":
  q=str(r["question"]).strip()
  ch=arc_options(r)
  if not 2<=len(ch)<=5:raise ValueError("bad ARC option count")
  matching=[idx for idx,(label,_) in enumerate(ch) if str(label)==str(r["answerKey"])]
  if len(matching)!=1:raise ValueError("ARC source answer key not in choices")
  gold=chr(65+matching[0])
  opts="\n".join(chr(65+i)+") "+str(z[1]) for i,z in enumerate(ch))
  prompt="Choose the correct option. Output its letter only.\nQuestion: "+q+"\n"+opts+"\nAnswer:\n"
 else:raise ValueError("unsupported dataset")
 if not q or len(prompt)>30000 or not gold:raise ValueError("malformed source prompt")
 return prompt,gold
def main():
 if ROOT.is_symlink() or not ROOT.is_dir():raise ValueError("bad D50 root")
 manifest=ROOT/"source_manifest.json"
 sources=json.loads(manifest.read_text())
 if sources.get("schema")!="beglin-hcp-p18-public-source-identity-v1" or len(sources.get("sources",[]))!=3:
  raise ValueError("source pins missing")
 candidates=[];receipt=[]
 for src in sorted(sources["sources"],key=lambda x:x["dataset"]):
  name=src["dataset"]; f=pathlib.Path(src["local_path"])
  if f.is_symlink() or not f.is_file() or filesha(f)!=src["sha256"]:
   raise ValueError("source data hash mismatch "+name)
  raw=[]
  for index,r in enumerate(rows(f)):
   if not isinstance(r,dict):raise ValueError("parquet data row malformed")
   rh=hashobj(r)
   priority=hashbytes((str(SEED)+"|"+name+"|"+rh).encode("utf-8"))
   raw.append((priority,rh,index,r))
  raw.sort(key=lambda t:(t[0],t[1],t[2]))
  if len(raw)<COUNT:raise ValueError("source lacks required external items "+name)
  selected=raw[:COUNT]
  if len({x[1] for x in selected})!=COUNT:
   raise ValueError("selected source contains duplicate records; no silent resampling")
  for index,(_,rh,orig_index,r) in enumerate(selected):
   prompt,gold=normalize(name,r)
   candidates.append({"item_id":f"p18-{name}-{index:03d}","dataset":name,
      "source_repo":src["repo_id"],"dataset_revision":src["revision"],
      "raw_parquet_sha256":src["sha256"],
      "record_sha256":rh,"source_row_ordinal":orig_index,
      "prompt":prompt,"reference_answer":gold,
      "task_scoring":"numeric_exact" if name=="gsm8k" else "letter_exact",
      "source_text":r})
  receipt.append({"dataset":name,"full_source_count":len(raw),"selection_n":COUNT,
        "selection_algorithm":"sha256(seed|dataset|canonical_json_record_sha256) lexicographic first N",
        "sample_record_sha256":[r[1] for r in selected],
        "parquet_sha256":src["sha256"],"revision":src["revision"]})
 private={"schema":"beglin-hcp-p18-public-candidate-question-bank-v1",
   "selection_seed":SEED,"candidate_not_tokenized":True,
   "model_inference_performed":False,"raw_source_manifest_sha256":filesha(manifest),
   "items":candidates,"production_touched":False}
 rawpath=ROOT/"candidate_bank_PRIVATE.json"
 receiptpath=ROOT/"candidate_selection_receipt.json"
 if rawpath.exists() or receiptpath.exists():raise FileExistsError("Never replace frozen P18 candidate selection")
 with rawpath.open("x",encoding="utf-8") as h:json.dump(private,h,sort_keys=True,ensure_ascii=False,indent=2);h.write("\n")
 out={"schema":"beglin-hcp-p18-source-selected-480-v1",
   "source_manifest_sha256":filesha(manifest),"selection_seed":SEED,
   "selection_count":len(candidates),"private_question_bank_sha256":filesha(rawpath),
   "sources":receipt,"tokenization_complete":False,
   "quality_metrics_observed":False,"automatic_live_promotion":False,"production_touched":False}
 with receiptpath.open("x",encoding="utf-8") as h:json.dump(out,h,sort_keys=True,indent=2);h.write("\n")
 print(canon({"selected":len(candidates),"by_source":[(r["dataset"],r["selection_n"],r["full_source_count"]) for r in receipt],
    "private_bank_sha256":filesha(rawpath),"selection_receipt_sha256":filesha(receiptpath),
    "tokenization_complete":False}))
if __name__=="__main__":main()
