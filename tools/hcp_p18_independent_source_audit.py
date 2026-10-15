#!/usr/bin/env python3
"""Independent P18 D50 public-source-to-480-candidate audit, no model scoring."""
import collections,hashlib,json,pathlib,sys
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
sys.path.insert(0,str(ROOT/"vendor"))
import duckdb
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for data in iter(lambda:f.read(1<<20),b""):h.update(data)
 return h.hexdigest()
def canonical(row):
 return json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(",",":")).encode("utf-8")
def digest(x):
 return hashlib.sha256(x).hexdigest()
def verify():
 sourcepath=ROOT/"source_manifest.json"
 candidatepath=ROOT/"candidate_bank_PRIVATE.json"
 receiptpath=ROOT/"candidate_selection_receipt.json"
 source=json.loads(sourcepath.read_text())
 bank=json.loads(candidatepath.read_text())
 rec=json.loads(receiptpath.read_text())
 if source["schema"]!="beglin-hcp-p18-public-source-identity-v1" or rec["selection_count"]!=480 or len(bank["items"])!=480:
  raise ValueError("P18 expected public-source cohort absent")
 if rec["private_question_bank_sha256"]!=sha(candidatepath) or rec["source_manifest_sha256"]!=sha(sourcepath):
  raise ValueError("P18 receipt SHA did not bind actual source and private bank")
 bysource=collections.defaultdict(list)
 for item in bank["items"]:bysource[item["dataset"]].append(item)
 total=0;reports=[]
 for sf in source["sources"]:
  ds=sf["dataset"];p=pathlib.Path(sf["local_path"])
  if p.is_symlink() or sha(p)!=sf["sha256"]:raise ValueError("source parquet changed "+ds)
  cur=duckdb.connect().execute("SELECT * FROM read_parquet(?)",[str(p)])
  cols=[v[0] for v in cur.description]
  candidate=[]
  for ix,values in enumerate(cur.fetchall()):
   row=dict(zip(cols,values))
   rowsha=digest(canonical(row))
   rank=digest((str(bank["selection_seed"])+"|"+ds+"|"+rowsha).encode())
   candidate.append((rank,rowsha,ix,row))
  candidate.sort(key=lambda t:(t[0],t[1],t[2]))
  expected=candidate[:160]
  actual=bysource[ds]
  if len(actual)!=160 or len({x["item_id"] for x in actual})!=160:
   raise ValueError("duplicate/missing P18 "+ds)
  sr=[z for z in rec["sources"] if z["dataset"]==ds]
  if len(sr)!=1 or sr[0]["sample_record_sha256"]!=[x[1] for x in expected]:
   raise ValueError("selection receipt did not match SHA sorting "+ds)
  for i,(row,item) in enumerate(zip(expected,actual)):
   _,rowsha,ix,raw=row
   if (item["record_sha256"]!=rowsha or item["source_row_ordinal"]!=ix
       or item["source_text"]!=raw or item["dataset_revision"]!=sf["revision"]
       or item["raw_parquet_sha256"]!=sf["sha256"]
       or item["item_id"]!=f"p18-{ds}-{i:03d}"
       or not item["prompt"].endswith("\nAnswer:\n")):
    raise ValueError("P18 wrong private candidate identity or prompt boundary "+ds)
   gold=item["reference_answer"]
   if ds=="mmlu":
    if gold!=chr(65+int(raw["answer"])):raise ValueError("MMLU answer tampering")
   elif ds=="gsm8k":
    ref=str(raw["answer"]).rsplit("####",1)[1].strip().replace(",","")
    if gold!=ref:raise ValueError("GSM8K answer tampering")
   elif ds=="arc_challenge":
    x=raw["choices"]
    pairs=list(zip(x["label"],x["text"])) if isinstance(x,dict) else [(v["label"],v["text"]) for v in x]
    matches=[n for n,pair in enumerate(pairs) if str(pair[0])==str(raw["answerKey"])]
    if len(matches)!=1 or gold!=chr(65+matches[0]):
     raise ValueError("ARC source label mapping error")
   else:raise ValueError("unknown source")
   total+=1
  reports.append({"dataset":ds,"source_rows":len(candidate),"sampled_rows":len(actual),"audited_rows":len(actual),
   "parquet_sha256":sf["sha256"]})
 result={"schema":"beglin-hcp-p18-independent-public-source-cohort-audit-v1",
   "status":"P18_SOURCE_SELECTION_INDEPENDENT_AUDIT_PASS",
   "audited":total,"source_manifest_sha256":sha(sourcepath),
   "private_candidate_bank_sha256":sha(candidatepath),
   "selection_receipt_sha256":sha(receiptpath),
   "by_source":reports,"original_model_executed":False,
   "native_AB_executed":False,"tokenization_complete":False,"production_touched":False}
 if total!=480:raise ValueError("P18 cohort short")
 out=ROOT/"candidate_independent_audit.json"
 with out.open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":result["status"],"audited":total,"audit_sha256":sha(out)},sort_keys=True))
if __name__=="__main__":verify()
