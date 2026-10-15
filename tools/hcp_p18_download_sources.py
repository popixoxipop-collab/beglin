#!/usr/bin/env python3
"""Fetch only three public, pinned P18 test shards to D50; no model or serving access."""
import hashlib,json,os,pathlib,sys,time
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
os.environ["HF_HOME"]=str(ROOT/"hf-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
from huggingface_hub import hf_hub_download
SOURCES=(
 ("gsm8k","openai/gsm8k","740312add88f781978c0658806c59bc2815b9866",
  "main/test-00000-of-00001.parquet","MIT"),
 ("mmlu","cais/mmlu","c30699e8356da336a370243923dbaf21066bb9fe",
  "all/test-00000-of-00001.parquet","MIT"),
 ("arc_challenge","allenai/ai2_arc","210d026faf9955653af8916fad021475a3f00453",
  "ARC-Challenge/test-00000-of-00001.parquet","CC-BY-SA-4.0")
)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def main():
 if ROOT.is_symlink() or not ROOT.is_dir() or not str(ROOT).startswith("/Volumes/D50/TailnetCommander/work/"):
  raise RuntimeError("P18 target must be canonical, existing D50 root")
 rows=[]
 for slug,repo,rev,filename,license in SOURCES:
  path=pathlib.Path(hf_hub_download(repo_id=repo,filename=filename,revision=rev,
    repo_type="dataset",local_dir=str(ROOT/"source_parquet"/slug)))
  if path.is_symlink() or not path.is_file() or path.stat().st_size>30_000_000:
   raise RuntimeError("source file is not a safe bounded regular shard: "+slug)
  rows.append({"dataset":slug,"repo_id":repo,"revision":rev,"path_in_repo":filename,
   "license":license,"sha256":sha(path),"size_bytes":path.stat().st_size,
   "local_path":str(path),"source_url":"https://huggingface.co/datasets/"+repo+"/tree/"+rev})
 out={"schema":"beglin-hcp-p18-public-source-identity-v1",
      "role":"public benchmark source download only, no scoring",
      "downloaded_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
      "sources":rows,"production_touched":False}
 target=ROOT/"source_manifest.json"
 if target.exists():
  old=json.loads(target.read_text())
  if [(r["dataset"],r["sha256"]) for r in old["sources"]] != [(r["dataset"],r["sha256"]) for r in rows]:
   raise RuntimeError("existing source manifest differs; abort not overwrite")
 else:
  with target.open("x",encoding="utf-8") as f:json.dump(out,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"sources":[{"dataset":x["dataset"],"sha256":x["sha256"],
   "size_bytes":x["size_bytes"],"revision":x["revision"]} for x in rows],
   "manifest_sha256":sha(target),"manifest_path":str(target)},sort_keys=True))
if __name__=="__main__":main()
