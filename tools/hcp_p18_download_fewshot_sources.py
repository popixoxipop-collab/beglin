#!/usr/bin/env python3
"""Fetch only P18 few-shot source splits, pinned to already selected test revisions.
All bytes land on D50. Never contacts production or executes model inference.
"""
from __future__ import annotations
import hashlib,json,os,pathlib
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
os.environ["HF_HOME"]=str(ROOT/"hf-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
from huggingface_hub import hf_hub_download

PINS=(
 ("mmlu","cais/mmlu","c30699e8356da336a370243923dbaf21066bb9fe","all/dev-00000-of-00001.parquet","MIT","dev"),
 ("arc_challenge","allenai/ai2_arc","210d026faf9955653af8916fad021475a3f00453","ARC-Challenge/validation-00000-of-00001.parquet","CC-BY-SA-4.0","validation"),
 ("gsm8k","openai/gsm8k","740312add88f781978c0658806c59bc2815b9866","main/train-00000-of-00001.parquet","MIT","train")
)
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def main():
 if not ROOT.is_dir() or ROOT.is_symlink():raise RuntimeError("unexpected P18 D50 root")
 if sha(ROOT/"source_manifest.json")!="ca453fd6776f20c9ea4279e98bffb1ffd97e9f1963c1425bc874a46eaaf150a4":
  raise RuntimeError("P18 pinned test manifest changed")
 target=ROOT/"dev_source_manifest.json"
 if target.exists():raise FileExistsError("Refuse to overwrite or redownload existing dev manifest")
 out=[]
 for name,repo,rev,file,license,split in PINS:
  p=pathlib.Path(hf_hub_download(repo_id=repo,filename=file,revision=rev,repo_type="dataset",
                               local_dir=str(ROOT/"source_parquet"/(name+"-dev"))))
  if p.is_symlink() or not p.is_file() or p.stat().st_size>20*1024*1024:
   raise ValueError("Source is not bounded regular file: "+name)
  out.append({"dataset":name,"repo_id":repo,"revision":rev,"split":split,"path_in_repo":file,
              "license":license,"sha256":sha(p),"size_bytes":p.stat().st_size,"local_path":str(p)})
 manifest={"schema":"beglin-hcp-p18-public-dev-source-pin-v1","purpose":"0/3/5-shot prompt examples; not evaluation examples",
           "test_source_manifest_sha256":"ca453fd6776f20c9ea4279e98bffb1ffd97e9f1963c1425bc874a46eaaf150a4",
           "source_pins":out,"production_touched":False,"quality_claim":False}
 with target.open("x") as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"dev_pins":[{"dataset":x["dataset"],"revision":x["revision"],"split":x["split"],
                              "bytes":x["size_bytes"],"sha256":x["sha256"]} for x in out],
                   "manifest_sha256":sha(target)},sort_keys=True))
if __name__=="__main__":main()
