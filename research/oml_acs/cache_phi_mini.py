#!/usr/bin/env python3
"""Pin and cache official public MIT Phi-4-mini for authorized OML-ACS pilot.
Never changes production routes, weights, or applies trust_remote_code.
"""
from pathlib import Path
import os,json,time,hashlib
from huggingface_hub import snapshot_download, HfApi
MODEL="microsoft/Phi-4-mini-instruct"
REV="cfbefacb99257ffa30c83adab238a50856ac3083"
ROOT=Path("/mnt/g/TailnetCommander/work/oml_acs_hf")
os.environ["HF_HOME"]=str(ROOT)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
def main():
    begin=time.perf_counter()
    meta=HfApi().model_info(MODEL,revision=REV)
    if meta.sha!=REV or meta.gated:
        raise RuntimeError("revision changed or gated; no automatic downloads")
    print("PHI_PREFLIGHT pinned_sha",REV,"gated",meta.gated,"license",(meta.card_data.license if meta.card_data else None),flush=True)
    files=["config.json","generation_config.json","model.safetensors.index.json","model-00001-of-00002.safetensors","model-00002-of-00002.safetensors","tokenizer.json","tokenizer_config.json","special_tokens_map.json"]
    path=Path(snapshot_download(repo_id=MODEL,revision=REV,repo_type="model",allow_patterns=files,max_workers=2))
    if path.name!=REV:
        raise RuntimeError("unexpected snapshot path/branch")
    expected={"model-00001-of-00002.safetensors","model-00002-of-00002.safetensors","config.json","tokenizer_config.json","tokenizer.json","model.safetensors.index.json"}
    missing=sorted(x for x in expected if not (path/x).is_file())
    if missing: raise RuntimeError("incomplete checkpoint: "+str(missing))
    rec={"schema":"oml-acs-local-model-cache/1","status":"CACHED_PINNED_NOT_EVALUATED",
         "model":MODEL,"revision":REV,"snapshot_path":str(path),
         "files":{x:{"size_bytes":(path/x).stat().st_size} for x in files if (path/x).is_file()},
         "weight_bytes":sum((path/x).stat().st_size for x in expected if x.endswith(".safetensors")),
         "elapsed_s":round(time.perf_counter()-begin,2),"no_external_paid_api":True,"production_touched":False}
    print("PHI_CACHE_RECEIPT",json.dumps(rec,sort_keys=True),flush=True)
if __name__=="__main__":
    main()
