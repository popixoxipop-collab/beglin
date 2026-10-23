#!/usr/bin/env python3
"""Qv16: independent from Qv14 4+24 validation cases, freeze 48 new requests BEFORE C forward.
This is a mechanism/latency experiment, NOT a new quality model evaluation.
Choice labels are required solely for post-switch scoring; routing uses prefix length only.
"""
import os,sys,struct,hashlib,json,pathlib,importlib.util
from datasets import Dataset
from tokenizers import Tokenizer
HOME=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/beglin-qv13-github-commit-20261023")
Q14=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv14-inprocess-20261023")
Q16=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv16-cost-aware-prewarm-20261023")
COHORT=Q16/"independent_arc_validation48"
SRC=HOME/"tools/qv14_freeze_dynamic_validation4.py"
def digest(x):return hashlib.sha256(x).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(262144),b""):h.update(chunk)
 return h.hexdigest()
def encode(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
def save_new(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def main():
 if COHORT.exists():raise FileExistsError("Qv16 frozen48 already exists: no refreeze")
 sp=importlib.util.spec_from_file_location("qv14_frozen_source_selector",str(SRC))
 mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 for key,path in [("arrow",mod.ARROW),("token",mod.TOKEN),("source",mod.SOURCE),("random",mod.RANDOM73)]:
  if file_sha(path)!=mod.SHAS[key]:raise RuntimeError("pinned source drift "+key)
 old4=Q14/"mechanistic_arc_validation4/frozen_inputs_private.json"
 old24=Q14/"replication_arc_validation24/frozen_inputs_private.json"
 if file_sha(old4)!="e12ba4ae07dba6476c4452357ce77b32f2a4c7ba64c5c1bb013e74cf1245e032":raise RuntimeError("old4 changed")
 if file_sha(old24)!="e836a4bb6ea86c43f5a9c28ecbbc235c4c9d7fd306dfc006511027e3bddf3dc6":raise RuntimeError("old24 changed")
 locked4=json.loads(old4.read_text())["rows"]
 locked24=json.loads(old24.read_text())["exact_input_rows"]
 excluded={v["original_source_id"] for v in locked4+locked24}
 if len(excluded)!=28:raise RuntimeError("28 old validation questions not distinct")
 tok=Tokenizer.from_file(str(mod.TOKEN))
 raw=Dataset.from_file(str(mod.ARROW))
 eligible=[v for r in raw if (v:=mod.row(r,tok)) is not None and v["original_source_id"] not in excluded]
 def key(v):return digest(("BEGLIN-QV16/independent-ARC-validation48/"+v["original_source_id"]+"/"+v["question_sha256"]).encode())
 short=sorted([v for v in eligible if v["prefix_len"]<85],key=key)[:24]
 long=sorted([v for v in eligible if v["prefix_len"]>=85],key=key)[:24]
 if len(short)!=24 or len(long)!=24:raise RuntimeError("24 short and 24 long required")
 chosen=[v for i in range(24) for v in [short[i],long[i]]]
 if len({v["original_source_id"] for v in chosen})!=48:raise RuntimeError("duplicate source IDs")
 profile=["source_qv8" if v["prefix_len"]<85 else "random73" for v in chosen]
 if profile!=["source_qv8","random73"]*24:raise RuntimeError("mask sequence invalid")
 os.umask(0o077);COHORT.mkdir(parents=True,mode=0o700)
 for mode in ["dynamic","static_source","static_random"]:
  for i,v in enumerate(chosen):
   ids=struct.pack("<"+"i"*len(v["prefix_ids"]),*v["prefix_ids"])
   answer=(" ".join(str(v["suffix_ids"][ch]) for ch in "ABCD")+" "+str("ABCD".index(v["correct"]))+"\n").encode()
   save_new(COHORT/mode/"eval/ppl_work"/f"w{i}.i32",ids)
   save_new(COHORT/mode/"eval/ppl_work"/f"w{i}.choices",answer)
   if i==0:save_new(COHORT/mode/"prompt_ids.i32",ids)
 for name,p in [("source_qv8",mod.SOURCE),("random73",mod.RANDOM73)]:
  save_new(COHORT/"masks"/f"{name}.txt",p.read_bytes())
 private={"schema":"BEGLIN-Qv16-original-ARC-validation48-new-mechanistic-requests/1",
 "status":"FROZEN_BEFORE_ANY_QV16_NATIVE_FORWARD",
 "original_arrow_sha256":mod.SHAS["arrow"],"original_tokenizer_sha256":mod.SHAS["token"],
 "excluded_Qv14_pilot4_sha256":file_sha(old4),"excluded_Qv14_validation24_sha256":file_sha(old24),
 "excluded_Qv14_28_source_IDs":True,
 "source_Qv8_mask_sha256":mod.SHAS["source"],"random73_mask_sha256":mod.SHAS["random"],
 "length_selector_threshold":85,
 "mask_selection_has_no_access_to_outputs_or_answers":True,
 "sample_selection":"SHA-order 24 eligible short and 24 eligible long original validation questions excluding all previous 28, interleaved S/L",
 "ordered_assigned_profiles":profile,
 "requests":chosen}
 packed=encode(private);save_new(COHORT/"frozen_inputs_private.json",packed)
 pub={"schema":"BEGLIN-Qv16-48-independent-mechanical-prewarm-benchmark-public-freeze/1",
 "status":"INPUTS_AND_MASKS_SEALED_BEFORE_ANY_QV16_MODEL_OUTPUT",
 "model":"Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
 "dataset":"ARC-Challenge validation, 48 source-distinct from Qv14 previous 28, unrelated to Qv13 test",
 "original_dataset_rows":len(raw),"eligible_after_excluding_previous_Qv14_28":len(eligible),"n":48,
 "lengths":[v["prefix_len"] for v in chosen],"profiles":profile,
 "private_lock_sha256":digest(packed),"unique_source_IDs_SHA256":digest(encode([v["original_source_id"] for v in chosen])),
 "original_cached_validation_arrow_sha256":mod.SHAS["arrow"],
 "original_tokenizer_sha256":mod.SHAS["token"],
 "source_mask_sha256":mod.SHAS["source"],"random_mask_sha256":mod.SHAS["random"],
 "cache_modes":"Qv15 on-demand 96MiB vs Qv16 explicit-startup prewarm 96MiB; compare cold first request, subsequent warm requests and process peak RSS",
 "intended_cold_prewarm_semantics":"40 alternate tensor quantizations BEFORE first user request; upfront cost is NOT removed, only moved from first swap to explicit startup",
 "quality_generalization_claim":False,"Metal_or_parallel_client_claim":False,
 "no_original_model_kernel_or_operating_release_change":True}
 rawpub=encode(pub);save_new(COHORT/"frozen_manifest_PUBLIC.json",rawpub)
 print("BEGLIN_QV16_48_SOURCE_FROZEN",json.dumps({"n":48,"short":24,"long":24,"eligible_remaining":len(eligible),
 "private_sha256":digest(packed),"public_sha256":digest(rawpub),
 "minimum_tokens":min(v["prefix_len"] for v in chosen),"maximum_tokens":max(v["prefix_len"] for v in chosen),
 "file_path":str(COHORT)},sort_keys=True),flush=True)
if __name__=="__main__":main()
