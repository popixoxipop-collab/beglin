#!/usr/bin/env python3
"""Qv14 24-request CPU dynamic-switching replication input freeze.
Original ARC-Challenge VALIDATION, distinct from the earlier 4-request pilot.
Fixed token-length threshold and 28n8 masks chosen before all native outcomes.
"""
import importlib.util,pathlib,json,hashlib,os,struct
from datasets import Dataset
from tokenizers import Tokenizer
R=pathlib.Path(__file__).resolve().parents[1]
S=R/"tools/qv14_freeze_dynamic_validation4.py"
spec=importlib.util.spec_from_file_location("qv14_previous_frozen_protocol",S)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
LAB=m.LAB
OUT=LAB/"replication_arc_validation24"
def sha(x):return hashlib.sha256(x).hexdigest()
def frozen(p,data):
 p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o600)
def enc(o):return (json.dumps(o,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
def main():
 if OUT.exists():raise FileExistsError("24-request input cohort already frozen; no refreeze")
 for key,p in [("arrow",m.ARROW),("token",m.TOKEN),("source",m.SOURCE),("random",m.RANDOM73)]:
  if m.shaf(p)!=m.SHAS[key]:raise RuntimeError("original Qv14 frozen source drift: "+key)
 old=m.LAB/"mechanistic_arc_validation4/frozen_inputs_private.json"
 d=json.loads(old.read_text());oldids={x["original_source_id"] for x in d["rows"]}
 tok=Tokenizer.from_file(str(m.TOKEN))
 allrows=Dataset.from_file(str(m.ARROW))
 eligible=[x for r in allrows if (x:=m.row(r,tok)) is not None and x["original_source_id"] not in oldids]
 def key(x):return sha(("QV14/replication24/original-ARC-valid/"+x["original_source_id"]+"/"+x["question_sha256"]).encode())
 short=sorted((x for x in eligible if x["prefix_len"]<m.THRESHOLD),key=key)[:12]
 long=sorted((x for x in eligible if x["prefix_len"]>=m.THRESHOLD),key=key)[:12]
 if len(short)!=12 or len(long)!=12:raise RuntimeError("must have 12+12 frozen")
 selected=[v for i in range(12) for v in (short[i],long[i])]
 if len({x["original_source_id"] for x in selected})!=24:raise RuntimeError("duplicate original question")
 decisions=["source_qv8" if x["prefix_len"]<85 else "random73" for x in selected]
 if decisions!=["source_qv8","random73"]*12:raise RuntimeError("profile alternation not frozen")
 os.umask(0o077);OUT.mkdir(parents=True,mode=0o700)
 for mode in ("dynamic","static_source","static_random"):
  for i,x in enumerate(selected):
   raw=struct.pack("<"+"i"*len(x["prefix_ids"]),*x["prefix_ids"])
   choices=(" ".join(str(x["suffix_ids"][k]) for k in "ABCD")+" "+str("ABCD".index(x["correct"]))+"\n").encode()
   frozen(OUT/mode/"eval/ppl_work"/f"w{i}.i32",raw)
   frozen(OUT/mode/"eval/ppl_work"/f"w{i}.choices",choices)
   if i==0:frozen(OUT/mode/"prompt_ids.i32",raw)
 for k,p in (("source_qv8",m.SOURCE),("random73",m.RANDOM73)):
  frozen(OUT/"masks"/(k+".txt"),p.read_bytes())
 private={"schema":"BEGLIN-Qv14-original-ARC-validation24-frozen-replication/1",
 "source":"allenai/ai2_arc ARC-Challenge validation cached arrow",
 "original_arrow_sha256":m.SHAS["arrow"],"original_tokenizer_sha256":m.SHAS["token"],
 "original_Qv14_4pilot_private_sha256":m.shaf(old),
 "original_4pilot_question_ids_excluded":True,
 "source_mask_sha256":m.SHAS["source"],"random73_mask_sha256":m.SHAS["random"],
 "source_only_length_selector_threshold":85,"sequence_of_selected_masks":decisions,
 "selection":"12 shortest? no: SHA-first 12 eligible original source questions of each length stratum; paired S/L interleaved; no labels/logits used",
 "exact_input_rows":selected,
 "model_output_not_seen_at_freeze":True}
 b=enc(private);frozen(OUT/"frozen_inputs_private.json",b)
 public={"schema":"BEGLIN-Qv14-original-ARC-validation24-mechanistic-replication-pre-forward-PUBLIC/1",
 "state":"FROZEN_SOURCE_ONLY_24_ALTERNATING_REQUESTS_BEFORE_NATIVE_OUTPUTS",
 "n":24,"dataset":"ARC-Challenge validation separate from Qv13 test",
 "previous_original_4pilot_excluded":True,
 "original_source_arrow_sha256":m.SHAS["arrow"],
 "tokenizer_sha256":m.SHAS["token"],"length_threshold":85,
 "length_range":[min(x["prefix_len"] for x in selected),max(x["prefix_len"] for x in selected)],
 "length_order":[x["prefix_len"] for x in selected],
 "profile_order":decisions,
 "private_sha256":sha(b),
 "exact_source_Qv8_SHA256":m.SHAS["source"],
 "exact_random73_SHA256":m.SHAS["random"],
 "actual_C_not_yet_executed_at_freeze":True,
 "quality_benefit_not_hypothesized":True,
 "static_reference_and_dynamic_runner_same_raw_prompt_answer_files":True,
 "no_operating_release_change":True}
 bb=enc(public);frozen(OUT/"frozen_manifest_PUBLIC.json",bb)
 print("QV14_REPLICATION_24_PRELOCKED",json.dumps({"n":24,"short":12,"long":12,
 "private_sha256":sha(b),"public_sha256":sha(bb),"original_eligible_after_four_excluded":len(eligible),"path":str(OUT)},sort_keys=True))
if __name__=="__main__":main()
