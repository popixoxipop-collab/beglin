#!/usr/bin/env python3
"""BEGLIN P18 source-pinned 480-item EOE/D50 admission & resumable view.

Read-only. No model import, no native / BF16 forward, no queue mutation.
Counts only immutably verified existing source receipts; never certifies
completion or chooses items after observing scores.
"""
from __future__ import annotations
import hashlib,json,pathlib,sys
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
sys.path.insert(0,str(ROOT))
from hcp_p18_eoe_d50_model_copy_seal import check_state as copy_incomplete
FROZEN="c5dc5ff1971dde5ca0809c458bd218a42905d0ca8836db7c84603abb06f401df"
PREVIOUS_PROGRESS="dae63937830b7e382bb1e59c145df97fe597f2efa3aee98e578d404435b4d0b3"
REQUIRED_MODEL_RECEIPTS=("EOE_P18_BF16_MODEL_COPY_RECEIPT.json","EOE_P18_MOE_BASE_COPY_RECEIPT.json")
def sha(p):
 p=pathlib.Path(p)
 if not p.is_file() or p.is_symlink():raise ValueError("missing original provenance file")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def view():
 if sha(ROOT/"p18_multishot_frozen_PRIVATE.json")!=FROZEN or sha(ROOT/"P18_480_LIVE_REAL_EVIDENCE_PROGRESS_20261015.json")!=PREVIOUS_PROGRESS:
  raise RuntimeError("frozen corpus or prior real 6-question coverage evidence changed")
 bank=json.loads((ROOT/"p18_multishot_frozen_PRIVATE.json").read_text())
 if len(bank["items"])!=480 or sum(i["gold_token_count"] for i in bank["items"])!=673:
  raise RuntimeError("not original frozen 480 test items/673 gold token positions")
 previous=json.loads((ROOT/"P18_480_LIVE_REAL_EVIDENCE_PROGRESS_20261015.json").read_text())
 if previous["directly_attested_original_BF16_plus_AB_questions"]!=6:
  raise RuntimeError("previous six original real question receipts not attested")
 exists={x["id"] for x in previous["question_observations"]}
 if len(exists)!=6 or not exists.issubset({x["item_id"] for x in bank["items"]}):
  raise RuntimeError("prior source question identity overlap or missing")
 new_dir=ROOT/"eoe_480_stream"
 copied=copy_incomplete()
 receipts={}
 for name in REQUIRED_MODEL_RECEIPTS:
  p=ROOT/name
  receipts[name]=p.is_file() and not p.is_symlink()
 has_both=(not copied and all(receipts.values()))
 completed=[]
 for idx,item in enumerate(bank["items"]):
  dir=new_dir/("unit_%03d"%idx)
  if (dir/"independent_audit.json").is_file():
   record=json.loads((dir/"independent_audit.json").read_text())
   if (record.get("item_id")!=item["item_id"] or record.get("index")!=idx or
      record.get("source_frozen_480_sha256")!=FROZEN or
      record.get("status")!="P18_REAL_BF16_PLUS_NATIVE_AB_INDEPENDENT_D50_PASS_NOT_TASK"):
    raise ValueError("EOE source evaluation unit has unauthenticated receipt")
   completed.append(item["item_id"])
 old_order={q["item_id"]:idx for idx,q in enumerate(bank["items"])}
 targets=[q for q in bank["items"] if q["item_id"] not in exists and q["item_id"] not in completed]
 nxt=targets[0] if targets else None
 out={
  "schema":"beglin-hcp-p18-EOE-offloaded-480-admission-status-v1",
  "status":"P18_EOE_MODELS_PENDING_ORIGINAL_SHA_COPY" if not has_both else "P18_EOE_MODEL_COPIES_READY_FOR_SAFE_ITEM_RESOURCE_CHECK",
  "model_copy_missing_or_partial_count":len(copied),
  "model_copies_sha_sealed":receipts,
  "original_frozen_test_items":480,
  "original_frozen_gold_positions":673,
  "prior_verified_actual_original_BF16_and_AB_questions":6,
  "new_D50_port_evaluation_items":len(completed),
  "next_pre_registered_source_index":old_order[nxt["item_id"]] if nxt else None,
  "next_pre_registered_item_id":nxt["item_id"] if nxt else None,
  "question_count_remaining_for_full_NLL":480-6-len(completed),
  "full_teacher_BF16_A_B_task_generation_certified":False,
  "three_warm_latency_trials_certified":False,
  "stratified_480_question_bootstrap_gate_certified":False,
  "XOX_storage_cleanup_performed":False,
  "source_XOX_and_EOE_original_model_changes":False,
  "production_touched":False,"automatic_live_promotion":False}
 print(json.dumps(out,sort_keys=True,indent=2))
if __name__=="__main__":view()
