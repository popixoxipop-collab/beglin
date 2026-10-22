#!/usr/bin/env python3
"""Projection packed weight cost for 0/1/2-n8 pair experiments, source-only.

Uses original BEGLIN Qwen28 n8/Q4 genuine kernel-byte ledger. These are
weights+group-scale projected bytes, NOT measured RSS, sustained latency,
or an equal-resource comparison to official 28-n8 masks.
"""
import hashlib,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/"research/oml_acs"))
import q_v2_async_activation_milp as v2
PRE=R/"configs/research/BEGLIN_QV12_TWO_N8_INTERACTION_PRELOCK_20261023.json"
OUT=R/"docs/research/BEGLIN_QV12_ONE_TWO_N8_PACKED_WEIGHT_COST_PUBLIC_20261023.json"
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 if OUT.exists():raise FileExistsError("source ledger append-only")
 cfg=json.loads(PRE.read_text())
 if sha(PRE.read_bytes())!="a565bdec341d675ce52a37132b06355760c2fef96b8e4329c17508a237e4374f":
  raise RuntimeError("Qv12 source prelock changed")
 role,b,stages,serial,mem=v2.verified_sources()
 q4=sum(28*mem[r]["q4"] for r in v2.ROLES)
 if q4!=736985088:raise RuntimeError("original 28Qwen Q4 projection byte ledger drift")
 costs={}
 for a in cfg["planned_arms"]:
  rows=a.split("__") if a!="q4" else []
  selected=[]
  for item in rows:
   name,layer=item.rsplit("_",1)
   if name not in mem or not layer.isdecimal() or not 0<=int(layer)<28:
    raise RuntimeError("unregistered source tensor")
   selected.append((name,int(layer)))
  extra=sum(mem[r]["n8"]-mem[r]["q4"] for r,l in selected)
  costs[a]={"n8_role_count":len(selected),"q4_role_count":196-len(selected),
    "q4_baseline_native_projection_bytes":q4,"incremental_n8_weight_bytes":extra,
    "projected_packed_projection_weight_bytes":q4+extra,
    "not_runtime_RSS_or_model_act_bytes":True}
 pub={"schema":"BEGLIN-Qv12-real-original-Apple-Qwen28-source-kernel-weight-costs/1",
  "status":"SOURCE_ONLY_7ARM_ACTUAL_Q4_N8_WEIGHT_AND_GROUP_SCALE_LEDGER",
  "original_BEGLIN_Q4_projection_baseline_bytes":q4,
  "source_PRELOCK_SHA256":sha(PRE.read_bytes()),
  "native_original_kernel_cost_verified":True,"separate_original_role_byte_costs":{
   r:{"q4":mem[r]["q4"],"n8":mem[r]["n8"],"increment_n8":mem[r]["n8"]-mem[r]["q4"]}
   for r in ("down_proj","v_proj","k_proj")},
  "native_one_two_bit_arm_costs":costs,
  "resource_fairness_note":"Different roles add different packed bytes, cannot interpret high absolute improvement as equal-byte improvement.",
  "not_official_28n8_serving_configuration":True,
  "actual_max_RSS_not_measured":True,
  "live_precision_auto_promotion":False}
 raw=(json.dumps(pub,sort_keys=True,indent=2)+"\n").encode()
 with OUT.open("xb") as f:f.write(raw);f.flush()
 print("QV12_REAL_KERNEL_PACKED_WEIGHT_BYTES_FROZEN",json.dumps({
    "sha":sha(raw),"costs":{k:v["projected_packed_projection_weight_bytes"] for k,v in costs.items()}},sort_keys=True))
if __name__=="__main__":main()
