#!/usr/bin/env python3
"""Independent read-only audit for P9 native-cache crossover result file."""
import argparse,hashlib,json,math,pathlib,random,statistics
SEED=20261014
EXPECTED_STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
EXPECTED_ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def check(result,quality,*,full=True):
 r=json.loads(pathlib.Path(result).read_text());q=json.loads(pathlib.Path(quality).read_text())
 assert r["schema"]=="beglin-hcp-p9-real-native-optin-packed-cache-v1"
 assert r["status"]=="REAL_XOX_RESEARCH_PAIRED_CACHE_ON_OFF_ONLY"
 n,blocks=(12,4) if full else (2,2)
 assert (r["prompt_domains"],r["blocks"])==(n,blocks)
 assert (r["full_admissions"],r["paired_admissions"])==(n*blocks*8,n*blocks*4)
 assert r["all_on_off_sequence_SHA_equal"] and r["all_P6_original_A_B_sequence_SHA_verified"]
 assert r["production_touched"] is False and r["automatic_live_promotion"] is False
 assert r["live_preimage"]==r["live_postimage"]
 assert r["live_preimage"]=={"stage":EXPECTED_STAGE,"route":EXPECTED_ROUTE,"pids":[35481,35493]}
 assert r["source_commit"]=="a46154d8d3598ef1c7b0d51cd1d4447090c89ac4"
 assert r["P9_MLX_source_SHA256"]=="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
 assert r["model_index_sha256"]=="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
 assert digest(quality)=="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
 original={u["prompt_id"]:{x:u[x]["sequence_sha256"] for x in ("A","B")} for u in q["records"]}
 assert len(original)==12
 cells={}
 for row in r["records"]:
  k=(row["block"],row["prompt_id"],row["position"],row["policy"])
  if row["prompt_id"] not in original:raise ValueError("unknown frozen input")
  if row["seq_sha256"]!=original[row["prompt_id"]][row["policy"]]:
   raise ValueError("unmatched P6 sealed real token sequence")
  assert row["generated_count"] in (1,2,3,4) and 104<=row["input_tokens"]<=112
  assert row["rss_bytes"]<=14*1024**3
  for tm in ("admission_ms","roundtrip_ms","engine_ms","transition_ms"):
   assert math.isfinite(float(row[tm])) and float(row[tm])>=0
  assert row["engine_ms"]>0
  c=cells.setdefault(k,{})
  mode="on" if row["cache_enabled"] else "off"
  assert mode not in c
  c[mode]=row
 assert len(cells)==n*blocks*4
 diffs=[];blocks_data={}
 for k,m in cells.items():
  assert set(m)=={"on","off"}
  a,b=m["off"],m["on"]
  assert a["switched"]==b["switched"] and a["policy_order"]==b["policy_order"]
  if a["switched"]:
   delta=b["admission_ms"]-a["admission_ms"]
   diffs.append(delta)
   blocks_data.setdefault(k[0],[]).append(delta)
 assert len(blocks_data)==blocks and len(diffs)>=n*blocks
 assert len(r["native_events"])==2*blocks
 for row in r["native_events"]:
  if row["enabled"]:
   assert row["HIT name="]>=4 and row["STORE name="]>=4
  else:
   assert row["HIT name="]==row["STORE name="]==0
  assert row["SOURCE_MISMATCH"]==row["BUDGET_SKIP"]==row["STORE_SKIPPED"]==0
 rng=random.Random(SEED);means=[statistics.mean(blocks_data[k]) for k in sorted(blocks_data)]
 sims=sorted(statistics.mean([means[rng.randrange(blocks)] for _ in range(blocks)]) for __ in range(10000))
 reported=r["summary"]["switch_paired_delta_on_minus_off_ms"]
 assert abs(statistics.mean(diffs)-reported["mean"])<1e-8
 assert abs(sims[250]-reported["temporal_block_bootstrap_ci95_ms"][0])<1e-8
 assert abs(sims[9750]-reported["temporal_block_bootstrap_ci95_ms"][1])<1e-8
 audit={"schema":"beglin-hcp-p9-independent-evidence-audit-v1","native_result_sha256":digest(result),
  "frozen_P6_quality_SHA256":digest(quality),"raw_tokens_absent":True,
  "actual_native_admissions":len(r["records"]),"strict_matched_pairs":len(cells),
  "cache_mutation_error_count":0,
  "paired_switch_on_minus_off_mean_ms":statistics.mean(diffs),
  "block_bootstrap_95ci_ms":[sims[250],sims[9750]],
  "source_P6_token_SHA_verified":True,"ACK_and_operator_route_untouched":True,
  "production_promotion_allowed":False}
 return audit
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--result",type=pathlib.Path,required=True)
 p.add_argument("--quality",type=pathlib.Path,required=True)
 p.add_argument("--output",type=pathlib.Path,required=True)
 p.add_argument("--smoke",action="store_true")
 a=p.parse_args();v=check(a.result,a.quality,full=not a.smoke)
 with a.output.open("x") as f:json.dump(v,f,indent=2,sort_keys=True);f.write("\n")
 print("P9_INDEPENDENT_AUDIT_PASS",json.dumps(v,sort_keys=True))
