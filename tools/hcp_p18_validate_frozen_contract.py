#!/usr/bin/env python3
"""P18 frozen scoring-contract preflight/negative tests. Never runs inference."""
import copy,hashlib,json,pathlib,sys
ROOT=pathlib.Path("/Volumes/D50/TailnetCommander/work/hcp-p18-external-multimetric-20261015")
sys.path.insert(0,str(ROOT))
import hcp_p18_quality_gate as quality
P_SHA="77e12dfff7fef48894c28413b3a8d7b9bd4bfbcc723c26d88ed1d02b4231484c"
C_SHA="19e7080b7208b8c7dd6fd06a656bb400dd2a9f72ca5f5674a80f5cc9a654cf65"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fail(call,name):
 try:call()
 except (ValueError,KeyError,TypeError):return
 raise AssertionError("P18 negative safety gate unexpectedly accepted: "+name)
def main():
 ppath=ROOT/"p18_protocol_FROZEN.json";cpath=ROOT/"p18_gate_cohort_PRIVATE.json"
 if sha(ppath)!=P_SHA or sha(cpath)!=C_SHA:raise RuntimeError("frozen P18 content SHA mismatch")
 p=quality.frozen_protocol(json.loads(ppath.read_text()))
 c=json.loads(cpath.read_text())
 checked=quality.frozen_cohort(c,p)
 if len(checked)!=480 or p.get("cohort_sha256")!=C_SHA:
  raise RuntimeError("P18 preregistered protocol/cohort mismatch")
 n=0
 m=copy.deepcopy(p);m["statistics"]["nll_margin_nats_per_gold_token"]=.051
 fail(lambda:quality.frozen_protocol(m),"margin changed");n+=1
 m=copy.deepcopy(p);m["status"]="DRAFT"
 fail(lambda:quality.frozen_protocol(m),"unfrozen protocol");n+=1
 m=copy.deepcopy(p);m["datasets"]["gsm8k"]["n"]=159
 fail(lambda:quality.frozen_protocol(m),"sample changed");n+=1
 m=copy.deepcopy(c);m["items"].pop()
 fail(lambda:quality.frozen_cohort(m,p),"missing item");n+=1
 m=copy.deepcopy(c);m["items"][0]["shot_count"]=4
 fail(lambda:quality.frozen_cohort(m,p),"unsupported shots");n+=1
 m=copy.deepcopy(c);m["items"][0]["dataset_revision"]="f"*64
 fail(lambda:quality.frozen_cohort(m,p),"dataset revision drift");n+=1
 m=copy.deepcopy(c);m["items"][0]["gold_token_count"]=1000
 fail(lambda:quality.frozen_cohort(m,p),"gold beyond budget");n+=1
 m=copy.deepcopy(c);m["items"][0]["prompt_token_sha256"]=""
 fail(lambda:quality.frozen_cohort(m,p),"token SHA missing");n+=1
 # Evidence-free evaluation is forbidden: no scores and no independent raw audit.
 scores={"schema":quality.SCHEMA,"protocol_sha256":P_SHA,"cohort_sha256":C_SHA,"records":[]}
 fail(lambda:quality.validate_rows(p,checked,scores,P_SHA,C_SHA),"no real audited logits");n+=1
 out={"schema":"beglin-hcp-p18-preregistered-contract-audit-v1",
      "status":"P18_FROZEN_PROTOCOL_AND_480_COHORT_CONTRACT_PASS_NOT_NATIVE",
      "protocol_sha256":P_SHA,"cohort_sha256":C_SHA,
      "independent_480_input_audit_sha256":p["independent_480_token_audit_sha256"],
      "frozen_scoring_margins":{"NLL_nats_per_token":.05,"task_accuracy_delta":-.05,
                               "latency_geomean_ratio":1.10,"latency_empirical_p95_ratio":1.15},
      "negative_fail_closed_checks":n,
      "raw_BF16_logit_observations":0,"raw_native_A_B_observations":0,
      "native_4096_context_verified":False,"production_touched":False,
      "automatic_live_promotion":False}
 path=ROOT/"p18_frozen_protocol_contract_receipt.json"
 if path.exists():raise FileExistsError("No overwrites of frozen protocol receipt")
 with path.open("x") as f:json.dump(out,f,sort_keys=True,indent=2);f.write("\n")
 print(json.dumps({"status":out["status"],"frozen_p_sha":P_SHA,"cohort_sha":C_SHA,
                   "negative_checks":n,"receipt_sha256":sha(path)},sort_keys=True))
if __name__=="__main__":main()
