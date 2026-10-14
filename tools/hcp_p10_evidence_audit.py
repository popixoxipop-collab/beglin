#!/usr/bin/env python3
"""P10 independent read-only evidence audit. No native inference or service writes."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,pathlib,random,statistics

SEED=20261014
BANK="7523f8e8ef5d2dea21391ad18ab575606b5804299386bf715a9647c815758083"
FROZEN="93f6f9f21983cb0c93e78b813b2972724aff45501b773943be2020574ada3b9f"
P6="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
BINARY="103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd"

def hash_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def qlinear(values,frac):
    arr=sorted(float(v) for v in values)
    t=(len(arr)-1)*frac;k=math.floor(t);j=math.ceil(t)
    return arr[k]+(arr[j]-arr[k])*(t-k)
def check(base:pathlib.Path,repo:pathlib.Path):
    def load(name):
        p=base/(name+".json")
        if not p.is_file() or p.is_symlink():raise ValueError("P10 missing evidence "+name)
        return json.loads(p.read_text()),hash_file(p)
    bank=repo/"experiments/hcp_p10/new_12domain_holdout.json"
    frozen=repo/"experiments/hcp_p10/holdout_token_manifest.json"
    if hash_file(bank)!=BANK or hash_file(frozen)!=FROZEN:raise ValueError("P10 source/freeze changed")
    q=json.loads(frozen.read_text())
    if q.get("first_use_only_relative_to_original_P4") is not True or q.get("count")!=12:
        raise ValueError("P10 not frozen before evaluation")
    if len({r["token_sha256"] for r in q["records"]})!=12:
        raise ValueError("P10 token duplication")

    results={};digests={}
    for name in ("new_holdout","restart","two_slots","steady_tail"):
        results[name],digests[name]=load(name)

    h=results["new_holdout"];restart=results["restart"];slots=results["two_slots"];tail=results["steady_tail"]
    if h.get("schema")!="beglin-hcp-p10-new-short-answer-native-parity-v1":
        raise ValueError("wrong holdout contract")
    if h["total_native_requests"]!=48 or h["paired_on_off_generations"]!=24:
        raise ValueError("holdout incomplete")
    if h["token_manifest_sha256"]!=FROZEN or h["source_bank_sha256"]!=BANK:
        raise ValueError("not the sealed holdout corpus")
    matches={};seen=set()
    for r in h["records"]:
        key=(r["item_id"],r["policy"],r["mode"])
        if key in seen:raise ValueError("duplicate holdout cell")
        seen.add(key)
        matches.setdefault((r["item_id"],r["policy"]),{})[r["mode"]]=r
        if (r["policy"] not in ("A","B") or r["mode"] not in ("on","off") or
            not 1<=r["generated_length"]<=12 or not 20<=r["tokens_in"]<=112):
            raise ValueError("invalid first-use response")
    if len(matches)!=24 or any(set(x)!={"on","off"} for x in matches.values()):
        raise ValueError("incomplete paired heldout cells")
    for a in matches.values():
        if any(a["on"][k]!=a["off"][k] for k in ("output_sequence_sha256","generated_length",
                    "elementary_reference_substring_match")):
            raise ValueError("P10 holdout cache changed output")
    if h.get("BF16_reference_measured") is not False:raise ValueError("false teacher claim")
    count={p:sum(v["off"]["elementary_reference_substring_match"] for key,v in matches.items() if key[1]==p)
           for p in ("A","B")}

    if (restart.get("schema")!="beglin-hcp-p10-worker-restart-recovery-v1" or
        restart.get("status")!="IDLE_CRASH_AND_GRACEFUL_NATIVE_RESTART_COMPLETED" or
        restart["successfully_recovered_requests"]!=9 or
        restart["inflight_txn_crash_not_tested"] is not True):
        raise ValueError("restart contract missing")
    rr=restart["restart_records"]
    if len(rr)!=2 or not all(x["new_process"] and x["policy_reset_to_A"] for x in rr):
        raise ValueError("restart did not reset worker policy and process")
    if (rr[0]["stale_request_refused"] is not True or
        rr[0]["old_policy_epoch"]<=rr[0]["new_runtime_epoch"]):
        raise ValueError("restart reused stale transaction")
    for k in ("STORE name=","HIT name="):
        if restart["postrestart_cache_events"][k]<1:raise ValueError("restart cache not populated")
    if (slots.get("schema")!="beglin-hcp-p10-two-slot-real-native-mla-v1" or
        slots.get("status")!="TWO_SLOT_BATCH_PASS" or slots["batch_admissions"]!=8 or
        slots["inferred_sequences"]!=16 or slots["slots"]!=2 or
        slots["max_worker_rss_bytes"]>=14*1024**3):
        raise ValueError("two-slot evidence failed")
    if not all(x["batch_reqs"]==2 and x["both_outputs_match_P6"] for x in slots["records"]):
        raise ValueError("two-slot source sequence error")

    if tail.get("schema")!="beglin-hcp-p10-repeated-steady-block-tail-v1":
        raise ValueError("wrong steady tail schema")
    rows=tail["rows"]
    if len(rows)!=96 or len(tail["run_receipts"])!=8:
        raise ValueError("P10 steady states incomplete")
    idx={};byblock={}
    for r in rows:
        if r["policy"]!="A" or r.get("seq_sha_match_P6") is not True or r["cache_enabled"]!=r["enabled"]:
            raise ValueError("P10 steady policy or seq mismatch")
        key=(r["block"],r["prompt_id"],r["enabled"])
        if key in idx or r.get("switched",False):
            raise ValueError("P10 duplicate/transition cell")
        idx[key]=r
        byblock.setdefault((r["block"],r["enabled"]),[]).append(r["admission_ms"])
    if len(idx)!=96 or len(byblock)!=8 or any(len(v)!=12 for v in byblock.values()):
        raise ValueError("P10 missing steady pair")
    means=[]
    for b in range(4):
        vals=[]
        ids=sorted({k[1] for k in idx if k[0]==b})
        if len(ids)!=12:raise ValueError("P10 non-disjoint timing domain set")
        for prompt in ids:
            vals.append(idx[(b,prompt,True)]["admission_ms"]-
                        idx[(b,prompt,False)]["admission_ms"])
        means.append(statistics.mean(vals))
    rng=random.Random(SEED)
    draws=10000
    ci=sorted(statistics.mean([means[rng.randrange(4)] for _ in range(4)]) for __ in range(draws))
    true_mean=statistics.mean(means)
    ps=tail["paired"]
    if (abs(true_mean-ps["mean_on_minus_off_ms"])>1e-7 or
        abs(ci[250]-ps["block_bootstrap95_ci_ms"][0])>1e-7 or
        abs(ci[9750]-ps["block_bootstrap95_ci_ms"][1])>1e-7):
        raise ValueError("P10 steady block bootstrap mismatched")
    for enabled in (False,True):
        subset=[r["admission_ms"] for r in rows if r["enabled"]==enabled]
        stat=tail["summary"]["on" if enabled else "off"]
        if (len(subset)!=stat["n"] or
            abs(qlinear(subset,.95)-stat["p95_ms"])>1e-7):
            raise ValueError("P10 reported p95 not reproducible")
    for receipt in tail["run_receipts"]:
        c=receipt["cache_events"]
        if receipt["cache_enabled"]:
            if c["STORE name="]<4 or c["HIT name="]<4:raise ValueError("missing packed cache")
        else:
            if c["STORE name="] or c["HIT name="]:raise ValueError("unexpected cache")
        if any(c[k] for k in ("SOURCE_MISMATCH","STORE_SKIPPED","BUDGET_SKIP")):
            raise ValueError("cache corruption/budget error")

    status={
        "schema":"beglin-hcp-p10-independent-integrity-audit-v1",
        "status":"P10_SCOPE_RESEARCH_EVIDENCE_INDEPENDENTLY_RECALCULATED",
        "result_sha256":digests,
        "source_holdout_sha256":BANK,"frozen_token_manifest_sha256":FROZEN,
        "separate_new_domains":12,"new_heldout_native_admissions":48,
        "new_heldout_cache_parity_pairs":len(matches),
        "elementary_rubric_substring_match_A":count["A"],
        "elementary_rubric_substring_match_B":count["B"],
        "exploratory_rubric_not_external_accuracy":True,
        "BF16_teacher_not_run":True,
        "idle_crash_and_graceful_restart_success":True,
        "inflight_rollback_not_retested_by_P10":True,
        "two_slot_native_batch_admissions":8,"two_slot_source_sha_matched_claimed_by_runner":True,
        "independent_raw_two_slot_tokens_not_available":True,
        "steady_tail_requests":len(rows),"steady_tail_repeated_blocks":4,
        "steady_mean_on_minus_off_ms":true_mean,
        "steady_block_ci95_ms":[ci[250],ci[9750]],
        "steady_p95_off_ms":tail["summary"]["off"]["p95_ms"],
        "steady_p95_on_ms":tail["summary"]["on"]["p95_ms"],
        "main_merged":False,"automatic_live_promotion":False}
    return status

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",type=pathlib.Path,required=True)
    ap.add_argument("--repository",type=pathlib.Path,required=True)
    ap.add_argument("--output",type=pathlib.Path,required=True)
    a=ap.parse_args()
    report=check(a.evidence,a.repository)
    with a.output.open("x") as f:json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
    print("P10_AUDIT_PASS",json.dumps(report,sort_keys=True),flush=True)
