#!/usr/bin/env python3
"""BEGLIN Qv13 postpilot 240 robust, paired original-C NLL audit.
Produces ONLY aggregate statistics. No model reruns, no mask training/selection.
"""
import argparse,hashlib,json,math,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
LAB=pathlib.Path("/Volumes/D50/TailnetCommander/work/beglin-qv13-20261023")
ARMS=("q4","source_qv8","random73","random137","random211","random389","random577","original_v2")
RANDS=("random73","random137","random211","random389","random577")
DOMAINS=("arc128","ob128")
PRE_SHA="5cb944375fa271c4528b4846f4b3c22cdf8179d4a38a2f5c17babbc03c4df92a"
PRIVATE_SHA="79826eb944ac2a4f87b637a430471f069a3789da37cdb12a9b7d1060d49db9b8"
PUBLIC_SHA="9d045e1056e5ec8bc3a093a8a3d198277cbbc4083fb054a929b566cb2bb03c8c"
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(1<<18),b""):h.update(chunk)
 return h.hexdigest()
def need(cond,msg):
 if not cond:raise RuntimeError("FAIL_CLOSED_"+msg)
def med(x):return statistics.median(x)
def trim(x,p=.10):
 y=sorted(x);k=int(len(y)*p);return sum(y[k:len(y)-k])/(len(y)-2*k)
def winsor(x,p=.10):
 y=sorted(x);k=int(len(y)*p);return sum(max(y[k],min(y[-k-1],v)) for v in x)/len(x)
def pctl(x,p=.95):
 return sorted(x)[min(len(x)-1,int(p*len(x)))]
def run():
 pre=ROOT/"configs/research/BEGLIN_QV13_ACTUAL_NATIVE_EVAL_PRELOCK_20261023.json"
 pub=ROOT/"docs/research/BEGLIN_QV13_ORIGINAL_NATIVE_256_PREEXEC_LOCK_PUBLIC_20261023.json"
 lockp=LAB/"qv13_lock_private.json"
 need(sha(pre)==PRE_SHA,"prelock")
 need(sha(pub)==PUBLIC_SHA,"public_freeze")
 need(sha(lockp)==PRIVATE_SHA,"private_lock")
 lock=json.loads(lockp.read_text())
 need(lock["prelock_sha256"]==PRE_SHA,"prelock_binding")
 import importlib.util
 auditpath=ROOT/"tools/qv13_native_log_crosscheck.py"
 spec=importlib.util.spec_from_file_location("qv13_independent_original_native_C",auditpath)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for domain in DOMAINS:
  status=mod.scan(domain)
  need(status["status"]=="ALL_EIGHT_NATIVE_C_LOG_REPARSE_PASS","incomplete_C_"+domain)
 cases={m:{d:{} for d in DOMAINS} for m in ARMS}
 for domain in DOMAINS:
  manifest=LAB/"native_runs"/domain/"input_manifest.json"
  need(sha(manifest)==lock["batches"][domain]["manifest_sha256"],"manifest_"+domain)
  for arm in ARMS:
   receipt=LAB/"native_runs"/domain/arm/"native_qv13_original_C_receipt.json"
   d=json.loads(receipt.read_text())
   need(d["n"]==128 and d["preexecution_lock_sha256"]==PRIVATE_SHA,"receipt_"+arm+domain)
   for name,arr in (("nll",d["article_fullvocab_gold_NLL"]),("correct",d["article_correct"])):
    need(len(arr)==128,"raw_length")
    cases[arm][domain][name]=arr[8:]
 for arm in ARMS:
  need(sum(len(cases[arm][d]["nll"]) for d in DOMAINS)==240,"split_size")
 src=[z for d in DOMAINS for z in cases["source_qv8"][d]["nll"]]
 summary={}
 for arm in ARMS:
  xs=[z for d in DOMAINS for z in cases[arm][d]["nll"]]
  acc=[z for d in DOMAINS for z in cases[arm][d]["correct"]]
  summary[arm]={"n":240,"correct":sum(acc),
    "mean_gold_token_fullvocab_NLL":statistics.mean(xs),
    "median_gold_token_fullvocab_NLL":med(xs),
    "p95_gold_token_fullvocab_NLL":pctl(xs),
    "count_gold_token_NLL_over5":sum(v>5 for v in xs),
    "count_gold_token_NLL_over10":sum(v>10 for v in xs),
    "projection_packed_bytes":lock["original_q4_packed_bytes"] if arm=="q4" else lock["masks"][arm]["projection_packed_bytes"]}
 comparison={}
 for arm in RANDS+("q4","original_v2"):
  other=[z for d in DOMAINS for z in cases[arm][d]["nll"]]
  delta=[x-y for x,y in zip(other,src)]
  top=sorted(delta,reverse=True)
  overall=sum(delta)
  comparison[arm]={
   "interpretation":"positive comparator_minus_Qv8 favors Qv8 for NLL",
   "n":240,"mean_comparator_minus_Qv8_NLL":statistics.mean(delta),
   "median_paired_diff":med(delta),
   "trimmed10pct_paired_mean":trim(delta),
   "winsor10pct_paired_mean":winsor(delta),
   "Qv8_lower_NLL_items":sum(v>0 for v in delta),
   "comparator_lower_NLL_items":sum(v<0 for v in delta),
   "top5_positive_diffs_fraction_of_net_sum":sum(top[:5])/overall if abs(overall)>1e-12 else None,
   "per_dataset_mean_diff":{d:statistics.mean([a-b for a,b in zip(cases[arm][d]["nll"],cases["source_qv8"][d]["nll"])]) for d in DOMAINS},
   "Qv8_correct_minus_comparator":summary["source_qv8"]["correct"]-summary[arm]["correct"]
  }
 avg_random={d:[sum(cases[arm][d]["nll"][i] for arm in RANDS)/5 for i in range(120)] for d in DOMAINS}
 aggregate={
  "mean_random_minus_Qv8_NLL_by_dataset":{d:statistics.mean([a-b for a,b in zip(avg_random[d],cases["source_qv8"][d]["nll"])]) for d in DOMAINS},
  "Qv8_vs_average_random_mean_NLL":statistics.mean([a-b for d in DOMAINS for a,b in zip(avg_random[d],cases["source_qv8"][d]["nll"])]),
 }
 return {
  "schema":"BEGLIN-Qv13-postpilot-240-robust-NLL-accuracy-native-C-independently-reparsed/1",
  "status":"ALL_16_NATIVE_C_LOGS_REPARSED_ROBUST_EXPLORATORY_NO_DYNAMISM_CLAIM",
  "data":"Frozen ARC-Challenge test 120 + OpenBookQA test 120, first 8 per domain excluded",
  "n":240,"original_Qwen_layers":28,
  "source_prelock_sha256":PRE_SHA,"private_preexecution_lock_sha256":PRIVATE_SHA,
  "source_public_freeze_sha256":PUBLIC_SHA,
  "per_arm":summary,"paired_comparison":comparison,"five_random_control_aggregate":aggregate,
  "limitations":[
   "Descriptive robust statistics; not a new preregistered primary hypothesis",
   "NLL is gold next-token full-vocab softmax, not conditional four-option NLL",
   "Models use fixed masks; dynamic request-level selection not demonstrated",
   "Results can be driven by a small set of outlier high-loss questions",
   "Public test questions may have appeared in model pretraining",
   "Original prereg explanation has SHA/order errata (preserved separately)",
   "Pairwise 240 selection was defined after observing 16 pilot results",
   "Latency and GPU precision-switch costs are not evaluated"],
  "raw_item_text_tokens_and_losses_not_published":True,
  "no_main_merge_or_runtime_release_change":True
 }
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--output",type=pathlib.Path)
 args=p.parse_args();result=run();s=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
 if args.output:
  args.output.parent.mkdir(parents=True,exist_ok=True)
  with args.output.open("xb") as f:f.write(s)
  print(json.dumps({"status":result["status"],"path":str(args.output),"sha256":hashlib.sha256(s).hexdigest(),"n":240},sort_keys=True))
 else:print(s.decode())
if __name__=="__main__":main()
