#!/usr/bin/env python3
"""P6 isolated real-worker acceptance: all admission paths, rejection and replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import statistics
import urllib.request
from pathlib import Path

import precision_context as pc
import precision_closed_loop as pcl
import precision_observability as po
import precision_closed_loop_acceptance_xox as p2
import precision_e2e_cost_acceptance_xox as p4
import precision_observability_acceptance_xox as p6
import production_serving_supervisor as base
import production_serving_supervisor_persistent as ps


def live_snapshot():
    with urllib.request.urlopen('http://127.0.0.1:18765/healthz', timeout=5) as f:
        health=json.load(f)
    return {
        'status': health['status'],
        'route_generation': health['route_generation'],
        'route_manifest_sha256': health['route_manifest_sha256'],
        'worker_pids': {k:v['pid'] for k,v in health['workers'].items()},
        'binary_sha256': p6.sha('/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu'),
        'adaptive_precision_enabled':health['adaptive_precision_enabled'],
        'auto_promotion_enabled':health['auto_promotion_enabled'],
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', required=True)
    ap.add_argument('--cost-evidence', required=True)
    ap.add_argument('--cost-sha256', required=True)
    ap.add_argument('--root', required=True)
    args=ap.parse_args()
    binary=Path(args.binary).resolve(strict=True)
    root=Path(args.root).resolve()
    if root.parent!=Path('/Users/xox/vdsp_serving') or not root.name.startswith('precision-observability-review-'):
        raise SystemExit('only a dedicated P6 scratch directory is allowed')
    root.mkdir(exist_ok=False)
    source=p6.head()
    subprocess.run(['git','-C',str(Path(__file__).resolve().parents[1]),'diff','--exit-code','HEAD'],check=True,stdout=subprocess.DEVNULL)
    cost=p6.load_cost(args.cost_evidence,args.cost_sha256,p6.sha(binary))
    candidates,trigger,combo,source_evidence=p2.evidence_snapshot()
    pre=live_snapshot()
    ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/'candidate')
    out={'schema':'beglin-p6-review-acceptance-v1','source_head':source,
         'binary_sha256':p6.sha(binary),'cost_evidence_sha256':args.cost_sha256,
         'production_touched':False,'production_before':pre,'source_evidence':source_evidence}
    successful=[]
    try:
        worker.start(); pid=worker.health()['pid']
        worker.configure_precision_closed_loop(candidates=candidates,trigger_evidence=trigger,
            combined_policy_evidence=p4.accepted_rows(combo),cost_evidence=cost,policy_e2e_weight=1.)
        lineage=root/'lineage.jsonl'; snapshot=root/'summary.json'
        worker.configure_precision_observability(lineage_path=lineage,snapshot_path=snapshot,strict=True)
        prompt=base._read_first_certified_prompt(); request=[(prompt,10)]
        successful.append(worker.submit_with_closed_loop_precision(request,signal={},admission_id='no-risk'))
        successful.append(worker.submit_with_closed_loop_precision(request,
            signal={'low_margin':True,'margin':.010715},admission_id='risk'))
        successful.append(worker.submit_with_closed_loop_precision(request,signal={},admission_id='hold'))
        successful.append(worker.submit_with_precision_policy(request,target_policy=p2.STARTUP,admission_id='restore'))
        ack_before=p6.sha(worker.ack_path); txn_before=p6.sha(worker.txn_path)
        rejected=False
        try:
            worker.submit_with_closed_loop_precision(request,
                signal={'high_entropy':True,'entropy':.5},admission_id='unaccepted-policy')
        except pcl.PrecisionClosedLoopError:
            rejected=True
        if not rejected:
            raise AssertionError('unaccepted policy was not rejected')
        if p6.sha(worker.ack_path)!=ack_before or p6.sha(worker.txn_path)!=txn_before:
            raise AssertionError('rejection changed the native transaction or ACK')
        # Raw A is deliberately a negative reference fixture. P6 observes it;
        # finite logits are not a claim of reference-token correctness.
        successful.append(worker.submit_base(request))
        successful.append(worker.submit(request))
        successful.append(worker.submit(request))
        successful.append(worker.submit_with_precision_policy(request,target_policy=p2.STARTUP,admission_id='final-restore'))
        observer=worker.precision_observability
        rows=observer.read_records(); po.verify_records(rows)
        summary=observer.summary()
        replay=po.summarize(rows)
        restarted=po.PrecisionObservability(lineage_path=lineage,snapshot_path=snapshot)
        persisted=json.loads(snapshot.read_text())
        expected_outcomes={'SUCCESS':8,'REJECTED':1}
        expected_paths={'closed_loop':4,'explicit_policy':2,'direct':1,'adaptive':2}
        risk_tokens=[successful[i]['responses'][0][8] for i in (1,5,6)]
        record_ms=[r['precision_observability']['recording_wall_ms'] for r in successful]
        final=worker.health()
        post=live_snapshot()
        assertions={
            'exactly_one_row_per_admission':len(rows)==9,
            'all_successes_recorded':all(r['precision_observability']['status']=='RECORDED' for r in successful),
            'outcome_counts':summary['outcome_counts']==expected_outcomes,
            'all_paths_covered':summary['admission_path_counts']==expected_paths,
            'pass_counts':summary['inference_pass_histogram']=={'0':1,'1':6,'2':2},
            'extra_pass_rate':summary['extra_pass_rate']==.25,
            'cache_counts':summary['cache_hits']==6 and summary['cache_misses']==2,
            'trigger_counts':summary['trigger_counts']=={'low_margin':3,'high_entropy':1},
            'observed_reference_recoveries':risk_tokens==[1224,1224,1224],
            'same_pid':all(r['worker_pid']==pid for r in rows) and final['pid']==pid,
            'restored_startup':final['runtime_policy_hash']==pc.policy_hash(p2.STARTUP),
            'exact_replay':summary==replay==persisted==restarted.summary(),
            'source_and_binary_identity':all(r['runtime_identity']['source_commit']==source and
                r['runtime_identity']['binary_sha256']==out['binary_sha256'] for r in rows),
            'no_response_payload_in_lineage':all('responses' not in r['outcome'] for r in rows),
            'production_unchanged':pre==post,
        }
        out.update(status='PASS' if all(assertions.values()) else 'FAIL',assertions=assertions,
                   pid=pid,summary=summary,risk_reference_tokens=risk_tokens,
                   negative_control_token8=successful[4]['responses'][0][8],
                   recording_wall_ms={'count':len(record_ms),'median':statistics.median(record_ms),'max':max(record_ms)},
                   lineage_sha256=p6.sha(lineage),snapshot_sha256=p6.sha(snapshot),
                   lineage_path=str(lineage),snapshot_path=str(snapshot),production_after=post)
    except Exception as exc:
        out.update(status='FAIL',error_type=type(exc).__name__,error=str(exc))
    finally:
        worker.stop(force=True)
    raw=json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n'
    (root/'result.json').write_text(raw)
    print(json.dumps({'status':out['status'],'source_head':source,'pid':out.get('pid'),
                      'assertions':out.get('assertions'),'summary':out.get('summary'),
                      'error':out.get('error'),'result_sha256':hashlib.sha256(raw.encode()).hexdigest()},indent=2))
    return 0 if out['status']=='PASS' else 2

if __name__=='__main__':
    raise SystemExit(main())
