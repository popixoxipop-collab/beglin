#!/usr/bin/env python3
"""P7 accelerated XOX soak: repeated precision transitions, rotation and fault boundaries."""
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
    ap.add_argument('--binary',required=True); ap.add_argument('--cost-evidence',required=True); ap.add_argument('--cost-sha256',required=True); ap.add_argument('--root',required=True); ap.add_argument('--admissions',type=int,default=60)
    args=ap.parse_args(); binary=Path(args.binary).resolve(strict=True); root=Path(args.root).resolve()
    if root.parent!=Path('/Users/xox/vdsp_serving') or not root.name.startswith('precision-p7-soak-'): raise SystemExit('unsafe P7 root')
    root.mkdir(exist_ok=False); source=p6.head(); cost=p6.load_cost(args.cost_evidence,args.cost_sha256,p6.sha(binary)); candidates,trigger,combo,src=p2.evidence_snapshot(); pre=live_snapshot(); ps.PERSISTENT_BINARY=binary
    worker=ps.AdaptivePersistentRouteWorker(route=base.candidate_route(),root=root/'candidate'); out={'schema':'beglin-p7-soak-v1','source_head':source,'production_before':pre,'production_touched':False}
    try:
      worker.start(); pid=worker.health()['pid']; worker.configure_precision_closed_loop(candidates=candidates,trigger_evidence=trigger,combined_policy_evidence=p4.accepted_rows(combo),cost_evidence=cost,policy_e2e_weight=1.)
      lineage=root/'lineage.jsonl'; snapshot=root/'summary.json'; worker.configure_precision_observability(lineage_path=lineage,snapshot_path=snapshot,strict=True); identity=dict(worker.precision_observability.identity); worker.precision_observability=po.PrecisionObservability(lineage_path=lineage,snapshot_path=snapshot,strict=True,identity=identity,max_active_records=10)
      prompt=base._read_first_certified_prompt(); req=[(prompt,10)]; tokens=[]
      for i in range(args.admissions):
        mode=i%6
        if mode==0: r=worker.submit_with_closed_loop_precision(req,signal={},admission_id=f'p7-{i:04d}-base')
        elif mode==1: r=worker.submit_with_closed_loop_precision(req,signal={'low_margin':True,'margin':.010715},admission_id=f'p7-{i:04d}-risk')
        elif mode==2: r=worker.submit_with_closed_loop_precision(req,signal={},admission_id=f'p7-{i:04d}-hold')
        elif mode==3: r=worker.submit_with_precision_policy(req,target_policy=p2.STARTUP,admission_id=f'p7-{i:04d}-restore')
        elif mode==4: r=worker.submit(req)
        else: r=worker.submit_with_precision_policy(req,target_policy=p2.STARTUP,admission_id=f'p7-{i:04d}-final')
        tokens.append(r['responses'][0][8])
        if worker.health()['pid']!=pid: raise AssertionError('worker PID changed during soak')
      summary=worker.precision_observability.summary(); rotation=worker.precision_observability.rotation_status(); restarted=po.PrecisionObservability(lineage_path=lineage,snapshot_path=snapshot,max_active_records=10,identity=identity)
      post=live_snapshot(); assertions={'admission_count':summary['admissions']==args.admissions,'same_pid':worker.health()['pid']==pid,'rotated':rotation['sealed_segments']>=5,'all_segments_retained':rotation['retained_segment_files']==rotation['sealed_segments'],'restart_summary_equal':restarted.summary()==summary,'finite':summary['finite_logits_rate']==1.0,'extra_passes':summary['extra_pass_admissions']>=5,'cache_reuse':summary['cache_hits']>summary['cache_misses'],'startup_restored':worker.health()['runtime_policy_hash']==pc.policy_hash(p2.STARTUP),'production_unchanged':pre==post}
      out.update(status='PASS' if all(assertions.values()) else 'FAIL',assertions=assertions,pid=pid,summary=summary,rotation=rotation,token8_histogram={str(x):tokens.count(x) for x in sorted(set(tokens))},production_after=post)
    except Exception as exc: out.update(status='FAIL',error_type=type(exc).__name__,error=str(exc))
    finally: worker.stop(force=True)
    raw=json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n'; (root/'result.json').write_text(raw); print(json.dumps({'status':out['status'],'source_head':source,'assertions':out.get('assertions'),'summary':out.get('summary'),'rotation':out.get('rotation'),'error':out.get('error'),'result_sha256':hashlib.sha256(raw.encode()).hexdigest()},indent=2)); return 0 if out['status']=='PASS' else 2

if __name__=='__main__':
    raise SystemExit(main())
