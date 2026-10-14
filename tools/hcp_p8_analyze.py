"""Read-only P6 cost decomposition and explicitly NON-native workload replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import pathlib
import statistics
from hcp_p8_governor import Governor, Runtime, Forecast, digest

SPEED_SHA='a27a1181f9fa4aec33d4d3a09cad25ed9e04585a779d8e7e57fbbed03071a31e'
QUALITY_SHA='2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104'
ORDER_SHA='43c5cae9a3c72c1986bfd013915f325375a9d959c0520bee9e201ce5dbf5bb6f'
POLICY={'A':'dbee11614bb74073e0751bbf9fad0f67f0b99396256cb88589df80701f363ef1',
        'B':'e471dfb075ca86cd0c5341b972258df7991847bb3f1557b69447c8b6168042ae'}


def load_pinned(path, expected):
    data=pathlib.Path(path).read_bytes()
    actual=hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError('source SHA mismatch: '+pathlib.Path(path).name)
    return json.loads(data)


def decompose(row):
    fields=('transition_ms','engine_ms','roundtrip_ms','admission_wall_ms')
    vals=[row[k] for k in fields]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in vals):
        raise ValueError('non-finite or malformed timing')
    if vals[0]<0 or any(a>b+1e-6 for a,b in zip(vals,vals[1:])):
        raise ValueError('timing nesting violated')
    out={'native_transition_before_ACK_ms':vals[0],
         'remaining_native_including_ACK_ms':vals[1]-vals[0],
         'worker_IPC_polling_ms':vals[2]-vals[1],
         'python_policy_wrapper_ms':vals[3]-vals[2]}
    if abs(sum(out.values())-vals[3])>1e-6:
        raise ValueError('cost decomposition does not close')
    return out


def replay(records, plans, strategy):
    """Lookup estimator: not a new native trace and not causal speed evidence.

    Workload order is frozen P6; alternating A/B proposals are a synthetic stress
    case, NOT a learned predictor. Fixed arms assume pre-warmed initial policies.
    """
    index={(r['block'],r['prompt_index'],r['policy'],r['phase']):r for r in records}
    current='B' if strategy=='fixed_B' else 'A'
    last=-1; costs=[]; switches=0; rejected=0; selections=[]
    for i,item in enumerate(plans):
        proposed='A' if i%2==0 else 'B'
        if strategy=='naive_follow':
            target=proposed
        elif strategy=='minimum_hold_8':
            target=proposed if i-last>8 else current
        elif strategy in ('fixed_A','cost_quality_governor'):
            # With the current confidence interval and unverified B quality,
            # the conservative gate never authorizes a change away from A.
            target='A'
        elif strategy=='fixed_B':
            target='B'
        else:
            raise ValueError('unknown replay arm')
        moved=target!=current
        if moved:
            switches+=1;last=i
        if target!=proposed:rejected+=1
        row=index[(item['block'],item['prompt_index'],target,'switch' if moved else 'steady')]
        costs.append(row['admission_wall_ms'])
        selections.append(target);current=target
    return {'requests':len(plans),'switches':switches,'proposals_not_followed':rejected,
        'estimated_total_admission_ms':sum(costs),'estimated_mean_admission_ms':statistics.mean(costs),
        'selected_policy_counts':{p:selections.count(p) for p in ('A','B')},
        'method':'archived_cost_cell_lookup','new_native_execution':False,
        'initial_policy_setup_cost_included':False}


def analyze(directory):
    root=pathlib.Path(directory)
    speed=load_pinned(root/'speed.json',SPEED_SHA)
    quality=load_pinned(root/'quality.json',QUALITY_SHA)
    plans=load_pinned(root/'speed-order-preregistered.json',ORDER_SHA)
    records=speed['records']
    if speed['status']!='COMPLETE' or quality['status']!='COMPLETE' or len(records)!=288:
        raise ValueError('incomplete original source')
    keys={(r['block'],r['prompt_index'],r['policy'],r['phase']) for r in records}
    expected={(b,p,k,f) for b in range(6) for p in range(12) for k in ('A','B') for f in ('switch','steady')}
    if len(keys)!=len(records) or keys!=expected or len(plans)!=72:
        raise ValueError('duplicate/missing source comparison cells')
    if {(r['block'],r['prompt_index']) for r in plans}!={(b,p) for b in range(6) for p in range(12)}:
        raise ValueError('frozen workload incomplete')
    qindex={r['prompt_id']:r for r in quality['records']}
    for row in records:
        decompose(row)
        if row['phase']=='steady' and row['transition_ms']!=0:
            raise ValueError('steady admission has a transition')
        if (not row['sequence_matches_quality'] or row['sequence_sha256']!=
            qindex[row['prompt_id']][row['policy']]['sequence_sha256']):
            raise ValueError('output lineage mismatch')
    boundaries={}
    for phase in ('switch','steady'):
        selected=[r for r in records if r['phase']==phase]
        parts=[decompose(r) for r in selected]
        boundaries[phase]={'admissions':len(selected),
            'mean_ms':{k:statistics.mean(r[k] for r in parts) for k in parts[0]},
            'mean_total_admission_ms':statistics.mean(r['admission_wall_ms'] for r in selected)}
    paired=speed['summary']['steady']['paired_comparisons']['admission_wall_ms']
    ci=paired['crossed_prompt_block_bootstrap_95ci_ms']
    gain_lower=-ci[1]
    switch_upper=max(r['transition_ms'] for r in records)
    context=digest({'speed_sha256':SPEED_SHA,'quality_sha256':QUALITY_SHA,'scope':'offline-only'})
    gate=Governor(Runtime(POLICY['A'],0,'OFFLINE_NO_NATIVE_WORKER',context))
    forecast=Forecast(POLICY['A'],POLICY['B'],context,SPEED_SHA,gain_lower,switch_upper,4.,8,72)
    decision=gate.plan(POLICY['B'],allowed=frozenset({POLICY['A']}),forecast=forecast)
    gate.abort(decision)
    cost_decision=gate.plan(POLICY['B'],allowed=frozenset(POLICY.values()),forecast=forecast)
    gate.abort(cost_decision)
    arms={name:replay(records,plans,name) for name in
          ('fixed_A','fixed_B','naive_follow','minimum_hold_8','cost_quality_governor')}
    arms['fixed_AF']={'status':'NOT_MEASURED','reason':'P6 speed data contain only A and B',
                      'new_native_execution':False}
    return {'schema':'beglin-hcp-p8-offline-decomposition-and-governor-v1',
        'source_hashes':{'speed':SPEED_SHA,'quality':QUALITY_SHA,'order':ORDER_SHA},
        'validated_source_cells':288,'cost_boundary_decomposition':boundaries,
        'native_transition_internal_phases_measured':False,
        'static_source_findings':{
            'transition_timer_ends_before_durable_ACK':True,
            'qNg64_CPU_cache_hit_still_calls_MLX_bind':True,
            'MLX_5_6bit_bind_reallocates_and_repacks':True,
            'dominant_internal_phase_proven':False},
        'candidate_gate':{'decision':decision.action,'reason':decision.reason,
            'cost_gate_independently':cost_decision.reason,
            'steady_B_minus_A_mean_ms':paired['mean_B_minus_A_ms'],
            'steady_B_minus_A_ci95_ms':ci,
            'positive_saving_lower_bound_ms':gain_lower,
            'observed_switch_upper_bound_ms':switch_upper,
            'reliable_break_even_requests':None,
            'quality_permission_granted':False},
        'lookup_replay':arms,
        'replay_assumptions':['synthetic alternating policy proposals; original P6 input order',
            'latencies looked up by block/prompt/policy/switch flag',
            'no new native execution; not causal or production speed evidence',
            'initial policy establishment/warmup cost not included',
            'previously used corpus, not a new untouched holdout'],
        'native_profile_status':'NOT_EXECUTED_STORAGE_GATE',
        'native_gate':{'host':'xox','free_gb':15.41,'reserve_gb':20,'reason':'no_eligible_storage_pool'},
        'fixed_AF_A_B_dynamic_real_comparison_complete':False,
        'speedup_proven':False,'production_touched':False,'automatic_live_promotion':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence',required=True,type=pathlib.Path)
    ap.add_argument('--output',required=True,type=pathlib.Path)
    args=ap.parse_args()
    result=analyze(args.evidence)
    with args.output.open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,sort_keys=True,allow_nan=False))
