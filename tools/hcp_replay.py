#!/usr/bin/env python3
"""Beglin H-JEPA-inspired two-time-scale compute policy replay (proposal only).

This does not implement JEPA representations or modify a model. Upper policy
chooses a certified-shaped precision policy for a multi-request horizon; lower
policy enforces the current request's observed risk by falling back to the
incumbent. CANNOT infer quality from cost evidence. The replay's risk sequence
is SYNTHETIC and all calculated inference times are sums of archived per-policy
p50 measurements, NOT native execution of those counterfactual policies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import precision_context as pc

SCHEMA = 'beglin-hcp-offline-cost-replay-v1'
FIXTURE_RISKS = ('calm', 'bursts', 'alternating')
MODES = ('static_risk_compatible', 'flat', 'hierarchical')
HORIZONS = (4, 16, 64, 256, 512)

class HCPError(ValueError): pass

def finite_positive(value, name):
    try: n=float(value)
    except (ValueError, TypeError) as err: raise HCPError(f'{name}: not numeric') from err
    if not math.isfinite(n) or n<=0: raise HCPError(f'{name}: positive finite required')
    return n

def validate_cost_record(item):
    if item.get('source')!='XOX_MEASURED_ARCHIVED_COST' or item.get('source_status')!='PASS' or item.get('production_touched') is not False:
        raise HCPError('not read-only PASS XOX archived cost evidence')
    profiles={}
    for row in item.get('accepted_policy_profiles',[]):
        policy=pc.normalize_policy(row['policy'])
        key=pc.policy_hash(policy)
        if key != row.get('policy_hash') or key in profiles:
            raise HCPError('policy hash mismatch or duplicate')
        steady=finite_positive(row.get('steady_roundtrip_p50_ms'), 'steady p50')
        engine=finite_positive(row.get('steady_engine_p50_ms'), 'engine p50')
        if int(row.get('expected_inference_passes',0))!=1:
            raise HCPError('counterfactual adaptive/recovery passes cannot be priced as one')
        transitions={}
        for tr in row.get('transitions',[]):
            from_hash=tr.get('from_policy_hash')
            state=tr.get('cache_state')
            if not isinstance(from_hash,str) or len(from_hash)!=64 or state not in ('cold','warm'):
                raise HCPError('invalid transition identity/cache state')
            price=finite_positive(tr.get('p50_transition_ms'), 'transition p50')
            tkey=(from_hash,state)
            if tkey in transitions: raise HCPError('duplicate transition')
            transitions[tkey]=price
        profiles[key]={'policy':policy,'steady_ms':steady,'engine_ms':engine,'transitions':transitions}
    base=item.get('startup_policy_hash')
    if base not in profiles or len(profiles)!=2:
        raise HCPError('this P0 only supports 2 fully enumerated policy profiles and startup')
    keys={tuple((row['role'],row['layer']) for row in p['policy']) for p in profiles.values()}
    if len(keys)!=1:raise HCPError('precision policy shape changed')
    alternate=next(k for k in profiles if k!=base)
    if (base,'cold') not in profiles[alternate]['transitions'] or (alternate,'warm') not in profiles[base]['transitions']:
        raise HCPError('missing cold entry or warm restore transition evidence')
    if profiles[alternate]['steady_ms']<=0:raise HCPError('invalid alt')
    return {'base':base,'alternate':alternate,'profiles':profiles,
       'evidence_sha':item['file_sha256'],'file':item.get('relative_to_cost_root','')}

def load_inventory(path):
    data=json.loads(Path(path).read_text())
    if data.get('schema')!='beglin-hcp-xox-evidence-preflight-v1' or data.get('production_touched') is not False:
        raise HCPError('invalid P0 read-only cost evidence inventory')
    items=data.get('results')
    if not isinstance(items,list) or not items:raise HCPError('no measured XOX cost evidence available')
    out=[];seen=set()
    for row in items:
        item=validate_cost_record(row)
        if row['file_sha256'] in seen:raise HCPError('duplicate cost file')
        seen.add(row['file_sha256'])
        out.append(item)
    labels={(x['base'],x['alternate']) for x in out}
    if len(labels)!=1:raise HCPError('incompatible policies across cost snapshots')
    return out,hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make_risk_sequence(kind,n=2048,seed=20261014):
    if kind not in FIXTURE_RISKS or n<20:raise HCPError('invalid synthetic risk scenario')
    if kind=='calm':return [False]*n
    if kind=='alternating':return [i%2==0 for i in range(n)]
    rng=random.Random(seed)
    risks=[rng.random()<.01 for _ in range(n)]
    for start,length in ((300,9),(960,15),(1600,11)):
        for i in range(start,min(start+length,n)):risks[i]=True
    return risks

def transition_price(profiles,cur,to,seen):
    if to==cur:return 0.0
    mode='warm' if to in seen else 'cold'
    tr=profiles[to]['transitions'].get((cur,mode))
    if tr is None:
        # Fail closed: a missing transition never gets zero or estimated latency.
        return None
    return tr

def chosen_if_one_request(profiles,base,cur,seen,risk):
    options=[next(k for k in profiles if k != base)] if risk else list(profiles)
    ranked=[]
    for p in options:
        tr=transition_price(profiles,cur,p,seen)
        if tr is not None:ranked.append((tr+profiles[p]['steady_ms'],tr,p))
    if not ranked:raise HCPError('no eligible executable policy')
    return min(ranked)[2]

def upper_plan(profiles,base,cur,seen,horizon,past_risks,current_risk):
    """No future risk labels are accessible: only current and past observations."""
    if current_risk or any(past_risks[-16:]):return next(k for k in profiles if k!=base)
    scored=[]
    for policy in profiles:
        tr=transition_price(profiles,cur,policy,seen)
        if tr is None:continue
        # Horizon is a policy amortization hypothesis; these are NOT JEPA latents.
        scored.append((tr+horizon*profiles[policy]['steady_ms'],policy))
    if not scored:raise HCPError('no eligible upper plan')
    return min(scored)[1]

def replay(model,measured,risks,*,mode,horizon=1):
    if mode not in MODES or horizon<1:raise HCPError('invalid comparison arm')
    if (model['base'],model['alternate'])!=(measured['base'],measured['alternate']):
        raise HCPError('policy identity mismatch between plan and evaluation')
    base=model['base'];risk_policy=model['alternate'];ref=model['profiles'];actual=measured['profiles']
    cur=base;seen={base};planned=base;valid_until=0; history=[]
    transitions=0;upper_calls=0;guards=0;alt_requests=0
    total_ms=0.0;decision_ns=0
    for t,risk in enumerate(risks):
        if type(risk) is not bool:raise HCPError('risk records must be boolean synthetic flags')
        if mode=='static_risk_compatible':desired=risk_policy
        elif mode=='flat':
            t0=time.perf_counter_ns()
            desired=chosen_if_one_request(ref,base,cur,seen,risk)
            decision_ns+=time.perf_counter_ns()-t0
            upper_calls+=1
        else:
            if t>=valid_until or (risk and planned!=risk_policy):
                t0=time.perf_counter_ns()
                planned=upper_plan(ref,base,cur,seen,horizon,history,risk)
                valid_until=t+horizon
                decision_ns+=time.perf_counter_ns()-t0
                upper_calls+=1
            desired=planned
            if risk and desired!=risk_policy:
                desired=risk_policy;guards+=1
                valid_until=t+1
        if risk and desired!=risk_policy:raise HCPError('risk policy evidence guard violated')
        tr=transition_price(actual,cur,desired,seen)
        if tr is None:
            # Missing-cost transitions cannot be priced; never bypass a risk rule.
            if risk:raise HCPError('required risk transition lacks cost evidence')
            desired=base
            tr=transition_price(actual,cur,base,seen)
            if tr is None:raise HCPError('no measured rollback transition')
        total_ms+=tr+actual[desired]['steady_ms']
        if desired!=cur:transitions+=1
        if desired!=base:alt_requests+=1
        cur=desired;seen.add(cur);history.append(risk)
    return {'mode':mode,'horizon':horizon if mode=='hierarchical' else None,
        'modeled_sum_of_archived_p50_ms':round(total_ms,4),
        'planner_cpu_elapsed_ms':round(decision_ns/1e6,4),
        'request_count':len(risks),'alternate_policy_requests':alt_requests,
        'transition_count':transitions,'upper_planning_calls':upper_calls,
        'low_level_risk_guards':guards,
        'quality_certification_per_event':False,'live_execution':False}

def run_experiment(inventory,*,model_path=None,scenario_size=2048):
    records,inv_sha=load_inventory(inventory)
    # Original first-archive calibration vs chronological/experimental holdout is
    # descriptive only; binary, cache and load differ between archived snapshots.
    comparisons=[]
    for planning_reference in records:
        for evaluation_record in records:
            for scenario in FIXTURE_RISKS:
                risks=make_risk_sequence(scenario,scenario_size)
                baseline=replay(planning_reference,evaluation_record,risks,mode='static_risk_compatible')
                rows=[baseline,replay(planning_reference,evaluation_record,risks,mode='flat')]
                for horizon in HORIZONS:
                    rows.append(replay(planning_reference,evaluation_record,risks,mode='hierarchical',horizon=horizon))
                for row in rows:
                    row['modeled_delta_vs_static_risk_compatible_percent']=round(
                        100*(row['modeled_sum_of_archived_p50_ms']/baseline['modeled_sum_of_archived_p50_ms']-1),6)
                comparisons.append({
                    'planning_cost_file':planning_reference['file'],
                    'planning_cost_sha256':planning_reference['evidence_sha'],
                    'evaluation_cost_file':evaluation_record['file'],
                    'evaluation_cost_sha256':evaluation_record['evidence_sha'],
                    'same_archive':planning_reference['evidence_sha']==evaluation_record['evidence_sha'],
                    'scenario':scenario,'synthetic_risks':True,
                    'hypothetical_quality_permission':True,'arms':rows})
    oos=[item for item in comparisons if not item['same_archive']]
    changes=[row['modeled_delta_vs_static_risk_compatible_percent'] for item in oos
             for row in item['arms'] if row['mode']=='hierarchical' and row['horizon']==512]
    summary={'h512_oos_modeled_delta_percent_min':round(min(changes),6),
             'h512_oos_modeled_delta_percent_median':round(statistics.median(changes),6),
             'h512_oos_modeled_delta_percent_max':round(max(changes),6),
             'h512_oos_scenarios_improved':sum(x<0 for x in changes),
             'h512_oos_scenarios_worsened':sum(x>0 for x in changes),
             'h512_oos_scenarios_equal':sum(x==0 for x in changes)}
    return {'schema':SCHEMA,'status':'COUNTERFACTUAL_COST_ONLY_NOT_CERTIFIED',
      'source':'XOX archived measured cost p50 profiles + SYNTHETIC risk sequences',
      'inventory_sha256':inv_sha,'planning_reference_design':'all ordered pairs of 10 archived cost snapshots, descriptive simulation',
      'archive_count':len(records),'risk_scenarios':list(FIXTURE_RISKS),
      'requests_per_scenario':scenario_size,
      'policies':[records[0]['base'],records[0]['alternate']],
      'models_by_horizon':list(HORIZONS),
      'uncertainty':'Different archived runs use different binaries/cache conditions; summing p50 is a modeled scenario, not real measured multi-request speedup.',
      'no_event_conditioned_certificates_found_in_cost_inventory':True,
      'hypothetical_event_rule':'risk requires alternate B, based on Beglin low_margin selector code; NOT per-event evidence in input trace',
      'no_actual_token_trajectories_or_logits':True,
      'quality_and_promotion_gate':'BLOCKED',
      'production_touched':False,'automatic_live_promotion':False,
      'summary':summary,
      'comparisons':comparisons}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    x=run_experiment(args.inventory)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
    print('HCP_REPLAY',json.dumps({**x['summary'],'status':x['status'],
         'archive_count':x['archive_count'],'scenario_count':len(x['comparisons']),
         'planning_reference_design':x['planning_reference_design']},sort_keys=True))
if __name__=='__main__':main()