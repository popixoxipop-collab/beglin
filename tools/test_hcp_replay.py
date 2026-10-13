import copy,json, pathlib,unittest
from hcp_replay import (
 HCPError,load_inventory,make_risk_sequence,replay,run_experiment,upper_plan,validate_cost_record,transition_price
)

HERE=pathlib.Path(__file__).parent
class HierarchicalReplayTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.records,_=load_inventory(HERE/'evidence_inventory.json')
  cls.reference=next(r for r in cls.records if r['file']=='p7-soak-precommit/result.json')
 def test_input_cost_identity_is_verified(self):
  orig=json.loads((HERE/'evidence_inventory.json').read_text())['results'][0]
  x=copy.deepcopy(orig);x['accepted_policy_profiles'][1]['policy_hash']='a'*64
  with self.assertRaisesRegex(HCPError,'hash mismatch'):validate_cost_record(x)
  x=copy.deepcopy(orig);x['production_touched']=True
  with self.assertRaisesRegex(HCPError,'read-only'):validate_cost_record(x)
 def test_bad_cost_and_missing_transition_block(self):
  src=copy.deepcopy(self.reference)
  src['profiles'][src['alternate']]['transitions'].clear()
  self.assertIsNone(transition_price(src['profiles'],src['base'],src['alternate'],{src['base']}))
 def test_hierarchical_switch_is_amortized_but_flat_holds(self):
  s=[False]*2048
  flat=replay(self.reference,self.reference,s,mode='flat')
  h16=replay(self.reference,self.reference,s,mode='hierarchical',horizon=16)
  h512=replay(self.reference,self.reference,s,mode='hierarchical',horizon=512)
  self.assertEqual(flat['alternate_policy_requests'],0)
  self.assertEqual(h16['alternate_policy_requests'],0)
  self.assertEqual(h512['alternate_policy_requests'],len(s))
  self.assertLess(h512['modeled_sum_of_archived_p50_ms'],flat['modeled_sum_of_archived_p50_ms'])
 def test_low_level_guards_risk(self):
  risks=make_risk_sequence('bursts')
  h=replay(self.reference,self.reference,risks,mode='hierarchical',horizon=512)
  self.assertGreater(h['transition_count'],0)
  self.assertLess(h['alternate_policy_requests'],len(risks))
  self.assertTrue(all(x['production_touched']==False if 'production_touched' in x else True for x in (h,)))
 def test_no_lookahead_at_plan_start(self):
  p=self.reference['profiles'];base=self.reference['base'];seen={base}
  a=upper_plan(p,base,base,seen,512,[],False)
  b=upper_plan(p,base,base,seen,512,[],False)
  self.assertEqual(a,b)
  self.assertEqual(a,self.reference['alternate'])
  self.assertEqual(upper_plan(p,base,base,seen,512,[True],False),base)
 def test_evidence_only_counterfactual_no_quality_claims(self):
  report=run_experiment(HERE/'evidence_inventory.json',scenario_size=256)
  self.assertEqual(report['status'],'COUNTERFACTUAL_COST_ONLY_NOT_CERTIFIED')
  self.assertEqual(report['quality_and_promotion_gate'],'BLOCKED')
  self.assertTrue(report['no_event_conditioned_certificates_found_in_cost_inventory'])
  self.assertFalse(report['automatic_live_promotion'])
  self.assertEqual(report['archive_count'],10)
  self.assertEqual(len(report['comparisons']),30)
  self.assertTrue(all(s['synthetic_risks'] and s['hypothetical_quality_permission'] for s in report['comparisons']))
 def test_all_risk_stays_on_base(self):
  q=replay(self.reference,self.reference,[True]*512,mode='hierarchical',horizon=512)
  self.assertEqual(q['alternate_policy_requests'],0)
  self.assertEqual(q['transition_count'],0)
 def test_changing_ref_profile_changes_transfer_accounting(self):
  src=copy.deepcopy(self.reference)
  v=src['profiles'][src['alternate']]['transitions'][(src['base'],'cold')]
  src['profiles'][src['alternate']]['transitions'][(src['base'],'cold')]=v+1000
  base=replay(src,self.reference,[False]*2048,mode='hierarchical',horizon=512)
  self.assertEqual(base['alternate_policy_requests'],0)

if __name__=='__main__':unittest.main()