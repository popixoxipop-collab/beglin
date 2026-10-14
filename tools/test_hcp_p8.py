"""Fixture/state and exact-source tests; never native inference claims."""
import dataclasses
import pathlib
import tempfile
import unittest
from hcp_p8_governor import Governor, Runtime, Forecast, digest
from hcp_p8_profile_patch import patch,to_cb128,strip_observer,sha,CB128,validate_phases
from hcp_p8_analyze import decompose,load_pinned
A='a'*64;B='b'*64;C='c'*64;E='e'*64

class GovernorTests(unittest.TestCase):
    def setUp(self):self.g=Governor(Runtime(A,7,'fixture-worker',C))
    def forecast(self,**changes):
        return dataclasses.replace(Forecast(A,B,C,E,100.,330.,4.,8,100),**changes)
    def ack(self,d,**changes):
        out={'worker':'fixture-worker','context':C,'active_policy_hash':d.target,
            'weight_epoch':d.before.epoch+(d.action=='PROPOSE'),
            'verified_by_native_controller':True,'status':'REBIND_SET_APPLIED','txn_id':'fixture-txn'}
        out.update(changes);return out
    def hold(self,n=8):
        for _ in range(n):
            d=self.g.plan(A,allowed=frozenset({A,B}));self.g.mark_dispatched(d)
            self.g.complete(d,self.ack(d))
    def propose(self,**kw):
        return self.g.plan(B,allowed=frozenset({A,B}),forecast=self.forecast(**kw))
    def test_01_initial_hold(self):self.assertEqual(self.propose().reason,'minimum_hold')
    def test_02_minimum_hold_expires(self):
        self.hold();self.assertEqual(self.propose().action,'PROPOSE')
    def test_03_unknown_quality_stays(self):
        d=self.g.plan(B,allowed=frozenset({A}),forecast=self.forecast())
        self.assertEqual(d.reason,'candidate_quality_or_authority_not_verified')
    def test_04_invalid_incumbent_blocks(self):
        d=self.g.plan(B,allowed=frozenset({B}),forecast=self.forecast())
        self.assertEqual(d.action,'BLOCK')
        with self.assertRaises(RuntimeError):self.g.mark_dispatched(d)
    def test_05_missing_cost(self):self.assertEqual(self.g.plan(B,allowed=frozenset({A,B})).reason,'no_cost_evidence')
    def test_06_nonpositive_bound(self):self.assertEqual(self.propose(gain_lower_ms=0).reason,'no_positive_confidence_bound')
    def test_07_ci_crosses_zero(self):self.assertEqual(self.propose(gain_lower_ms=-5.762349390625001).reason,'no_positive_confidence_bound')
    def test_08_negative_cost_rejected(self):
        with self.assertRaises(ValueError):self.forecast(switch_upper_ms=-1)
    def test_09_nan_rejected(self):
        with self.assertRaises(ValueError):self.forecast(gain_lower_ms=float('nan'))
    def test_10_infinity_rejected(self):
        with self.assertRaises(ValueError):self.forecast(switch_upper_ms=float('inf'))
    def test_11_context_bound(self):self.assertEqual(self.propose(context='d'*64).reason,'cost_evidence_identity_mismatch')
    def test_12_expiry(self):
        self.hold();self.assertEqual(self.propose(valid_through=0).reason,'cost_evidence_expired')
    def test_13_horizon_bound(self):self.assertEqual(self.propose(horizon=2).reason,'forecast_horizon_too_short')
    def test_14_unamortized(self):
        self.hold();self.assertEqual(self.propose(gain_lower_ms=2).reason,'switch_cost_not_amortized')
    def test_15_exact_cost_boundary_not_enough(self):
        self.hold();self.assertEqual(self.propose(gain_lower_ms=41.75).reason,'switch_cost_not_amortized')
    def test_16_does_not_commit_on_proposal(self):
        self.hold();d=self.propose();self.assertEqual(self.g.runtime.policy,A);self.assertEqual(self.g.runtime.epoch,7)
    def test_17_valid_ACK_commits(self):
        self.hold();d=self.propose();self.g.mark_dispatched(d);self.g.complete(d,self.ack(d),expected_txn='fixture-txn')
        self.assertEqual(self.g.runtime.policy,B);self.assertEqual(self.g.runtime.epoch,8)
    def test_18_failed_ACK_keeps_state(self):
        self.hold();d=self.propose();self.g.mark_dispatched(d)
        with self.assertRaises(RuntimeError):self.g.complete(d,self.ack(d,status='REBIND_SET_FAILED'),expected_txn='fixture-txn')
        self.assertEqual(self.g.runtime.policy,A)
    def test_19_wrong_epoch(self):
        self.hold();d=self.propose();self.g.mark_dispatched(d)
        with self.assertRaises(RuntimeError):self.g.complete(d,self.ack(d,weight_epoch=99),expected_txn='fixture-txn')
    def test_20_wrong_worker(self):
        d=self.g.plan(A,allowed=frozenset({A}));self.g.mark_dispatched(d)
        with self.assertRaises(RuntimeError):self.g.complete(d,self.ack(d,worker='other'))
    def test_21_wrong_context(self):
        d=self.g.plan(A,allowed=frozenset({A}));self.g.mark_dispatched(d)
        with self.assertRaises(RuntimeError):self.g.complete(d,self.ack(d,context='d'*64))
    def test_22_wrong_transaction(self):
        self.hold();d=self.propose();self.g.mark_dispatched(d)
        with self.assertRaises(RuntimeError):self.g.complete(d,self.ack(d),expected_txn='wrong')
    def test_23_verifier_not_attested(self):
        d=self.g.plan(A,allowed=frozenset({A}));self.g.mark_dispatched(d)
        with self.assertRaises(RuntimeError):self.g.complete(d,self.ack(d,verified_by_native_controller=False))
    def test_24_duplicate_plan_blocked(self):
        self.g.plan(A,allowed=frozenset({A}))
        with self.assertRaises(RuntimeError):self.g.plan(A,allowed=frozenset({A}))
    def test_25_duplicate_dispatch_blocked(self):
        d=self.g.plan(A,allowed=frozenset({A}));self.g.mark_dispatched(d)
        with self.assertRaises(RuntimeError):self.g.mark_dispatched(d)
    def test_26_before_dispatch_abort(self):
        d=self.g.plan(A,allowed=frozenset({A}));self.g.abort(d);self.assertEqual(self.g.admission,0)
    def test_27_timeout_does_not_allow_abort_or_retry(self):
        d=self.g.plan(A,allowed=frozenset({A}));self.g.mark_dispatched(d)
        with self.assertRaises(RuntimeError):self.g.abort(d)
        with self.assertRaises(RuntimeError):self.g.plan(A,allowed=frozenset({A}))
    def test_28_replayed_complete_blocked(self):
        d=self.g.plan(A,allowed=frozenset({A}));self.g.mark_dispatched(d);self.g.complete(d,self.ack(d))
        with self.assertRaises(RuntimeError):self.g.complete(d,self.ack(d))
    def test_29_rate_limit(self):
        self.g=Governor(Runtime(A,7,'fixture-worker',C),min_hold=1,window=64,max_switches=1)
        self.hold(1);d=self.propose();self.g.mark_dispatched(d);self.g.complete(d,self.ack(d),expected_txn='fixture-txn')
        for _ in range(2):
            h=self.g.plan(B,allowed=frozenset({A,B}));self.g.mark_dispatched(h);self.g.complete(h,self.ack(h))
        f=self.forecast(from_policy=B,to_policy=A)
        d=self.g.plan(A,allowed=frozenset({A,B}),forecast=f)
        self.assertEqual(d.reason,'switch_rate_limit')
    def test_30_policy_identity_mismatch(self):self.assertEqual(self.propose(from_policy=B).reason,'cost_evidence_identity_mismatch')
    def test_31_settings_invalid(self):
        with self.assertRaises(ValueError):Governor(Runtime(A,0,'x',C),min_hold=0)
    def test_32_bad_sha(self):
        with self.assertRaises(ValueError):Runtime('bad',0,'x',C)
    def test_33_bool_epoch(self):
        with self.assertRaises(ValueError):Runtime(A,True,'x',C)
    def test_34_complete_without_dispatch(self):
        d=self.g.plan(A,allowed=frozenset({A}))
        with self.assertRaises(RuntimeError):self.g.complete(d,self.ack(d))

class CostTests(unittest.TestCase):
    def test_35_cost_sum(self):
        r={'transition_ms':330.,'engine_ms':850.,'roundtrip_ms':868.,'admission_wall_ms':872.}
        p=decompose(r);self.assertEqual(sum(p.values()),872.);self.assertEqual(p['python_policy_wrapper_ms'],4.)
    def test_36_bad_nesting(self):
        with self.assertRaises(ValueError):decompose({'transition_ms':5.,'engine_ms':4.,'roundtrip_ms':3.,'admission_wall_ms':2.})
    def test_37_nan_cost(self):
        with self.assertRaises(ValueError):decompose(dict.fromkeys(('transition_ms','engine_ms','roundtrip_ms','admission_wall_ms'),float('nan')))
    def test_38_source_pin_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'x.json';p.write_text('{}')
            with self.assertRaises(ValueError):load_pinned(p,'0'*64)
    def test_39_phase_sum(self):
        r={'schema':'beglin-p8-native-phases-v1','targets':2,'entry_to_ack_ms':9.}
        for k in ('precheck_ms','synchronize_ms','snapshot_ms','qng64_cache_lookup_or_build_ms','mlx_bind_ms','bind_verify_and_loop_ms','epoch_reset_ms','metadata_commit_ms','durable_ack_ms'):r[k]=1.
        self.assertTrue(validate_phases(r))
        r['entry_to_ack_ms']=8.
        with self.assertRaises(ValueError):validate_phases(r)
    def test_40_phase_nan(self):
        with self.assertRaises(ValueError):validate_phases({'schema':'beglin-p8-native-phases-v1','targets':2})

class ExactSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=pathlib.Path(__file__).parent/'original_qwen_infer.c'
        if not p.is_file():p=pathlib.Path(__file__).parent.parent/'qwen_infer.c'
        if not p.is_file():raise unittest.SkipTest('full P5 C source required for source tests')
        cls.path=p;cls.original=p.read_text();cls.base=to_cb128(cls.original)
    def test_41_roundtrip_preserves_every_byte(self):self.assertEqual(strip_observer(patch(self.base)),self.base)
    def test_42_hash_exact(self):self.assertEqual(sha(self.base),CB128)
    def test_43_wrong_source_rejected(self):
        with self.assertRaises(ValueError):patch(self.base+' ')
    def test_44_double_patch_rejected(self):
        with self.assertRaises(ValueError):patch(patch(self.base))
    def test_45_ACK_rollback_still_present(self):
        p=patch(self.base)
        for key in ('REBIND_SET_APPLIED','REBIND_SET_FAILED','mlx_gpu_restore_binding_snapshot','moe_gpu_txn_mark_terminal'):
            self.assertEqual(p.count(key),self.base.count(key))
    def test_46_gate_not_widened(self):
        p=patch(self.base);self.assertIn('#define MOE_CBATCH_MAXPOS 128',p);self.assertIn('#define MOE_MAXPOS 160',p)
    def test_47_native_phase_schema(self):
        p=patch(self.base);self.assertIn('durable_ack_ms',p);self.assertIn('O_NOFOLLOW',p)
    def test_48_original_file_unchanged(self):
        patch(self.base)
        self.assertEqual(self.path.read_text(),self.original)

if __name__=='__main__':unittest.main(verbosity=2)
