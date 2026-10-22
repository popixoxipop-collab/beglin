#!/usr/bin/env python3
"""Independent (read-only) BEGLIN Qv2 measured-async-stage MILP budget audit.

Validates source-only 448 stage options, pinned frozen mask, original BF16
activation quality, observed concurrent stage costs, both serial/async
budgets, evaluation exclusion and preserved source-only solver optimum.
Runs on Alienware WSL without model forward or target evaluation.
"""
from __future__ import annotations
import hashlib,json,math,pathlib,sys,unittest
from collections import Counter
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
try:
    import q_v2_async_activation_milp as m
except ModuleNotFoundError as exc:
    # macOS EOE system Python has no SciPy. This is a supported, explicit
    # host capability skip, not an untested claim: Alienware WSL SciPy/HiGHS
    # still runs all eight source-only MILP integrity tests as required.
    if exc.name != "scipy":
        raise
    m = None

ROOT=m.ROOT if m is not None else pathlib.Path(__file__).resolve().parents[2]
PUB=ROOT/"configs/research/BEGLIN_QV2_ASYNC_ACTIVATION_PRE_EVAL_FROZEN_POLICY_20261022.json"
PRIV=(m.OUT/"qv2_async_activation_allocator_receipt.json") if m is not None else ROOT/"__scipy_unavailable__"
MASK=(m.OUT/"qv2_async_activation_28n8_frozen.txt") if m is not None else ROOT/"__scipy_unavailable__"
def sha(b):return hashlib.sha256(b).hexdigest()

@unittest.skipIf(m is None, "SciPy MILP unavailable on this host; required tests run on WSL SciPy host")
class Qv2MILPIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze=json.loads(PUB.read_text())
        cls.raw=json.loads(PRIV.read_text())
        cls.quality=json.loads(m.QUAL.read_text())
        cls.stages=json.loads(m.STAGE.read_text())
        cls.roles={p['role']:p for p in json.loads(m.ROLE.read_text())['role_kernel_cost_measured']}
        cls.sel={(role,int(l)) for role,l in cls.raw['optimizer']['selected_ids']}
    def test_exact_196_and_28_8bit(self):
        self.assertEqual(len(self.sel),28)
        self.assertEqual(self.raw['optimizer']['n4_tensors'],168)
        self.assertEqual(self.raw['optimizer']['milp_stage_variable_count'],448)
        self.assertEqual(len(self.raw['optimizer']['layer_stage_options']),28*4)
        self.assertEqual(sum(self.freeze['role_counts'].values()),28)
    def test_original_sources_and_pre_evaluation_freeze_sha(self):
        self.assertEqual(self.freeze['status'],'PRE_EVALUATION_FROZEN_SOURCE_ONLY')
        self.assertEqual(self.freeze['research_PRIVATE_receipt_sha256'],sha(PRIV.read_bytes()))
        self.assertEqual(self.freeze['selected_policy_sha256'],sha(MASK.read_bytes()))
        self.assertEqual(self.raw['status'],'SOURCE_ONLY_PROVED_MILP_OPTIMUM__NOT_YET_MODEL_EVALUATED')
        self.assertTrue(self.raw['output_test_window_not_used'])
        self.assertTrue(self.raw['parallel_execution_full_model_unverified'])
    def test_same_28_mask_as_selected_public(self):
        exposed={(role,int(l)) for role,l in self.freeze['selected_n8_role_layer_ids']}
        self.assertEqual(exposed,self.sel)
        self.assertEqual(Counter(role for role,_ in self.sel),Counter(self.freeze['role_counts']))
        self.assertEqual(MASK.stat().st_mode&0o777,0o600)
        self.assertEqual(PRIV.stat().st_mode&0o777,0o600)
    def test_exact_native_packed_bytes_independent(self):
        expected=28*sum(p['q4_with_scales_bytes'] for p in self.roles.values())
        for role,layer in self.sel:
            p=self.roles[role]
            expected+=p['n8_with_scales_bytes']-p['q4_with_scales_bytes']
        self.assertEqual(expected,self.raw['optimizer']['estimated_projection_weight_bytes'])
        self.assertEqual(expected,self.freeze['packed_projection_weight_bytes'])
        self.assertLessEqual(expected,m.MEMORY_LIMIT)
    def test_async_stage_dependency_and_measured_wall_time(self):
        total=0.
        seen=set()
        for layer in range(28):
            def bit(role):return '8' if (role,layer) in self.sel else '4'
            for stage,roles in m.GROUPS:
                mask=''.join(bit(r) for r in roles)
                if len(roles)>1:
                    entry=self.stages['data'][stage][mask]
                    wall=entry['best_ms']
                    self.assertLessEqual(wall,entry['serial_ms']+1e-8)
                else:
                    r=self.roles[roles[0]]
                    wall=r['n8_repacked_neon_median_ms'] if mask=='8' else min(r['q4_neon_1thread_median_ms'],r['q4_neon_6thread_median_ms'])
                total+=wall
                seen.add((layer,stage))
        self.assertEqual(len(seen),112)
        self.assertAlmostEqual(total,self.raw['optimizer']['isolated_async_CPU_stage_total_ms'],places=6)
        self.assertLessEqual(total,50.)
        self.assertTrue(self.raw['optimizer']['async_stage_makespan_is_not_full_model_walltime'])
    def test_qual_source_diag_moment_proxy_sum(self):
        scores=self.quality['quality_upgrade_by_role_layer']
        value=sum(scores[role][l] for role,l in self.sel)
        self.assertAlmostEqual(value,self.raw['optimizer']['source_diag_activation_quality_gain'],places=9)
        self.assertEqual(self.quality['moments_SHA'],'ee7b177b4dde984a5d83e0cdb5983d2ae00c34ee752fef9113a79568a1d12663')
    def test_serial_original_calibrated_limit(self):
        delta=sum(self.roles[role]['n8_repacked_neon_median_ms']-
            min(self.roles[role]['q4_neon_1thread_median_ms'],self.roles[role]['q4_neon_6thread_median_ms'])
            for role,layer in self.sel)
        ms=m.FULL_Q4_MS+m.GAMMA_SERIAL*delta
        self.assertAlmostEqual(ms,self.raw['optimizer']['original_fullmodel_serial_estimated_ms'],places=6)
        self.assertLessEqual(ms,m.SERIAL_CEILING_MS)
    def test_mip_zero_gap_and_no_final_quality_claim(self):
        self.assertEqual(self.raw['optimizer']['source_only_solver_status'],0)
        self.assertLessEqual(self.raw['optimizer']['mip_gap'],1e-9)
        self.assertEqual(self.raw['scientific_Q_verdict'],'NO_VERDICT')
        self.assertTrue(self.raw['production_main_unchanged'])
        self.assertTrue(self.raw['prior_blocked_full_model_tensor_intervention_not_retried'])
if __name__=='__main__':
    unittest.main(verbosity=2)
