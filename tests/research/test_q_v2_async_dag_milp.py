#!/usr/bin/env python3
"""BEGLIN Qv2 source-only 3584-choice async-DAG + serial MILP audit.

No model inference, no evaluation prompt opening. Require provenance-hardened
activation input moments, real native stage timing, exact solved policies,
and revalidate MILP optimality on the original source-only objective.
"""
from __future__ import annotations
import hashlib,json,pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_activation_async_dag_milp as impl

class KernelAwareAsyncMILPQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=impl.LAB
        cls.p=cls.root/"source_only_DAG_MILP_receipt.json"
        assert cls.p.is_file() and (cls.p.stat().st_mode&0o777)==0o600
        cls.data=json.loads(cls.p.read_text())
        cls.cfg,cls.q,cls.stages,cls.roles,cls.sourcehash=impl.collect_profiles()
    def test_locked_sources_and_held_set_unseen(self):
        self.assertEqual(self.data["status"],"OPTIMAL_SOURCE_ONLY_POLICIES_FROZEN_BEFORE_EVAL")
        self.assertEqual(self.data["original_source_sha256"]["prelock"],self.sourcehash[0])
        self.assertEqual(self.data["original_source_sha256"]["quality_scores"],self.sourcehash[1])
        self.assertTrue(self.data["eval_prompts_not_loaded_for_selection"])
        self.assertTrue(self.data["blocked_fullmodel_per_tensor_interventions_not_retried"])
    def test_all_12_measured_masks_and_independent_pool_modes(self):
        masks=self.stages["stages"]
        self.assertEqual(len(masks["qkv"]),8)
        self.assertEqual(len(masks["ffn_gu"]),4)
        for stage,combos in masks.items():
            for mask,record in combos.items():
                self.assertLessEqual(record["best_ms"],record["serial_q4_6t_ms"]+1e-8)
                self.assertIn(record["best_mode"],("serial_q4_6t","parallel_q4_1t","parallel_q4_6t"))
    def test_verified_old_role_balanced_random_identity(self):
        gamma,base,old=impl.calibrate(self.stages["stages"],self.roles)
        text="".join(f"{role} {idx}\n" for role in impl.ROLE_ORDER for idx in old[role])
        self.assertEqual(impl.sha(text.encode()),impl.OLD_ROLE_RANDOM_SHA)
        self.assertGreater(gamma,0)
        self.assertGreater(base,0)
    def test_mask_geometry_bit_options(self):
        self.assertEqual(impl.N_MASKS,128)
        self.assertEqual(impl.mask_bits(0),(4,)*7)
        self.assertEqual(impl.mask_bits(127),(8,)*7)
        self.assertEqual(impl.mask_bits(1),(*((4,)*6),8))
    def test_each_certified_policy_constraints_and_SHA(self):
        policies={q["name"]:q for q in self.data["policies"]}
        self.assertEqual(set(policies),{"async_DAG","serial_only"})
        for name,policy in policies.items():
            with self.subTest(policy=name):
                self.assertEqual(policy["status"],"MILP_OPTIMAL_CERTIFIED")
                self.assertEqual(policy["solver"]["status"],0)
                self.assertLessEqual(policy["solver"]["mip_gap"],1e-5)
                selected=policy["n8_role_layer_pairs"]
                self.assertEqual(len(selected),28)
                self.assertEqual(len({tuple(x) for x in selected}),28)
                self.assertEqual(sum(policy["n8_role_counts"].values()),28)
                self.assertEqual(len(policy["native_mask_for_28_layers"]),28)
                self.assertLessEqual(policy["projection_bytes"],830570496)
                self.assertLessEqual(policy["predicted_native_serial_ms"],100+1e-5)
                self.assertLessEqual(policy["predicted_hypothetical_parallel_graph_ms"],
                                  policy["predicted_native_serial_ms"]+1e-5)
                frozen=self.root/f"{name}_policy.txt"
                self.assertEqual(frozen.stat().st_mode&0o777,0o600)
                self.assertEqual(impl.sha(frozen.read_bytes()),policy["frozen_policy_sha256"])
        self.assertLessEqual(policies["async_DAG"]["predicted_hypothetical_parallel_graph_ms"],85.00001)
        self.assertLessEqual(policies["serial_only"]["predicted_native_serial_ms"],85.00001)
    def test_independent_exact_source_only_MILP_reproduction(self):
        base,choices,quality=impl.option_bank(self.q["quality_upgrade_by_role_layer"],self.stages,self.roles)
        gamma,serialbase,_=impl.calibrate(self.stages["stages"],self.roles)
        policies={p["name"]:p for p in self.data["policies"]}
        for name in ("async_DAG","serial_only"):
            candidate=impl.solve_one(name,self.cfg,base,choices,quality,gamma,serialbase)
            self.assertEqual(candidate["frozen_policy_sha256"],policies[name]["frozen_policy_sha256"])
            self.assertAlmostEqual(candidate["source_activation_local_quality_gain"],
                                   policies[name]["source_activation_local_quality_gain"],places=8)
    def test_async_and_serial_optimal_quality_vs_constraints(self):
        p={x["name"]:x for x in self.data["policies"]}
        self.assertNotEqual(p["async_DAG"]["frozen_policy_sha256"],p["serial_only"]["frozen_policy_sha256"])
        self.assertGreaterEqual(p["async_DAG"]["source_activation_local_quality_gain"],
                                p["serial_only"]["source_activation_local_quality_gain"]-1e-8)
        self.assertFalse(self.data["production_model_unchanged"] is False)
    def test_private_evidence_idempotent(self):
        self.assertEqual(self.root.stat().st_mode&0o777,0o700)
        for file in self.root.iterdir():
            self.assertEqual(file.stat().st_mode&0o777,0o600)
        with self.assertRaises(FileExistsError):impl.main()
if __name__=="__main__":unittest.main(verbosity=2)
