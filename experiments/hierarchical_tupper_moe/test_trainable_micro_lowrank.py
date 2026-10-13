"""Rank-8 residual compression and heldout nonleakage regression tests."""
import unittest
import numpy as np
from trainable_micro_router import fit_one_expert,lowrank_residual_model,predict_lowrank_residual
from prompt_holdout_micro_eval import paired_improvement_ci

class LowrankCorrectorTests(unittest.TestCase):
    def test_constant_residual_is_exact(self):
        rng=np.random.default_rng(201)
        x=rng.normal(size=(20,32))
        missing=np.tile(np.arange(16,dtype=float),(20,1))
        model=lowrank_residual_model(x,missing,rank=8)
        z=predict_lowrank_residual(model,x[:4])
        np.testing.assert_allclose(z,missing[:4],atol=1.e-8)

    def test_rank8_is_smaller_than_full_ridge(self):
        rng=np.random.default_rng(202)
        x=rng.normal(size=(34,64))
        y=rng.normal(size=(34,8,64))
        val=rng.normal(size=(8,8,64))
        p=fit_one_expert(x[:26],y[:26],x[26:],val,.5)
        self.assertIn("dynamic_rank8_residual",p)
        self.assertEqual(p["dynamic_rank8_residual"].shape,(8,64))
        mem=p["estimated_fp32_storage_by_method"]
        self.assertLess(mem["dynamic_rank8_residual"],mem["dynamic_ridge_residual"])
        self.assertTrue(np.all(np.isfinite(p["dynamic_rank8_residual"])))

    def test_router_prediction_never_uses_holdout_targets(self):
        rng=np.random.default_rng(203)
        x=rng.normal(size=(23,48))
        cal=rng.normal(size=(15,12,48))
        hold=rng.normal(size=(8,12,48))
        orig=fit_one_expert(x[:15],cal,x[15:],hold,.5)
        bad=fit_one_expert(x[:15],cal,x[15:],hold*999,.5)
        np.testing.assert_array_equal(orig["dynamic_mask"],bad["dynamic_mask"])
        self.assertEqual(orig["estimated_fp32_storage_by_method"],bad["estimated_fp32_storage_by_method"])

    def test_paired_bootstrap_grouped_by_prompt(self):
        keys=[("a",0),("a",1),("b",0),("c",0),("d",0)]
        target=np.ones((5,6))
        base=target+.8
        good=target+.5
        row=paired_improvement_ci(keys,target,base,good,draws=100)
        self.assertGreater(row["relative_l2_reduction"],0)
        self.assertGreater(row["prompt_block_bootstrap_95ci"][0],0)

if __name__=="__main__":unittest.main()
