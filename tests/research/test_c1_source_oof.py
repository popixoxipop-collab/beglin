import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
from evaluate_c1_source_oof import shared_fold_plan,source_oof,paired_bootstrap,source_axis,compute_case
from acs_core import weights_from_source_errors

def synthetic_model(n=200,hidden=4,anchors=5):
    rng=np.random.default_rng(123)
    classlabel=np.array([0]*100+[1]*100)
    arr=rng.normal(size=(n,hidden))+classlabel[:,None]*np.array([1.,0.,0.,0.])[None,:]
    a=rng.normal(size=(anchors,hidden))
    a=a/np.linalg.norm(a,axis=1)[:,None]
    return dict(X=arr,ids=[f"{i:064x}" for i in range(n)],axis=np.array(["sst2_positive_vs_negative"]*n),
         split=np.array(["behavior_train"]*n),y=classlabel,mu=np.zeros(hidden),A=a,rev="a"*40)

class OOFUnitTests(unittest.TestCase):
 def setUp(self):
    self.x=synthetic_model()
 def test_fold_stratified_and_total(self):
    folds=shared_fold_plan(self.x,"sst2_positive_vs_negative")
    self.assertEqual(len(folds["indices"]),200)
    for fold in range(5):
        labels=folds["labels"][folds["fold_ids"]==fold]
        self.assertEqual(((labels==1).sum(),(labels==0).sum()),(20,20))
 def test_fold_determinism(self):
    a=shared_fold_plan(self.x,"sst2_positive_vs_negative")
    b=shared_fold_plan(self.x,"sst2_positive_vs_negative")
    self.assertEqual(a["manifest_sha256"],b["manifest_sha256"])
 def test_source_oof_not_in_sample(self):
    f=shared_fold_plan(self.x,"sst2_positive_vs_negative")
    o=source_oof(self.x,"sst2_positive_vs_negative",f)
    self.assertEqual(len(o["error"]),200)
    self.assertTrue(np.isin(o["error"],[0,1]).all())
    self.assertEqual(len(o["thresholds"]),5)
 def test_source_model_fails_missing_class(self):
    z=dict(self.x)
    z["y"]=np.zeros_like(self.x["y"])
    with self.assertRaises(RuntimeError):
        shared_fold_plan(z,"sst2_positive_vs_negative")
 def test_paired_bootstrap_identical_scores_zero_difference(self):
    y=np.array([0]*50+[1]*50)
    s=np.linspace(0,1,100)
    result=paired_bootstrap({"uniform":s,"reliability_only":s,"correlated_error":s},y,draws=50)
    self.assertEqual(result["correlated_minus_uniform"]["lower"],0)
    self.assertEqual(result["correlated_minus_reliability_only"]["upper"],0)
 def test_correlated_and_diagonal_weight_simplex(self):
    e=np.column_stack((np.array([0,1]*100),np.array([0,0,1,1]*50)))
    for dependence in (0,1):
        w=weights_from_source_errors(e,dependence=dependence)
        self.assertEqual(len(w),2)
        self.assertTrue(np.all(w>=0))
        self.assertAlmostEqual(w.sum(),1,places=8)
 def test_unknown_axis_fail_closed(self):
    with self.assertRaises(ValueError):
        compute_case("qwen","not-a-valid-axis",{})
 def test_missing_target_fails_closed(self):
    with self.assertRaises(ValueError):
        compute_case("unknown","sst2_positive_vs_negative",{})
 def test_public_protocol_count(self):
    from evaluate_c1_source_oof import PARTICIPANTS,AXES
    self.assertEqual((len(PARTICIPANTS),len(AXES)),(3,2))
if __name__=="__main__":unittest.main(verbosity=2)
