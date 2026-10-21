import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
from audit_two_model_anchors import center_kernel, linear_cka_gram, upper_pearson, metrics

class GeometryTests(unittest.TestCase):
    def test_center_kernel_row_and_column_means_zero(self):
        k=np.array([[1.,2.,0.],[2.,5.,4.],[0.,4.,9.]])
        h=center_kernel(k)
        self.assertTrue(np.allclose(h.mean(0),0))
        self.assertTrue(np.allclose(h.mean(1),0))
    def test_equal_gram_has_cka_one(self):
        k=np.array([[1.,0.2,0.3],[0.2,1.,0.7],[0.3,0.7,1.]])
        self.assertAlmostEqual(linear_cka_gram(k,k),1.)
        self.assertAlmostEqual(upper_pearson(k,k),1.)
    def test_permuted_alignment_less(self):
        q=np.eye(18,dtype=float)
        q[:,0]=q[:,0]+np.linspace(0,1.0,18)
        q=q/np.linalg.norm(q,axis=1)[:,None]
        m=q.copy()
        out=metrics(q,m,repeats=39,seed=42)
        self.assertAlmostEqual(out["alignment_CKA_biased"],1.,places=7)
        self.assertGreater(out["alignment_CKA_biased"],out["CKA_row_shuffle_mean"])
        self.assertAlmostEqual(out["offdiag_pairwise_cosine_pearson"],1.,places=7)
        self.assertGreater(out["offdiag_pairwise_cosine_pearson"],out["offdiag_pearson_shuffle_mean"])
    def test_fails_closed_wrong_shape_and_nonunit(self):
        with self.assertRaises(ValueError):
            metrics(np.eye(3),np.eye(4))
        with self.assertRaises(ValueError):
            metrics(np.eye(3)*2,np.eye(3))
        with self.assertRaises(ValueError):
            metrics(np.eye(3)*np.nan,np.eye(3))
    def test_no_values_from_hidden_final_labels(self):
        q=np.eye(10);m=q.copy()
        out=metrics(q,m,repeats=4)
        self.assertEqual(out["shuffle_count"],4)
        self.assertEqual(out["row_count"],10)

if __name__=="__main__":unittest.main(verbosity=2)
