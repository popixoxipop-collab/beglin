"""Deterministic algebra/anti-leakage smoke tests. No model claims."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "oml_acs"))
from acs_core import (AnchorProjector, source_axis_direction, canonical_direction, canonical_mean_raw,
                      weights_from_source_errors, candidate_probs, mixture_nll, simplex_projection)

class ACSMathTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[12., 4., 3.], [8., 4., 3.], [10., 6., 3.], [10., 2., 3.]])
        self.p = AnchorProjector.fit(self.a)

    def test_exact_paper_rows_and_mu(self):
        self.assertTrue(np.allclose(self.p.mu, [10., 4., 3.]))
        self.assertTrue(np.allclose(np.linalg.norm(self.p.ahat, axis=1), 1.0))

    def test_point_and_direction_have_distinct_centering(self):
        v = np.array([1., 1., 0.])
        self.assertTrue(np.allclose(self.p.project(v, treat_as_point=False),
                                    self.p.ahat @ (v / np.linalg.norm(v))))
        self.assertTrue(np.allclose(self.p.project(v, treat_as_point=True),
                                    self.p.ahat @ ((v - self.p.mu) / np.linalg.norm(v - self.p.mu))))
        self.assertFalse(np.allclose(self.p.project(v, treat_as_point=True),
                                     self.p.project(v, treat_as_point=False)))

    def test_axis_difference_is_translation_invariant(self):
        p = np.array([[2., 5.], [3., 6.]])
        n = np.array([[1., 0.], [0., 1.]])
        self.assertTrue(np.allclose(source_axis_direction(p, n),
                                    source_axis_direction(p + 9, n + 9)))

    def test_backprojection_is_adjoint_and_unit(self):
        c = np.array([0., 1., 0., -1.])
        expected = self.p.ahat.T @ c
        expected /= np.linalg.norm(expected)
        self.assertTrue(np.allclose(self.p.backproject(c), expected))
        self.assertAlmostEqual(np.linalg.norm(self.p.backproject(c)), 1.)

    def test_uniform_canonical_equals_eq4_direction(self):
        u = np.array([[1., 0., 0., 0.], [0., 2., 0., 0.]])
        expected = np.array([0.5, 1., 0., 0.])
        expected /= np.linalg.norm(expected)
        self.assertTrue(np.allclose(canonical_direction(u), expected))

    def test_weighted_canonical(self):
        u = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.assertTrue(np.allclose(canonical_direction(u, [1., 0.]), [1., 0., 0., 0.]))

    def test_literal_eq4_canonical_mean_preserves_scale(self):
        u = np.array([[2., 0., 0., 0.], [0., 3., 0., 0.]])
        got = canonical_mean_raw(u)
        self.assertTrue(np.allclose(got, [1., 1.5, 0., 0.]))
        self.assertNotAlmostEqual(np.linalg.norm(got), 1.0)
        self.assertTrue(np.allclose(canonical_direction(u), got / np.linalg.norm(got)))

    def test_literal_eq5_adjoint_is_not_normalized(self):
        c = np.array([0., 1., 0., -1.])
        raw = self.p.ahat.T @ c
        self.assertTrue(np.allclose(self.p.backproject(c, normalize_output=False), raw))
        self.assertTrue(np.allclose(self.p.backproject(c, normalize_output=True), raw / np.linalg.norm(raw)))

    def test_weighted_literal_mean(self):
        u = np.array([[3., 0., 0., 0.], [0., 2., 0., 0.]])
        self.assertTrue(np.allclose(canonical_mean_raw(u, weights=[3., 1.]), [2.25, 0.5, 0., 0.]))

    def test_wrong_anchor_count_and_degenerate_fail_closed(self):
        with self.assertRaises(ValueError):
            self.p.backproject([1., 2.])
        with self.assertRaises(ValueError):
            AnchorProjector.fit(np.ones((3, 4)))

    def test_zero_norm_source_axis_fails(self):
        with self.assertRaises(ValueError):
            source_axis_direction([[1., 0.]], [[1., 0.]])

    def test_simplex(self):
        w = simplex_projection([2., 0., -3.])
        self.assertTrue(np.all(w >= 0))
        self.assertAlmostEqual(w.sum(), 1.0)

    def test_correlated_sources_are_not_counted_twice(self):
        e1 = np.array([1., 1., 0., 0., 1., 1., 0., 0.])
        e3 = np.array([1., 0., 1., 0., 0., 1., 0., 1.])
        e = np.stack((e1, e1, e3), axis=1)
        wr = weights_from_source_errors(e, dependence=0.0)
        wc = weights_from_source_errors(e, dependence=1.0)
        self.assertTrue(np.allclose(wr, np.ones(3)/3, atol=1e-3))
        self.assertGreater(wc[2], wc[0])
        self.assertTrue(np.allclose(wc[0], wc[1], atol=1e-6))
        self.assertAlmostEqual(wc.sum(), 1.0, places=7)

    def test_error_domain_rejected(self):
        with self.assertRaises(ValueError):
            weights_from_source_errors([[0,1],[2,0],[1,0]])

    def test_candidate_prob_and_mixture_nll(self):
        p = candidate_probs([-1., -1.])
        self.assertTrue(np.allclose(p, [0.5, 0.5]))
        arr = np.array([[[0.9,0.1],[0.2,0.8]], [[0.8,0.2],[0.1,0.9]]])
        val = mixture_nll(arr, [0,1], [0.5,0.5])
        self.assertTrue(np.isfinite(val))
        with self.assertRaises(ValueError):
            mixture_nll(arr, [0,5], [0.5,0.5])

if __name__ == "__main__":
    unittest.main(verbosity=2)
