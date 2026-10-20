"""ACS algebra and reliability weighting. Pure NumPy; does not load or mutate models.

Research baseline: Kim & Han, arXiv:2605.09875v1, eqs. (1)-(5).
Target label access is intentionally absent from the ACS reconstruction API.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

EPS = 1e-10

def _array(x, name: str, ndim: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != ndim or any(d < 1 for d in a.shape) or not np.isfinite(a).all():
        raise ValueError(f"{name} requires finite {ndim}-D nonempty data")
    return a

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("zero or nonfinite direction")
    return v / norm

@dataclass(frozen=True)
class AnchorProjector:
    """Rows of A-hat are mean-centered, individually L2-normalized anchors."""
    mu: np.ndarray
    ahat: np.ndarray

    @classmethod
    def fit(cls, anchor_activations) -> "AnchorProjector":
        a = _array(anchor_activations, "anchors", 2)
        if a.shape[0] < 2:
            raise ValueError("at least two anchors needed")
        mu = a.mean(axis=0)
        centered = a - mu
        norms = np.linalg.norm(centered, axis=1)
        if (norms <= EPS).any():
            raise ValueError("zero-norm centered anchor; remove duplicates/degenerate samples")
        return cls(mu=mu, ahat=centered / norms[:, None])

    @property
    def hidden_dim(self) -> int:
        return self.ahat.shape[1]

    @property
    def anchor_count(self) -> int:
        return self.ahat.shape[0]

    def project(self, vector, *, treat_as_point: bool) -> np.ndarray:
        """pi_pt(h)=Ahat unit(h-mu); pi_dir(v)=Ahat unit(v).
        Treating a difference direction as an absolute point is an error.
        """
        v = _array(vector, "activation", 1)
        if len(v) != self.hidden_dim:
            raise ValueError("wrong hidden dimension")
        if not isinstance(treat_as_point, bool):
            raise TypeError("treat_as_point must be explicit bool")
        return self.ahat @ unit(v - self.mu if treat_as_point else v)

    def backproject(self, canonical_direction, *, normalize_output: bool = True) -> np.ndarray:
        """Transpose/adjoint projection (not an inverse).

        normalize_output=False matches displayed ACS paper Eq. (5) literally.
        normalize_output=True retains our earlier normalized steering variant.
        The text also discusses unit-normalizing axes; report which mode ran.
        """
        c = _array(canonical_direction, "canonical_direction", 1)
        if c.shape[0] != self.anchor_count:
            raise ValueError("incompatible anchors/order")
        raw = self.ahat.T @ c
        if normalize_output not in (True, False):
            raise TypeError("normalize_output must be bool")
        return unit(raw) if normalize_output else raw

def source_axis_direction(positives, negatives) -> np.ndarray:
    p, n = _array(positives, "positives", 2), _array(negatives, "negatives", 2)
    if p.shape[1] != n.shape[1]:
        raise ValueError("positive/negative hidden dimensions differ")
    return unit(p.mean(axis=0) - n.mean(axis=0))

def canonical_mean_raw(projected_directions, weights=None) -> np.ndarray:
    """Literal displayed Eq. (4): mean without additional L2 normalization.

    For C1 source reliability weighting, pass fixed nonnegative weights.
    This is a scale-preserving reference, separate from steering-normalized axes.
    """
    u = _array(projected_directions, "projected_directions", 2)
    n = u.shape[0]
    w = np.full(n, 1.0 / n) if weights is None else _array(weights, "weights", 1)
    if w.shape != (n,) or (w < 0).any() or float(w.sum()) <= EPS:
        raise ValueError("weights must be a nonnegative vector, one per source")
    return w @ u / w.sum()

def canonical_direction(projected_directions, weights=None) -> np.ndarray:
    """Normalized axis *variant*, used in earlier G1 engineering tests.

    The displayed Eq. (4) itself is returned by canonical_mean_raw().
    The paper text describes unit-normalization yet omits it in Eqs. (4)-(5).
    Preserve both to test the effect rather than silently pick one.
    """
    return unit(canonical_mean_raw(projected_directions, weights))

def simplex_projection(v) -> np.ndarray:
    """Euclidean projection onto nonnegative weights summing to 1."""
    x = _array(v, "weights", 1)
    u = np.sort(x)[::-1]
    c = np.cumsum(u) - 1.0
    valid = u - c / np.arange(1, len(u) + 1) > 0
    rho = np.where(valid)[0][-1]
    theta = c[rho] / (rho + 1)
    return np.maximum(x - theta, 0)

def weights_from_source_errors(errors, *, dependence: float = 1.0,
                               ridge: float = 1e-6, iterations: int = 2000) -> np.ndarray:
    """Source-only OOF errors (n_questions, n_models), NOT target labels.
    Minimize w' E[e e'] w on simplex; dependence=0 drops off-diagonal
    co-errors (reliability-only), dependence=1 retains correlations.
    This is a defined candidate estimator, not a claimed optimal oracle.
    """
    e = _array(errors, "out-of-fold source errors", 2)
    if e.shape[0] < 3 or e.shape[1] < 2 or (e < 0).any() or (e > 1).any():
        raise ValueError("errors must be >=3 samples, >=2 models, in [0,1]")
    if not 0 <= dependence <= 1 or ridge < 0 or iterations < 1:
        raise ValueError("invalid weight parameters")
    R = (e.T @ e) / e.shape[0]
    R = (1 - dependence) * np.diag(np.diag(R)) + dependence * R
    R += ridge * np.eye(e.shape[1])
    max_ev = max(float(np.linalg.eigvalsh(R)[-1]), EPS)
    w = np.full(e.shape[1], 1.0 / e.shape[1])
    for _ in range(iterations):
        next_w = simplex_projection(w - (R @ w) / max_ev)
        if np.linalg.norm(next_w - w) < 1e-12:
            break
        w = next_w
    return w

def candidate_probs(candidate_log_likelihoods):
    """Normalize common candidate-string log-likelihood scores per question."""
    s = _array(candidate_log_likelihoods, "candidate scores", 1)
    z = np.exp(s - s.max())
    return z / z.sum()

def mixture_nll(model_candidate_probs, gold_indices, weights) -> float:
    p = _array(model_candidate_probs, "probabilities", 3)
    # dimensions (questions, models, common_candidates)
    labels = np.asarray(gold_indices, dtype=int)
    w = _array(weights, "weights", 1)
    if labels.shape != (p.shape[0],) or w.shape != (p.shape[1],):
        raise ValueError("label/weight dimensionality mismatch")
    if not np.allclose(p.sum(axis=-1), 1, atol=1e-6) or (p < 0).any():
        raise ValueError("each candidate distribution must be normalized")
    if (w < 0).any() or w.sum() <= EPS or (labels < 0).any() or (labels >= p.shape[2]).any():
        raise ValueError("invalid weights or labels")
    mixed = np.einsum("qmc,m->qc", p, w / w.sum())
    return float(-np.log(np.maximum(mixed[np.arange(len(labels)), labels], 1e-300)).mean())
