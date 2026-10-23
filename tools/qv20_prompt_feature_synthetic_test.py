#!/usr/bin/env python3
"""BEGLIN Qv20 synthetic-only prompt feature/LOOCV fallback contract.
Does NOT read, score, fit to, or change the locked ARC/OpenBookQA heldout data.
Run: python3 -B tools/qv20_prompt_feature_synthetic_test.py
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "qv20_fit_prompt_policy.py"
spec = importlib.util.spec_from_file_location("BEGLIN_QV20_FIT_UNIT", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert (module.ALPHA, module.WINSOR_NLL, module.SWITCH_MARGIN,
        module.MIN_LOOCV_IMPROVEMENT, module.MIN_USAGE_EACH) == (6.0, 2.0, 0.1, 0.03, 4)
rng = np.random.default_rng(20261024)
checked = 0

for n in list(range(18, 209, 7)) + [18, 84, 85, 86, 208]:
    for pattern in range(4):
        if pattern == 0:
            ids = rng.integers(0, 151936, size=n).tolist()
        elif pattern == 1:
            ids = [151935] * n
        elif pattern == 2:
            ids = [j % 3 for j in range(n)]
        else:
            ids = [99999 if j % 2 else 100000 for j in range(n)]
        actual = module.feat(ids)
        independent = [math.log(n / 85.0),
                       len(set(ids)) / n,
                       sum(v >= 100000 for v in ids) / n,
                       sum(ids) / (151936.0 * n),
                       sum(ids[j] == ids[j-1] for j in range(1, n)) / (n-1)]
        assert len(actual) == 5
        assert all(math.isfinite(v) for v in actual)
        assert all(abs(a-b) < 1e-14 for a, b in zip(actual, independent))
        checked += 1

# Sign: positive random73-NLL minus Qv8-NLL means use Qv8.
assert module.choose(0.10001, 1) == 0
assert module.choose(-0.10001, 0) == 1
assert module.choose(0.1, 1) == 1
assert module.choose(-0.1, 0) == 0

features = np.asarray([
    module.feat(rng.integers(0, 151936, size=(60 if i % 2 == 0 else 115)).tolist())
    for i in range(32)
])
uninformative = np.zeros(32)
model = module.ridge(features, uninformative)
assert all(abs(module.predict(f, model)) < 1e-12 for f in features)
assert all(module.choose(module.predict(f, model), 0) == 0 for f in features)
assert all(module.choose(module.predict(f, model), 1) == 1 for f in features)

print("QV20_SYNTHETIC_ONLY_POLICY_CONTRACT_PASS", json.dumps({
    "feature_vectors_verified": checked,
    "strict_positive_negative_prediction_margin": True,
    "uninformative_calibration_falls_back_to_best_static": True,
    "real_model_outputs_used": False,
    "evaluation_labels_or_frozen_tokens_opened": False,
    "trained_real_policy_created": False,
    "fit_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
}, sort_keys=True))
