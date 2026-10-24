#!/usr/bin/env python3
"""Synthetic-only parity contract for the actual Qv20 C selector generator.

Imports the original qv20_build_prompt_selector_native.py.choice_source
from this checkout, compiles generated C on the CI host, and compares it with
a separately implemented Python selector. Does not load any real model,
calibration receipts, heldout samples, or private datasets.
"""
import hashlib
import importlib.util
import json
import math
import pathlib
import random
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
GENERATOR = ROOT / "qv20_build_prompt_selector_native.py"
spec = importlib.util.spec_from_file_location("original_qv20_builder", GENERATOR)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def independent_python(ids, policy):
    if not policy["dynamic_policy_admitted_from_calibration"]:
        return int(policy["static_fallback_index"])
    n = len(ids)
    features = [
        math.log(n / 85.0), len(set(ids)) / n,
        sum(x >= 100000 for x in ids) / n,
        sum(ids) / (151936.0 * n),
        sum(ids[i] == ids[i-1] for i in range(1, n)) / (n-1),
    ]
    beta = policy["coefficients_intercept_and_slopes"]
    mu = policy["feature_means"]
    scale = policy["feature_scales"]
    delta = beta[0] + sum(
        beta[i+1] * (features[i]-mu[i]) / scale[i] for i in range(5)
    )
    threshold = policy["profile_Qv8_if_predicted_delta_greater_than"]
    return 0 if delta > threshold else 1 if delta < -threshold else policy["static_fallback_index"]


def make_cases():
    rng = random.Random(20261024)
    samples = []
    for n in list(range(18, 209, 7)) + [18, 84, 85, 86, 208]:
        samples.extend((
            [rng.randrange(151936) for _ in range(n)],
            [151935] * n,
            [j % 3 for j in range(n)],
            [100000 if j % 2 else 99999 for j in range(n)],
        ))
    assert len(samples) == 132
    return samples


def check_c_selector(policy, samples, directory):
    body = build.choice_source(policy)
    assert body.count("static int qv20_source_only_choice") == 1
    source = (
        "#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n"
        + body
        + """
int main(void) {
  int n;
  while (scanf("%d", &n) == 1) {
    if (n < 18 || n > 208) return 5;
    int ids[208];
    for (int j = 0; j < n; j++) {
      if (scanf("%d", &ids[j]) != 1) return 6;
    }
    printf("%d\\n", qv20_source_only_choice(ids, n));
  }
  return 0;
}
"""
    )
    exe = directory / ("qv20_" + hashlib.sha256(source.encode()).hexdigest()[:12])
    cc = subprocess.run(
        ["cc", "-O2", "-std=c11", "-o", str(exe), "-x", "c", "-", "-lm"],
        input=source, text=True, capture_output=True, check=False,
    )
    assert cc.returncode == 0, cc.stderr[-1000:]
    inputs = "".join(str(len(v)) + " " + " ".join(map(str, v)) + "\n" for v in samples)
    run = subprocess.run([str(exe)], input=inputs, text=True, capture_output=True, check=False)
    assert run.returncode == 0, run.stderr[-1000:]
    actual = [int(line) for line in run.stdout.splitlines()]
    expected = [independent_python(v, policy) for v in samples]
    assert actual == expected, "Original emitted Qv20 C disagrees with independent Python"
    return actual


def main():
    base = {
        "dynamic_policy_admitted_from_calibration": True,
        "static_fallback_index": 0,
        "coefficients_intercept_and_slopes": [0.005, 0.12, -0.32, 0.18, -0.11, 0.07],
        "feature_means": [0.0, 0.5, 0.3, 0.4, 0.1],
        "feature_scales": [0.4, 0.3, 0.15, 0.2, 0.08],
        "profile_Qv8_if_predicted_delta_greater_than": 0.10,
    }
    cases = make_cases()
    completed = 0
    with tempfile.TemporaryDirectory(prefix="qv20_codegen_parity_") as temp:
        for admitted in (False, True):
            for fallback in (0, 1):
                policy = dict(base, dynamic_policy_admitted_from_calibration=admitted,
                              static_fallback_index=fallback)
                check_c_selector(policy, cases, pathlib.Path(temp))
                completed += len(cases)
        for boundary in (0.1, -0.1, 0.1000001, -0.1000001):
            policy = dict(base, static_fallback_index=1,
                          coefficients_intercept_and_slopes=[boundary, 0, 0, 0, 0, 0])
            check_c_selector(policy, cases[:2], pathlib.Path(temp))
            completed += 2
    print("QV20_SYNTHETIC_ORIGINAL_GENERATOR_C_PYTHON_PARITY_PASS",
          json.dumps({
              "source_sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
              "total_decisions": completed,
              "synthetic_only": True,
              "calibration_or_heldout_items_used": 0,
              "actual_Qwen28_native_inference_proven": False,
          }, sort_keys=True))


if __name__ == "__main__":
    main()
