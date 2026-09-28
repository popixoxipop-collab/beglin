from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path

from .artifact import compare_identity, validate_run_artifact
from .corpus import load_corpus
from .evaluate import classify_pair, load_policy
from .metrics import evaluate_pair, metric_completeness
from .prepare_matrix import build_plan

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "configs" / "quality" / "deterministic_corpus_v1.json"
POLICY_PATH = ROOT / "configs" / "quality" / "quality_policy_v1.json"
MATRIX_PATH = ROOT / "configs" / "quality" / "p9_quality_matrix_v1.json"


def task_output(entry: dict) -> str:
    task = entry["task"]
    if task["type"] == "exact_match":
        return task["expected"]
    if task["type"] == "contains_all":
        return "alpha beta"
    if task["type"] == "numeric_tolerance":
        return str(int(task["expected"]))
    return "OK"


def make_run(corpus: dict, *, kind: str, n: int | None, quality_missing: bool = False) -> dict:
    identity = {
        "source_commit": "a" * 40,
        "source_tree": "b" * 64,
        "binary_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "model_id": "deepseek-v2-lite",
        "hardware_id": "xox-apple-silicon",
        "backend": "mlx_metal",
        "seed": 0,
        "prompt_corpus_sha256": corpus["corpus_sha256"],
        "runtime_config_hash": "e" * 64,
        "policy_hash": None if kind == "reference" else ("f" * 64),
    }
    requests = []
    for index, entry in enumerate(corpus["entries"]):
        for repetition in (0, 1):
            use_ppl = entry["use_for_perplexity"]
            requests.append({
                "request_id": f"{kind}-{n}-{entry['id']}-{repetition}",
                "prompt_id": entry["id"],
                "repetition": repetition,
                "context_tokens": 3000 if "long_context" in entry["tags"] else 64,
                "output_token_ids": [1000 + index, 2000 + index],
                "output_text": task_output(entry),
                "finite_logits": True,
                "token_logprobs": None,
                "nll_sum": (1.01 if kind == "candidate" else 1.0) if use_ppl else None,
                "nll_token_count": 10 if use_ppl else None,
                "corrections": 1 if kind == "candidate" else 0,
                "promotions": 1 if kind == "candidate" else 0,
                "quality_score": None if quality_missing and index == 0 and repetition == 0 else (
                    0.94 if kind == "candidate" else 0.95
                ),
                "activation_summaries": [{
                    "role": "shared_up_proj",
                    "layer": 3,
                    "expert_id": None,
                    "mean_abs": 1.01 if kind == "candidate" else 1.0,
                    "rms": 2.02 if kind == "candidate" else 2.0,
                }],
                "router": {
                    "near_tie_count": 2 if kind == "candidate" else 1,
                    "decision_count": 100,
                    "near_tie_rate": None,
                },
            })

    return validate_run_artifact({
        "schema": "beglin-quality-run/1",
        "run_id": f"synthetic-{kind}-{n}",
        "variant": {
            "kind": kind,
            "precision": "safe_baseline" if kind == "reference" else f"n{n}",
            "n": n,
        },
        "identity": identity,
        "requests": requests,
    })


def main() -> None:
    corpus = load_corpus(CORPUS_PATH)
    policy = load_policy(POLICY_PATH)
    reference = make_run(corpus, kind="reference", n=None)
    candidate = make_run(corpus, kind="candidate", n=6)

    pair = evaluate_pair(reference, candidate, corpus)
    assert pair["status"] == "MEASURED"
    assert compare_identity(reference, candidate)["compatible"] is True
    assert metric_completeness(pair)["complete"] is True

    metrics = pair["metrics"]
    assert metrics["finite_logits"]["rate"] == 1.0
    assert metrics["candidate_determinism"]["rate"] == 1.0
    assert metrics["reference_output_match"]["rate"] == 1.0
    assert metrics["candidate_task_score"]["value"] == 1.0
    assert metrics["long_context_stability"]["rate"] == 1.0
    assert metrics["correction_count"] == len(candidate["requests"])
    assert metrics["promotion_count"] == len(candidate["requests"])
    assert metrics["candidate_router_near_tie"]["rate"] == 0.02
    assert abs(metrics["activation_drift"]["max_abs_rms_relative_delta"] - 0.01) < 1e-12
    assert metrics["perplexity_ratio"] is not None
    assert metrics["candidate_quality_score"] == 0.94

    gate = classify_pair(pair, policy)
    assert gate["status"] == "MEASURED_POLICY_NOT_FROZEN"

    incomplete = make_run(corpus, kind="candidate", n=5, quality_missing=True)
    incomplete_pair = evaluate_pair(reference, incomplete, corpus)
    incomplete_gate = classify_pair(incomplete_pair, policy)
    assert incomplete_gate["status"] == "INCOMPLETE_MISSING_METRICS"
    assert "candidate_quality_score" in incomplete_gate["missing_required_metrics"]

    mismatch = deepcopy(candidate)
    mismatch["identity"]["binary_sha256"] = "0" * 64
    mismatch_pair = evaluate_pair(reference, mismatch, corpus)
    assert mismatch_pair["status"] == "CONTROL_MISMATCH"
    assert mismatch_pair["identity"]["differing_fields"] == ["binary_sha256"]

    no_p8 = build_plan(MATRIX_PATH, CORPUS_PATH, None)
    assert no_p8["status"] == "P9_QUALITY_HARNESS_READY"
    assert no_p8["execution_allowed"] is False
    assert [cell["n"] for cell in no_p8["cells"]] == [None, 4, 5, 6, 7]

    with tempfile.TemporaryDirectory(prefix="beglin-p9-selftest-") as temp:
        p8_path = Path(temp) / "p8.json"
        p8_path.write_text(json.dumps({
            "schema": "beglin-p8-integration/1",
            "status": "P8_INTEGRATION_PASS",
            "gates": {
                "runtime": "P8_RUNTIME_VALID_8_OF_8",
                "policy": "P8_POLICY_ATTRIBUTION_8_OF_8",
                "evidence": "P8_EVIDENCE_INDEPENDENT_PASS",
            },
            "identity": {
                "source_commit": "a" * 40,
                "source_tree": "b" * 64,
                "binary_sha256": "c" * 64,
                "checkpoint_sha256": "d" * 64,
                "model_id": "deepseek-v2-lite",
                "hardware_id": "xox-apple-silicon",
                "backend": "mlx_metal",
                "seed": 0,
                "runtime_config_hash": "e" * 64,
            },
        }) + "\n", encoding="utf-8")
        after_p8 = build_plan(MATRIX_PATH, CORPUS_PATH, p8_path)
        assert after_p8["status"] == "P9_EXECUTION_PLAN_READY"
        # Agent D intentionally leaves the physical runner adapter unwired until A0 integrates P8.
        assert after_p8["execution_allowed"] is False

    print(
        "P9 quality harness PASS "
        "ppl=PASS task=PASS quality=PASS finite=PASS deterministic=PASS "
        "correction_promotion=PASS activation=PASS near_tie=PASS long_context=PASS "
        "missing_metric_fail_closed=PASS p8_gate=PASS"
    )


if __name__ == "__main__":
    main()
