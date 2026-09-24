#!/usr/bin/env python3
import unittest

import precision_context as pc
from precision_planner_v3 import evaluate_candidate


def h(ch):
    return ch * 64


def context(backend="mlx_metal", binary="a"):
    return pc.ExecutionContext(
        schema="precision-context-v3",
        model_id="m",
        architecture="arch",
        checkpoint_sha256=h("1"),
        tokenizer_sha256=h("2"),
        base_artifact_sha256=h("3"),
        backend=backend,
        device_fingerprint="dev",
        binary_sha256=h(binary),
        build_manifest_sha256=h("4"),
        kernel_revision="k",
        execution_mode="online",
        runtime_config_sha256=h("5"),
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        correction_mode="off",
    )


BASE = [{"role": "shared_up_proj", "layer": 3, "n": 5}]


class PlannerV3Tests(unittest.TestCase):
    def test_missing_evidence_requires_backend_preflight(self):
        got = evaluate_candidate(
            context=context(),
            current_policy=BASE,
            current_epoch=7,
            role="shared_down_proj",
            layer=4,
            n=6,
            preflight_evidence=None,
        )
        self.assertEqual(got["action"], "NEEDS_BACKEND_PREFLIGHT")

    def test_cpu_or_other_context_evidence_cannot_admit_gpu(self):
        ctx = context()
        got = evaluate_candidate(
            context=ctx,
            current_policy=BASE,
            current_epoch=7,
            role="shared_down_proj",
            layer=4,
            n=6,
            preflight_evidence={
                "context_hash": context("cpu").context_hash,
            },
        )
        self.assertEqual(got["action"], "EVIDENCE_CONTEXT_MISMATCH")

    def test_old_policy_preimage_is_stale(self):
        ctx = context()
        got = evaluate_candidate(
            context=ctx,
            current_policy=BASE,
            current_epoch=7,
            role="shared_down_proj",
            layer=4,
            n=6,
            preflight_evidence={
                "context_hash": ctx.context_hash,
                "baseline_policy_hash": "old",
            },
        )
        self.assertEqual(got["action"], "STALE_PREFLIGHT_PREIMAGE")

    def test_pass_requires_exact_candidate_and_epoch(self):
        ctx = context()
        candidate = BASE + [{"role": "shared_down_proj", "layer": 4, "n": 6}]
        evidence = {
            "context_hash": ctx.context_hash,
            "baseline_policy_hash": pc.policy_hash(BASE),
            "requested_policy_hash": pc.policy_hash(candidate),
            "applied_policy_hash": pc.policy_hash(candidate),
            "expected_epoch": 7,
            "observed_epoch": 8,
            "status": "passed",
            "pass": True,
        }
        got = evaluate_candidate(
            context=ctx,
            current_policy=BASE,
            current_epoch=7,
            role="shared_down_proj",
            layer=4,
            n=6,
            preflight_evidence=evidence,
        )
        self.assertEqual(got["action"], "ADMIT_ONE_TARGET")
        self.assertEqual(got["backend"], "mlx_metal")

    def test_same_candidate_from_different_binary_context_is_rejected(self):
        ctx = context(binary="a")
        other = context(binary="b")
        candidate = BASE + [{"role": "shared_down_proj", "layer": 4, "n": 6}]
        got = evaluate_candidate(
            context=ctx,
            current_policy=BASE,
            current_epoch=7,
            role="shared_down_proj",
            layer=4,
            n=6,
            preflight_evidence={
                "context_hash": other.context_hash,
                "baseline_policy_hash": pc.policy_hash(BASE),
                "requested_policy_hash": pc.policy_hash(candidate),
                "applied_policy_hash": pc.policy_hash(candidate),
                "expected_epoch": 7,
                "observed_epoch": 8,
                "status": "passed",
                "pass": True,
            },
        )
        self.assertEqual(got["action"], "EVIDENCE_CONTEXT_MISMATCH")


if __name__ == "__main__":
    unittest.main()
