#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import manual_canary_contract as mc
import production_adapter_bridge as bridge


def proposal():
    return {
        "architecture": "mla",
        "backend": "mlx_metal",
        "baseline_policy_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "binary_sha256": "6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392",
        "budget": {
            "max_duration_ms": 30000,
            "max_memory_bytes": 7890145280,
            "max_requests": 18,
            "max_tokens": 256,
        },
        "candidate_policy_hash": "0e048b8c5c50a1c005ea573caadd6ccdf99797c24b9c443e993fe0d7b0b27141",
        "checkpoint_sha256": "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898",
        "environment_id": "xox-vdsp-gpu-precision",
        "evidence_refs": [
            {
                "kind": "G4_A_B_R",
                "run_id": "f-20261002T091915Z-32080",
                "sha256": "d3a814ccd685cb4bdb2bdad6e0c7561b3ba54b123ed32395168227846c162b9e",
            },
            {
                "kind": "G6_RESTART_CANARY",
                "run_id": "f-20261002T091915Z-32080",
                "sha256": "f06a9b6b537f7db317421ab5a6922e1c3c239c9a9f975324e2480501ba5829ee",
            },
        ],
        "expected_epoch": 0,
        "kill_switch_scope": "single-target",
        "mode": "dry_run",
        "model_revision": "deepseek-v2-lite",
        "production_write_allowed": False,
        "proposal_id": "beglin-shared-up-l3-n6-20261002",
        "proposer": "A0-planner",
        "restart_instance_id": "pid-59196",
        "rollback_plan": "restore exact runtime baseline preimage",
        "schema": "manual-canary-proposal-v1",
        "single_target": {
            "after_n": 6,
            "before_n": None,
            "layer": 3,
            "role": "shared_up_proj",
        },
        "source_commit": "330954b27f146b8a17db2cb353c3e620968bad5e",
    }


def runtime_preimage():
    return {
        "active_policy": [],
        "active_policy_hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "weight_epoch": 0,
        "ack_sha256": "67365f4df26d895b05a683f0f63e1f26808bf849a3d4274a2f3b6159cd868c8d",
        "worker_instance_id": "pid-59196",
    }


def candidate_policy():
    return [{"role": "shared_up_proj", "layer": 3, "n": 6}]


def attestation_verification():
    return {
        "schema": "beglin-github-attestation-verification/1",
        "status": "VERIFIED",
        "artifact_sha256": "2b9cd86d6455f3e2a2d3ed88eaf9011d72ca12cf2d5baaaf1f627e9a3c306ab5",
        "repository": "popixoxipop-collab/beglin",
        "signer_workflow": bridge.EXPECTED_SIGNER_WORKFLOW,
        "source_ref": bridge.EXPECTED_SOURCE_REF,
        "verified_attestations": 1,
    }


def github_seal():
    return {
        "schema": "beglin-agent-e-github-oidc-seal/1",
        "status": "VERIFIED_GITHUB_OIDC_ATTESTATION_SUBJECT",
        "github": {
            "repository": "popixoxipop-collab/beglin",
            "actor": "popixoxipop-collab",
            "actor_id": "261965176",
            "sha": "163d7ecd0edaca36bfb68f69a75bbbd711e02ad5",
            "ref": "refs/heads/agent-e-github-approval-seal-20261002",
            "workflow": "Agent E GitHub Approval Attestation",
            "run_id": "37010516604",
            "run_attempt": "1",
        },
        "approval_file_sha256": "54be8b73c69db7e42f2407bdcef2ced6a817e34a4177817f6e5edcbc942c7dc2",
        "proposal_digest": "726ae550c7519307e771d1f9fb23157a7f75338921082f7a64282be56c9cbc15",
        "measured_capture_sha256": "caec06441a9fab41fd0f98cb8fc027931671903b001e28561e94d4dd82d4ab2f",
        "private_approval_commit": "7dc3398c87ffbe8e9bf2496c6d16e762ea787094",
        "private_verification_run_id": "37010332740",
        "execution_enabled": False,
        "production_write_allowed": False,
        "auto_promotion_enabled": False,
        "production_bridge_reviewed": False,
        "ssh_detached_signature_used": False,
        "approval_path": "github_account_plus_github_oidc",
    }


def plan():
    return bridge.build_bridge_plan(
        proposal=proposal(),
        runtime_preimage=runtime_preimage(),
        candidate_policy=candidate_policy(),
        github_seal=github_seal(),
        measured_capture_sha256="caec06441a9fab41fd0f98cb8fc027931671903b001e28561e94d4dd82d4ab2f",
        attestation_verification=attestation_verification(),
    )


def good_result():
    return {
        "schema": "mlx-isolated-restart-canary-result-v1",
        "source_commit": "330954b27f146b8a17db2cb353c3e620968bad5e",
        "binary_sha256": "6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392",
        "checkpoint_sha256": "1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898",
        "candidate_policy_hash": "0e048b8c5c50a1c005ea573caadd6ccdf99797c24b9c443e993fe0d7b0b27141",
        "requested_policy_applied": True,
        "returncode": 0,
        "finite_logits": True,
        "reference_token_match": True,
        "requests": 12,
        "tokens": 120,
        "duration_ms": 4500,
        "memory_bytes": 6400000000,
    }


class BridgePlanTests(unittest.TestCase):
    def test_actual_proposal_digest_is_stable(self):
        self.assertEqual(
            mc.proposal_digest(proposal()),
            "726ae550c7519307e771d1f9fb23157a7f75338921082f7a64282be56c9cbc15",
        )

    def test_build_plan_binds_measured_github_approval(self):
        got = plan()
        self.assertEqual(got["status"], "READY_FOR_ISOLATED_RESTART_CANARY")
        self.assertTrue(got["candidate_worker_launch_allowed"])
        self.assertFalse(got["production_routing_switch_allowed"])
        self.assertFalse(got["baseline_worker_mutation_allowed"])
        self.assertEqual(got["canary_requests"], 12)
        self.assertEqual(got["target"]["before_n"], None)
        self.assertEqual(got["target"]["after_n"], 6)
        self.assertEqual(len(got["plan_sha256"]), 64)

    def test_tampered_approval_digest_fails(self):
        seal = github_seal()
        seal["proposal_digest"] = "0" * 64
        with self.assertRaises(bridge.ProductionAdapterBridgeError):
            bridge.build_bridge_plan(
                proposal=proposal(),
                runtime_preimage=runtime_preimage(),
                candidate_policy=candidate_policy(),
                github_seal=seal,
                measured_capture_sha256="caec06441a9fab41fd0f98cb8fc027931671903b001e28561e94d4dd82d4ab2f",
                attestation_verification=attestation_verification(),
            )

    def test_missing_attestation_verification_fails(self):
        with self.assertRaises(bridge.ProductionAdapterBridgeError):
            bridge.validate_github_approval_seal(
                github_seal(),
                proposal_digest=mc.proposal_digest(proposal()),
                measured_capture_sha256="caec06441a9fab41fd0f98cb8fc027931671903b001e28561e94d4dd82d4ab2f",
                attestation_verification=None,
            )

    def test_runtime_worker_mismatch_fails(self):
        runtime = runtime_preimage()
        runtime["worker_instance_id"] = "pid-other"
        with self.assertRaises(bridge.ProductionAdapterBridgeError):
            bridge.build_bridge_plan(
                proposal=proposal(),
                runtime_preimage=runtime,
                candidate_policy=candidate_policy(),
                github_seal=github_seal(),
                measured_capture_sha256="caec06441a9fab41fd0f98cb8fc027931671903b001e28561e94d4dd82d4ab2f",
                attestation_verification=attestation_verification(),
            )


class VerdictTests(unittest.TestCase):
    def test_good_isolated_candidate_never_auto_promotes(self):
        got = bridge.evaluate_isolated_candidate(plan(), good_result())
        self.assertEqual(got["status"], "CANARY_PASS_REVIEW_REQUIRED")
        self.assertEqual(got["decision"], "NO_AUTO_PROMOTION")
        self.assertTrue(got["candidate_worker_must_exit"])
        self.assertTrue(got["baseline_remains_active"])
        self.assertFalse(got["production_routing_switch_allowed"])

    def test_regression_fails_closed(self):
        result = good_result()
        result["reference_token_match"] = False
        got = bridge.evaluate_isolated_candidate(plan(), result)
        self.assertEqual(got["status"], "ROLLBACK_REQUIRED")
        self.assertIn("reference_token_match", got["reasons"])
        self.assertTrue(got["baseline_remains_active"])

    def test_budget_overrun_fails_closed(self):
        result = good_result()
        result["memory_bytes"] = 8000000000
        got = bridge.evaluate_isolated_candidate(plan(), result)
        self.assertEqual(got["status"], "ROLLBACK_REQUIRED")
        self.assertIn("memory_bytes_budget", got["reasons"])

    def test_production_switch_is_unconditionally_disabled(self):
        with self.assertRaises(bridge.ProductionRoutingDisabled):
            bridge.execute_production_routing_switch(plan())


class AttestationCliTests(unittest.TestCase):
    def test_gh_verifier_uses_repo_workflow_ref_and_hosted_runner_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact = root / "seal.json"
            artifact.write_text("{}")
            gh = root / "gh"
            gh.write_text("")
            seen = {}

            def fake_runner(cmd, **kwargs):
                seen["cmd"] = cmd
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=json.dumps([{"verificationResult": {}}]), stderr=""
                )

            got = bridge.verify_github_attestation(
                artifact,
                gh_path=gh,
                runner=fake_runner,
            )
            self.assertEqual(got["status"], "VERIFIED")
            cmd = seen["cmd"]
            self.assertIn("--repo", cmd)
            self.assertIn(bridge.EXPECTED_REPO, cmd)
            self.assertIn("--signer-workflow", cmd)
            self.assertIn(bridge.EXPECTED_SIGNER_WORKFLOW, cmd)
            self.assertIn("--source-ref", cmd)
            self.assertIn(bridge.EXPECTED_SOURCE_REF, cmd)
            self.assertIn("--deny-self-hosted-runners", cmd)


if __name__ == "__main__":
    unittest.main()
