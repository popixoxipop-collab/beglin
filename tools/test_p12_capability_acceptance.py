#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_capability as mc
import p12_capability_acceptance as acceptance
import p12_pipeline_bridge as bridge
from test_model_capability_p12 import qwen_fixture


class CapabilityAcceptanceTests(unittest.TestCase):
    def fixture(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = qwen_fixture(Path(td.name) / "qwen")
        source = mc.inspect_model_source(root)
        descriptor = mc.build_architecture_descriptor(source)
        provisional = mc.compile_model_capabilities(root, backend="mlx_metal")
        target = next(
            row["canonical_target_key"]
            for row in provisional["tensor_role_graph"]["nodes"]
            if row["role"] == "Q_PROJ" and row["layer"] == 0
        )

        def evidence(component, marker, *, backend=None, target_key=None,
                     supported_n=None, mutation_mode=None):
            row = {
                "schema": mc.VERIFICATION_EVIDENCE_SCHEMA,
                "status": "VERIFIED",
                "component": component,
                "architecture_id": descriptor["architecture_id"],
                "checkpoint_identity_sha256": source["checkpoint_identity_sha256"],
                "evidence_sha256": marker * 64,
                "run_id": f"p12-acceptance-{component}",
                "kind": "UNIT_TEST_ACCEPTANCE",
            }
            if backend is not None:
                row["backend"] = backend
            if target_key is not None:
                row["target_key"] = target_key
            if supported_n is not None:
                row["supported_n"] = list(supported_n)
            if mutation_mode is not None:
                row["mutation_mode"] = mutation_mode
            return row

        return {
            "root": root,
            "target": target,
            "runtime": evidence("backend_runtime", "1", backend="mlx_metal"),
            "loader": evidence("loader", "2"),
            "tokenizer": evidence("tokenizer", "3"),
            "qng64": evidence(
                "qng64_runtime", "4", backend="mlx_metal",
                target_key=target, supported_n=[5, 6],
            ),
            "mutation": evidence(
                "mutation_runtime", "5", backend="mlx_metal",
                target_key=target, supported_n=[5, 6],
                mutation_mode="HOT_REBIND_SINGLE",
            ),
            "provenance_sha": "6" * 64,
        }

    def test_materializes_p11_only_with_exact_verified_target(self):
        f = self.fixture()
        bundle, result = acceptance.materialize(
            model_path=str(f["root"]),
            backend="mlx_metal",
            runtime_evidence=f["runtime"],
            loader_evidence=f["loader"],
            tokenizer_evidence=f["tokenizer"],
            qng64_evidence=[f["qng64"]],
            mutation_evidence=[f["mutation"]],
            target_key=f["target"],
            requested_n=5,
            provenance_evidence_sha256=f["provenance_sha"],
            runtime_state={
                "backend": "mlx_metal",
                "target_key": f["target"],
                "precision_n": 5,
                "worker_pid": 123,
                "weight_epoch": 7,
            },
        )
        self.assertEqual(result["status"], "P11_CAPABILITY_PREIMAGE_READY")
        self.assertEqual(
            result["p10_binding"]["canary_strategy"],
            "SAME_WORKER_PRECISION_CANARY",
        )
        self.assertEqual(result["p10_binding"]["requested_n"], 5)
        self.assertEqual(
            result["p11_capability_preimage"]["runtime_precision_n"], 5
        )
        self.assertEqual(bundle["p8_p11_eligibility"]["status"], "FULL")
        self.assertFalse(result["production_write_allowed"])

    def test_runtime_precision_mismatch_is_rejected(self):
        f = self.fixture()
        with self.assertRaisesRegex(
            bridge.PipelineCapabilityError, "runtime precision mismatch"
        ):
            acceptance.materialize(
                model_path=str(f["root"]),
                backend="mlx_metal",
                runtime_evidence=f["runtime"],
                loader_evidence=f["loader"],
                tokenizer_evidence=f["tokenizer"],
                qng64_evidence=[f["qng64"]],
                mutation_evidence=[f["mutation"]],
                target_key=f["target"],
                requested_n=5,
                provenance_evidence_sha256=f["provenance_sha"],
                runtime_state={
                    "backend": "mlx_metal",
                    "target_key": f["target"],
                    "precision_n": 6,
                    "worker_pid": 123,
                    "weight_epoch": 7,
                },
            )

    def test_missing_target_evidence_stops_before_p11(self):
        f = self.fixture()
        bundle, result = acceptance.materialize(
            model_path=str(f["root"]),
            backend="mlx_metal",
            runtime_evidence=f["runtime"],
            loader_evidence=f["loader"],
            tokenizer_evidence=f["tokenizer"],
            target_key=f["target"],
            requested_n=5,
            provenance_evidence_sha256=f["provenance_sha"],
        )
        self.assertEqual(
            result["status"], "VALIDATION_REQUIRED_BEFORE_P10_CANARY"
        )
        self.assertTrue(
            bundle["precision_search_targets"][0]["requires_validation"]
        )


if __name__ == "__main__":
    unittest.main()
