#!/usr/bin/env python3
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import inspect_model as im
import model_capability as mc
from test_model_capability_p12 import qwen_fixture, verification_evidence


class InspectRuntimePlanTests(unittest.TestCase):
    @staticmethod
    def _args(*, verify=False):
        return SimpleNamespace(
            verify_loader_sources=verify,
            tokenizer_executable=None,
            tokenizer_executable_sha256=None,
        )

    def test_runtime_report_preserves_bundle_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root=qwen_fixture(Path(td)/"m")
            bundle=mc.compile_model_capabilities(root)
            before=bundle["bundle_sha256"]
            report=im._runtime_plan_report(bundle,self._args())
            self.assertEqual(bundle["bundle_sha256"],before)
            self.assertEqual(report["loader"]["status"],"AVAILABLE")
            self.assertEqual(
                report["loader"]["plan"]["status"],"VALIDATION_REQUIRED"
            )
            self.assertFalse(
                report["loader"]["plan"]["execution_allowed"]
            )
            self.assertEqual(report["tokenizer"]["status"],"AVAILABLE")
            self.assertEqual(
                report["tokenizer"]["plan"]["status"],"VALIDATION_REQUIRED"
            )

    def test_verified_runtime_readiness_and_source_rehash(self):
        with tempfile.TemporaryDirectory() as td:
            root=qwen_fixture(Path(td)/"m")
            loader_evidence=verification_evidence(
                root,component="loader",evidence_byte="d"
            )
            tokenizer_evidence=verification_evidence(
                root,component="tokenizer",evidence_byte="e"
            )
            bundle=mc.compile_model_capabilities(
                root,
                loader_evidence=loader_evidence,
                tokenizer_evidence=tokenizer_evidence,
            )
            report=im._runtime_plan_report(bundle,self._args(verify=True))
            loader=report["loader"]
            tokenizer=report["tokenizer"]
            self.assertEqual(loader["status"],"AVAILABLE")
            self.assertEqual(loader["plan"]["status"],"READY")
            self.assertTrue(loader["plan"]["execution_allowed"])
            self.assertEqual(
                loader["source_verification"]["status"],
                "SOURCE_FILES_VERIFIED",
            )
            self.assertGreater(loader["source_verification"]["file_count"],0)
            self.assertEqual(tokenizer["status"],"AVAILABLE")
            self.assertEqual(
                tokenizer["plan"]["status"],"READY_IN_ENGINE"
            )

    def test_runtime_report_fails_closed_for_unsupported_loader(self):
        with tempfile.TemporaryDirectory() as td:
            root=qwen_fixture(Path(td)/"m")
            bundle=mc.compile_model_capabilities(root)
            bad=dict(bundle)
            bad_loader=dict(bundle["loader_contract"])
            bad_loader["status"]="UNSUPPORTED"
            bad_loader["loader_contract_sha256"]=mc.stable_identity_sha256(
                bad_loader
            )
            bad["loader_contract"]=bad_loader
            bad["bundle_sha256"]=mc.stable_identity_sha256(bad)
            report=im._runtime_plan_report(bad,self._args())
            self.assertEqual(report["loader"]["status"],"UNAVAILABLE")
            self.assertIn("unsupported",report["loader"]["error"].lower())


if __name__=="__main__":
    unittest.main()
