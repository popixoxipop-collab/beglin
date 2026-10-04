#!/usr/bin/env python3
import copy
import unittest

import model_capability as mc
import precision_p12_pipeline_bridge as bridge


class P12PipelineBridgeTests(unittest.TestCase):
    def evidence(self):
        return [{
            "kind": "REAL_GPU",
            "sha256": "e" * 64,
            "status": "VERIFIED",
            "ref": "fixture",
        }]

    def bundle(self, *, mutation="HOT_REBIND_SINGLE", eligibility="FULL"):
        target = "m/L3/shared_up"
        cell = {
            "target_key": target,
            "backend": "mlx_metal",
            "inference_status": "VERIFIED",
            "mutation_mode": mutation,
            "supported_n": [5, 6],
            "quant_formats": ["qNg64"],
            "evidence_refs": self.evidence(),
            "reason_code": "fixture",
        }
        bundle = mc.build_model_capability_bundle(
            model_id="m",
            checkpoint_identity="a" * 64,
            skeleton_sha256="b" * 64,
            architecture_status="KNOWN",
            tokenizer_status="IN_ENGINE_VERIFIED",
            loader_status="VERIFIED",
            backend_matrix=[cell],
        )
        if eligibility != bundle["p8_p11_eligibility"]:
            bundle = copy.deepcopy(bundle)
            bundle["p8_p11_eligibility"] = eligibility
            bundle["bundle_sha256"] = mc.sha256_json({
                k: v for k, v in bundle.items() if k != "bundle_sha256"
            })
        return bundle

    def test_hot_rebind_selects_same_worker_canary(self):
        bundle = self.bundle()
        gate = bridge.build_p8_capability_gate(
            model_capability_bundle=bundle,
            target_key="m/L3/shared_up",
            backend="mlx_metal",
            requested_n=5,
        )
        self.assertEqual(gate["canary_mode"], "SAME_WORKER_PRECISION_EPOCH")
        self.assertFalse(gate["production_write_allowed"])

    def test_restart_target_selects_isolated_restart_canary(self):
        bundle = self.bundle(mutation="RESTART_REQUIRED")
        gate = bridge.build_p8_capability_gate(
            model_capability_bundle=bundle,
            target_key="m/L3/shared_up",
            backend="mlx_metal",
            requested_n=5,
        )
        self.assertEqual(gate["canary_mode"], "ISOLATED_RESTART_CANARY")

    def test_denied_bundle_cannot_enter_p8(self):
        bundle = self.bundle(eligibility="DENIED")
        with self.assertRaisesRegex(
            bridge.P12PipelineBridgeError, "denies P8-P11"
        ):
            bridge.build_p8_capability_gate(
                model_capability_bundle=bundle,
                target_key="m/L3/shared_up",
                backend="mlx_metal",
                requested_n=5,
            )

    def test_unverified_or_unsupported_precision_is_rejected(self):
        bundle = self.bundle()
        with self.assertRaisesRegex(
            bridge.P12PipelineBridgeError, "unsupported"
        ):
            bridge.build_p8_capability_gate(
                model_capability_bundle=bundle,
                target_key="m/L3/shared_up",
                backend="mlx_metal",
                requested_n=13,
            )

    def test_bundle_sha_threads_p9_p10_p11(self):
        bundle = self.bundle()
        gate = bridge.build_p8_capability_gate(
            model_capability_bundle=bundle,
            target_key="m/L3/shared_up",
            backend="mlx_metal",
            requested_n=5,
        )
        p9 = bridge.bind_p9_certification(
            certification_bundle={
                "status": "MANUAL_REVIEW_CANDIDATE",
                "production_write_allowed": False,
                "automatic_live_promotion": False,
                "proposal_id": "p",
            },
            capability_gate=gate,
        )
        selection = bridge.select_p10_canary(
            p9_bundle=p9,
            model_capability_bundle=bundle,
        )
        p11 = bridge.build_p11_capability_binding(
            p10_final_gate={
                "status": "AWAITING_TRUSTED_PRODUCTION_APPROVAL",
                "production_cutover_allowed": False,
                "production_write_allowed": False,
            },
            p10_selection=selection,
            model_capability_bundle=bundle,
        )
        self.assertEqual(
            p9["model_capability_bundle_sha256"], bundle["bundle_sha256"]
        )
        self.assertEqual(
            selection["model_capability_bundle_sha256"], bundle["bundle_sha256"]
        )
        self.assertEqual(
            p11["model_capability_bundle_sha256"], bundle["bundle_sha256"]
        )
        self.assertFalse(p11["production_cutover_allowed"])

    def test_stale_bundle_is_rejected_at_p10(self):
        bundle = self.bundle()
        gate = bridge.build_p8_capability_gate(
            model_capability_bundle=bundle,
            target_key="m/L3/shared_up",
            backend="mlx_metal",
            requested_n=5,
        )
        p9 = bridge.bind_p9_certification(
            certification_bundle={
                "status": "MANUAL_REVIEW_CANDIDATE",
                "production_write_allowed": False,
                "automatic_live_promotion": False,
            },
            capability_gate=gate,
        )
        changed = copy.deepcopy(bundle)
        changed["backend_matrix"][0]["reason_code"] = "changed"
        changed["bundle_sha256"] = mc.sha256_json({
            k: v for k, v in changed.items() if k != "bundle_sha256"
        })
        with self.assertRaisesRegex(
            bridge.P12PipelineBridgeError, "stale"
        ):
            bridge.select_p10_canary(
                p9_bundle=p9,
                model_capability_bundle=changed,
            )


if __name__ == "__main__":
    unittest.main()
