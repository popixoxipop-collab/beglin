#!/usr/bin/env python3
import unittest

import capability_compiler_p12 as cc
import model_capability as mc


class CapabilityCompilerTests(unittest.TestCase):
    def graph(self, *, include_unknown=False):
        nodes = [{
            "source_tensor_name": "model.layers.3.self_attn.q_proj.weight",
            "role": "Q_PROJ",
            "layer": 3,
            "shape": [16, 16],
            "dtype": "F16",
            "source_quant_format": "F16",
            "mapping_status": "MAPPED",
        }]
        if include_unknown:
            nodes.append({
                "source_tensor_name": "model.layers.3.future.weight",
                "role": "UNMAPPED_LLAMA",
                "layer": 3,
                "shape": [16, 16],
                "dtype": "F16",
                "source_quant_format": "F16",
                "mapping_status": "UNSUPPORTED",
            })
        return mc.build_tensor_role_graph(model_id="m", nodes=nodes)

    def evidence(self, ch):
        return [{
            "kind": "CHECKPOINT_BACKEND_VALIDATION",
            "sha256": ch * 64,
            "status": "VERIFIED",
            "ref": "fixture",
        }]

    def test_no_evidence_never_promotes_backend_to_verified(self):
        report = cc.compile_capability_report(
            model_id="m",
            checkpoint_identity="a" * 64,
            skeleton_sha256="b" * 64,
            architecture_status="KNOWN",
            tokenizer_status="IMPLEMENTED_UNVERIFIED",
            loader_status="IMPLEMENTED_UNVERIFIED",
            tensor_role_graph=self.graph(),
        )
        bundle = report["model_capability_bundle"]
        self.assertEqual(bundle["p8_p11_eligibility"], "DENIED")
        self.assertIn(
            "no_verified_backend_target",
            bundle["eligibility_reasons"],
        )
        self.assertEqual(bundle["hot_rebind_targets"], [])
        self.assertEqual(bundle["precision_search_targets"], [])
        for cell in report["backend_capability"]["cells"]:
            self.assertEqual(cell["inference_status"], "IMPLEMENTED_UNVERIFIED")

    def test_verified_evidence_produces_cpu_restart_and_mlx_hot(self):
        target = "m/L3/q_proj"
        evidence = {
            (target, "cpu"): self.evidence("c"),
            (target, "mlx_metal"): self.evidence("d"),
        }
        report = cc.compile_capability_report(
            model_id="m",
            checkpoint_identity="a" * 64,
            skeleton_sha256="b" * 64,
            architecture_status="KNOWN",
            tokenizer_status="IN_ENGINE_VERIFIED",
            loader_status="VERIFIED",
            tensor_role_graph=self.graph(),
            verified_targets={
                "cpu": [target],
                "mlx_metal": [target],
            },
            verified_hot_targets=[target],
            evidence_by_target=evidence,
        )
        bundle = report["model_capability_bundle"]
        self.assertEqual(bundle["p8_p11_eligibility"], "FULL")
        self.assertEqual(bundle["hot_rebind_targets"], [target])
        self.assertEqual(bundle["restart_only_targets"], [])
        self.assertEqual(bundle["precision_search_targets"], [target])

        cells = {
            cell["backend"]: cell
            for cell in report["backend_capability"]["cells"]
        }
        self.assertEqual(cells["cpu"]["mutation_mode"], "RESTART_REQUIRED")
        self.assertEqual(cells["mlx_metal"]["mutation_mode"], "HOT_REBIND_SINGLE")
        self.assertEqual(cells["cpu"]["inference_status"], "VERIFIED")
        self.assertEqual(cells["mlx_metal"]["inference_status"], "VERIFIED")

        mutation = {
            row["backend"]: row
            for row in report["runtime_mutation"]
        }
        self.assertFalse(mutation["cpu"]["requires_quiesce"])
        self.assertTrue(mutation["mlx_metal"]["requires_quiesce"])
        self.assertTrue(mutation["mlx_metal"]["rollback_supported"])

    def test_unsupported_tensor_makes_evidence_backed_bundle_partial(self):
        target = "m/L3/q_proj"
        evidence = {
            (target, "cpu"): self.evidence("e"),
            (target, "mlx_metal"): self.evidence("f"),
        }
        report = cc.compile_capability_report(
            model_id="m",
            checkpoint_identity="a" * 64,
            skeleton_sha256="b" * 64,
            architecture_status="KNOWN",
            tokenizer_status="IN_ENGINE_VERIFIED",
            loader_status="VERIFIED",
            tensor_role_graph=self.graph(include_unknown=True),
            verified_targets={
                "cpu": [target],
                "mlx_metal": [target],
            },
            evidence_by_target=evidence,
        )
        bundle = report["model_capability_bundle"]
        self.assertEqual(bundle["p8_p11_eligibility"], "PARTIAL")
        self.assertEqual(
            bundle["unsupported_targets"],
            ["m/L3/unmapped_llama"],
        )

    def test_bundle_identity_is_deterministic(self):
        kwargs = dict(
            model_id="m",
            checkpoint_identity="a" * 64,
            skeleton_sha256="b" * 64,
            architecture_status="KNOWN",
            tokenizer_status="IMPLEMENTED_UNVERIFIED",
            loader_status="IMPLEMENTED_UNVERIFIED",
            tensor_role_graph=self.graph(),
        )
        a = cc.compile_capability_report(**kwargs)
        b = cc.compile_capability_report(**kwargs)
        self.assertEqual(
            a["model_capability_bundle"]["bundle_sha256"],
            b["model_capability_bundle"]["bundle_sha256"],
        )
        self.assertEqual(
            a["backend_capability"]["matrix_sha256"],
            b["backend_capability"]["matrix_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
