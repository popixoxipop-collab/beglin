#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_capability as mc
import p12_pipeline_bridge as bridge
from test_model_capability_p12 import qwen_fixture, write_safetensors


class PipelineBridgeTests(unittest.TestCase):
    def bundle(self, *, cpu_verified=False, mlx_verified=False):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = qwen_fixture(Path(td.name) / "qwen")
        source = mc.inspect_model_source(root)
        descriptor = mc.build_architecture_descriptor(source)

        def evidence(backend, marker):
            return {
                "schema": mc.VERIFICATION_EVIDENCE_SCHEMA,
                "status": "VERIFIED",
                "component": "backend_runtime",
                "architecture_id": descriptor["architecture_id"],
                "checkpoint_identity_sha256": source["checkpoint_identity_sha256"],
                "backend": backend,
                "evidence_sha256": marker * 64,
                "run_id": f"p12-test-{backend}",
                "kind": "TEST_RUNTIME_EVIDENCE",
            }

        return mc.compile_model_capabilities(
            root,
            cpu_runtime_evidence=evidence("cpu", "1") if cpu_verified else None,
            mlx_runtime_evidence=evidence("mlx_metal", "2") if mlx_verified else None,
        )

    @staticmethod
    def q_target(bundle):
        return next(
            row["canonical_target_key"]
            for row in bundle["tensor_role_graph"]["nodes"]
            if row["role"] == "Q_PROJ" and row["layer"] == 0
        )

    def test_runtime_verified_flag_without_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "qwen")
            with self.assertRaisesRegex(
                mc.ModelCapabilityError,
                "mlx_runtime_verified requires explicit mlx_runtime_evidence",
            ):
                mc.compile_model_capabilities(root, mlx_runtime_verified=True)

    def test_deepseek_verified_mlx_selects_same_worker_canary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "deepseek"
            root.mkdir(parents=True)
            (root / "config.json").write_text(__import__("json").dumps({
                "_name_or_path": "acme/deepseek-v2-p12-fixture",
                "model_type": "deepseek_v2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 256,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "n_shared_experts": 1,
                "first_k_dense_replace": 0,
            }, sort_keys=True))
            tensors = {
                "model.embed_tokens.weight": ("F16", [128, 64]),
                "model.norm.weight": ("F16", [64]),
                "lm_head.weight": ("F16", [128, 64]),
                "model.layers.0.self_attn.q_proj.weight": ("F16", [64, 64]),
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": ("F16", [64, 64]),
                "model.layers.0.self_attn.kv_b_proj.weight": ("F16", [64, 64]),
                "model.layers.0.self_attn.o_proj.weight": ("F16", [64, 64]),
                "model.layers.0.mlp.gate.weight": ("F16", [4, 64]),
                "model.layers.0.mlp.shared_experts.gate_proj.weight": ("F16", [128, 64]),
                "model.layers.0.mlp.shared_experts.up_proj.weight": ("F16", [128, 64]),
                "model.layers.0.mlp.shared_experts.down_proj.weight": ("F16", [64, 128]),
            }
            write_safetensors(root / "model.safetensors", tensors)
            source = mc.inspect_model_source(root)
            descriptor = mc.build_architecture_descriptor(source)
            runtime_evidence = {
                "schema": mc.VERIFICATION_EVIDENCE_SCHEMA,
                "status": "VERIFIED",
                "component": "backend_runtime",
                "architecture_id": descriptor["architecture_id"],
                "checkpoint_identity_sha256": source["checkpoint_identity_sha256"],
                "backend": "mlx_metal",
                "evidence_sha256": "3" * 64,
                "run_id": "p12-test-deepseek-mlx",
                "kind": "TEST_RUNTIME_EVIDENCE",
            }
            bundle = mc.compile_model_capabilities(
                root, mlx_runtime_evidence=runtime_evidence
            )
            target = next(
                row["canonical_target_key"]
                for row in bundle["tensor_role_graph"]["nodes"]
                if row["role"] == "SHARED_UP" and row["layer"] == 0
            )
            p8 = bridge.bind_p8_target(
                bundle, target_key=target, backend="mlx_metal", requested_n=5
            )
            p9 = bridge.bind_p9_certification(
                bundle, p8_binding=p8, provenance_evidence_sha256="9" * 64
            )
            p10 = bridge.select_p10_canary(bundle, p9_binding=p9)
            self.assertEqual(p10["status"], "READY_FOR_P10_CANARY")
            self.assertEqual(
                p10["canary_strategy"], "SAME_WORKER_PRECISION_CANARY"
            )
            self.assertEqual(p10["backend_inference_status"], "VERIFIED")
            self.assertEqual(p10["qng64_status"], "VERIFIED")

    def test_qwen_cpu_p10_selects_restart_canary(self):
        bundle = self.bundle(cpu_verified=True)
        target = self.q_target(bundle)
        p8 = bridge.bind_p8_target(
            bundle, target_key=target, backend="cpu", requested_n=5
        )
        p9 = bridge.bind_p9_certification(
            bundle, p8_binding=p8, provenance_evidence_sha256="b" * 64
        )
        p10 = bridge.select_p10_canary(bundle, p9_binding=p9)
        self.assertEqual(p10["canary_strategy"], "ISOLATED_RESTART_CANARY")
        self.assertEqual(p10["status"], "READY_FOR_P10_CANARY")

    def test_unverified_mlx_requires_runtime_validation_before_p10(self):
        bundle = self.bundle()
        target = self.q_target(bundle)
        p8 = bridge.bind_p8_target(
            bundle, target_key=target, backend="mlx_metal", requested_n=5
        )
        p9 = bridge.bind_p9_certification(
            bundle, p8_binding=p8, provenance_evidence_sha256="c" * 64
        )
        p10 = bridge.select_p10_canary(bundle, p9_binding=p9)
        self.assertEqual(
            p10["canary_strategy"], "ISOLATED_RUNTIME_VALIDATION"
        )
        self.assertEqual(
            p10["status"], "VALIDATION_REQUIRED_BEFORE_P10_CANARY"
        )

    def test_requested_precision_outside_capability_is_rejected(self):
        bundle = self.bundle()
        target = self.q_target(bundle)
        with self.assertRaisesRegex(
            bridge.PipelineCapabilityError, "requested n=4 unsupported"
        ):
            bridge.bind_p8_target(
                bundle, target_key=target, backend="mlx_metal", requested_n=4
            )

    def test_tampered_p8_binding_is_rejected_by_p9(self):
        bundle = self.bundle(mlx_verified=True)
        target = self.q_target(bundle)
        p8 = bridge.bind_p8_target(
            bundle, target_key=target, backend="mlx_metal", requested_n=5
        )
        tampered = dict(p8)
        tampered["requested_n"] = 6
        with self.assertRaisesRegex(
            bridge.PipelineCapabilityError, "p8_binding_sha256 mismatch"
        ):
            bridge.bind_p9_certification(
                bundle,
                p8_binding=tampered,
                provenance_evidence_sha256="d" * 64,
            )

    def test_stale_bundle_is_rejected(self):
        bundle = self.bundle(mlx_verified=True)
        target = self.q_target(bundle)
        p8 = bridge.bind_p8_target(
            bundle, target_key=target, backend="mlx_metal", requested_n=5
        )
        stale = copy.deepcopy(bundle)
        stale["model_id"] = "tampered-model"
        stale["bundle_sha256"] = mc.stable_identity_sha256(stale)
        with self.assertRaisesRegex(
            bridge.PipelineCapabilityError,
            "P8 binding uses stale model capability bundle",
        ):
            bridge.bind_p9_certification(
                stale,
                p8_binding=p8,
                provenance_evidence_sha256="e" * 64,
            )

    def test_non_deepseek_full_bundle_reaches_p11_capability_preimage(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "qwen-full")
            source = mc.inspect_model_source(root)
            descriptor = mc.build_architecture_descriptor(source)

            def evidence(component, marker, backend=None):
                row = {
                    "schema": mc.VERIFICATION_EVIDENCE_SCHEMA,
                    "status": "VERIFIED",
                    "component": component,
                    "architecture_id": descriptor["architecture_id"],
                    "checkpoint_identity_sha256": source["checkpoint_identity_sha256"],
                    "evidence_sha256": marker * 64,
                    "run_id": f"p12-full-{component}",
                    "kind": "TEST_RUNTIME_EVIDENCE",
                }
                if backend is not None:
                    row["backend"] = backend
                return row

            bundle = mc.compile_model_capabilities(
                root,
                backend="mlx_metal",
                mlx_runtime_evidence=evidence(
                    "backend_runtime", "4", backend="mlx_metal"
                ),
                tokenizer_evidence=evidence("tokenizer", "5"),
                loader_evidence=evidence("loader", "6"),
            )
            self.assertEqual(bundle["p8_p11_eligibility"]["status"], "FULL")
            target = self.q_target(bundle)
            p8 = bridge.bind_p8_target(
                bundle, target_key=target, backend="mlx_metal", requested_n=5
            )
            p9 = bridge.bind_p9_certification(
                bundle, p8_binding=p8, provenance_evidence_sha256="7" * 64
            )
            p10 = bridge.select_p10_canary(bundle, p9_binding=p9)
            self.assertEqual(p10["status"], "READY_FOR_P10_CANARY")
            self.assertEqual(
                p10["canary_strategy"], "SAME_WORKER_PRECISION_CANARY"
            )
            p11 = bridge.build_p11_capability_preimage(
                bundle,
                p10_binding=p10,
                runtime_state={
                    "model_capability_bundle_sha256": bundle["bundle_sha256"],
                    "checkpoint_identity_sha256": bundle[
                        "checkpoint_identity_sha256"
                    ],
                    "backend": "mlx_metal",
                    "target_key": target,
                    "worker_pid": 1234,
                    "weight_epoch": 9,
                },
            )
            self.assertEqual(
                p11["status"], "AWAITING_TRUSTED_PRODUCTION_APPROVAL"
            )
            self.assertFalse(p11["production_cutover_allowed"])
            self.assertEqual(
                p11["model_capability_bundle_sha256"], bundle["bundle_sha256"]
            )

    def test_partial_bundle_cannot_materialize_p11_preimage(self):
        bundle = self.bundle(mlx_verified=True)
        target = self.q_target(bundle)
        p8 = bridge.bind_p8_target(
            bundle, target_key=target, backend="mlx_metal", requested_n=5
        )
        p9 = bridge.bind_p9_certification(
            bundle, p8_binding=p8, provenance_evidence_sha256="f" * 64
        )
        p10 = bridge.select_p10_canary(bundle, p9_binding=p9)
        self.assertEqual(bundle["p8_p11_eligibility"]["status"], "PARTIAL")
        with self.assertRaisesRegex(
            bridge.PipelineCapabilityError, "P11 denied"
        ):
            bridge.build_p11_capability_preimage(
                bundle,
                p10_binding=p10,
                runtime_state={
                    "model_capability_bundle_sha256": bundle["bundle_sha256"],
                    "checkpoint_identity_sha256": bundle[
                        "checkpoint_identity_sha256"
                    ],
                    "backend": "mlx_metal",
                    "target_key": target,
                },
            )


if __name__ == "__main__":
    unittest.main()
