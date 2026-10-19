#!/usr/bin/env python3
"""N6 regression: loader evidence that does not transfer to main's engine must not read VERIFIED / READY.

Refs N6 (issue pending).  The real MLX-affine loader evidence in attestations/ is a record about a
binary built from a commit that is not an ancestor of main and whose engine wiring (g_st_mlx_affine4,
st_dequant_logical_f32) is absent from main's qwen_infer.c.  The provenance addendum next to it says
so (engine_wiring_in_main=false).  Before the fix model_capability.py / loader_runtime_v1.py derived
loader VERIFIED and plan READY from the evidence file alone.

Stand-ins (the real Qwen 3B MLX checkpoint is not in the repo): the model directory is a synthetic
MLX-affine qwen2 fixture, and inspect_model_source() is wrapped so the source carries the REAL
evidence's checkpoint_identity_sha256.  Everything else is the real code on the real evidence file.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loader_runtime_v1 as lrv
import model_capability as mc
import model_evidence_registry as mer
from test_model_capability_p12 import verification_evidence, write_safetensors

# The fixture only needs tensor dtypes/shapes (payload is zeros); no quantizer runs, so no
# residual / error-feedback compensation is applicable to it.
Q4_QUANTIZER_USED = "none_shape_only_fixture_zero_payload_no_residual_error_feedback_applicable"

ATTEST = Path(__file__).resolve().parents[1] / "attestations"
VERIFICATION = ATTEST / "BEGLIN_P12_QWEN3B_MLX_AFFINE_LOADER_VERIFICATION_2026-10-04.json"
ACCEPTANCE = ATTEST / "BEGLIN_P12_QWEN3B_MLX_AFFINE_LOADER_ACCEPTANCE_2026-10-04.json"
ADDENDUM = ATTEST / "BEGLIN_P12_QWEN3B_MLX_AFFINE_LOADER_PROVENANCE_ADDENDUM_2026-10-19.json"


def mlx_affine_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "_name_or_path": "acme/qwen2-mlx-q4",
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    }, sort_keys=True))
    (root / "tokenizer.json").write_text('{"version":"1.0"}')

    def packed(name: str, out: int, inn: int) -> dict:
        return {
            f"{name}.weight": ("U32", [out, inn // 8]),
            f"{name}.scales": ("BF16", [out, inn // 64]),
            f"{name}.biases": ("BF16", [out, inn // 64]),
        }

    tensors: dict = {"model.norm.weight": ("BF16", [64])}
    tensors.update(packed("model.embed_tokens", 128, 64))
    tensors.update(packed("lm_head", 128, 64))
    layer = "model.layers.0."
    for role, (out, inn) in {
        "self_attn.q_proj": (64, 64), "self_attn.k_proj": (16, 64),
        "self_attn.v_proj": (16, 64), "self_attn.o_proj": (64, 64),
        "mlp.gate_proj": (128, 64), "mlp.up_proj": (128, 64),
        "mlp.down_proj": (64, 128),
    }.items():
        tensors.update(packed(layer + role, out, inn))
    write_safetensors(root / "model.safetensors", tensors)
    return root


def compile_pinned(root: Path, checkpoint_identity: str, **kwargs) -> dict:
    """compile_model_capabilities() with the source pinned to a given checkpoint identity."""
    real = mc.inspect_model_source

    def pinned(path):
        source = real(path)
        source["checkpoint_identity_sha256"] = checkpoint_identity
        source["source_manifest_sha256"] = mc.stable_identity_sha256(source)
        return source

    with patch.object(mc, "inspect_model_source", pinned):
        return mc.compile_model_capabilities(root, **kwargs)


class RealRepoEvidenceTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = mlx_affine_fixture(Path(self._td.name) / "mlx-q4")
        self.evidence = json.loads(VERIFICATION.read_text())

    def test_real_evidence_files_and_addendum_are_consistent(self):
        # The gate is keyed on evidence_sha256; pin the chain evidence -> acceptance file -> addendum.
        self.assertEqual(mc.sha256_file(ACCEPTANCE), self.evidence["evidence_sha256"])
        addendum = json.loads(ADDENDUM.read_text())
        self.assertEqual(addendum["schema"], mc.PROVENANCE_ADDENDUM_SCHEMA)
        self.assertIs(addendum["findings"]["engine_wiring_in_main"], False)
        bound = {row.get("evidence_sha256_in_record") or row.get("sha256")
                 for row in addendum["clarifies"]}
        self.assertIn(self.evidence["evidence_sha256"], bound)

    def test_real_loader_evidence_is_not_verified_for_main_engine(self):
        bundle = compile_pinned(
            self.root, self.evidence["checkpoint_identity_sha256"],
            loader_evidence=self.evidence,
        )
        loader = bundle["loader_contract"]
        self.assertEqual(loader["status"], "VERIFIED_FORMAT_ONLY")
        self.assertEqual(loader["source_quantization"]["status"], "VERIFIED_FORMAT_ONLY")
        self.assertIs(loader["evidence_provenance"]["engine_wiring_in_main"], False)
        self.assertEqual(loader["evidence_provenance"]["addendum"], ADDENDUM.name)
        self.assertEqual(loader["evidence_provenance"]["addendum_sha256"], mc.sha256_file(ADDENDUM))
        self.assertIn("EVIDENCE_NOT_BOUND_TO_ENGINE_WIRING_IN_MAIN", loader["unsupported_reason_codes"])

        eligibility = bundle["p8_p11_eligibility"]
        self.assertFalse(eligibility["p11_allowed"])
        self.assertIn("LOADER_NOT_VERIFIED", eligibility["reasons"])

        plan = lrv.plan_from_bundle(bundle)
        self.assertEqual(plan["status"], "VALIDATION_REQUIRED")
        self.assertFalse(plan["execution_allowed"])
        with self.assertRaises(lrv.LoaderRuntimeError):
            lrv.require_executable_plan(plan)

    def test_registry_path_is_downgraded_the_same_way(self):
        registry = mer.VerificationEvidenceRegistry.from_paths([VERIFICATION])
        resolved = registry.resolve(
            component="loader", architecture_id="qwen2",
            checkpoint_identity_sha256=self.evidence["checkpoint_identity_sha256"],
        )
        self.assertIsNotNone(resolved)
        bundle = compile_pinned(
            self.root, self.evidence["checkpoint_identity_sha256"], loader_evidence=resolved
        )
        self.assertEqual(bundle["loader_contract"]["status"], "VERIFIED_FORMAT_ONLY")
        self.assertEqual(lrv.plan_from_bundle(bundle)["status"], "VALIDATION_REQUIRED")

    def test_other_loader_evidence_for_the_same_model_shape_is_unchanged(self):
        # Control: evidence with no addendum behaves exactly as before (VERIFIED -> READY).
        other = verification_evidence(self.root, component="loader", evidence_byte="6")
        bundle = mc.compile_model_capabilities(self.root, loader_evidence=other)
        self.assertEqual(bundle["loader_contract"]["status"], "VERIFIED")
        self.assertNotIn("evidence_provenance", bundle["loader_contract"])
        self.assertEqual(bundle["loader_contract"]["unsupported_reason_codes"], [])
        self.assertEqual(lrv.plan_from_bundle(bundle)["status"], "READY")


class AddendumMechanicsTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        self.root = mlx_affine_fixture(self.tmp / "mlx-q4")
        self.evidence = verification_evidence(self.root, component="loader", evidence_byte="7")
        self.dir = self.tmp / "attest"
        self.dir.mkdir()

    def _addendum(self, name="X_PROVENANCE_ADDENDUM_T.json", *, sha="7" * 64, wiring=False):
        (self.dir / name).write_text(json.dumps({
            "schema": mc.PROVENANCE_ADDENDUM_SCHEMA,
            "clarifies": [{"evidence_sha256_in_record": sha, "sha256": "a" * 64}],
            "findings": {"engine_wiring_in_main": wiring},
        }))

    def _status(self):
        with patch.object(mc, "ATTESTATIONS_DIR", self.dir):
            return mc.compile_model_capabilities(
                self.root, loader_evidence=self.evidence
            )["loader_contract"]["status"]

    def test_addendum_with_wiring_false_downgrades_exact_evidence(self):
        self._addendum()
        self.assertEqual(self._status(), "VERIFIED_FORMAT_ONLY")

    def test_addendum_with_wiring_true_does_not_downgrade(self):
        self._addendum(wiring=True)
        self.assertEqual(self._status(), "VERIFIED")

    def test_addendum_for_a_different_evidence_hash_does_not_downgrade(self):
        self._addendum(sha="8" * 64)
        self.assertEqual(self._status(), "VERIFIED")

    def test_unreadable_addendum_fails_closed(self):
        (self.dir / "X_PROVENANCE_ADDENDUM_BAD.json").write_text("{not json")
        with patch.object(mc, "ATTESTATIONS_DIR", self.dir):
            with self.assertRaises(mc.ModelCapabilityError):
                mc.compile_model_capabilities(self.root, loader_evidence=self.evidence)

    def test_malformed_addendum_findings_fail_closed(self):
        (self.dir / "X_PROVENANCE_ADDENDUM_BAD.json").write_text(json.dumps({
            "schema": mc.PROVENANCE_ADDENDUM_SCHEMA, "findings": "oops",
        }))
        with patch.object(mc, "ATTESTATIONS_DIR", self.dir):
            with self.assertRaises(mc.ModelCapabilityError):
                mc.compile_model_capabilities(self.root, loader_evidence=self.evidence)

    def test_no_loader_evidence_never_reads_the_attestations_dir(self):
        (self.dir / "X_PROVENANCE_ADDENDUM_BAD.json").write_text("{not json")
        with patch.object(mc, "ATTESTATIONS_DIR", self.dir):
            bundle = mc.compile_model_capabilities(self.root)
        self.assertEqual(bundle["loader_contract"]["status"], "IMPLEMENTED_UNVERIFIED")

    def test_missing_attestations_dir_leaves_evidence_unchanged(self):
        # Documented limitation: without the attestations/ directory (e.g. an npm install, whose
        # package.json "files" does not ship it) the gate cannot see the addendum.
        with patch.object(mc, "ATTESTATIONS_DIR", self.dir / "absent"):
            status = mc.compile_model_capabilities(
                self.root, loader_evidence=self.evidence
            )["loader_contract"]["status"]
        self.assertEqual(status, "VERIFIED")


if __name__ == "__main__":
    unittest.main()
