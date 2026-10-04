#!/usr/bin/env python3
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import loader_runtime_v1 as lr
import model_capability as mc


class LoaderRuntimeTests(unittest.TestCase):
    def _bundle(self, root: Path, *, status="VERIFIED", source_format="GGUF"):
        if source_format == "GGUF":
            primary = root / "model.gguf"
            primary.write_bytes(b"GGUF-test-fixture")
            shards = []
        elif source_format == "SAFETENSORS_SINGLE":
            primary = root / "model.safetensors"
            primary.write_bytes(b"safe-single")
            shards = []
        elif source_format == "SAFETENSORS_SHARDED":
            primary = root / "model.safetensors.index.json"
            primary.write_text('{"weight_map":{"w":"model-00001-of-00001.safetensors"}}')
            shard = root / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"safe-shard")
            shards = [shard]
        else:
            primary = root / "legacy.bin"
            primary.write_bytes(b"legacy")
            shards = []

        paths = [primary, *shards]
        records = [
            {
                "name": p.name,
                "path": str(p.resolve()),
                "size_bytes": p.stat().st_size,
                "sha256": mc.sha256_file(p),
            }
            for p in paths
        ]
        checkpoint = mc.sha256_json(
            [{"name": r["name"], "sha256": r["sha256"], "size_bytes": r["size_bytes"]} for r in records]
        )
        source = {
            "schema": "beglin-model-source-v1",
            "model_id": "fixture",
            "model_revision": "",
            "source_format": source_format,
            "root_path": str(root.resolve()),
            "primary_path": str(primary.resolve()),
            "config_path": None,
            "tokenizer_paths": [],
            "shard_paths": [str(p.resolve()) for p in shards],
            "file_hashes": records,
            "checkpoint_identity_sha256": checkpoint,
            "tensor_count": 0,
            "metadata": {},
            "config": {},
            "tensor_inventory": [],
            "immutable": True,
        }
        source["source_manifest_sha256"] = mc.stable_identity_sha256(source)

        loader = {
            "schema": "beglin-loader-contract-v1",
            "architecture_id": "qwen2",
            "source_format": source_format,
            "status": status,
            "mmap_strategy": "MMAP" if source_format == "GGUF" else "SHARD_AWARE",
            "transcode_strategy": "BEGLIN_QNG64_WHEN_ELIGIBLE",
            "cache_strategy": "CONTENT_IDENTITY",
            "encountered_formats": ["Q4_K"],
            "unsupported_formats": [],
            "source_quantization": None,
            "memory_preflight": {
                "status": "REQUIRES_RUNTIME_PROBE",
                "source_total_bytes": sum(r["size_bytes"] for r in records),
                "weight_storage_bytes": sum(r["size_bytes"] for r in records),
                "storage_lower_bound_bytes": sum(r["size_bytes"] for r in records),
                "peak_resident_bytes_estimate": None,
                "workspace_bytes_estimate": None,
                "mmap_candidate": source_format == "GGUF",
                "requires_runtime_probe": True,
                "estimate_kind": "STORAGE_LOWER_BOUND_ONLY",
            },
            "unsupported_reason_codes": [],
            "verification_evidence": (
                {"schema": "beglin-verification-evidence-v1", "status": "VERIFIED"}
                if status == "VERIFIED" else None
            ),
            "silent_dense_fallback_allowed": False,
        }
        loader["loader_contract_sha256"] = mc.stable_identity_sha256(loader)

        bundle = {
            "schema": "beglin-model-capability-bundle-v1",
            "model_id": "fixture",
            "checkpoint_identity_sha256": checkpoint,
            "source_manifest": source,
            "architecture_descriptor": {
                "schema": "beglin-architecture-descriptor-v1",
                "architecture_id": "qwen2",
            },
            "loader_contract": loader,
            "automatic_live_promotion": False,
            "production_write_allowed": False,
        }
        bundle["bundle_sha256"] = mc.stable_identity_sha256(bundle)
        return bundle, primary, shards

    def test_verified_gguf_plan_and_source_verification(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, _, _ = self._bundle(Path(td), status="VERIFIED", source_format="GGUF")
            plan = lr.compile_runtime_plan(bundle)
            self.assertEqual(plan["status"], "READY")
            self.assertTrue(plan["execution_allowed"])
            self.assertEqual(plan["execution_mode"], "GGUF_DIRECT")
            self.assertFalse(plan["silent_dense_fallback_allowed"])
            result = lr.verify_source_files(plan)
            self.assertEqual(result["status"], "SOURCE_FILES_VERIFIED")
            self.assertEqual(result["file_count"], 1)
            lr.require_executable_plan(plan)

    def test_unverified_loader_cannot_execute(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, _, _ = self._bundle(
                Path(td), status="IMPLEMENTED_UNVERIFIED", source_format="GGUF"
            )
            plan = lr.compile_runtime_plan(bundle)
            self.assertEqual(plan["status"], "VALIDATION_REQUIRED")
            self.assertFalse(plan["execution_allowed"])
            with self.assertRaises(lr.LoaderRuntimeError):
                lr.require_executable_plan(plan)

    def test_sharded_safetensors_binds_index_and_shards(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, _, shards = self._bundle(
                Path(td), status="VERIFIED", source_format="SAFETENSORS_SHARDED"
            )
            plan = lr.compile_runtime_plan(bundle)
            self.assertEqual(plan["execution_mode"], "SAFETENSORS_SHARDED")
            self.assertEqual(plan["shard_paths"], [str(p.resolve()) for p in shards])
            result = lr.verify_source_files(plan)
            self.assertEqual(result["file_count"], 2)

    def test_source_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, primary, _ = self._bundle(Path(td), status="VERIFIED", source_format="GGUF")
            plan = lr.compile_runtime_plan(bundle)
            primary.write_bytes(b"changed")
            with self.assertRaisesRegex(lr.LoaderRuntimeError, "drift"):
                lr.verify_source_files(plan)

    def test_symlink_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle, primary, _ = self._bundle(root, status="VERIFIED", source_format="GGUF")
            plan = lr.compile_runtime_plan(bundle)
            target = root / "target.gguf"
            target.write_bytes(primary.read_bytes())
            primary.unlink()
            primary.symlink_to(target)
            with self.assertRaisesRegex(lr.LoaderRuntimeError, "symlink"):
                lr.verify_source_files(plan)

    def test_plan_hash_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, _, _ = self._bundle(Path(td), status="VERIFIED", source_format="GGUF")
            plan = lr.compile_runtime_plan(bundle)
            tampered = copy.deepcopy(plan)
            tampered["execution_mode"] = "SAFETENSORS_SINGLE"
            with self.assertRaisesRegex(lr.LoaderRuntimeError, "hash mismatch"):
                lr.verify_runtime_plan(tampered)

    def test_unsupported_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            bundle, _, _ = self._bundle(Path(td), status="UNSUPPORTED", source_format="GGUF")
            with self.assertRaisesRegex(lr.LoaderRuntimeError, "unsupported"):
                lr.compile_runtime_plan(bundle)


if __name__ == "__main__":
    unittest.main()
