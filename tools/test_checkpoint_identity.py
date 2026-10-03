#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import checkpoint_identity as ci
import gpu_shadow_materialize as gm


class CheckpointIdentityTests(unittest.TestCase):
    def make_index(self, root: Path):
        shard_a = root / "model-00001-of-00002.safetensors"
        shard_b = root / "model-00002-of-00002.safetensors"
        shard_a.write_bytes(b"a" * 8192)
        shard_b.write_bytes(b"b" * 12288)
        index = root / "model.safetensors.index.json"
        index.write_text(json.dumps({
            "metadata": {"total_size": shard_a.stat().st_size + shard_b.stat().st_size},
            "weight_map": {
                "layer.a": shard_a.name,
                "layer.b": shard_b.name,
                "layer.c": shard_a.name,
            },
        }, sort_keys=True))
        return index, shard_a, shard_b

    def test_two_pass_identity_matches_existing_materializer_v1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index, _, _ = self.make_index(root)
            expected, expected_manifest = gm.checkpoint_identity_from_safetensors(
                str(index)
            )
            got = ci.verify_checkpoint_identity(
                index,
                allowed_root=root,
                passes=2,
                chunk_bytes=4096,
            )
            self.assertEqual(got["status"], "VERIFIED")
            self.assertEqual(got["checkpoint_sha256"], expected)
            self.assertEqual(got["identity_manifest"], expected_manifest)
            self.assertEqual(len(got["verification_passes"]), 2)
            self.assertEqual(
                got["verification_passes"][0]["checkpoint_sha256"],
                got["verification_passes"][1]["checkpoint_sha256"],
            )

    def test_single_file_identity_matches_existing_materializer_v1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"single-file-checkpoint" * 512)
            expected, expected_manifest = gm.checkpoint_identity_from_safetensors(
                str(checkpoint)
            )
            got = ci.verify_checkpoint_identity(
                checkpoint,
                allowed_root=root,
                passes=2,
                chunk_bytes=4096,
            )
            self.assertEqual(got["checkpoint_sha256"], expected)
            self.assertEqual(got["identity_manifest"], expected_manifest)

    def test_shard_change_between_passes_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index, _, shard_b = self.make_index(root)

            def mutate(pass_index):
                self.assertEqual(pass_index, 1)
                shard_b.write_bytes(b"changed" * 4096)

            with self.assertRaises(ci.InputChangedError):
                ci.verify_checkpoint_identity(
                    index,
                    allowed_root=root,
                    passes=2,
                    chunk_bytes=4096,
                    between_passes=mutate,
                )

    def test_index_change_between_passes_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index, shard_a, shard_b = self.make_index(root)

            def mutate(pass_index):
                self.assertEqual(pass_index, 1)
                index.write_text(json.dumps({
                    "weight_map": {
                        "layer.a": shard_b.name,
                        "layer.b": shard_a.name,
                    }
                }))

            with self.assertRaises(ci.InputChangedError):
                ci.verify_checkpoint_identity(
                    index,
                    allowed_root=root,
                    passes=2,
                    chunk_bytes=4096,
                    between_passes=mutate,
                )

    def test_parent_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({
                "weight_map": {"layer.a": "../outside.safetensors"},
            }))
            with self.assertRaises(ci.CheckpointIdentityError):
                ci.verify_checkpoint_identity(
                    index,
                    allowed_root=root,
                    passes=2,
                    chunk_bytes=4096,
                )

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = root.parent / (root.name + "-outside.safetensors")
            try:
                outside.write_bytes(b"outside")
                link = root / "model-00001-of-00001.safetensors"
                link.symlink_to(outside)
                index = root / "model.safetensors.index.json"
                index.write_text(json.dumps({
                    "weight_map": {"layer.a": link.name},
                }))
                with self.assertRaises(ci.CheckpointIdentityError):
                    ci.verify_checkpoint_identity(
                        index,
                        allowed_root=root,
                        passes=2,
                        chunk_bytes=4096,
                    )
            finally:
                try:
                    outside.unlink()
                except FileNotFoundError:
                    pass

    def test_missing_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({
                "weight_map": {"layer.a": "missing.safetensors"},
            }))
            with self.assertRaises(ci.CheckpointIdentityError):
                ci.verify_checkpoint_identity(
                    index,
                    allowed_root=root,
                    passes=2,
                    chunk_bytes=4096,
                )

    def test_invalid_weight_map_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({
                "weight_map": {"layer.a": 123},
            }))
            with self.assertRaises(ci.CheckpointIdentityError):
                ci.verify_checkpoint_identity(
                    index,
                    allowed_root=root,
                    passes=2,
                    chunk_bytes=4096,
                )

    def test_output_is_atomic_json_and_never_marks_production_write(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index, _, _ = self.make_index(root)
            output = root / "out" / "checkpoint_identity.json"
            got = ci.verify_checkpoint_identity(
                index,
                output=output,
                allowed_root=root,
                passes=2,
                chunk_bytes=4096,
            )
            saved = json.loads(output.read_text())
            self.assertEqual(saved["checkpoint_sha256"], got["checkpoint_sha256"])
            self.assertFalse(saved["production_write_allowed"])
            self.assertEqual(saved["schema"], ci.VERIFICATION_SCHEMA)

    def test_pass_count_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"x" * 8192)
            with self.assertRaises(ci.CheckpointIdentityError):
                ci.verify_checkpoint_identity(checkpoint, allowed_root=root, passes=1)
            with self.assertRaises(ci.CheckpointIdentityError):
                ci.verify_checkpoint_identity(checkpoint, allowed_root=root, passes=4)


if __name__ == "__main__":
    unittest.main()
