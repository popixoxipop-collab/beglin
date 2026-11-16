#!/usr/bin/env python3
"""Synthetic JE01 full-checkpoint attestation tests; NEVER download real model."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import je01_checkpoint_attest as je


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class AttestationTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = pathlib.Path(self.t.name)
        self.allowed = self.root / "hub" / "models--deepseek"
        self.snap = self.allowed / "snapshots" / "local"
        self.snap.mkdir(parents=True)
        self.bin = self.root / "original_engine"
        self.bin.write_bytes(b"mock-binary-no-native-execution")
        write(self.snap / "config.json", json.dumps({
            "model_type": "deepseek_v2", "n_routed_experts": 64,
            "num_experts_per_tok": 6,
            "norm_topk_prob": False, "first_k_dense_replace": 1}))
        write(self.snap / "tokenizer.json", '{"version":"mock-only"}\n')
        write(self.snap / "model.safetensors.index.json", json.dumps({
            "metadata": {"total_size": 6},
            "weight_map": {
                "model.layers.1.mlp.experts.0": "shard01.safetensors",
                "model.layers.1.mlp.experts.1": "shard02.safetensors"
            }}))
        (self.snap / "shard01.safetensors").write_bytes(b"one")
        (self.snap / "shard02.safetensors").write_bytes(b"two")
        self.params = dict(model_snapshot=self.snap, allowed_model_root=self.allowed,
                           runtime_binary=self.bin,
                           expected_runtime_sha256=je.sha256_file(self.bin),
                           engine_git_commit="a"*40)

    def test_identity_and_sha_completeness(self):
        out = je.inspect(**self.params)
        self.assertEqual(out["status"], "DISCOVERY_ONLY")
        self.assertEqual(out["model"]["num_experts_per_tok"], 6)
        self.assertEqual(len(out["model"]["shards"]), 2)
        self.assertFalse(out["runtime_parity"] == "PASS")
        self.assertEqual(len(out["checkpoint_identity_sha256"]), 64)

    def test_same_bytes_same_identity(self):
        self.assertEqual(je.inspect(**self.params), je.inspect(**self.params))
        prior = je.inspect(**self.params)
        pinned = je.inspect(**self.params, expected_checkpoint_identity=prior["checkpoint_identity_sha256"])
        self.assertEqual(pinned["status"], "HASH_PINNED_ONLY")
        self.assertEqual(pinned["checkpoint_identity_sha256"], prior["checkpoint_identity_sha256"])

    def test_changed_shard_is_new_identity(self):
        prior = je.inspect(**self.params)
        (self.snap / "shard01.safetensors").write_bytes(b"not-one")
        with self.assertRaisesRegex(ValueError, "IDENTITY_DRIFT"):
            je.inspect(**self.params, expected_checkpoint_identity=prior["checkpoint_identity_sha256"])

    def test_binary_mismatch_denied(self):
        self.bin.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "BINARY_SHA_DRIFT"):
            je.inspect(**self.params)

    def test_missing_shard_denied(self):
        (self.snap / "shard01.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "MISSING_OR_NONFILE"):
            je.inspect(**self.params)

    def test_non_moe_denied(self):
        write(self.snap / "config.json", '{"model_type":"qwen2"}')
        with self.assertRaisesRegex(ValueError, "NOT_DEEPSEEK_V2"):
            je.inspect(**self.params)

    def test_wrong_expert_count_denied(self):
        write(self.snap / "config.json", '{"model_type":"deepseek_v2","n_routed_experts":16,"num_experts_per_tok":2}')
        with self.assertRaisesRegex(ValueError, "UNEXPECTED_ROUTED_EXPERT"):
            je.inspect(**self.params)

    def test_wrong_topk_is_not_deepseek_v2_lite(self):
        write(self.snap / "config.json", json.dumps({
            "model_type": "deepseek_v2", "n_routed_experts": 64,
            "num_experts_per_tok": 8, "norm_topk_prob": False,
            "first_k_dense_replace": 1}))
        with self.assertRaisesRegex(ValueError, "UNEXPECTED_ROUTED"):
            je.inspect(**self.params)

    def test_topk_renorm_mode_must_be_exact(self):
        write(self.snap / "config.json", json.dumps({
            "model_type": "deepseek_v2", "n_routed_experts": 64,
            "num_experts_per_tok": 6, "norm_topk_prob": True,
            "first_k_dense_replace": 1}))
        with self.assertRaisesRegex(ValueError, "UNEXPECTED_ROUTED"):
            je.inspect(**self.params)

    def test_traversal_shard_denied(self):
        write(self.snap / "model.safetensors.index.json", json.dumps(
            {"weight_map": {"a": "../secret.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "SHARD_PATH_INVALID"):
            je.inspect(**self.params)

    def test_symlink_outside_allowed_root_denied(self):
        outside = self.root / "outside.safetensors"
        outside.write_bytes(b"fake-outside")
        (self.snap / "shard01.safetensors").unlink()
        (self.snap / "shard01.safetensors").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "OUTSIDE_ALLOWED_ROOT"):
            je.inspect(**self.params)

    def test_empty_index_denied(self):
        write(self.snap / "model.safetensors.index.json", '{"weight_map":{}}')
        with self.assertRaisesRegex(ValueError, "WEIGHT_MAP_EMPTY"):
            je.inspect(**self.params)

    def test_git_commit_must_be_pinned(self):
        self.params["engine_git_commit"] = "main"
        with self.assertRaisesRegex(ValueError, "GIT_COMMIT_UNPINNED"):
            je.inspect(**self.params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
