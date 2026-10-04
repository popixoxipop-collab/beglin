#!/usr/bin/env python3
from pathlib import Path
import copy
import tempfile
import unittest

import capability_diff as cd
import model_capability as mc
from test_model_capability_p12 import qwen_fixture


class CapabilityDiffTests(unittest.TestCase):
    def test_same_bundle_is_unchanged_and_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root=qwen_fixture(Path(td)/"m")
            bundle=mc.compile_model_capabilities(root,backend="cpu")
            first=cd.build_diff(
                bundle,bundle,left_backend="cpu",right_backend="cpu"
            )
            second=cd.build_diff(
                bundle,bundle,left_backend="cpu",right_backend="cpu"
            )
            self.assertTrue(first["same_checkpoint"])
            self.assertTrue(first["same_architecture"])
            self.assertEqual(first["target_counts"]["changed"],0)
            self.assertEqual(first["target_counts"]["added"],0)
            self.assertEqual(first["target_counts"]["removed"],0)
            self.assertGreater(first["target_counts"]["unchanged"],0)
            self.assertEqual(first["diff_sha256"],second["diff_sha256"])

    def test_cpu_mlx_diff_aligns_same_semantic_targets(self):
        with tempfile.TemporaryDirectory() as td:
            root=qwen_fixture(Path(td)/"m")
            cpu=mc.compile_model_capabilities(root,backend="cpu")
            mlx=mc.compile_model_capabilities(root,backend="mlx_metal")
            diff=cd.build_diff(
                cpu,mlx,left_backend="cpu",right_backend="mlx_metal"
            )
            self.assertTrue(diff["same_checkpoint"])
            self.assertTrue(diff["same_architecture"])
            self.assertEqual(diff["target_counts"]["added"],0)
            self.assertEqual(diff["target_counts"]["removed"],0)
            self.assertGreater(diff["target_counts"]["changed"],0)
            self.assertTrue(diff["target_changes"])
            sample=diff["target_changes"][0]
            self.assertEqual(sample["left_backend"],"cpu")
            self.assertEqual(sample["right_backend"],"mlx_metal")
            self.assertEqual(len(sample["semantic_key"]),3)

    def test_tampered_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=qwen_fixture(Path(td)/"m")
            bundle=mc.compile_model_capabilities(root,backend="cpu")
            bad=copy.deepcopy(bundle)
            bad["model_id"]="tampered"
            with self.assertRaisesRegex(cd.CapabilityDiffError,"hash mismatch"):
                cd.build_diff(
                    bad,bundle,left_backend="cpu",right_backend="cpu"
                )


if __name__=="__main__":
    unittest.main()
