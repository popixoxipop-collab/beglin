#!/usr/bin/env python3
"""JE01 shadow 384-token one-slot build preparation safety tests."""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import je01_isolated_context384_stage as je


class Context384Safety(unittest.TestCase):
    def test_memory_only_one_slot(self):
        self.assertEqual(je.CONTEXT, 384)
        self.assertEqual(je.SLOTS, 1)
        self.assertEqual(je.kv_bytes(), 251_658_240)  # 240 MiB
        self.assertEqual(je.kv_bytes(slots=64), 16_106_127_360)  # 15 GiB
        self.assertLess(je.kv_bytes(), je.MAX_ALLOWED_KV_BYTES)
        self.assertGreater(je.kv_bytes(slots=64), je.MAX_ALLOWED_KV_BYTES)

    def test_full_real_target_331_fits_context(self):
        p = ROOT / "docs" / "research" / "JE01_REAL_512_FULL_TARGET_CONTEXT_20261117.json"
        import json
        report = json.loads(p.read_text())
        self.assertEqual(report["input_row_manifest_sha256"], je.EXPECTED_DATASET_MANIFEST_SHA)
        self.assertEqual(report["receipt_sha256"], "88a756cee1f6d6e4f20413ced90fcd62c6d7ca3cb013435c756e08e4f10b4cbc")
        self.assertEqual([report["cohorts"][x]["count"] for x in (
            "calibration","same_source_holdout","ood_holdout")], [256,128,128])
        self.assertEqual(max(x["max_bos_plus_complete_gold_or_all_4_endings"]
                             for x in report["cohorts"].values()), 331)
        self.assertEqual(sum(x["fits_context"]["384"] for x in report["cohorts"].values()), 512)
        self.assertEqual(sum(x["fits_context"]["256"] for x in report["cohorts"].values()), 505)
        self.assertEqual(report["native_full_model_forward"], "NOT_RUN")

    def test_actual_source_anchors_change_only_expected_fields(self):
        q=(ROOT/"qwen_infer.c").read_text()
        m=(ROOT/"mlx_moe.cpp").read_text()
        c=(ROOT/"CMakeLists.txt").read_text()
        result=je.patch_sources(q,m,c)
        self.assertEqual(result["qwen_infer.c"].count("#define MOE_MAXPOS 384"), 1)
        self.assertEqual(result["qwen_infer.c"].count("#define MOE_CBATCH_MAXPOS 384"), 1)
        self.assertIn("static int g_moe_cb_slots_cap = 1;",result["qwen_infer.c"])
        self.assertIn("if (B != 1) { fprintf(stderr, \"FATAL: QWEN_MOE_CB_SLOTS",
                      result["qwen_infer.c"])
        self.assertIn("if (B != 1) return 0;", result["mlx_moe.cpp"])
        self.assertIn("if (A != 1 || !slot || slot[0] != 0 || !spos",
                      result["mlx_moe.cpp"])
        self.assertIn("#define MLA_L0_MAXPOS 384", result["mlx_moe.cpp"])
        self.assertIn("mlx_gpu_je01_materialize_fused_mla_kv", result["mlx_moe.cpp"])
        self.assertNotIn("mlx_gpu_je01_materialize_fused_mla_kv",m)
        self.assertIn(je.REQUIRED_BINARY_NAME, result["CMakeLists.txt"])
        self.assertIn("add_executable(je01_native_context384_probe", result["CMakeLists.txt"])
        self.assertTrue((ROOT/"tools"/"je01_native_metal384_probe.cpp").is_file())
        self.assertIn("#define GQA_L0_MAXPOS",result["mlx_moe.cpp"])
        self.assertNotIn("#define GQA_L0_MAXPOS 384",result["mlx_moe.cpp"])
        self.assertNotIn("install(",result["CMakeLists.txt"])
        self.assertEqual((ROOT/"qwen_infer.c").read_text(),q)
        self.assertEqual((ROOT/"mlx_moe.cpp").read_text(),m)
        self.assertEqual((ROOT/"CMakeLists.txt").read_text(),c)

    def test_exact_anchor_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"UNPINNED_SOURCE_ANCHOR"):
            je.exactly_one("no match", "#define MOE_MAXPOS 160",
                           "#define MOE_MAXPOS 384", where="probe")
        with self.assertRaisesRegex(ValueError,"UNPINNED_SOURCE_ANCHOR"):
            je.exactly_one("x x", "x", "y", where="duplicate")

    def test_git_checkout_staged_read_only_with_untracked_file_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            home=pathlib.Path(root)
            src=home/"source"
            src.mkdir()
            src.joinpath("qwen_infer.c").write_text(
                '#define MOE_MAXPOS 160\n#define MOE_CBATCH_MAXPOS 32\n'
                'static int g_moe_cb_slots_cap = MOE_BATCH_MAX;\n' +
                ('if (B < 1 || B > MOE_BATCH_MAX) { fprintf(stderr, "FATAL: QWEN_MOE_CB_SLOTS\n' * 4))
            src.joinpath("mlx_moe.cpp").write_text(
                '#define MLA_L0_MAXPOS 32\n'
                'if (B < 1 || B > 64) return 0;\n'
                'if (A < 1 || A > 64 || l < 0 || l >= MLA_MAXLAYERS) return 0;\n'
                '    g_fused_kv_inited_B = B;\n}\n\n// D-gpu-5-hotpath:')
            src.joinpath("CMakeLists.txt").write_text(
                'add_executable(qwen_infer_gpu\n a)\n'
                'target_link_libraries(qwen_infer_gpu PRIVATE mlx)\n')
            src.joinpath("untracked-secret.txt").write_text("PRIVATE KEY - MUST NOT COPY")
            def call(args):
                return subprocess.run(args,cwd=src,capture_output=True,
                                      check=True,text=True).stdout.strip()
            call(["git","init","-q"])
            call(["git","add","qwen_infer.c","mlx_moe.cpp","CMakeLists.txt"])
            call(["git","-c","user.email=je01@example.invalid",
                  "-c","user.name=JE01 test","commit","-qm","fixture"])
            commit=call(["git","rev-parse","HEAD"])
            target=home/"isolated"
            report=je.stage(checkout=src,scratch=target,expected_commit=commit)
            self.assertEqual(report["stage_status"],"SOURCE_STAGED_NOT_COMPILED")
            self.assertFalse(report["weight_or_production_edits"])
            self.assertFalse(report["production_install_allowed"])
            self.assertEqual(report["short_prefix_logit_parity"],"NOT_TESTED")
            self.assertEqual(report["standalone_launch_env"],{"QWEN_MOE_CB_SLOTS":"1"})
            self.assertFalse((target/"untracked-secret.txt").exists())
            self.assertFalse((target/".git").exists())
            self.assertIn("#define MOE_CBATCH_MAXPOS 384",
                          (target/"qwen_infer.c").read_text())
            self.assertIn("#define MOE_CBATCH_MAXPOS 32",
                          (src/"qwen_infer.c").read_text())
            with self.assertRaisesRegex(ValueError,"DESTINATION_MUST_BE_NEW"):
                je.stage(checkout=src,scratch=target,expected_commit=commit)
            with self.assertRaisesRegex(ValueError,"SOURCE_DESTINATION_CONFLICT"):
                je.stage(checkout=src,scratch=src/"child",expected_commit=commit)
            with self.assertRaisesRegex(ValueError,"CHECKOUT_HEAD_DRIFT"):
                je.stage(checkout=src,scratch=home/"other",expected_commit="a"*40)
            (src/"qwen_infer.c").write_text("dirty")
            with self.assertRaisesRegex(ValueError,"CHECKOUT_NOT_CLEAN|UNPINNED_SOURCE_ANCHOR"):
                je.stage(checkout=src,scratch=home/"dirty",expected_commit=commit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
