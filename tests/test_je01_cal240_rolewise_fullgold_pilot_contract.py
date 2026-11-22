#!/usr/bin/env python3
"""Synthetic static safety/ABI contract for JE01 frozen seven-expert gold pilot.

Does not load pretrained weights or run model inference. Independent
cross-run NLL/heldout claims MUST NOT be derived from this test.
"""
from __future__ import annotations
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools/je01_eoe_cal240_rolewise_gold16_paired_runner.py"
TEXT = SRC.read_text()
TREE = ast.parse(TEXT, str(SRC))


class TestJE01RealCal240RolewiseGoldPilotContract(unittest.TestCase):
    def test_original_27layer_source_and_exact_binary_sha_pins(self):
        required = [
            'CANDIDATE_SHA="db028b252d489861349259ec7cd08c87086e617d9f331b1de4a5f660bed79f8f"',
            'CANDIDATE_FILE_SHA="ada43c5e233aa4154d37de5988aa22662ed16496363e84183a784c4c588a63ac"',
            'AF_SHA="9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"',
            'je01_cal240_rolewise_native_stage_20261122_v1',
            'JE01_CAL240_ROLEWISE_STAGE_RECEIPT.json',
            'source_real_CAL240_rolewise_BF16_source_sha256',
        ]
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, TEXT)

    def test_off_first_true_original_output_sha_parity_and_new_marker(self):
        for marker in [
            'choices=["off","rolewise"]',
            'if mode=="rolewise":',
            '"PASS_ORIGINAL27_CAL240_ROLEWISE_BINARY_OFF_GOLD_PARITY"',
            'sha(target)!=sha(BASE/"gold_original_nll.bin")',
            'JE01_NO_OVERLAY_ORIGINAL_GOLD_OUTPUT_SHA_NOT_EXACT',
            'JE01_CAL240_FROZEN_ROLEWISE_SEVEN_EXPERTS_REGISTERED',
        ]:
            with self.subTest(marker=marker):
                self.assertIn(marker, TEXT)

    def test_required_full_gold_and_memory_guards(self):
        for marker in [
            'GOLD_MAGIC=b"JE01_ORIGINAL_AF_FULL_GOLD_NLL_V1',
            'META_MAGIC=b"JE01_ORIGINAL_FULL_GOLD_TARGETS_META_V1',
            '656', 'gold_meta.bin',
            'QWEN_MOE_GPU_CB_REQS',
            'JE01_REAL_Q35_BUNDLE',
            'JE01_REAL_Q35_TSV',
            'if free<35:stop=',
            'if memfree()<55:',
            'JE01_REAL_ORIGINAL27_TIMEOUT420',
            'JE01_MEMORY_FREE_BELOW35',
        ]:
            with self.subTest(marker=marker):
                self.assertIn(marker, TEXT)

    def test_scope_and_repository_native_original_untouched(self):
        for marker in [
            '"real_OOD_and_same_source_128":"NOT_MEASURED"',
            '"meets_preregistered_adoption_gate":"NOT_EVALUATED"',
            '"true_source_heldout_ood_labels_accessed":False',
            '"production_checkpoint_modified":False',
            '"scientific_scope":"CAL240_FROZEN_SELECTION_ALREADY_OBSERVED_TRAIN16_IS_EXPLORATORY_PARITY_ONLY_NO_HOLDOUT"',
        ]:
            with self.subTest(marker=marker):
                self.assertIn(marker, TEXT)
        self.assertNotIn('native_4_experts_rebound_only', TEXT)
        self.assertNotIn('choices=["off","mix4"]', TEXT)

    def test_parser_and_core_functions_exist(self):
        functions = {node.name for node in TREE.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"sha", "canon", "signed", "original_656", "preflight", "run"}.issubset(functions))
        self.assertEqual(sum(isinstance(node, ast.FunctionDef) and node.name=="run"
                             for node in TREE.body), 1)


if __name__=="__main__":
    unittest.main()
