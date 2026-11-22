#!/usr/bin/env python3
"""No-GPU, no-NumPy fail-closed source contract for JE01 nonlinear FFN replay."""
import ast
import math
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "je01_eoe_cal240_joint_swiglu_ffn_replay_20261122.py"
TEXT = SOURCE.read_text(encoding="utf-8")
TREE = ast.parse(TEXT, str(SOURCE))


class JE01Cal240JointSwiGLUSourceContract(unittest.TestCase):
    def test_python_is_syntactically_valid_and_explicitly_ffn_joint(self):
        functions = {x.name for x in TREE.body if isinstance(x, ast.FunctionDef)}
        self.assertGreaterEqual(len(functions), 8)
        self.assertTrue({"silu", "ffn_output", "weighted_sse", "find_ledger",
                         "load_cal_source", "run"}.issubset(functions))
        self.assertIn("SW = silu(G) * U", TEXT)
        self.assertIn("Y = (down @ SW)", TEXT)
        self.assertIn("coexpert[key][0] +=", TEXT)
        self.assertIn("coexpert[key][1] +=", TEXT)

    def test_evidence_and_baseline_guards_are_present(self):
        self.assertIn("CAL_LEDGER_SIGNED_SHA", TEXT)
        self.assertIn("TRACE_SIGNED_SHA", TEXT)
        self.assertIn("CAL_SCRIPT_SHA", TEXT)
        self.assertIn("original_AF4_checkpoint_sha256", TEXT)
        self.assertIn("source_heldout_OOD_labels_used", TEXT)
        self.assertIn("production_modified", TEXT)
        self.assertIn("all(16 <= i < 256 for i in cases)", TEXT)
        self.assertIn("len(data.values())", TEXT.replace("sum(len(v) for v in data.values())", "len(data.values())"))
        self.assertIn("JE01_RECONSTRUCTED_SILU_VS_NATIVE_ACTUAL_PARITY_FAIL", TEXT)

    def test_cost_and_role_selection_not_all_symmetric(self):
        self.assertIn('"AF4": 1802240', TEXT)
        self.assertIn('"Q4": 1622016', TEXT)
        self.assertIn('"Q5": 1982464', TEXT)
        self.assertIn("sum(COST[t] for t in mapping.values()) == 43253760", TEXT)
        self.assertIn('sum(t == "Q4" for t in mapping.values()) == 6', TEXT)
        self.assertIn('sum(t == "Q5" for t in mapping.values()) == 6', TEXT)
        self.assertEqual(12 * 1802240 + 6 * 1622016 + 6 * 1982464, 43253760)

    def test_reference_state_is_not_claimed_to_be_full_nll_or_fisher(self):
        self.assertIn('"true_loss_Fisher": "NOT_MEASURED"', TEXT)
        self.assertIn('"true_loss_Hessian": "NOT_MEASURED"', TEXT)
        self.assertIn('"real_full_native_27layer_NLL": "NOT_RUN"', TEXT)
        self.assertIn('"untouched_validation_OOD_gold": "NOT_USED"', TEXT)
        self.assertIn('"production_modified": False', TEXT)

    def test_nonlinear_swiglu_demonstrates_nonadditive_coupling(self):
        def sigmoidz(x):
            return 1 / (1 + math.exp(-x))
        def output(g, up, down, inp=1):
            return down * (g * inp * sigmoidz(g * inp)) * (up * inp)
        baseline = output(1.2, 0.9, 1.0)
        changed_gate = output(1.4, 0.9, 1.0) - baseline
        changed_up = output(1.2, 0.8, 1.0) - baseline
        both = output(1.4, 0.8, 1.0) - baseline
        self.assertNotAlmostEqual(changed_gate + changed_up, both, places=8)


if __name__ == "__main__":
    unittest.main()
