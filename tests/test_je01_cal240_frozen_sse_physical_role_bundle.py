#!/usr/bin/env python3
"""Offline, synthetic JE01 frozen 240-TRAIN physical role-bundle contracts."""
from __future__ import annotations
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "je01_eoe_cal240_frozen_sse_exactbyte_role_bundle_20261122.py"
TEXT = SRC.read_text(encoding="utf-8")
TREE = ast.parse(TEXT, str(SRC))


def policy_only():
    names = {"BUDGET", "COST", "ROLES", "FROZEN", "Q4_SET", "Q5_SET"}
    nodes = []
    for n in TREE.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in n.targets
        ):
            nodes.append(n)
        if isinstance(n, ast.FunctionDef) and n.name == "tier_for":
            nodes.append(n)
    env = {}
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, str(SRC), "exec"), env, env)
    return env


class TestJE01Cal240RolewiseBundle(unittest.TestCase):
    def test_frozen_24role_policy_and_exact_physical_bytes(self):
        x = policy_only()
        d = {(l, e, role): x["tier_for"](l, e, role)
             for l, e in x["FROZEN"] for role in x["ROLES"]}
        self.assertEqual(len(d), 24)
        self.assertEqual(len(set(d)), 24)
        self.assertEqual(sum(v == "AF4" for v in d.values()), 12)
        self.assertEqual(sum(v == "Q4" for v in d.values()), 6)
        self.assertEqual(sum(v == "Q5" for v in d.values()), 6)
        self.assertEqual(sum(x["COST"][v] for v in d.values()), 43_253_760)
        self.assertEqual(x["BUDGET"], 43_253_760)
        self.assertEqual(x["COST"]["AF4"] - x["COST"]["Q4"],
                         x["COST"]["Q5"] - x["COST"]["AF4"])

    def test_changed_roles_match_actual_cal240_frozen_sse_dp(self):
        x = policy_only()
        d = {(l, e, role): x["tier_for"](l, e, role)
             for l, e in x["FROZEN"] for role in x["ROLES"]}
        expected_q4 = {(l, e, r) for l, e in ((12, 13), (13, 22), (13, 28))
                       for r in ("gate_proj", "up_proj")}
        expected_q5 = {(20, 17, "down_proj"), (21, 61, "down_proj"),
                       (24, 60, "down_proj")} | {
                           (24, 26, r) for r in ("gate_proj", "up_proj", "down_proj")}
        self.assertEqual({k for k, v in d.items() if v == "Q4"}, expected_q4)
        self.assertEqual({k for k, v in d.items() if v == "Q5"}, expected_q5)

    def test_pinned_actual_source_and_no_gold_access_guards(self):
        required = (
            "35f9b96767d3e2b40ab6dfc9b79d99bf1ccd392124804cf41e63b38af6c77dfb",
            "b911aef00007b1f9b44214996353f93c7218fe51c82daf91966e450ebba460b6",
            "9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4",
            "863fb9c1d5a17fc810564610739bd1c53f727addee99a7966f65404b8725a414",
            "source_heldout_OOD_labels_used", "training_first16_prior_pilot_excluded_from_precision_selection",
            "sum(COST[x] for x in selected.values()) == BUDGET",
            "JE01_CAL240_ROLE_POLICY_NOT_EXACTLY_FROZEN",
            "JE01_NATIVE_ORIGINAL_AFFINE4_PACKED_SHA_CHANGED",
            "JE01_ROLE_ORIGINAL_BF16_SHA_CHANGED",
            "native.bf16_selected", "native.read_bf16_headers",
            "lib.gguf_quantize_q4g64_error_feedback", "lib.gguf_quantize_qNg64",
            "JE01_FROZEN_PHYSICAL_BUNDLE_NOT_EXACT_43253760",
            '"native_full_model_overlay": "NOT_YET_BOUND"',
            '"true_full_model_gold_NLL": "NOT_MEASURED"',
            '"heldout_OOD_gold_accessed": False', '"production_modified": False')
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, TEXT)

    def test_q5_native_ctypes_abi_is_six_arguments(self):
        """Regression for real 14/24 role builder failure: Q5 ABI was typed as 7."""
        functions = [node for node in TREE.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "load_original_sources"]
        self.assertEqual(len(functions), 1)
        assignments = [
            node for node in ast.walk(functions[0])
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute)
                    and target.attr == "argtypes"
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "gguf_quantize_qNg64"
                    for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertIsInstance(assignments[0].value, ast.List)
        self.assertEqual(len(assignments[0].value.elts), 6)
        self.assertIn("lib.gguf_quantize_qNg64.restype = None", TEXT)

    def test_research_file_is_create_only_never_prod_mutation(self):
        self.assertIn('not output.exists() and not output.is_symlink()', TEXT)
        self.assertIn('binary.open("xb")', TEXT)
        self.assertIn('output.parent == WORK', TEXT)
        self.assertIn('spans[-1][1] == BUDGET', TEXT)
        self.assertIn('all(spans[i][1] == spans[i + 1][0]', TEXT)


if __name__ == "__main__":
    unittest.main()
