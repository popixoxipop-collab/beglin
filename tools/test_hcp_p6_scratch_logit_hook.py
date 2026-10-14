#!/usr/bin/env python3
"""P6 hook isolation regression; no native model invoked."""
import pathlib
import tempfile
import unittest

from hcp_p5_scratch_capacity128 import patch as p5patch
from hcp_p6_scratch_logit_hook import apply, patch_text, HOOK, ANCHOR

SOURCE=pathlib.Path(__file__).resolve().parents[1]/"qwen_infer.c"

class P6CaptureTest(unittest.TestCase):
    def test_only_verified_p5_cb128_source(self):
        c=p5patch(SOURCE.read_text())
        amended=patch_text(c)
        self.assertEqual(amended.replace(HOOK,""),c)
        self.assertEqual(amended.count("BEGLIN_HCP_P6_LOGITS_OUTPUT"),1)
        self.assertIn("BEGLIN_HCP_P6_CAPTURE_ROOT",amended)
        self.assertIn("REBIND_SET_APPLIED",amended)
        self.assertIn("#define MOE_CBATCH_MAXPOS 128",amended)
    def test_double_patch_and_unknown_source_denied(self):
        c=p5patch(SOURCE.read_text())
        with self.assertRaisesRegex(ValueError,"SHA"):
            patch_text(c+" ")
        with self.assertRaisesRegex(ValueError,"SHA"):
            patch_text(patch_text(c))
        with self.assertRaisesRegex(ValueError,"SHA"):
            patch_text(SOURCE.read_text())
    def test_owned_scratch_only_and_no_original_mutation(self):
        original=SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)
            path=root/"qwen_infer.c"
            path.write_text(p5patch(original.decode()))
            receipt=apply(root)
            self.assertEqual(receipt["capture_scope"],"single_last_prefill_position_float32_per_request")
            self.assertFalse(receipt["production_touched"])
            self.assertEqual(path.read_text().count("P6 logit write incomplete"),1)
            self.assertEqual(SOURCE.read_bytes(),original)
    def test_missing_scratch_denied(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"regular scratch"):
                apply(pathlib.Path(temp))

if __name__=="__main__":unittest.main()
