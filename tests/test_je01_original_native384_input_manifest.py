#!/usr/bin/env python3
"""JE01 original-tokenizer raw 384 native inputs fail-closed offline tests."""
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import je01_original_native384_input_manifest as je
import je01_data_freeze as df
from je01_native_token_feasibility import prompt_for


class Original384NativeInputTests(unittest.TestCase):
    def test_real_frozen_512_source_ids_are_truly_locked(self):
        path=ROOT/"docs"/"research"/"JE01_FROZEN_WIKITEXT_HELLASWAG_512_20261117.json"
        original=json.loads(path.read_text())
        je.verify_manifest(original)
        self.assertEqual(original["manifest_sha256"],je.SOURCE_SHA)
        self.assertEqual([original["cohorts"][k]["count"] for k in
            ("calibration","same_source_holdout","ood_holdout")],[256,128,128])
        self.assertEqual(sum(v["count"] for v in original["cohorts"].values()),512)
        self.assertFalse(original["gold_labels_used_in_sampling"])
        self.assertFalse(original["native_model_forward_observed"])
        self.assertEqual(je.BOS,100000)
        self.assertEqual(je.CAP,384)
        self.assertNotEqual(je.TOKENIZER_SHA,
            "c6276ee9e3cfa46033f54dc46db3f6dba0c6a1f6a2e547543124f33b44873db4")

    def test_packs_genuine_raw_int32_le_bos(self):
        b=je.pack_native_input([549,4345,280])
        self.assertEqual(b,struct.pack("<4i",100000,549,4345,280))
        self.assertEqual(struct.unpack("<4i",b),(100000,549,4345,280))
        self.assertEqual(len(b),16)

    def test_384_context_reserves_BOS_and_one_gold_position(self):
        packed=je.pack_native_input([7]*382)
        self.assertEqual(len(packed),4*383)
        with self.assertRaisesRegex(ValueError,"EXCEEDS_384"):
            je.pack_native_input([7]*383)

    def test_invalid_token_types_and_ranges_fail(self):
        for tokens in ([],None,"12",[1.0],[True],[-1],[102400],["3"]):
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    je.pack_native_input(tokens)

    def test_cannot_alter_original_manifest_sha_or_cohort_counts(self):
        path=ROOT/"docs"/"research"/"JE01_FROZEN_WIKITEXT_HELLASWAG_512_20261117.json"
        original=json.loads(path.read_text())
        original["cohorts"]["calibration"]["count"]=255
        with self.assertRaisesRegex(ValueError,"CANONICAL_SHA_MISMATCH"):
            je.verify_manifest(original)

    def test_ood_label_changes_do_not_affect_prompt(self):
        row={"ctx":"  A person  walks  to  the   park ",
             "endings":["one","two","three","four"],"label":0}
        raw=prompt_for(df,"ood_holdout",row)
        row["label"]=2
        self.assertEqual(raw,prompt_for(df,"ood_holdout",row))
        self.assertEqual(raw,"A person walks to the park")

    def test_generator_explicitly_withholds_native_NLL(self):
        source=(ROOT/"tools"/"je01_original_native384_input_manifest.py").read_text()
        self.assertIn('"native_gold_nll":"NOT_IMPLEMENTED"',source)
        self.assertIn('"real_model_forward":"NOT_RUN"',source)
        self.assertIn('if output_root.exists() or output_root.is_symlink()',source)
        self.assertIn('data.freeze(pages)',source)
        self.assertIn('data.digest(raw_tok_ids)!=SOURCE_TOKENS[cohort]',source)
        self.assertNotIn("QWEN_MOE_PROMOTION_FILE",source)


if __name__=="__main__":
    unittest.main(verbosity=2)
