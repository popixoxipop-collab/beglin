"""Preregistered 120 prompt corpus invariants; no inference or downloads."""
import json
import pathlib
import tempfile
import unittest
from capture_prompt_suite import preregistered_prompts, PROMPT_BANK_FILES

BANK=pathlib.Path(__file__).parent/"prompt_bank"

class PromptBank120Test(unittest.TestCase):
    def test_fixed_prompt_splits_and_domain_balance(self):
        rows=preregistered_prompts(BANK)
        self.assertEqual(len(rows),120)
        self.assertEqual(len({x[1].casefold() for x in rows}),120)
        self.assertEqual([sum(x[0]==s for x in rows) for s in ("train","validation","holdout")],[72,24,24])
        domains={x[2] for x in rows}
        self.assertEqual(len(domains),12)
        for domain in domains:
            self.assertEqual([sum(x[0]==s and x[2]==domain for x in rows) for s in ("train","validation","holdout")],[6,2,2])

    def test_duplicate_text_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            dst=pathlib.Path(root)
            for fn in PROMPT_BANK_FILES:
                o=json.loads((BANK/fn).read_text())
                vals=list(o.values())
                vals[0][1]=vals[0][0]
                (dst/fn).write_text(json.dumps(o))
            with self.assertRaisesRegex(ValueError,"duplicat"):
                preregistered_prompts(dst)

    def test_missing_bank_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                preregistered_prompts(pathlib.Path(root))

if __name__=="__main__":unittest.main()
