#!/usr/bin/env python3
from pathlib import Path
import re
import unittest

SRC=Path(__file__).resolve().parents[1]/"qwen_infer.c"

def gpu_gate():
    text=SRC.read_text()
    start=text.index("static int run_moe_gpu_cbatch_online_gate")
    brace=text.index("{",start)
    depth=0
    for i in range(brace,len(text)):
        if text[i]=="{":
            depth+=1
        elif text[i]=="}":
            depth-=1
            if depth==0:
                return text[start:i+1]
    raise AssertionError("unterminated GPU online gate")

class PersistentSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fn=gpu_gate()

    def test_opt_in_gate_only(self):
        self.assertIn('getenv("QWEN_MOE_GPU_PERSIST_STDIN")',self.fn)
        self.assertIn("if (persistent)",self.fn)

    def test_protocol_is_line_oriented_and_has_quit(self):
        self.assertIn("fgets(line, sizeof line, stdin)",self.fn)
        self.assertIn('"QUIT"',self.fn)
        self.assertIn("PERSIST_RESPONSE",self.fn)
        self.assertIn("fflush(stdout)",self.fn)

    def test_subsequent_batches_skip_warmup_pass_zero(self):
        self.assertIn(
            "int pass_begin = (persistent && persistent_batch_index > 0) ? 1 : 0;",
            self.fn,
        )
        self.assertIn("persistent_batch_index++;",self.fn)

    def test_one_shot_cleanup_still_exists(self):
        self.assertRegex(
            self.fn,
            re.compile(r"persistent_done:\s*free\(x_embed\); free\(gpu_logits\);\s*return 1;")
        )

    def test_existing_online_result_marker_preserved(self):
        self.assertIn(
            "RESULT: MoE GPU V5h online cbatch gate complete",
            self.fn,
        )

if __name__=="__main__":
    unittest.main()
