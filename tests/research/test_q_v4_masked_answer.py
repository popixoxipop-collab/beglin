"""Offline contract tests for research-only Qv4 masked assistant-suffix NLL."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v4_masked_answer_native as q

class Qv4MaskedAnswerNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=q.config()

    def windows(self):
        out=[]
        for i,item in enumerate(self.cfg["examples"]):
            for lab in ("A","B"):
                out.append({"idx":len(out),"case":item["id"],"answer":lab,
                    "is_correct":lab==item["correct"],"prefix":25})
        return out

    def fake(self,windows,good_loss=0.3,bad_loss=1.5):
        rows=[];total=0.
        for w in windows:
            loss=good_loss if w["is_correct"] else bad_loss
            rows.append("[qv4-masked] window %d prefix %d nll %.17g tokens 1" %
                        (w["idx"],w["prefix"],loss))
            total+=loss
        n=len(windows)
        rows.append("qv4_masked_nll %.17g tokens %d mean %.17g ppl %.17g" %
                    (total,n,total/n,math.exp(total/n)))
        return "\n".join(rows)

    def test_prelock_is_exploratory_and_frozen(self):
        self.assertEqual(self.cfg["status"],"FROZEN_BEFORE_NATIVE_SCORES")
        self.assertFalse(self.cfg["production_promotion"])
        self.assertEqual(len(self.cfg["examples"]),8)

    def test_labels_balanced_and_ordered(self):
        self.assertEqual(sum(x["correct"]=="A" for x in self.cfg["examples"]),4)
        self.assertEqual(sum(x["correct"]=="B" for x in self.cfg["examples"]),4)
        self.assertEqual(len({x["question"] for x in self.cfg["examples"]}),8)

    def test_hash_locked_native_sources(self):
        for kind,path in q.OLD_SOURCES.items():
            self.assertEqual(q.sha(path.read_bytes()),q.SOURCE_SHA[kind])

    def test_patch_preserves_original_ppl_tail(self):
        source=q.OLD_SOURCES["mixed"].read_text()
        updated=q.guarded_patch(source)
        self.assertEqual(updated.count('"masked_nll"'),1)
        self.assertEqual(updated.split(q.PATCH_ANCHOR,1)[1],
                         source.split(q.PATCH_ANCHOR,1)[1])

    def test_patch_refuses_duplicate(self):
        updated=q.guarded_patch(q.OLD_SOURCES["base"].read_text())
        with self.assertRaises(RuntimeError):
            q.guarded_patch(updated)

    def test_patch_refuses_ambiguous_source(self):
        with self.assertRaises(RuntimeError):
            q.guarded_patch(q.PATCH_ANCHOR+q.PATCH_ANCHOR)

    def test_pairwise_accuracy_with_correct_only_suffix(self):
        rows=self.windows()
        score=q.parse_native_log(self.fake(rows),rows)
        self.assertEqual(score["choice_accuracy"],1.)
        self.assertEqual(score["masked_suffix_tokens"],16)
        self.assertAlmostEqual(score["correct_answer_mean_nll"],.3)
        self.assertAlmostEqual(score["mean_correct_logprob_margin"],1.2)

    def test_pairwise_accuracy_catches_wrong_answers(self):
        rows=self.windows()
        score=q.parse_native_log(self.fake(rows,good_loss=1.5,bad_loss=.3),rows)
        self.assertEqual(score["choice_accuracy"],0.)

    def test_wrong_prefix_is_rejected(self):
        rows=self.windows()
        log=self.fake(rows).replace("window 0 prefix 25","window 0 prefix 24")
        with self.assertRaises(RuntimeError):
            q.parse_native_log(log,rows)

    def test_wrong_token_count_is_rejected(self):
        rows=self.windows()
        log=self.fake(rows).replace("tokens 1","tokens 2",1)
        with self.assertRaises(RuntimeError):
            q.parse_native_log(log,rows)

    def test_missing_window_is_rejected(self):
        rows=self.windows()
        with self.assertRaises(RuntimeError):
            q.parse_native_log(self.fake(rows),rows[:-1])

    def test_tampered_aggregate_is_rejected(self):
        rows=self.windows()
        log=self.fake(rows).replace("qv4_masked_nll 14.4","qv4_masked_nll 15.4")
        with self.assertRaises(RuntimeError):
            q.parse_native_log(log,rows)

    def test_nll_nonfinite_is_rejected(self):
        rows=self.windows()
        log=self.fake(rows).replace("nll 0.29999999999999999","nll NaN",1)
        with self.assertRaises(RuntimeError):
            q.parse_native_log(log,rows)

if __name__=="__main__":
    unittest.main(verbosity=2)
