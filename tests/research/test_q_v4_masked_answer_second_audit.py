"""Independent verifier checks; do not import the Qv4 policy evaluator."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v4_masked_answer_independent_audit as a

class MaskedAnswerSecondAuditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cfg=json.loads(a.PRELOCK.read_text())
        cls.windows=[]
        for obj in cfg["examples"]:
            for letter in ("A","B"):
                cls.windows.append({"case":obj["id"],"prefix":30,"answer":letter,
                                     "is_correct":letter==obj["correct"]})

    def transcript(self,mode="f32",correct=.15,incorrect=2.7):
        counts=a.EXPECTED_COUNTS[mode]
        rows=["[Q-NATIVE-RESEARCH] projection_role_tensors: f32=%s n8=%s q4=%s" % counts]
        nlls=[]
        for i,x in enumerate(self.windows):
            nll=correct if x["is_correct"] else incorrect
            rows.append("[qv4-masked] window %d prefix 30 nll %.17g tokens 1"%(i,nll))
            nlls.append(nll)
        total=sum(nlls); n=len(nlls)
        rows.append("qv4_masked_nll %.17g tokens %d mean %.17g ppl %.17g"%(total,n,total/n,math.exp(total/n)))
        return "\n".join(rows)

    def test_independent_per_case_nll_and_accuracy(self):
        d=a.interpret(self.transcript(),self.windows,"f32")
        self.assertEqual(d["accuracy"],1.)
        self.assertAlmostEqual(d["correct_only_nll"],.15)
        self.assertAlmostEqual(d["mean_answer_margin"],2.55)

    def test_distinct_policy_counts(self):
        for mode in a.POLICIES:
            self.assertEqual(a.interpret(self.transcript(mode),self.windows,mode)["accuracy"],1.)

    def test_wrong_projection_tier_fails_closed(self):
        with self.assertRaises(RuntimeError):
            a.interpret(self.transcript("q4"),self.windows,"v3_cov")

    def test_prefix_tampering_detected(self):
        with self.assertRaises(RuntimeError):
            a.interpret(self.transcript().replace("window 0 prefix 30","window 0 prefix 29"),self.windows,"f32")

    def test_denominator_tampering_detected(self):
        with self.assertRaises(RuntimeError):
            a.interpret(self.transcript().replace("tokens 16 mean","tokens 15 mean"),self.windows,"f32")

    def test_missing_suffix_score_detected(self):
        with self.assertRaises(RuntimeError):
            a.interpret(self.transcript().replace("[qv4-masked] window 5","[fake] window 5"),self.windows,"f32")

    def test_nonfinite_suffix_score_detected(self):
        with self.assertRaises(RuntimeError):
            a.interpret(self.transcript().replace("nll 0.14999999999999999","nll nan",1),self.windows,"f32")

    def test_source_policy_hashes(self):
        self.assertEqual(a.sha(a.PRELOCK),a.PROTO_HASH)
        self.assertEqual(a.sha(a.LAB/"inputs_frozen.json"),a.WINDOW_HASH)

if __name__=="__main__":
    unittest.main(verbosity=2)
