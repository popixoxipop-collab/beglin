import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
from build_behavior_train_dev import canonical_sha, select_rows, validate_rows, DATASETS

class BehaviorSourceTest(unittest.TestCase):
    def records(self,n=400):
        out=[]
        for i in range(n):
            out.append({"sentence":f"Source review example number {i} is different.",
                        "label":i%2})
        return out
    def test_pinned_binary_selection_balanced(self):
        s=DATASETS[0]; used=set()
        a,meta=select_rows(records=self.records(),config=s,split="behavior_train",
                      n_per_class=30,seed=42,occupied=used,anchor_hashes=set())
        self.assertEqual(len(a),60)
        self.assertEqual(sum(x["class_label"] for x in a),30)
        self.assertEqual(len(used),60)
    def test_deterministic_row_identity(self):
        s=DATASETS[0]
        a,_=select_rows(records=self.records(),config=s,split="behavior_train",
                      n_per_class=20,seed=42,occupied=set(),anchor_hashes=set())
        b,_=select_rows(records=self.records(),config=s,split="behavior_train",
                      n_per_class=20,seed=42,occupied=set(),anchor_hashes=set())
        self.assertEqual([x["text_sha256"] for x in a],[x["text_sha256"] for x in b])
    def test_anchor_overlap_is_filtered(self):
        s=DATASETS[0]
        anchor={canonical_sha("Source review example number 0 is different.")}
        a,_=select_rows(records=self.records(),config=s,split="behavior_train",
                      n_per_class=20,seed=42,occupied=set(),anchor_hashes=anchor)
        self.assertNotIn(anchor.pop(),{x["canonical_text_sha256"] for x in a})
    def test_split_fail_closed(self):
        with self.assertRaises(ValueError):
            select_rows(records=self.records(),config=DATASETS[0],split="behavior_final",
                        n_per_class=20,seed=42,occupied=set(),anchor_hashes=set())
    def test_no_duplicate_across_selections(self):
        used=set()
        a,_=select_rows(records=self.records(),config=DATASETS[0],split="behavior_train",
                      n_per_class=20,seed=42,occupied=used,anchor_hashes=set())
        b,_=select_rows(records=self.records(),config=DATASETS[0],split="behavior_dev",
                      n_per_class=20,seed=43,occupied=used,anchor_hashes=set())
        self.assertEqual(len(a+b),len({x["canonical_text_sha256"] for x in a+b}))
    def test_insufficient_samples_rejected(self):
        with self.assertRaises(RuntimeError):
            select_rows(records=self.records(10),config=DATASETS[0],
                        split="behavior_train",n_per_class=20,seed=42,
                        occupied=set(),anchor_hashes=set())
    def test_validate_requires_full_quotas(self):
        with self.assertRaises(AssertionError):
            validate_rows([],set())
if __name__=="__main__":unittest.main(verbosity=2)
