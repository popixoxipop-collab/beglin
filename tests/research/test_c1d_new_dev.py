import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
from build_c1d_new_dev import tidy,canonical_sha,pick,SPEC,N

class SourceTests(unittest.TestCase):
 def test_html_whitespace(self):self.assertEqual(tidy("A <br /> b   c"),"A b c")
 def test_canonical_duplicate(self):self.assertEqual(canonical_sha(" HELLO x "),canonical_sha("hello    X"))
 def test_labels_are_fixed(self):
  self.assertEqual(SPEC[0]['positive'],1)
  self.assertEqual(SPEC[0]['negative'],0)
  self.assertEqual(SPEC[1]['positive'],17)
  self.assertEqual(SPEC[1]['negative'],2)
 def test_select_balances_synthetic_dataset(self):
  source=[dict(text=f"Unique strong emotional sentence number {i}",label=i%2) for i in range(200)]
  out=pick(SPEC[0],source,np.random.default_rng(73),set())
  self.assertEqual((len(out),sum(r['class_label'] for r in out)),(100,50))
 def test_anchor_collision_excluded(self):
  source=[dict(text=f"Unique long review number {i} is a different example",label=i%2) for i in range(300)]
  blocked=canonical_sha(source[0]['text'])
  out=pick(SPEC[0],source,np.random.default_rng(73),{blocked})
  self.assertNotIn(blocked,{r['canonical_sha256'] for r in out})
 def test_insufficient_examples(self):
  with self.assertRaises(RuntimeError):
   pick(SPEC[0],[dict(text="Unique but far too few examples",label=1)],np.random.default_rng(73),set())
 def test_multilabel_goemotions_excluded(self):
  source=[]
  for i in range(300):
   lab=17 if i%2==0 else 2
   source.append(dict(text=f"Emotion source comment number {i} is very different",labels=[lab]))
  source[0]['labels']=[17,2]
  out=pick(SPEC[1],source,np.random.default_rng(74),set())
  self.assertTrue(all(row['class_label'] in (0,1) for row in out))
if __name__=="__main__":unittest.main(verbosity=2)
