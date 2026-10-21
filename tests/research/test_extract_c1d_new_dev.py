import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
from extract_c1d_new_dev import MODELS,load_rows,run,DATA_SHA

class Contract(unittest.TestCase):
 def test_models_three(self):self.assertEqual(set(MODELS),{"qwen","mistral","phi"})
 def test_dims(self):self.assertEqual(sorted(k['hidden'] for k in MODELS.values()),[1536,3072,4096])
 def test_sha(self):self.assertEqual(len(DATA_SHA),64)
 def test_unknown_alias(self):
  with self.assertRaises(ValueError):run("unknown")
 def test_class_balance(self):
  rows=load_rows()
  self.assertEqual(len(rows),200)
  self.assertEqual(sum(x['class_label'] for x in rows),100)
 def test_data_scope(self):self.assertTrue(all(x['source_split']=="train" for x in load_rows()))
if __name__=="__main__":unittest.main(verbosity=2)
