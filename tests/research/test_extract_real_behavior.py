import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
from extract_real_behavior import SOURCES, sha, BEHAVIOR_ROOT_SHA, extract

class BehaviorActivationContracts(unittest.TestCase):
 def test_model_architecture_pinned(self):
  self.assertEqual(SOURCES["qwen"]["hidden"],1536)
  self.assertEqual(SOURCES["mistral"]["hidden"],4096)
  self.assertEqual(SOURCES["phi"]["hidden"],3072)
 def test_revisions_are_commit_hashes(self):
  for data in SOURCES.values():
   self.assertEqual(len(data["revision"]),40)
   int(data["revision"],16)
 def test_no_implicit_models(self):
  with self.assertRaises(ValueError):
   extract("unapproved")
 def test_sha_is_stable(self):
  self.assertEqual(sha(b"a"*2),sha(b"aa"))
 def test_release_contract_not_final(self):
  self.assertEqual(BEHAVIOR_ROOT_SHA,"b99fb5d7e8c909e6821f555ac590b36ff52393b9d4e7a01828ee936c615bd274")
if __name__=="__main__":unittest.main(verbosity=2)
