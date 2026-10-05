#!/usr/bin/env python3
import array,tempfile,unittest
from pathlib import Path
from qt_qwen25_uniform_fixture import build,write_fixture
class TestUniformFixture(unittest.TestCase):
 def test_n_changes_planes_and_manifest_is_deterministic(self):
  vals=array.array("f",[0.01*((i%17)-8) for i in range(2*64)])
  with tempfile.TemporaryDirectory() as td:
   a=build(vals,2,64,4); b=build(vals,2,64,6)
   self.assertNotEqual(a["planes.bin"],b["planes.bin"])
   m1=write_fixture(Path(td)/"a",a,{"schema":"beglin-qt-qwen25-uniform-fixture-v1","tensor_name":"x","tensor_sha256":"0"*64,"shape":[2,64],"dtype":"F32","n":4,"group_size":64,"production_touched":False,"automatic_live_promotion":False})
   m2=write_fixture(Path(td)/"b",a,{"schema":"beglin-qt-qwen25-uniform-fixture-v1","tensor_name":"x","tensor_sha256":"0"*64,"shape":[2,64],"dtype":"F32","n":4,"group_size":64,"production_touched":False,"automatic_live_promotion":False})
   self.assertEqual(m1["manifest_sha256"],m2["manifest_sha256"])
if __name__=="__main__": unittest.main()
