#!/usr/bin/env python3
import unittest
from qt_build_qpolicy import build
class TestQPolicy(unittest.TestCase):
 def test_candidate_is_deterministic_and_unapproved(self):
  q={"schema":"beglin-qheatmap-v2","generation":1,"heatmap_sha256":"a"*64,"cells":[
   {"target_key":"qwen2.5/L0/q_proj/row=0/group64=1","recommended_n":6,"confidence":1.0,"error_by_n":{"6":0.0001}},
   {"target_key":"qwen2.5/L0/q_proj/row=0/group64=0","recommended_n":4,"confidence":1.0,"error_by_n":{"4":0.0002}}]}
  s={"schema":"beglin-qt-qwen25-qheatmap-summary-v1","heatmap_sha256":"a"*64,"tensor_name":"x","tensor_sha256":"b"*64,"current_n":5}
  a=build(q,s); b=build(q,s)
  self.assertEqual(a,b); self.assertEqual(a["distribution"],{"4":1,"6":1}); self.assertFalse(a["approved_by_beval"]); self.assertFalse(a["production_write_allowed"])
  self.assertEqual([c["qgroup_index"] for c in a["cells"]],[0,1])
if __name__=="__main__": unittest.main()
