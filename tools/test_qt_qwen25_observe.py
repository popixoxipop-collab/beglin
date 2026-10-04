#!/usr/bin/env python3
import array,unittest
from qt_qwen25_observe import observe
class QwenObservationTests(unittest.TestCase):
 def test_deterministic_and_group_addressed(self):
  vals=array.array("f",[0.01*((i%17)-8) for i in range(2*64)])
  a=observe(vals,2,64,[4,5,6]); b=observe(vals,2,64,[6,4,5])
  self.assertEqual(a,b); self.assertEqual(len(a),6)
  self.assertEqual(a[0]["target_key"],"qwen2.5/L0/q_proj/row=0/group64=0")
  self.assertEqual({x["candidate_n"] for x in a},{4,5,6})
if __name__=="__main__": unittest.main()
