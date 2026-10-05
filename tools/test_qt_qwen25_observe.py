#!/usr/bin/env python3
import array,unittest
from qt_qwen25_observe import observe,build_outputs

class QwenObservationTests(unittest.TestCase):
 def vals(self):
  return array.array("f",[0.01*((i%17)-8) for i in range(2*64)])

 def test_deterministic_and_group_addressed(self):
  vals=self.vals()
  a=observe(vals,2,64,[4,5,6]); b=observe(vals,2,64,[6,4,5])
  self.assertEqual(a,b); self.assertEqual(len(a),6)
  self.assertEqual(a[0]["target_key"],"qwen2.5/L0/q_proj/row=0/group64=0")
  self.assertEqual({x["candidate_n"] for x in a},{4,5,6})

 def test_qheatmap_outputs_are_deterministic(self):
  vals=self.vals()
  a=build_outputs(vals,2,64,"F32",[4,5,6],tensor_sha256="0"*64)
  b=build_outputs(vals,2,64,"F32",[6,5,4],tensor_sha256="0"*64)
  self.assertEqual(a,b)
  obs,qmap,summary=a
  self.assertEqual(summary["group_cells"],2)
  self.assertEqual(summary["event_count"],6)
  self.assertEqual(sum(summary["recommended_n_distribution"].values()),2)
  self.assertEqual(qmap["heatmap_sha256"],summary["heatmap_sha256"])
  self.assertFalse(summary["production_touched"])
  self.assertFalse(summary["automatic_live_promotion"])

if __name__=="__main__": unittest.main()
