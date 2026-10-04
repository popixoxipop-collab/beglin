#!/usr/bin/env python3
import itertools,unittest
import qt_heatmap as h

EVENTS=[
 {"kind":"quant","target_key":"qwen2.5/L0/q_proj/row=0/group64=0","candidate_n":4,"current_n":5,"output_max_abs_error":8e-4},
 {"kind":"quant","target_key":"qwen2.5/L0/q_proj/row=0/group64=0","candidate_n":5,"current_n":5,"output_max_abs_error":1e-5},
 {"kind":"quant","target_key":"qwen2.5/L0/q_proj/row=0/group64=0","candidate_n":6,"current_n":5,"output_max_abs_error":5e-6},
 {"kind":"train","target_key":"qwen2.5/L0/q_proj/row=0/group64=0","delta_w_quant":.02,"grad":.5},
 {"kind":"train","target_key":"qwen2.5/L0/q_proj/row=0/group64=1","delta_w_quant":.001,"grad":.1},
]
class TestQTHeatmap(unittest.TestCase):
 def test_q_selects_minimum_safe_n(self):
  q,_=h.project(EVENTS); self.assertEqual(q["cells"][0]["recommended_n"],5)
 def test_t_ranks_sensitivity(self):
  _,t=h.project(EVENTS); d={x["target_key"]:x for x in t["cells"]}; self.assertGreater(d[list(d)[0]]["sensitivity"],d[list(d)[1]]["sensitivity"])
 def test_policy_gates_are_explicit(self):
  q,_=h.project(EVENTS,error_budget=1e-6,min_samples=99); self.assertEqual(q["cells"][0]["recommended_n"],6); self.assertEqual(q["cells"][0]["state"],"OBSERVED")
 def test_replay_is_order_deterministic(self):
  base=tuple(map(h.digest,h.project(EVENTS)))
  for ev in (list(reversed(EVENTS)),EVENTS[2:]+EVENTS[:2]):
   self.assertEqual(tuple(map(h.digest,h.project(ev))),base)
if __name__=="__main__": unittest.main()
