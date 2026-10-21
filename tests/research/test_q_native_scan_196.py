#!/usr/bin/env python3
"""Pure unit tests for native source-only quantization proxy and role-matched policy."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"oml_acs"))
from q_native_scan_196 import decode_q4, decode_n8, policy_candidates, ROLES, NUM_LAYERS, UPGRADES_PER_ROLE

class RealTensorPolicyTests(unittest.TestCase):
    def test_q4_nibble_exact_semantics(self):
        # low nibble = element 0, high nibble = element 1, biased by 8
        b=np.zeros(32,dtype=np.uint8)
        b[0]=((15<<4)|0)
        scales=np.array([0.25],dtype=np.float32)
        decoded=decode_q4(b,scales,1,64).reshape(-1)
        self.assertEqual(float(decoded[0]),-2.0)
        self.assertEqual(float(decoded[1]),1.75)
        self.assertEqual(float(decoded[2]),-2.0)
    def test_n8_bitplane_le_semantics(self):
        # n=8 code index0 unsigned 255 -> 127; index1 unsigned 0 -> -128
        planes=np.zeros(64,dtype=np.uint8)
        for bit in range(8):
            planes[bit*8]=1
        decoded=decode_n8(planes,np.array([.125],dtype=np.float32),1,64).reshape(-1)
        self.assertAlmostEqual(float(decoded[0]),15.875)
        self.assertAlmostEqual(float(decoded[1]),-16.)
    def test_exact_role_budget_and_equal_cost(self):
        rows=[]
        for role,_ in ROLES:
            for layer in range(NUM_LAYERS):
                rows.append(dict(role=role,layer=layer,
                    squared_normalized_error_reduction_q4_minus_n8=float(layer),
                    q4_with_scales_bytes=100*(len(role)+1),
                    qng64_8_with_scales_bytes=200*(len(role)+1)))
        policy,total=policy_candidates(rows)
        self.assertEqual(total['proxy'],total['random'])
        for role,_ in ROLES:
            self.assertEqual(policy[role]['proxy_n8_layers'],[24,25,26,27])
            self.assertEqual(len(policy[role]['seed73_random_n8_layers']),UPGRADES_PER_ROLE)
    def test_repeatable_seed(self):
        rows=[dict(role=role,layer=layer,
             squared_normalized_error_reduction_q4_minus_n8=float((layer*7)%29),
             q4_with_scales_bytes=100,qng64_8_with_scales_bytes=150)
             for role,_ in ROLES for layer in range(NUM_LAYERS)]
        a,_=policy_candidates(rows);b,_=policy_candidates(rows)
        self.assertEqual(a,b)
    def test_incomplete_roles_fail_closed(self):
        rows=[dict(role=ROLES[0][0],layer=layer,
             squared_normalized_error_reduction_q4_minus_n8=float(layer),
             q4_with_scales_bytes=100,qng64_8_with_scales_bytes=150)
             for layer in range(NUM_LAYERS)]
        with self.assertRaises(RuntimeError):
            policy_candidates(rows)
    def test_role_matched_cost_required(self):
        rows=[dict(role=role,layer=layer,
             squared_normalized_error_reduction_q4_minus_n8=float(layer),
             q4_with_scales_bytes=100,qng64_8_with_scales_bytes=200+layer)
             for role,_ in ROLES for layer in range(NUM_LAYERS)]
        with self.assertRaises(RuntimeError):
            policy_candidates(rows)

if __name__=="__main__":
    unittest.main(verbosity=2)
