#!/usr/bin/env python3
"""JE01 original-gate-on-real-reference-hidden 26x6x64 evidence contract tests."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import unittest

ROOT=Path(__file__).resolve().parents[1]
EVID=ROOT/"docs"/"research"
GATES="JE01_REAL_BF16_VS_EXTERNAL_CHAT_26_ROUTER_WEIGHT_PARITY_20261117.json"
TRACES="JE01_REAL_26_LAYER_ORIGINAL_BF16_ROUTER_ON_REFERENCE_HIDDEN_20261117.json"

def verify(source,expected):
    x=json.loads((EVID/source).read_text())
    sig=x.pop("receipt_sha256")
    canonical=hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),
                                allow_nan=False).encode()).hexdigest()
    if sig!=expected or canonical!=expected:
        raise AssertionError("JE01_REAL_PROVENANCE_CANONICAL_SHA_FAILED")
    return x


class OriginalGateAndReferenceHiddenState(unittest.TestCase):
    def test_all_26_original_BF16_gate_weights_tiny_difference(self):
        x=verify(GATES,"5865d0b0e1e9b0f6e55a2c05257db82a1c66c8c72eb16f1f2e19ad55c3c2d7d6")
        self.assertEqual(x["status"],"ALL_26_ROUTER_WEIGHT_NUMERIC_PARITY_PASS")
        self.assertEqual(x["n_routed_expert_gate_matrices"],26)
        self.assertEqual(x["n_matrix_scalar_coefficients"],26*64*2048)
        self.assertEqual(x["max_absolute_coefficient_difference"],2.9802322387695312e-8)
        self.assertLess(x["max_relative_rms_difference"],1e-6)
        self.assertEqual([p["layer"] for p in x["per_layer_router_weights"]],
                         list(range(1,27)))
        self.assertEqual(x["same_router_hidden_states"],"NOT_MEASURED")
        self.assertEqual(x["full_pretrained_BF16_model_forward"],"NOT_RUN")
        self.assertEqual(x["JE01_gold_token_NLL"],"NOT_MEASURED")
        self.assertFalse(x["same_model_tokenizer"])
        self.assertFalse(x["production_promotion_allowed"])

    def test_all_156_real_trace_events_exact_encoded_source_gate_reference_parity(self):
        a=verify(GATES,"5865d0b0e1e9b0f6e55a2c05257db82a1c66c8c72eb16f1f2e19ad55c3c2d7d6")
        z=verify(TRACES,"1d94acbbd801c52faa2ec546c8595b1feae00b48437fb8d922dda16aa0bb9623")
        self.assertEqual(z["status"],"REAL_26_LAYER_SOURCE_GATE_SAME_REFERENCE_HIDDEN_ROUTER_PASS")
        self.assertEqual(z["routed_layers_monitored"],26)
        self.assertEqual(z["total_real_reference_token_layer_events"],156)
        self.assertEqual(z["sample_token_positions_per_layer"],6)
        self.assertEqual(z["full_model_reference_logits_change_max_abs"],0.0)
        self.assertEqual(z["max_BF16_gate_vs_ref_gate_same_hidden_probability_difference"],0.0)
        self.assertEqual(z["hard_topk_selection_mismatches"],0)
        self.assertEqual(z["ambiguous_boundary_tie_events"],0)
        self.assertFalse(z["JE01_512_official_training_and_holdout_used"])
        self.assertEqual(z["original_BF16_full_model_logits_parity"],"NOT_TESTED")
        self.assertEqual(z["original_BEG_LIN_full_model_gold_NLL"],"NOT_MEASURED")
        self.assertFalse(z["production_write_allowed"])
        self.assertTrue(z["source_BF16_router_gates_converted_to_reference_fp16"])
        self.assertTrue(z["same_reference_model_hidden_states_for_both_router_gates"])
        self.assertEqual(z["original_BF16_model_hidden_states"],"NOT_USED_OR_MEASURED")
        self.assertEqual(z["original_BF16_full_checkpoint_identity_sha256"],
                         a["original_BF16_full_checkpoint_identity_sha256"])
        self.assertEqual(z["external_MLX_Chat4bit_checkpoint_identity_sha256"],
                         a["reference_MLX_Chat4bit_checkpoint_identity_sha256"])
        observed = {(e["layer"],e["token_position"]) for e in z["actual_reference_router_events"]}
        self.assertEqual(observed,{(l,t) for l in range(1,27) for t in range(6)})
        routers={k:str(v) for k,v in z["original_BF16_router_matrix_per_layer_sha256"].items()}
        self.assertEqual(routers,{str(x["layer"]):x["original_BF16_data_sha256"]
                                  for x in a["per_layer_router_weights"]})
        for e in z["actual_reference_router_events"]:
            p=struct.unpack("<64e",base64.b64decode(e["reference_fp16_softmax64_b64"],validate=True))
            q=struct.unpack("<64e",base64.b64decode(e["source_BF16_gate_on_reference_hidden_fp16_softmax64_b64"],validate=True))
            self.assertEqual(p,q)
            selected=e["chosen6_expert_ids"]
            self.assertEqual(len(selected),6)
            self.assertEqual(len(set(selected)),6)
            self.assertTrue(all(0<=i<64 for i in selected))
            self.assertLess(abs(sum(p)-1),0.001)
            self.assertGreater(e["boundary_margin"],0)
            self.assertGreater(min(p[i] for i in selected),
                               max(p[i] for i in range(64) if i not in selected))
            for i,v in zip(selected,e["selected_probs"]):
                self.assertAlmostEqual(p[i],v,places=5)
            self.assertEqual(e["source_reference_max_abs_prob_diff"],0.0)
        self.assertLess(z["mlx_gpu_peak_bytes"],z["mlx_memory_cap_bytes"])

if __name__=="__main__":
    unittest.main(verbosity=2)
