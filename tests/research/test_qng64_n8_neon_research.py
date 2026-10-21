import hashlib
import json
import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import qng64_n8_neon_independent_audit as q

class NativeN8NEONResearchQA(unittest.TestCase):
    def test_source_sha_guards_and_private_native_binary(self):
        n8,mixed=q.validate_native_sources()
        self.assertEqual(n8["source_original_scalar_c_sha256"],q.EXPECTED_NATIVE_SOURCE)
        self.assertTrue(n8["no_live_promotion"])
        self.assertFalse(mixed["production_touched"])
    def test_entire_model_8bit_output_parity(self):
        rec=q.verify_numeric(q.BASE/"n8/results/c_logits_last.f32",
                             q.NEON/"n8/results/c_logits_last.f32")
        self.assertLess(rec["max_abs_logits"],.001)
        self.assertLess(rec["relative_L2_logits"],1e-4)
        self.assertEqual(rec["argmax_id"],9707)
        self.assertEqual(rec["top10_overlap"],10)
    def test_frozen_mixed_output_parity(self):
        for name in ("proxy","random"):
            with self.subTest(name=name):
                v=q.verify_numeric(
                    q.MIXED/f"mixed_{name}/results/c_logits_last.f32",
                    q.NEON_MIX/f"mixed_{name}/results/c_logits_last.f32")
                self.assertLess(v["relative_L2_logits"],1e-4)
    def test_real_four_shape_extraction_pass(self):
        data=json.loads(q.secure(q.MULTI/"n8_neon_multishape_private_receipt.json"))
        self.assertEqual(len(data["examples"]),4)
        shapes={tuple(x["weight_shape"]) for x in data["examples"]}
        self.assertEqual(shapes,{(256,1536),(1536,1536),(8960,1536),(1536,8960)})
        for item in data["examples"]:
            self.assertGreater(item["relative_speedup"],5)
            self.assertLess(item["n8_neon_vs_original_scalar_maxabs"],.0005)
    def test_all_original_ppl_nearly_constant_and_speed_improves(self):
        for label,item in q.OBSERVED_NATIVE_RUNS.items():
            with self.subTest(label=label):
                self.assertLess(abs(item["PPL_NEON"]-item["PPL_original"])/item["PPL_original"],1e-5)
                self.assertGreater(item["single_token_old_ms"]/item["single_token_NEON_ms"],5)
    def test_private_frozen_receipt_and_no_overwrite(self):
        p=q.OUT
        self.assertEqual(p.stat().st_mode&0o777,0o600)
        d=json.loads(p.read_text())
        self.assertEqual(d["scientific_Q_accuracy_verdict"],"NO_VERDICT")
        self.assertFalse(d["main_or_production_modified"])
        with self.assertRaises(FileExistsError):
            q.run()
if __name__=="__main__":
    unittest.main(verbosity=2)
