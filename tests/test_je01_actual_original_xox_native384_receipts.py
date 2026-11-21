#!/usr/bin/env python3
"""JE01 original XOX real 384 native model, old32 token parity receipt tests."""
import hashlib
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parents[1]
EVID=HERE/"docs"/"research"/"JE01_ACTUAL_XOX_ORIGINAL_AF_NATIVE384_5_RUNS_20261117.json"
RECEIPT="3bef35d1fc892e651df74dc03f5bf9bda31824753411e2bfccf2af118d403ec0"
CURRENT_SHA="60a0b559496180d29b65d5a9cb86bba4d16a12238a140cdfdd157602fcc90437"
OLD_SHA="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
BUNDLES={"deepseek_moe_af.bin":(9814343680,"9e619a391c6eb438f1c0b82b06848578727a86ddebaca15b70d95183a73ae6b4"),
         "deepseek_moe_f32.bin":(14137344,"1cc5386c8b8773ee802e059be22f0b7b317dacfc3c9bbf0b724b4e411b596033"),
         "layout_af.txt":(25625,"ba3898f3bf366b1aec8471d9896885f101bf64b78c816828ea82bd7d58851c3e"),
         "layout_f32.txt":(6695,"2ecc5fcb2b4d26302fdd26e9f7d9167811b431a5649d467fa5e5912dbb5a5e4c"),
         "arch_config_moe.txt":(372,"43f205631553478b204f761d5894aaeec67100ed276aca762a24c62e63a91c50")}

def attested():
    d=json.loads(EVID.read_text())
    sig=d.pop("receipt_sha256")
    got=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),
                                  allow_nan=False).encode()).hexdigest()
    if sig!=RECEIPT or got!=RECEIPT:
        raise AssertionError("JE01_XOX_REAL384_RECEIPT_CANONICAL_SHA_INVALID")
    return d

class RealNative384XoxEvidence(unittest.TestCase):
    def test_model_is_original_XOX_AF_SHA_pinned_and_512_original_inputs(self):
        d=attested()
        self.assertEqual(d["status"],"PASS_384_NATIVE_BOUNDARY_SHORT_PARITY")
        self.assertEqual(d["xox_original_source_git_commit"],
                         "e827b522959a5bb01852686c76c4e65ac16b63ba")
        self.assertEqual(d["source_af_total_bytes"],9828513716)
        actual={x["name"]:(x["bytes"],x["sha256"]) for x in d["source_af_5_files"]}
        self.assertEqual(actual,BUNDLES)
        self.assertEqual(d["source_sha256_receipt_sha256"],
                         "8399be688722e6732c67ed9c0518ff808403e3ca5d82597aad7407906b915870")
        self.assertEqual(d["official_512_raw_native_inputs_verified"],512)
        self.assertEqual(d["original_512_source_receipt_sha256"],
                         "2f835daf453741d35e292dd2661b927e2a6196000c9917b62b68415968b5ce34")
        self.assertTrue(d["source_code_original_unmodified"])
        self.assertFalse(d["source_bitstream_promotion"])
        self.assertFalse(d["production_route_and_worker_modified"])

    def test_real_model_384_boundary_succeeded_without_kernel_or_cache_overflow(self):
        d=attested()
        expected={
            "native384_short":(10,279,"03493190422092cd4e32996395b767ae011a7a77b0da3df2df14d357b629c3c4"),
            "native384_long":(217,853,"8591db42a76cc308709d38edf38d702eb6523ea03b60c5fe0c780d39bd0ff7bd"),
            "native384_synthetic330":(330,3572,"7a5c21cc6400c69bb3460b16acc35372f374919f85dba251f5554f4a19cfb76b"),
            "native384_synthetic383":(383,3572,"ba83fcb72944619c936271aa8a78a1dd3273563a9528d2279ab8d4d89a054b9d")}
        for arm,(n,tok,receipt) in expected.items():
            with self.subTest(arm=arm):
                x=d["runs"][arm]
                self.assertEqual(x["status"],"REAL_MODEL_FORWARD_PASS")
                self.assertEqual(x["num_prefill_input_positions"],n)
                self.assertEqual(x["next_token_id"],tok)
                self.assertEqual(x["run_receipt_sha256"],receipt)
                self.assertEqual(x["original_binary_sha256"],CURRENT_SHA)
                self.assertEqual(x["resolved_model_layers"],27)
                self.assertEqual(x["bound_mlxtensors"],269)
                self.assertTrue(x["production_workers_unchanged"])
                self.assertGreaterEqual(x["lowest_memory_free_percent"],35)
                self.assertEqual(x["native_teacher_forced_NLL"],"NOT_MEASURED")
        self.assertEqual(d["runs"]["native384_synthetic383"]["num_prefill_input_positions"]+1,384)

    def test_operating32_and_isolated384_have_identical_first_token_on_same_real_ood_input(self):
        d=attested()
        a=d["runs"]["operating32_short"]
        b=d["runs"]["native384_short"]
        self.assertEqual(a["original_binary_sha256"],OLD_SHA)
        self.assertEqual(a["run_receipt_sha256"],
                         "114c7953a9b599da9174fa2974b5ca56cc5b3020060cc2c79a75b2bd8fe374e4")
        self.assertEqual(a["manifest_sha256"],b["manifest_sha256"])
        self.assertEqual(a["next_token_id"],b["next_token_id"])
        self.assertEqual(a["next_token_id"],279)
        self.assertTrue(a["production_workers_unchanged"])

    def test_no_unmeasured_quality_or_logit_parity_promotion_claim(self):
        d=attested()
        self.assertEqual(d["full_logit_tensor_equality"],"NOT_MEASURED")
        self.assertEqual(d["real_gold_token_NLL"],"NOT_MEASURED")
        self.assertEqual(d["JE01_512_full_model_qualities"],"NOT_MEASURED")
        self.assertEqual(d["native_eval_prompt_cohorts_processed"]["official_512_full_evaluation"],"NOT_RUN")
        self.assertEqual(d["native_eval_prompt_cohorts_processed"]["official_ood_short"],1)
        self.assertEqual(d["native_eval_prompt_cohorts_processed"]["official_calibration_long"],1)
        self.assertEqual(d["native_eval_prompt_cohorts_processed"]["synthetic_extra_boundary"],2)
        self.assertTrue(d["no_operating_release_activate"])

if __name__=="__main__":
    unittest.main(verbosity=2)
