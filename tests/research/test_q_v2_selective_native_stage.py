#!/usr/bin/env python3
"""Eight fail-closed BEGLIN Qv2 native stage-selective audit regressions."""
import hashlib
import json
import pathlib
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"research/oml_acs"))
import q_v2_selective_native_stage_audit as audit
import q_v2_selective_native_stage_abba as runner

class SelectiveNativeAsyncQv2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=audit.inspect()
    def test_one_SHA_pinned_source_model_mask_binary(self):
        self.assertEqual(self.data["original_BF16_snapshot_revision"],audit.ORIGINAL_BF16_REV)
        self.assertEqual(self.data["selected_196_tensor_bitmask_SHA256"],audit.ORIGINAL_MASK_SHA)
        self.assertEqual(self.data["single_native_binary_sha256"],audit.BIN_SHA)
    def test_all_8_scheduling_modes_and_two_reverse_sweeps(self):
        self.assertEqual(len(self.data["stage_mode_2_sweep_native_observations"]),8)
        for mode,info in self.data["stage_mode_2_sweep_native_observations"].items():
            self.assertEqual(len(info["samples_ms"]),2)
            self.assertGreater(info["mean_ms"],0)
    def test_real_QKV_stage_join_executed(self):
        j=json.loads(audit.secure(audit.RECEIPT))
        qkv=[x for x in j["real_native_two_reverse_sweeps"] if x["mode"]=="qkv6"]
        self.assertEqual(len(qkv),2)
        self.assertTrue(all(x["qkv_joined"]>0 and x["gateup_joined"]==0 for x in qkv))
    def test_real_gateup_stage_join_executed(self):
        j=json.loads(audit.secure(audit.RECEIPT))
        gu=[x for x in j["real_native_two_reverse_sweeps"] if x["mode"]=="gu6"]
        self.assertEqual(len(gu),2)
        self.assertTrue(all(x["gateup_joined"]>0 and x["qkv_joined"]==0 for x in gu))
    def test_original_serial_no_parallel_stage_join(self):
        j=json.loads(audit.secure(audit.RECEIPT))
        serial=[x for x in j["real_native_two_reverse_sweeps"] if x["mode"]=="serial"]
        self.assertEqual(len(serial),2)
        self.assertTrue(all(x["qkv_joined"]==0 and x["gateup_joined"]==0 for x in serial))
    def test_first_vocab_logits_bit_exact(self):
        self.assertTrue(self.data["serial_candidate_logits_151936_byte_identical"])
        j=json.loads(audit.secure(audit.RECEIPT))
        self.assertEqual(j["posthoc_dump"][0]["logits_sha256"],j["posthoc_dump"][1]["logits_sha256"])
    def test_same_131_public_PPL_and_not_quality_promotion(self):
        self.assertEqual(self.data["serial_candidate_same_131_position_PPL"],827.4999)
        self.assertEqual(self.data["PPL_relative_diff"],0)
        self.assertFalse(self.data["real_task_AUROC_heldout_tested"])
        self.assertFalse(self.data["actual_main_or_release_touched"])
        self.assertTrue(self.data["not_latency_per_token_dynamic_adaptation"])
    def test_source_data_stays_private_and_no_rerun_overwrite(self):
        self.assertEqual(audit.RECEIPT.stat().st_mode&0o777,0o600)
        self.assertEqual(audit.OUT.stat().st_mode&0o777,0o600)
        with self.assertRaises(FileExistsError):runner.main()
        with self.assertRaises(FileExistsError):audit.main()

if __name__=="__main__":
    unittest.main(verbosity=2)
