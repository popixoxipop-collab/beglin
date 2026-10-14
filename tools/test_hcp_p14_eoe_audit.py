#!/usr/bin/env python3
"""P14 fail-closed EOE evidence regression: dual native workers and BF16 scope."""
from __future__ import annotations
import json,pathlib,shutil,tempfile,unittest
from hcp_p14_eoe_summary_audit import audit

REPO=pathlib.Path(__file__).resolve().parents[1]
SOURCE=REPO/"experiments/hcp_p14/evidence/native_xox_20261014"

class P14EOEChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.dir=pathlib.Path(self.temp.name)/"native"
        shutil.copytree(SOURCE,self.dir)
        (self.dir/"MANIFEST.json").unlink(missing_ok=True)
    def tearDown(self):
        self.temp.cleanup()
    def check(self):return audit(self.dir)
    def rejected(self):
        with self.assertRaises((ValueError,KeyError,TypeError,RuntimeError,AssertionError)):
            self.check()
    def modify(self,name,callback):
        path=self.dir/name
        j=json.loads(path.read_text())
        callback(j)
        path.write_text(json.dumps(j,sort_keys=True)+"\n")
    def test_01_actual_sanitized_proof_passes(self):
        r=self.check()
        self.assertEqual(r["true_dual_native_responses_passed_in_original_XOX_audit"],48)
        self.assertEqual(r["original_XOX_time_overlap_pairs"],24)
        self.assertFalse(r["promote_production_allowed"])
    def test_02_missing_original_unknown_txn_denied(self):
        (self.dir/"quarantine/original_unknown.cmd").unlink()
        self.rejected()
    def test_03_ackless_crash_claimed_success_denied(self):
        self.modify("P14_AUDIT_SUMMARY.json",
          lambda d:d.update(real_before_ack_crash_escrow_original_outcome="APPLIED"))
        self.rejected()
    def test_04_healthy_peer_restarted_claim_denied(self):
        self.modify("P14_AUDIT_SUMMARY.json",
          lambda d:d.update(surviving_worker_PID_unchanged=False))
        self.rejected()
    def test_05_overlapping_calls_forgery_denied(self):
        self.modify("P14_AUDIT_SUMMARY.json",
          lambda d:d.update(actual_overlapping_admission_intervals=0))
        self.rejected()
    def test_06_oversubscription_with_forged_RSS_denied(self):
        self.modify("P14_AUDIT_SUMMARY.json",
          lambda d:d.update(peak_combined_native_worker_RSS_GiB=14))
        self.rejected()
    def test_07_changed_quarantine_command_denied(self):
        (self.dir/"quarantine/original_unknown.cmd").write_text("REBIND_SET tampered\n")
        self.rejected()
    def test_08_intent_missing_denied(self):
        (self.dir/"quarantine/INTENT.json").unlink()
        self.rejected()
    def test_09_verified_receipt_tamper_denied(self):
        self.modify("quarantine/VERIFIED.json",
          lambda d:d.update(no_automatic_retry=False))
        self.rejected()
    def test_10_marker_wrong_epoch_id_denied(self):
        (self.dir/"before_ack.fault.marker").write_text("before_ack epoch-"+("b"*32)+"\n")
        self.rejected()
    def test_11_BF16_full_forward_false_claim_denied(self):
        self.modify("BF16_TEACHER_FEASIBILITY.json",
          lambda d:d.update(full_BF16_teacher_forward_executed=True))
        self.rejected()
    def test_12_BF16_model_index_false_claim_denied(self):
        self.modify("BF16_TEACHER_FEASIBILITY.json",
          lambda d:d.update(model_index_sha256="0"*64))
        self.rejected()
    def test_13_fake_streaming_teacher_accuracy_denied(self):
        self.modify("BF16_TEACHER_FEASIBILITY.json",
          lambda d:d.update(exact_whole_BF16_teacher_logits_produced=True))
        self.rejected()
    def test_14_wrong_native_git_commit_denied(self):
        self.modify("preflight.json",
          lambda d:d.update(source_commit="0"*40))
        self.rejected()
    def test_15_bad_GPU_receipt_SHA_denied(self):
        self.modify("P14_AUDIT_SUMMARY.json",
          lambda d:d.update(raw_native_result_sha256="0"*64))
        self.rejected()
    def test_16_reduced_P6_verified_response_count_denied(self):
        self.modify("P14_AUDIT_SUMMARY.json",
          lambda d:d.update(native_generation_responses_verified_against_P6=47))
        self.rejected()
if __name__=="__main__":unittest.main()
