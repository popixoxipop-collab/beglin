#!/usr/bin/env python3
"""P11 forensic evidence tamper/negative tests, all offline with pinned P6 QA."""

from __future__ import annotations
import hashlib,json,os,pathlib,shutil,tempfile,unittest
from hcp_p11_forensic_audit import review

REPO=pathlib.Path(__file__).resolve().parents[1]
EVIDENCE=REPO/"experiments/hcp_p11/evidence/native_xox_20261014"
QUALITY=pathlib.Path(os.environ.get(
    "BEGLIN_P11_QUALITY_FILE",
    str(REPO.parents[1]/"BEGLIN_HCP_P5_P6_HANDOFF_EVIDENCE_2026-10-14/p6_repeat/quality.json")
))
P6_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"

class P11TamperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QUALITY.is_file() or hashlib.sha256(QUALITY.read_bytes()).hexdigest()!=P6_SHA:
            raise ValueError("need exact P6 original quality evidence (set BEGLIN_P11_QUALITY_FILE)")
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.root=pathlib.Path(self.t.name)/"native_xox_20261014"
        shutil.copytree(EVIDENCE,self.root)
        (self.root/"MANIFEST.json").unlink(missing_ok=True)
    def tearDown(self):self.t.cleanup()
    def audit(self):return review(self.root,QUALITY)
    def rejected(self):
        with self.assertRaises((ValueError,KeyError,AssertionError)):
            self.audit()
    def mutate(self,path,fn):
        p=self.root/path
        doc=json.loads(p.read_text())
        fn(doc)
        p.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n")
    def test_01_intact_sha_provenance(self):
        p=self.audit()
        self.assertEqual(len(p["first_bind_before_ack_forensic_sha256"]),2)
        self.assertTrue(p["original_interrupted_transaction_outcome_not_claimed_successful"])
        self.assertFalse(p["BF16_reference_full_model_logits_measured"])
    def test_02_native_marker_corrupted(self):
        (self.root/"first_bind.marker").write_text("before_ack epoch-"+("a"*32)+"\n")
        self.rejected()
    def test_03_escrow_transaction_corrupted(self):
        (self.root/"before_ack_quarantined_transaction.cmd").write_text("REBIND_SET tampered\n")
        self.rejected()
    def test_04_missing_durable_intent_blocks(self):
        (self.root/"forensic/first_bind_INTENT.json").unlink()
        self.rejected()
    def test_05_modified_intent_ack_sha_blocked(self):
        self.mutate("forensic/before_ack_INTENT.json",
                    lambda d:d.update(preimage_ack_sha256="0"*64))
        self.rejected()
    def test_06_modified_verified_txn_hash_blocked(self):
        self.mutate("forensic/first_bind_VERIFIED.json",
                    lambda d:d.update(source_command_sha256="0"*64))
        self.rejected()
    def test_07_wrong_compiled_source_preimage_blocked(self):
        self.mutate("preflight.json",lambda d:d.update(fault_binary_sha="1"*64))
        self.rejected()
    def test_08_original_crash_declared_success_blocked(self):
        self.mutate("before_ack.json",
                    lambda d:d.update(old_transaction_outcome="APPLIED"))
        self.rejected()
    def test_09_replayed_original_crash_txn_blocked(self):
        self.mutate("first_bind.json",
                    lambda d:d.update(old_transaction_never_replayed=False))
        self.rejected()
    def test_10_incorrect_recovery_sequence_blocked(self):
        self.mutate("first_bind.json",
                    lambda d:d["fresh_recovery"][0].update(sequence_sha256="0"*64))
        self.rejected()
    def test_11_missing_second_BF16_readback(self):
        (self.root/"bf16_tile_reread.json").unlink()
        self.rejected()
    def test_12_bad_readback_tile_hash_rejected(self):
        self.mutate("bf16_tile_reread.json",
                    lambda d:d["per_tensor_recheck"][0].update(readback_tile_weight_sha256="f"*64))
        self.rejected()
    def test_13_false_BF16_full_teacher_claim_rejected(self):
        self.mutate("bf16_source_tiles.json",
                    lambda d:d.update(whole_BF16_model_forward_executed=True))
        self.rejected()
    def test_14_wrong_BF16_source_index_rejected(self):
        self.mutate("bf16_tile_reread.json",
                    lambda d:d.update(source_index_sha256="0"*64))
        self.rejected()
    def test_15_unwitnessed_earlier_attempt_falsified_rejected(self):
        self.mutate("first_bind_attempt1_unmarked.json",
                    lambda d:d.update(status="TRUE_INFLIGHT_WINDOW_WITNESSED_AND_QUARANTINED"))
        self.rejected()
    def test_16_duplicate_crashed_txn_rejected(self):
        source=(self.root/"first_bind.marker").read_text().split()[1]
        dest=self.root/"before_ack.marker"
        dest.write_text(f"before_ack {source}\n")
        self.rejected()

if __name__=="__main__":unittest.main()
