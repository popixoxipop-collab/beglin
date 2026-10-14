#!/usr/bin/env python3
"""P11 independent crash-proof and partial BF16 oracle tamper rejection tests."""
from __future__ import annotations
import json,pathlib,shutil,tempfile,unittest
from hcp_p11_evidence_audit import audit

REPO=pathlib.Path(__file__).resolve().parents[1]
SRC=REPO/"experiments/hcp_p11/evidence/native_xox_20261014"
QUALITY=REPO/"experiments/hcp_p6/evidence/quality.json"
class P11EvidenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory()
  self.root=pathlib.Path(self.tmp.name)/"sample"
  shutil.copytree(SRC,self.root)
 def tearDown(self):self.tmp.cleanup()
 def mutate(self,file,fn):
  p=self.root/file
  row=json.loads(p.read_text())
  fn(row)
  p.write_text(json.dumps(row,sort_keys=True)+"\n")
 def blocked(self):
  with self.assertRaises((ValueError,KeyError,AssertionError,TypeError)):
   audit(self.root,QUALITY)
 def test_01_frozen_sha_proof_passes(self):
  a=audit(self.root,QUALITY)
  self.assertEqual(a["unique_ambiguous_transaction_count"],2)
  self.assertEqual(a["total_real_recovered_inferences"],6)
  self.assertFalse(a["production_promotion_allowed"])
 def test_02_missing_crash_marker_refused(self):
  (self.root/"first_bind.marker").unlink()
  self.blocked()
 def test_03_replaced_crash_command_refused(self):
  (self.root/"before_ack_quarantined_transaction.cmd").write_text("wrong")
  self.blocked()
 def test_04_false_committed_outcome_refused(self):
  self.mutate("before_ack.json",lambda j:j.update(old_transaction_outcome="COMMITTED"))
  self.blocked()
 def test_05_recovered_tokens_mismatch_refused(self):
  self.mutate("first_bind.json",lambda j:j["fresh_recovery"][1].update(sequence_sha256="f"*64))
  self.blocked()
 def test_06_missing_BF16_independent_reread_refused(self):
  (self.root/"bf16_tile_reread.json").unlink()
  self.blocked()
 def test_07_fake_full_teacher_claim_refused(self):
  self.mutate("bf16_source_tiles.json",lambda j:j.update(whole_BF16_model_forward_executed=True))
  self.blocked()
 def test_08_wrong_BF16_reread_sample_refused(self):
  self.mutate("bf16_tile_reread.json",lambda j:j["per_tensor_recheck"][0].update(readback_tile_weight_sha256="f"*64))
  self.blocked()
 def test_09_early_unmarked_is_not_crash_proof(self):
  self.mutate("first_bind_attempt1_unmarked.json",lambda j:j.update(status="PASS"))
  self.blocked()
 def test_10_wrong_first_policy_before_ACK_refused(self):
  self.mutate("first_bind.json",lambda j:j["quarantine_receipt"].update(preimage_policy_hash="0"*64))
  self.blocked()
 def test_11_wrong_pretend_native_ACK_refused(self):
  self.mutate("before_ack.json",lambda j:j.update(native_old_terminal_ack_for_txn=True))
  self.blocked()
 def test_12_wrong_BF16_dtype_refused(self):
  self.mutate("bf16_source_tiles.json",lambda j:j["samples"][0].update(source_dtype="float16"))
  self.blocked()

if __name__=="__main__":unittest.main()
