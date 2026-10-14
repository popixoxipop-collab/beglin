#!/usr/bin/env python3
"""P12 independent on-disk prestart journal + local BF16 accuracy audit tamper fixtures."""
from __future__ import annotations
import json,pathlib,shutil,tempfile,unittest
from hcp_p12_evidence_audit import audit
REPO=pathlib.Path(__file__).resolve().parents[1]
SOURCE=REPO/"experiments/hcp_p12/evidence/native_xox_20261014"
QUALITY=REPO/"experiments/hcp_p6/evidence/quality.json"

class P12NegativeAudits(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory()
  self.dir=pathlib.Path(self.tmp.name)/"archive"
  shutil.copytree(SOURCE,self.dir)
  (self.dir/"MANIFEST.json").unlink(missing_ok=True)
 def tearDown(self):self.tmp.cleanup()
 def check(self):return audit(self.dir,REPO,QUALITY)
 def reject(self):
  with self.assertRaises((ValueError,RuntimeError,KeyError,AssertionError,TypeError)):
   self.check()
 def mutate(self,p,edit):
  f=self.dir/p
  j=json.loads(f.read_text())
  edit(j)
  f.write_text(json.dumps(j,sort_keys=True)+"\n")
 def folders(self):
  return sorted((self.dir/"p12_journal").iterdir())
 def test_01_untampered_double_source_passes(self):
  d=self.check()
  self.assertEqual(d["native_terminal_transactions_verified"],2)
  self.assertEqual(d["new_generated_sequences_verified"],4)
  self.assertFalse(d["promotion_allowed"])
 def test_02_corrupt_terminal_command_refused(self):
  (self.folders()[0]/"original.cmd").write_text("REBIND_SET 0")
  self.reject()
 def test_03_terminal_ack_tampered_refused(self):
  self.mutate(self.folders()[1].relative_to(self.dir)/"original_ack.json",
    lambda d:d.update(weight_epoch=10))
  self.reject()
 def test_04_partial_intent_fsync_refused(self):
  (self.folders()[0]/"INTENT.json").unlink()
  self.reject()
 def test_05_deleted_verified_receipt_refused(self):
  (self.folders()[1]/"VERIFIED.json").unlink()
  self.reject()
 def test_06_forged_committed_status_refused(self):
  self.mutate(self.folders()[1].relative_to(self.dir)/"VERIFIED.json",
    lambda d:d.update(decision="TERMINAL_FAILED"))
  self.reject()
 def test_07_native_generation_divergence_refused(self):
  self.mutate("P12_native_prestart.json",
    lambda d:d["responses"][2].update(seq_sha256="f"*64))
  self.reject()
 def test_08_native_PID_reuse_refused(self):
  self.mutate("P12_native_prestart.json",
    lambda d:d["responses"][1].update(PID=d["responses"][0]["PID"]))
  self.reject()
 def test_09_native_source_hash_refused(self):
  self.mutate("P12_native_prestart.json",
    lambda d:d.update(p9_mlx_source_sha256="0"*64))
  self.reject()
 def test_10_native_ack_epoch_invalid_refused(self):
  self.mutate(self.folders()[0].relative_to(self.dir)/"original_ack.json",
    lambda d:d.update(changed_targets=1))
  self.reject()
 def test_11_original_full_teacher_claim_refused(self):
  self.mutate("p12_local_BF16_oracle.json",
    lambda d:d.update(full_BF16_teacher_forward_executed=True))
  self.reject()
 def test_12_fake_task_accuracy_claim_refused(self):
  self.mutate("p12_local_BF16_oracle.json",
    lambda d:d.update(task_accuracy_noninferiority_proven=True))
  self.reject()
 def test_13_source_quantizer_changed_refused(self):
  self.mutate("p12_local_BF16_oracle.json",
    lambda d:d.update(C_compiled_library_sha256="f"*64))
  self.reject()
 def test_14_native_C_unpack_disagreement_refused(self):
  self.mutate("p12_local_BF16_oracle.json",
    lambda d:d["records"][0]["quantization_bits"]["5"].update(
    native_C_vs_independent_unpack_projection_max_abs=999.0))
  self.reject()
 def test_15_wrong_BF16_n_not_in_5_6_refused(self):
  self.mutate("p12_local_BF16_oracle.json",
    lambda d:d["records"][1]["quantization_bits"]["5"].update(bits=7))
  self.reject()
 def test_16_wrong_model_checkpoint_identity_refused(self):
  self.mutate("p12_local_BF16_oracle.json",
    lambda d:d.update(model_index_sha256="0"*64))
  self.reject()
 def test_17_wrong_owner_binary_identity_refused(self):
  self.mutate("P12_OWNER_IDENTITY.json",
    lambda d:d.update(P9_binary_sha256="e"*64))
  self.reject()
 def test_18_preflight_production_route_changed_refused(self):
  self.mutate("preflight.json",
    lambda d:d.update(original_route_sha256="a"*64))
  self.reject()
 def test_19_duplicate_missing_archive_refused(self):
  shutil.rmtree(self.folders()[0])
  self.reject()
 def test_20_wrong_native_delta_pair_refused(self):
  self.mutate("P12_native_prestart.json",
    lambda d:d.update(verified_receipt_sha256=[]))
  self.reject()

if __name__=="__main__":unittest.main()
