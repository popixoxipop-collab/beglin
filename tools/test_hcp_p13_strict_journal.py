#!/usr/bin/env python3
"""P13 archived real native terminal journal tamper/semantic defense regression.

Using actual P12 two native REBIND_SET ACK journal folders, including attacks
that recompute INTENT and VERIFIED SHA after altering old ACK (so the earlier
P12 hash-only check can pass). P13 must reject incorrect semantic state.
"""
from __future__ import annotations
import hashlib,json,pathlib,shutil,tempfile,unittest
from hcp_p12_prestart_journal import verify_archive_folder
from hcp_p13_strict_journal import verify_strict_terminal

REPO=pathlib.Path(__file__).resolve().parents[1]
ORIGINAL=REPO/"experiments/hcp_p12/evidence/native_xox_20261014/p12_journal"

class P13SemanticsTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.root=pathlib.Path(self.t.name)/"p13"
        shutil.copytree(ORIGINAL,self.root)
        self.folder=sorted(self.root.iterdir())[0]
    def tearDown(self):self.t.cleanup()
    def mutate_ack(self,mutator):
        ackfile=self.folder/"original_ack.json"
        obj=json.loads(ackfile.read_text())
        mutator(obj)
        ackfile.write_text(json.dumps(obj,sort_keys=True)+"\n")
        digest=hashlib.sha256(ackfile.read_bytes()).hexdigest()
        for name in ("INTENT.json","VERIFIED.json"):
            p=self.folder/name
            doc=json.loads(p.read_text())
            doc["ack_sha256"]=digest
            p.write_text(json.dumps(doc,sort_keys=True)+"\n")
    def strict_refuse_even_with_forged_hashes(self):
        # The earlier P12 archive verifier only checks digest consistency.
        verify_archive_folder(self.folder)
        with self.assertRaises((ValueError,RuntimeError,KeyError,AssertionError)):
            verify_strict_terminal(self.folder)
    def test_01_original_real_native_terminals_pass(self):
        result=[verify_strict_terminal(d) for d in self.root.iterdir()]
        self.assertEqual(len(result),2)
        self.assertTrue(all(v["two_targets_verified"] for v in result))
    def test_02_ack_epoch_forgery_with_rehashed_receipts_rejected(self):
        self.mutate_ack(lambda a:a.update(weight_epoch=999))
        self.strict_refuse_even_with_forged_hashes()
    def test_03_ack_result_policy_forgery_with_rehashed_receipts_rejected(self):
        self.mutate_ack(lambda a:a.update(active_policy=[
          {"role":"shared_up_proj","layer":3,"n":6},
          {"role":"shared_down_proj","layer":26,"n":5}]))
        self.strict_refuse_even_with_forged_hashes()
    def test_04_ack_preimage_policy_forgery_with_rehashed_receipts_rejected(self):
        self.mutate_ack(lambda a:a.update(expected_policy_hash="f"*64))
        self.strict_refuse_even_with_forged_hashes()
    def test_05_wrong_second_target_bitwidth_with_rehashed_receipts_rejected(self):
        self.mutate_ack(lambda a:a["txn_targets"][1].update(target_n=7))
        self.strict_refuse_even_with_forged_hashes()
    def test_06_wrong_changed_targets_count_with_rehashed_receipts_rejected(self):
        self.mutate_ack(lambda a:a.update(changed_targets=1))
        self.strict_refuse_even_with_forged_hashes()
    def test_07_wrong_native_transaction_id_with_rehashed_receipts_rejected(self):
        self.mutate_ack(lambda a:a.update(txn_id="epoch-"+("d"*32)))
        self.strict_refuse_even_with_forged_hashes()
    def test_08_wrong_native_expected_epoch_with_rehashed_receipts_rejected(self):
        self.mutate_ack(lambda a:a.update(expected_epoch=110))
        self.strict_refuse_even_with_forged_hashes()
    def test_09_intent_status_forgery_even_when_SHA_untouched_rejected(self):
        f=self.folder/"INTENT.json"
        d=json.loads(f.read_text());d["terminal_status"]="REBIND_SET_FAILED"
        f.write_text(json.dumps(d,sort_keys=True)+"\n")
        verify_archive_folder(self.folder)
        with self.assertRaises((ValueError,RuntimeError)):
            verify_strict_terminal(self.folder)
    def test_10_wrong_src_epoch_in_intent_even_when_SHA_untouched_rejected(self):
        f=self.folder/"INTENT.json"
        d=json.loads(f.read_text());d["source_epoch"]+=10
        f.write_text(json.dumps(d,sort_keys=True)+"\n")
        verify_archive_folder(self.folder)
        with self.assertRaises((ValueError,RuntimeError)):
            verify_strict_terminal(self.folder)
    def test_11_missing_ACK_denied(self):
        (self.folder/"original_ack.json").unlink()
        with self.assertRaises((ValueError,RuntimeError)):
            verify_strict_terminal(self.folder)
    def test_12_archive_ACK_symlink_denied(self):
        p=self.folder/"original_ack.json"
        backup=self.folder/"forbidden"
        p.rename(backup)
        p.symlink_to(backup)
        with self.assertRaises((ValueError,RuntimeError)):
            verify_strict_terminal(self.folder)

if __name__=="__main__":unittest.main()
