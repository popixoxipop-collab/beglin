#!/usr/bin/env python3
"""P12 run-owned restart preflight journal fail-closed unit/regression."""
from __future__ import annotations
import json,os,pathlib,subprocess,sys,tempfile,unittest
import gpu_runtime_control as grc
import precision_context as pc
from hcp_p12_prestart_journal import (
    PrestartLease,parse_txn,verify_archive_folder,POLICY_A,POLICY_B,
    SOURCE_POLICY_HASH,TARGET_POLICY_HASH,TXN
)
from hcp_p11_quarantine import quarantine,verify_quarantine

class PrestartTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        base=pathlib.Path(self.tmp.name).resolve()/".tailnet-tmp"
        base.mkdir()
        self.root=base/"hcp-p11-inflight-p12-tests"
        self.worker=self.root/"workers"/"p12-candidate-0"
        self.worker.mkdir(parents=True)
        self.trans="epoch-"+("b"*32)
        self.cmd=(f"REBIND_SET {self.trans} 2 2 {SOURCE_POLICY_HASH} "
                  "shared_down_proj 26 5 6 shared_up_proj 3 6 5\n")
        with PrestartLease(self.root,self.worker) as owner:
            self.assertTrue(owner.inspect_for_start()["ready_for_fresh_start"])
    def tearDown(self):self.tmp.cleanup()
    def with_command(self,status,*,txn_id=None,epoch=None,policy=None,changed=None):
        (self.worker/"txn.cmd").write_text(self.cmd)
        desired=POLICY_B if status=="REBIND_SET_APPLIED" else POLICY_A
        ack={
            "schema":"gpu-precision-applied-v1","backend":"mlx_metal","correction_mode":"off",
            "status":status,"weight_epoch":3 if status=="REBIND_SET_APPLIED" else 2,
            "changed_targets":2 if status=="REBIND_SET_APPLIED" else 0,
            "active_policy":desired,"qng64_cache":[],
            "txn_id":self.trans,"expected_epoch":2,
            "expected_policy_hash":SOURCE_POLICY_HASH
        }
        if txn_id is not None:ack["txn_id"]=txn_id
        if epoch is not None:ack["weight_epoch"]=epoch
        if policy is not None:ack["active_policy"]=policy
        if changed is not None:ack["changed_targets"]=changed
        (self.worker/"applied_ack.json").write_text(json.dumps(ack,sort_keys=True)+"\n")
    def owner(self):return PrestartLease(self.root,self.worker)
    def blocked(self):
        with self.assertRaises((RuntimeError,ValueError)):
            with self.owner() as guard:guard.inspect_for_start()
    def test_01_fresh_start_allowed_and_owned(self):
        with self.owner() as guard:
            d=guard.inspect_for_start()
            self.assertEqual(d["state"],"CLEAN_FRESH_START")
            self.assertTrue(d["no_auto_replay"])
    def test_02_valid_terminal_applied_persisted_before_boot(self):
        self.with_command("REBIND_SET_APPLIED")
        with self.owner() as guard:
            d=guard.inspect_for_start()
            self.assertEqual(d["state"],"KNOWN_TERMINAL_JOURNALED")
            self.assertFalse((self.worker/"txn.cmd").exists())
            folder=self.worker/"p12_journal"/self.trans
            self.assertTrue(verify_archive_folder(folder)["no_auto_replay"])
    def test_03_valid_terminal_failed_persisted_not_replayed(self):
        self.with_command("REBIND_SET_FAILED")
        with self.owner() as guard:
            self.assertEqual(guard.inspect_for_start()["terminal"],"REBIND_SET_FAILED")
            f=verify_archive_folder(self.worker/"p12_journal"/self.trans)
            self.assertEqual(f["decision"],"TERMINAL_FAILED")
    def test_04_idempotent_crash_after_journal_before_start(self):
        self.with_command("REBIND_SET_APPLIED")
        with self.owner() as guard:guard.inspect_for_start()
        with self.owner() as guard:
            d=guard.inspect_for_start()
            self.assertEqual(d["state"],"CLEAN_FRESH_START")
            self.assertFalse((self.worker/"txn.cmd").exists())
    def test_05_unknown_no_terminal_ack_blocked(self):
        self.with_command("PROMOTION_APPLIED",txn_id=None)
        obj=json.loads((self.worker/"applied_ack.json").read_text())
        obj["txn_id"]=None
        (self.worker/"applied_ack.json").write_text(json.dumps(obj)+"\n")
        self.blocked()
    def test_06_unknown_can_be_quarantined_by_p11_then_new_start_allowed(self):
        self.with_command("PROMOTION_APPLIED")
        obj=json.loads((self.worker/"applied_ack.json").read_text())
        obj["txn_id"]=None
        (self.worker/"applied_ack.json").write_text(json.dumps(obj)+"\n")
        marker=self.root/"first_bind.marker"
        marker.write_text(f"first_bind {self.trans}\n")
        self.blocked()
        with self.owner() as guard:
            q=quarantine(self.root,self.worker,marker,pid_is_dead=True)
            self.assertFalse((self.worker/"txn.cmd").exists())
            self.assertTrue(verify_quarantine(self.worker)["no_automatic_retry"])
            d=guard.inspect_for_start()
            self.assertEqual(d["state"],"QUARANTINED_UNKNOWN_FRESH_START")
            self.assertTrue(d["prior_unknown_original_outcome"])
    def test_07_wrong_epoch_applied_blocked(self):
        self.with_command("REBIND_SET_APPLIED",epoch=2)
        self.blocked()
    def test_08_wrong_result_policy_blocked(self):
        self.with_command("REBIND_SET_APPLIED",policy=POLICY_A)
        self.blocked()
    def test_09_wrong_target_count_blocked(self):
        self.with_command("REBIND_SET_APPLIED",changed=1)
        self.blocked()
    def test_10_wrong_transaction_id_blocked(self):
        self.with_command("REBIND_SET_APPLIED",txn_id="epoch-"+("c"*32))
        self.blocked()
    def test_11_source_policy_forgery_blocked(self):
        self.with_command("REBIND_SET_APPLIED")
        p=self.worker/"txn.cmd"
        p.write_text(self.cmd.replace(SOURCE_POLICY_HASH,"a"*64))
        self.blocked()
    def test_12_wrong_target_bitwidth_blocked(self):
        self.with_command("REBIND_SET_APPLIED")
        p=self.worker/"txn.cmd"
        p.write_text(self.cmd.replace("shared_up_proj 3 6 5","shared_up_proj 3 6 7"))
        self.blocked()
    def test_13_partial_archive_blocks_restart(self):
        self.with_command("REBIND_SET_APPLIED")
        with self.owner() as g:g.inspect_for_start()
        folder=self.worker/"p12_journal"/self.trans
        (folder/"VERIFIED.json").unlink()
        self.blocked()
    def test_14_archive_tamper_blocks_restart(self):
        self.with_command("REBIND_SET_APPLIED")
        with self.owner() as g:g.inspect_for_start()
        folder=self.worker/"p12_journal"/self.trans
        (folder/"original.cmd").write_text("REPLACED")
        self.blocked()
    def test_15_second_controller_lock_denied_in_separate_process(self):
        with self.owner() as guard:
            script=("from hcp_p12_prestart_journal import PrestartLease\n"
                    "import sys\n"
                    "try:\n"
                    "  with PrestartLease(sys.argv[1],sys.argv[2]): pass\n"
                    "except RuntimeError: print('BLOCKED'); sys.exit(0)\n"
                    "sys.exit(3)\n")
            env={**os.environ,"PYTHONPATH":str(pathlib.Path(__file__).resolve().parent)}
            x=subprocess.run([sys.executable,"-c",script,str(self.root),str(self.worker)],
                             env=env,capture_output=True,text=True,timeout=5)
            self.assertEqual(x.returncode,0,x.stderr)
            self.assertIn("BLOCKED",x.stdout)
    def test_16_identity_mismatch_refused(self):
        with self.assertRaises(ValueError):
            PrestartLease(self.root,self.worker,model_sha="0"*64)
    def test_17_foreign_worker_root_refused(self):
        foreign=pathlib.Path(self.tmp.name)/"forbidden"
        foreign.mkdir()
        with self.assertRaises(ValueError):
            PrestartLease(self.root,foreign)
    def test_18_owner_identity_tamper_refused(self):
        meta=self.root/"P12_OWNER_IDENTITY.json"
        d=json.loads(meta.read_text());d["model_index_sha256"]="0"*64
        meta.write_text(json.dumps(d)+"\n")
        with self.assertRaises(RuntimeError):
            with self.owner():pass
    def test_19_unaccompanied_ack_terminal_refused(self):
        (self.worker/"applied_ack.json").write_text(json.dumps({
          "schema":"gpu-precision-applied-v1","backend":"mlx_metal","correction_mode":"off",
          "status":"REBIND_SET_APPLIED","weight_epoch":3,"changed_targets":2,
          "active_policy":POLICY_B,"qng64_cache":[],"txn_id":self.trans})+"\n")
        self.blocked()
    def test_20_archive_unacknowledged_when_unmatched_policy_denied(self):
        self.with_command("REBIND_SET_FAILED",policy=POLICY_B)
        self.blocked()
    def test_21_bad_ack_corrupted_json_denied(self):
        self.with_command("REBIND_SET_APPLIED")
        (self.worker/"applied_ack.json").write_text("{")
        self.blocked()
    def test_22_duplicate_terminal_receipt_archive_not_overwritten(self):
        self.with_command("REBIND_SET_APPLIED")
        with self.owner() as guard:guard.inspect_for_start()
        (self.worker/"txn.cmd").write_text(self.cmd)
        self.blocked()
    def test_23_journal_symlink_refused(self):
        symlink=self.worker/"p12_journal"
        target=pathlib.Path(self.tmp.name)/"foreign"
        target.mkdir();symlink.symlink_to(target,target_is_directory=True)
        self.blocked()
    def test_24_two_owner_lifetime_can_be_serial(self):
        with self.owner() as guard:guard.inspect_for_start()
        with self.owner() as guard:guard.inspect_for_start()

if __name__=="__main__":unittest.main()
