#!/usr/bin/env python3
"""P11 source reversal and fail-closed crash-ambiguous quarantine fixtures."""
from __future__ import annotations
import pathlib,hashlib,json,tempfile,unittest,sys
from hcp_p11_inflight_patch import patch,strip,C128_SHA,MARK
from hcp_p5_scratch_capacity128 import patch as to128
from hcp_p11_quarantine import inspect,quarantine,verify_quarantine
import precision_context as pc

SOURCE=pathlib.Path(__file__).resolve().parents[1]/"qwen_infer.c"
class SourceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cb128=to128(SOURCE.read_text()).encode()
  cls.modified=patch(cls.cb128)
 def test_01_pin_sealed_source(self):
  self.assertEqual(hashlib.sha256(self.cb128).hexdigest(),C128_SHA)
 def test_02_source_reversible(self):
  self.assertEqual(strip(self.modified),self.cb128)
 def test_03_double_patch_blocked(self):
  with self.assertRaises(ValueError):patch(self.modified)
 def test_04_unsealed_original_blocked(self):
  with self.assertRaises(ValueError):patch(SOURCE.read_bytes())
 def test_05_two_crash_windows_inside_rebind_set(self):
  text=self.modified.decode()
  s=text.index("static int moe_gpu_rebind_set_apply_quiescent")
  e=text.index("static int moe_gpu_demotion_pending",s)
  body=text[s:e]
  self.assertIn('p11_test_gate(cmd, "first_bind")',body)
  self.assertIn('p11_test_gate(cmd, "before_ack")',body)
  self.assertLess(body.index("bound_count++"),body.index('p11_test_gate(cmd, "first_bind")'))
  self.assertLess(body.index('p11_test_gate(cmd, "before_ack")'),body.index('"REBIND_SET_APPLIED"'))
 def test_06_bounded_test_gate_failclosed(self):
  text=self.modified.decode()
  self.assertIn("sleep(15)",text)
  self.assertIn("_exit(92)",text)
  self.assertIn("O_EXCL|O_NOFOLLOW",text)
  self.assertIn('strstr(file,"/hcp-p11-inflight-")',text)
 def test_07_production_math_and_rollback_unchanged(self):
  self.assertEqual(strip(self.modified),self.cb128)
  self.assertIn("mlx_gpu_restore_binding_snapshot(",self.modified.decode())

class ReconcileTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory()
  base=pathlib.Path(self.tmp.name)/".tailnet-tmp"
  base.mkdir()
  self.root=base/"hcp-p11-inflight-unit"
  self.worker=self.root/"workers"/"test-0"
  self.worker.mkdir(parents=True)
  self.txn="epoch-"+"a"*32
  self.marker=self.root/"first_bind.marker"
  self.hash=pc.policy_hash([
   {"role":"shared_down_proj","layer":26,"n":5},
   {"role":"shared_up_proj","layer":3,"n":6}
  ])
  self.ack={
   "schema":"gpu-precision-applied-v1","backend":"mlx_metal","correction_mode":"off",
   "status":"PROMOTION_APPLIED","weight_epoch":2,
   "active_policy":[
    {"role":"shared_down_proj","layer":26,"n":5},
    {"role":"shared_up_proj","layer":3,"n":6}],
   "qng64_cache":[]
  }
  self.marker.write_text(f"first_bind {self.txn}\n")
  self.cmd=f"REBIND_SET {self.txn} 2 2 {self.hash} shared_down_proj 26 5 6 shared_up_proj 3 6 5\n"
  (self.worker/"txn.cmd").write_text(self.cmd)
  (self.worker/"applied_ack.json").write_text(json.dumps(self.ack)+"\n")
 def tearDown(self):self.tmp.cleanup()
 def test_08_live_worker_quarantine_forbidden(self):
  with self.assertRaises(RuntimeError):quarantine(self.root,self.worker,self.marker,pid_is_dead=False)
 def test_09_real_ambiguous_preimage_inspection(self):
  p=inspect(self.root,self.worker,self.marker,pid_is_dead=True)
  self.assertFalse(p["terminal_ACK_for_interrupted_transaction"])
  self.assertEqual(p["expected_epoch"],2)
  self.assertEqual(p["preimage_policy_hash"],self.hash)
 def test_10_atomic_quarantine_retains_unreplayed_command(self):
  q=quarantine(self.root,self.worker,self.marker,pid_is_dead=True)
  self.assertEqual(q["status"],"CRASH_AMBIGUOUS_UNCOMMITTED_TRANSACTION_QUARANTINED")
  self.assertFalse((self.worker/"txn.cmd").exists())
  self.assertEqual(verify_quarantine(self.worker)["txn_id"],self.txn)
  self.assertEqual(pathlib.Path(q["quarantine_cmd_path"]).read_text(),self.cmd)
 def test_11_duplicate_quarantine_blocked(self):
  quarantine(self.root,self.worker,self.marker,pid_is_dead=True)
  with self.assertRaises(RuntimeError):quarantine(self.root,self.worker,self.marker,pid_is_dead=True)
 def test_12_missing_verified_receipt_failclosed(self):
  q=quarantine(self.root,self.worker,self.marker,pid_is_dead=True)
  (self.worker/"quarantine/VERIFIED.json").unlink()
  with self.assertRaises(RuntimeError):verify_quarantine(self.worker)
 def test_13_durable_command_hash_corruption_rejected(self):
  q=quarantine(self.root,self.worker,self.marker,pid_is_dead=True)
  pathlib.Path(q["quarantine_cmd_path"]).write_text("DELETED")
  with self.assertRaises(RuntimeError):verify_quarantine(self.worker)
 def test_14_marker_wrong_transaction_rejected(self):
  self.marker.write_text(f"first_bind epoch-{'b'*32}\n")
  with self.assertRaises(ValueError):inspect(self.root,self.worker,self.marker,pid_is_dead=True)
 def test_15_stale_ACK_wrong_epoch_rejected(self):
  self.ack["weight_epoch"]=3
  (self.worker/"applied_ack.json").write_text(json.dumps(self.ack)+"\n")
  with self.assertRaises(ValueError):inspect(self.root,self.worker,self.marker,pid_is_dead=True)
 def test_16_invalid_policy_transition_rejected(self):
  (self.worker/"txn.cmd").write_text(self.cmd.replace("shared_up_proj 3 6 5","shared_up_proj 3 5 6"))
  with self.assertRaises(ValueError):inspect(self.root,self.worker,self.marker,pid_is_dead=True)
 def test_17_foreign_workspace_rejected(self):
  outside=pathlib.Path(self.tmp.name)/"other"
  outside.mkdir()
  with self.assertRaises(ValueError):inspect(outside,self.worker,self.marker,pid_is_dead=True)
 def test_18_tampered_final_receipt_rejected(self):
  quarantine(self.root,self.worker,self.marker,pid_is_dead=True)
  p=self.worker/"quarantine/VERIFIED.json"
  data=json.loads(p.read_text());data["txn_id"]="epoch-"+("c"*32)
  p.write_text(json.dumps(data)+"\n")
  with self.assertRaises(RuntimeError):verify_quarantine(self.worker)

if __name__=="__main__":unittest.main()
