#!/usr/bin/env python3
"""P9 fault-only exact source/rollback fail-closed regression."""
import pathlib,hashlib,tempfile,unittest
from hcp_p9_fault_patch import patch,strip,P5_C128_SHA
from hcp_p5_scratch_capacity128 import patch as cb128
SOURCE=pathlib.Path(__file__).resolve().parents[1]/"qwen_infer.c"
class P9FaultTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.original=SOURCE.read_bytes()
  cls.p5=cb128(cls.original.decode()).encode()
  cls.failed=patch(cls.p5)
 def test_01_sealed_p5_source(self):
  self.assertEqual(hashlib.sha256(self.p5).hexdigest(),P5_C128_SHA)
 def test_02_source_exact_reversal(self):
  self.assertEqual(strip(self.failed),self.p5)
 def test_03_injected_in_middle_of_two_target_transaction(self):
  s=self.failed.decode()
  first=s.index("static int moe_gpu_rebind_set_apply_quiescent(")
  end=s.index("static int moe_gpu_demotion_pending(",first)
  body=s[first:end]
  self.assertEqual(body.count('fprintf(stderr, "P9_FAULT_AFTER_FIRST_BIND_ONCE'),1)
  self.assertIn("goto rollback; /* original snapshot restore",body)
  self.assertLess(body.index("P9_FAULT_AFTER_FIRST_BIND_ONCE"),
                  body.index("if (!mlx_gpu_reset_runtime_epoch())"))
 def test_04_unchanged_source_rejected(self):
  with self.assertRaises(ValueError):patch(self.original)
 def test_05_double_patch_rejected(self):
  with self.assertRaises(ValueError):patch(self.failed)
 def test_06_opt_in_once_only(self):
  s=self.failed.decode()
  self.assertIn('strcmp(f,"1")==0',s)
  self.assertIn("static int p9_fault_once = 0;",s)
  self.assertEqual(s.count("p9_fault_once=1;"),1)
 def test_07_stage_remains_original(self):
  self.assertEqual(SOURCE.read_bytes(),self.original)
 def test_08_rollback_and_terminal_ack_preserved(self):
  s=self.failed.decode()
  self.assertIn("mlx_gpu_restore_binding_snapshot(",s)
  self.assertIn("REBIND_SET_FAILED",s)
  self.assertIn("REBIND_SET_APPLIED",s)
if __name__=="__main__":unittest.main()
