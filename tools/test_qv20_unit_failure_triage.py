#!/usr/bin/env python3
import hashlib,unittest
import qv20_unit_failure_triage as t
PASS='✔ ok (10ms)\nℹ tests 1\nℹ pass 1\nℹ fail 0\nℹ cancelled 0\nℹ skipped 0\nℹ duration_ms 11\n'.encode()
FAIL='✖ failed (20ms)\nℹ tests 1\nℹ pass 0\nℹ fail 1\nℹ cancelled 0\nℹ skipped 0\nℹ duration_ms 23\n'.encode()
def receipt(data,state='PASS',code=0):
 return {'schema':'tc-development-test-receipt/1','profile':'unit','stdout_sha256':hashlib.sha256(data).hexdigest(),'state':state,'exit_code':code}
class Tests(unittest.TestCase):
 def test_empty(self):self.assertEqual(t.analyze(b'')['verdict'],'UNKNOWN_INCOMPLETE')
 def test_missing_summary(self):self.assertEqual(t.analyze('✔ ok\n'.encode())['verdict'],'UNKNOWN_INCOMPLETE')
 def test_unattested(self):self.assertEqual(t.analyze(PASS)['verdict'],'LOG_PASS_UNATTESTED')
 def test_verified_pass(self):self.assertEqual(t.analyze(PASS,receipt(PASS))['verdict'],'PASS')
 def test_fail(self):self.assertEqual(t.analyze(FAIL,receipt(FAIL,'FAIL',1))['verdict'],'FAIL')
 def test_bad_sha(self):self.assertEqual(t.analyze(PASS,receipt(FAIL))['verdict'],'EVIDENCE_MISMATCH')
 def test_cancelled(self):
  data=PASS.replace(b'cancelled 0',b'cancelled 1').replace(b'pass 1',b'pass 0')
  self.assertEqual(t.analyze(data)['verdict'],'FAIL')
 def test_bad_total(self):
  data=PASS.replace(b'tests 1',b'tests 2')
  self.assertEqual(t.analyze(data,receipt(data))['verdict'],'EVIDENCE_MISMATCH')
 def test_no_duration(self):
  data=PASS.replace('ℹ duration_ms 11\n'.encode(),b'')
  self.assertEqual(t.analyze(data)['verdict'],'UNKNOWN_INCOMPLETE')
 def test_failure_not_suppressed(self):
  data=FAIL.replace(b'fail 1',b'fail 0').replace(b'pass 0',b'pass 1')
  self.assertEqual(t.analyze(data,receipt(data))['verdict'],'EVIDENCE_MISMATCH')
if __name__=='__main__':unittest.main()
