#!/usr/bin/env python3
import os,tempfile,unittest
from pathlib import Path
from unittest import mock
import precision_observability as po
from test_gpu_precision_observability import decision,result

def stable():
 r=result(transitioned=False); r['precision_epoch'].update(before_policy_hash='1'*64,after_policy_hash='1'*64,before_epoch=2,after_epoch=2); return r

class P7FaultTests(unittest.TestCase):
 def test_manifest_write_failure_keeps_active_journal(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); o=po.PrecisionObservability(lineage_path=root/'l.jsonl',max_active_records=2)
   for i in range(2): o.record(admission_id=f'a{i}',worker_pid=1,request_count=1,decision=decision(False),result=stable())
   before=(root/'l.jsonl').read_bytes()
   real_open=Path.open
   def bad_open(path,*a,**kw):
    if str(path).endswith('manifest.jsonl') and a and 'a' in a[0]: raise OSError('fault-manifest')
    return real_open(path,*a,**kw)
   with mock.patch.object(Path,'open',bad_open):
    with self.assertRaises(OSError): o.record(admission_id='a2',worker_pid=1,request_count=1,decision=decision(False),result=stable())
   self.assertEqual((root/'l.jsonl').read_bytes(),before)
   self.assertEqual(o.summary()['admissions'],2)

 def test_active_reset_failure_does_not_lose_sealed_segment(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); o=po.PrecisionObservability(lineage_path=root/'l.jsonl',max_active_records=2)
   for i in range(2): o.record(admission_id=f'a{i}',worker_pid=1,request_count=1,decision=decision(False),result=stable())
   real_replace=os.replace
   def bad_replace(src,dst):
    if str(dst).endswith('l.jsonl'): raise OSError('fault-reset')
    return real_replace(src,dst)
   with mock.patch('precision_observability.os.replace',bad_replace):
    with self.assertRaises(OSError): o.record(admission_id='a2',worker_pid=1,request_count=1,decision=decision(False),result=stable())
   rows=o._segment_manifest_rows(); self.assertEqual(len(rows),1)
   self.assertTrue((o.archive_dir/rows[0]['segment_file']).is_file())
   self.assertEqual(o.summary()['admissions'],2)

 def test_tampered_manifest_fails_closed(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); o=po.PrecisionObservability(lineage_path=root/'l.jsonl',max_active_records=2)
   for i in range(3): o.record(admission_id=f'a{i}',worker_pid=1,request_count=1,decision=decision(False),result=stable())
   raw=o.manifest_path.read_text(); o.manifest_path.write_text(raw.replace('segment_file','segment_fyle',1))
   with self.assertRaises(po.PrecisionObservabilityError): po.PrecisionObservability(lineage_path=root/'l.jsonl',max_active_records=2)

if __name__=='__main__': unittest.main()
