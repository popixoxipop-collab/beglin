#!/usr/bin/env python3
import json,tempfile,unittest
from pathlib import Path
import precision_observability as po
from test_gpu_precision_observability import decision,result

class RotationTests(unittest.TestCase):
 def test_rotation_preserves_summary_and_duplicate_detection(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); obs=po.PrecisionObservability(lineage_path=root/'lineage.jsonl',max_active_records=3)
   prev_policy='1'*64; epoch=2
   for i in range(7):
    r=result(transitioned=False); r['precision_epoch'].update(before_policy_hash=prev_policy,after_policy_hash=prev_policy,before_epoch=epoch,after_epoch=epoch)
    obs.record(admission_id=f'a{i}',worker_pid=9,request_count=1,decision=decision(False),result=r)
   self.assertEqual(obs.summary()['admissions'],7)
   st=obs.rotation_status(); self.assertEqual(st['sealed_segments'],2); self.assertEqual(st['retained_segment_files'],2)
   with self.assertRaises(po.PrecisionObservabilityError):
    r=result(transitioned=False); r['precision_epoch'].update(before_policy_hash=prev_policy,after_policy_hash=prev_policy,before_epoch=epoch,after_epoch=epoch)
    obs.record(admission_id='a0',worker_pid=9,request_count=1,decision=decision(False),result=r)
   restarted=po.PrecisionObservability(lineage_path=root/'lineage.jsonl',max_active_records=3)
   self.assertEqual(restarted.summary(),obs.summary())

 def test_tampered_segment_fails_restart(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); obs=po.PrecisionObservability(lineage_path=root/'lineage.jsonl',max_active_records=2)
   for i in range(3):
    r=result(transitioned=False); r['precision_epoch'].update(before_policy_hash='1'*64,after_policy_hash='1'*64,before_epoch=2,after_epoch=2)
    obs.record(admission_id=f'a{i}',worker_pid=1,request_count=1,decision=decision(False),result=r)
   seg=next((root/'lineage.jsonl.segments').glob('segment-*.jsonl')); seg.write_bytes(seg.read_bytes()+b'{}\n')
   with self.assertRaises(po.PrecisionObservabilityError): po.PrecisionObservability(lineage_path=root/'lineage.jsonl',max_active_records=2)

if __name__=='__main__': unittest.main()
