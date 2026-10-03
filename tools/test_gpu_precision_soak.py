#!/usr/bin/env python3
import json, tempfile, unittest
from pathlib import Path
import precision_observability as po
import precision_soak as ps

class SoakTests(unittest.TestCase):
    def test_epoch_sequence_and_drift(self):
        rows=[{"pid":7,"before_epoch":i,"after_epoch":i+1} for i in range(5)]
        ps.verify_epoch_sequence(rows)
        rows[3]["before_epoch"]=99
        with self.assertRaises(ps.PrecisionSoakError): ps.verify_epoch_sequence(rows)

    def test_cache_plateau(self):
        rows=[{"cache_misses":1,"cache_bytes_added":100} for _ in range(4)]
        rows += [{"cache_misses":0,"cache_bytes_added":0} for _ in range(10)]
        ps.verify_cache_plateau(rows,4)
        rows[-1]["cache_misses"]=1
        with self.assertRaises(ps.PrecisionSoakError): ps.verify_cache_plateau(rows,4)

    def test_rss_drift(self):
        self.assertEqual(ps.rss_drift([100,120,110]),{"start":100,"end":110,"peak":120,"delta":10})

    def test_lineage_corruption_detected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"x.jsonl"
            p.write_text(json.dumps({"schema":po.SCHEMA,"admission_id":"x","prev_record_sha256":None,"record_sha256":"bad"})+"\n")
            self.assertTrue(ps.verify_lineage_failure(p))

    def test_rotation_retention(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"lineage.jsonl"
            obs=po.PrecisionObservability(lineage_path=p,max_bytes=900,max_rotated_files=2)
            decision={"signal":{"active_triggers":[]},"selected_policy":[],"changes":[],"selection":{"targets":[]}}
            for i in range(8):
                result={"finite_logits":True,"responses":[[i]],"precision_epoch":{
                    "before_policy_hash":"a"*64,"after_policy_hash":"a"*64,
                    "before_epoch":0,"after_epoch":0,"transitioned":False,
                    "changed_targets":[],"transition_cost":{},"inference_passes":1,
                    "engine_wall_ms":1.0,"roundtrip_ms":2.0}}
                obs.record(admission_id=f"a{i}",worker_pid=7,request_count=1,decision=decision,result=result)
            files=ps.assert_retention(p,2)
            self.assertLessEqual(len(files),2)
            po.verify_records(obs.read_records())

if __name__=="__main__": unittest.main()
