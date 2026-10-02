#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import persistent_gpu_worker as p


class PersistentProtocolTests(unittest.TestCase):
    def test_line_protocol_reuses_one_process_for_two_requests(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            fakebin=root/"fake-qwen"
            fakebin.write_text("#!/bin/sh\n")
            manifest=root/"manifest.txt"
            manifest.write_text("/tmp/fake.i32 10\n")

            child_code = r'''
import json,sys
idx=0
for line in sys.stdin:
    line=line.strip()
    if line=="QUIT":
        break
    rid,path=line.split()
    print("PERSIST_RESPONSE "+json.dumps({
        "schema":"beglin-gpu-persistent-response-v1",
        "status":"OK",
        "request_id":rid,
        "batch_index":idx,
        "requests":[[1,2,3]],
        "wall_ms":10.0,
        "finite_logits":True,
        "weight_epoch":1,
    }),flush=True)
    idx+=1
'''
            def factory(*args,**kwargs):
                return subprocess.Popen(
                    [sys.executable,"-u","-c",child_code],
                    text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,bufsize=1,
                )

            worker=p.PersistentGpuWorker(
                binary=fakebin,
                cwd=root,
                base_env={},
                route={
                    "route_id":"candidate",
                    "worker_instance_id":"persistent",
                    "endpoint":"local://candidate",
                    "source_commit":"a"*40,
                    "binary_sha256":"b"*64,
                    "checkpoint_sha256":"c"*64,
                    "policy_hash":"d"*64,
                    "role":"shared_up_proj","layer":3,"n":6,
                },
                generation=7,
                response_timeout_seconds=5,
                popen_factory=factory,
            )
            worker.start()
            pid=worker.pid
            first=worker.request(manifest_path=manifest,request_id="r1")
            second=worker.request(manifest_path=manifest,request_id="r2")
            self.assertEqual(worker.pid,pid)
            self.assertEqual(first["batch_index"],0)
            self.assertEqual(second["batch_index"],1)
            worker.close()


class ManagerTests(unittest.TestCase):
    def test_same_generation_reuses_and_new_generation_restarts(self):
        created=[]
        class FakeWorker:
            def __init__(self,**kwargs):
                self.kwargs=kwargs
                self.pid=100+len(created)
                self.closed=False
                created.append(self)
            def start(self): pass
            def _ensure_alive(self): return self
            def close(self): self.closed=True

        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            binary=root/"qwen"
            binary.write_text("x")
            ack_holder={}
            def env_builder(route,promo,ack,txn):
                promo.write_text("shared_up_proj 3 6\n")
                ack.write_text("{}")
                ack_holder[str(ack)]=route["policy_hash"]
                return {}
            def ack_reader(path):
                return {
                    "active_policy_hash":ack_holder[str(path)],
                    "ack_sha256":"e"*64,
                    "weight_epoch":1,
                }
            mgr=p.PersistentWorkerManager(
                binary=binary,cwd=root,env_builder=env_builder,
                ack_reader=ack_reader,timeout_seconds=1,
                worker_factory=FakeWorker,
            )
            route={
                "route_id":"candidate",
                "worker_instance_id":"persistent",
                "endpoint":"local://candidate",
                "source_commit":"a"*40,
                "binary_sha256":"b"*64,
                "checkpoint_sha256":"c"*64,
                "policy_hash":"d"*64,
                "role":"shared_up_proj","layer":3,"n":6,
            }
            w1=mgr.ensure({"generation":6,"route":route})
            w2=mgr.ensure({"generation":6,"route":route})
            self.assertIs(w1,w2)
            self.assertEqual(len(created),1)
            w3=mgr.ensure({"generation":7,"route":route})
            self.assertIsNot(w1,w3)
            self.assertTrue(w1.closed)
            self.assertEqual(len(created),2)
            self.assertEqual(mgr.startup_ack["active_policy_hash"],"d"*64)
            mgr.close()

if __name__=="__main__":
    unittest.main()
