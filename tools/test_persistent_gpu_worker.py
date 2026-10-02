#!/usr/bin/env python3
from pathlib import Path
import tempfile
import unittest

import persistent_gpu_worker as p


class ValidationTests(unittest.TestCase):
    def test_valid_batch(self):
        got=p.validate_batch([
            {"prompt_tokens":[1,2,3],"max_new_tokens":10},
            {"prompt_tokens":[4],"max_new_tokens":1},
        ])
        self.assertEqual(got,[([1,2,3],10),([4],1)])

    def test_empty_batch_rejected(self):
        with self.assertRaises(p.PersistentWorkerError):
            p.validate_batch([])

    def test_large_generation_rejected(self):
        with self.assertRaises(p.PersistentWorkerError):
            p.validate_batch([{"prompt_tokens":[1],"max_new_tokens":257}])


class AtomicProtocolTests(unittest.TestCase):
    def test_atomic_text_replaces_file(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"request.seq"
            p.atomic_text(path,"1\n")
            self.assertEqual(path.read_text(),"1\n")
            p.atomic_text(path,"2\n")
            self.assertEqual(path.read_text(),"2\n")

    def test_write_i32_is_little_endian(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"tokens.i32"
            p.write_i32(path,[1,256,-1])
            self.assertEqual(
                path.read_bytes(),
                b"\x01\x00\x00\x00\x00\x01\x00\x00\xff\xff\xff\xff",
            )


class SourceContractTests(unittest.TestCase):
    def test_qwen_persistent_protocol_markers_exist(self):
        src=(Path(__file__).resolve().parents[1]/"qwen_infer.c").read_text()
        for needle in (
            "QWEN_MOE_GPU_PERSIST_DIR",
            "moe_gpu_persist_wait_next",
            "beglin-gpu-persistent-ready-v1",
            "beglin-gpu-persistent-response-v1",
            "persist_last_seq = persist_seq",
            "persistent_warmed",
            "peak_rss_bytes",
        ):
            self.assertIn(needle,src)

    def test_historical_one_shot_result_marker_remains(self):
        src=(Path(__file__).resolve().parents[1]/"qwen_infer.c").read_text()
        self.assertIn(
            "RESULT: MoE GPU V5h online cbatch gate complete",
            src,
        )


if __name__=="__main__":
    unittest.main()
