#!/usr/bin/env python3
from pathlib import Path
import tempfile
import unittest

import production_serving_prewarm as prewarm


class FakeWorker:
    def __init__(self, snapshot):
        self.route_snapshot=snapshot
        self.pid=123
        self.startup_ms=456
        self.cleaned=False
    def cleanup(self):
        self.cleaned=True


def snapshot(gen=6, policy="a"*64):
    return {
        "generation":gen,
        "manifest_sha256":"b"*64,
        "route":{"policy_hash":policy},
    }


class PoolContractTests(unittest.TestCase):
    def make_pool(self, root):
        return prewarm.PrewarmPool(
            state_root=Path(root),
            repo=Path("/tmp/repo"),
            binary=Path("/tmp/bin"),
            moe_base=Path("/tmp/moe"),
            safetensors=Path("/tmp/model.json"),
            timeout_seconds=90,
        )

    def test_only_single_request_uses_prewarm(self):
        with tempfile.TemporaryDirectory() as td:
            pool=self.make_pool(td)
            snap=snapshot()
            fake=FakeWorker(snap)
            pool.target_snapshot=snap
            pool.ready_worker=fake
            self.assertIsNone(pool.acquire(snap, 12))
            self.assertIs(pool.ready_worker, fake)

    def test_matching_single_request_leases_ready_worker(self):
        with tempfile.TemporaryDirectory() as td:
            pool=self.make_pool(td)
            snap=snapshot()
            fake=FakeWorker(snap)
            pool.target_snapshot=snap
            pool.ready_worker=fake
            got=pool.acquire(snap,1)
            self.assertIs(got,fake)
            self.assertIsNone(pool.ready_worker)

    def test_status_exposes_generation_and_policy(self):
        with tempfile.TemporaryDirectory() as td:
            pool=self.make_pool(td)
            snap=snapshot()
            fake=FakeWorker(snap)
            pool.target_snapshot=snap
            pool.ready_worker=fake
            got=pool.status()
            self.assertTrue(got["ready"])
            self.assertEqual(got["ready_worker_pid"],123)
            self.assertEqual(got["target_generation"],6)
            self.assertEqual(got["target_policy_hash"],"a"*64)


class ParserTests(unittest.TestCase):
    def test_parses_native_generated_lines(self):
        text=(
            "[moe gpu cb online] req 0 prompt 0 tokens: 1 2 3\n"
            "[moe gpu cb online] req 1 prompt 0 tokens: 4 5\n"
        )
        self.assertEqual(
            prewarm._parse_generated(text),
            {0:[1,2,3],1:[4,5]},
        )


if __name__=="__main__":
    unittest.main()
