#!/usr/bin/env python3
import tempfile
import unittest

import precision_context as pc
from precision_control_state import ControlStore


BASE = [{"role": "shared_up_proj", "layer": 3, "n": 5}]
NEXT = BASE + [{"role": "shared_down_proj", "layer": 4, "n": 6}]


class ControlStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ControlStore(
            self.tmp.name, "modelrev", "mlx_metal"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_desired_and_applied_in_sync(self):
        self.store.set_desired(
            context_hash="ctx", policy=BASE, reason="test"
        )
        self.store.set_applied(
            context_hash="ctx", policy=BASE, epoch=3,
            txn_id="t1", ack_sha256="ack",
        )
        self.assertEqual(self.store.reconcile()["status"], "IN_SYNC")

    def test_policy_drift_is_visible(self):
        self.store.set_desired(
            context_hash="ctx", policy=NEXT, reason="test"
        )
        self.store.set_applied(
            context_hash="ctx", policy=BASE, epoch=3,
            txn_id="t1", ack_sha256="ack",
        )
        self.assertEqual(self.store.reconcile()["status"], "DRIFT")

    def test_context_mismatch_is_visible(self):
        self.store.set_desired(
            context_hash="ctx-a", policy=BASE, reason="test"
        )
        self.store.set_applied(
            context_hash="ctx-b", policy=BASE, epoch=3,
            txn_id="t1", ack_sha256="ack",
        )
        self.assertEqual(
            self.store.reconcile()["status"], "CONTEXT_MISMATCH"
        )

    def test_quarantine_is_context_and_precision_scoped(self):
        self.store.quarantine_target(
            context_hash="ctx",
            role="shared_down_proj",
            layer=4,
            n=6,
            reason="regression",
            evidence_id=9,
        )
        self.assertTrue(self.store.is_quarantined(
            context_hash="ctx",
            role="shared_down_proj",
            layer=4,
            n=6,
        ))
        self.assertFalse(self.store.is_quarantined(
            context_hash="ctx",
            role="shared_down_proj",
            layer=4,
            n=7,
        ))
        self.assertFalse(self.store.is_quarantined(
            context_hash="other",
            role="shared_down_proj",
            layer=4,
            n=6,
        ))

    def test_policy_hash_is_normalized(self):
        d = self.store.set_desired(
            context_hash="ctx",
            policy=list(reversed(NEXT)),
            reason="test",
        )
        self.assertEqual(d["policy_hash"], pc.policy_hash(NEXT))


if __name__ == "__main__":
    unittest.main()
