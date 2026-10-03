#!/usr/bin/env python3
import tempfile
import unittest

from backend_adapters import AppliedState
import gpu_observer_control as ctl
import gpu_runtime_control as grc
import precision_context as pc
from precision_control_state import ControlStore


BASE = [{"role": "shared_up_proj", "layer": 3, "n": 5}]
FAILED = BASE + [{"role": "shared_down_proj", "layer": 4, "n": 6}]


def ack(policy, epoch, status="PROMOTION_APPLIED", txn_id=None):
    return grc.normalize_runtime_ack({
        "schema": grc.ACK_SCHEMA,
        "status": status,
        "backend": "mlx_metal",
        "correction_mode": "off",
        "weight_epoch": epoch,
        "changed_targets": 1,
        "snapshot_count": len(policy),
        "txn_id": txn_id,
        "expected_epoch": None,
        "expected_n": None,
        "expected_policy_hash": None,
        "target_role": None,
        "target_layer": None,
        "active_policy": policy,
    })


def metrics(**overrides):
    value = {
        "requests_completed": 100,
        "tokens_evaluated": 1000,
        "near_tie_events": 10,
        "reference_checks_attempted": 10,
        "effective_attribution_checks": 10,
        "attribution_hits": 2,
        "run_errors": 0,
        "median_margin": 0.02,
    }
    value.update(overrides)
    return value


def evidence(policy, epoch, *, replay=True, **metric_overrides):
    return ctl.evidence_from_runtime_ack(
        context_hash="ctx",
        runtime_ack=ack(policy, epoch),
        metrics=metrics(**metric_overrides),
        target_replay_pass=replay,
    )


class FakeAdapter:
    name = "mlx_metal"

    def __init__(self, policy, epoch):
        self.policy = pc.normalize_policy(policy)
        self.epoch = int(epoch)
        self.requests = []
        self.terminal_ack = None

    def query_applied_state(self):
        return AppliedState(
            backend=self.name,
            epoch=self.epoch,
            policy=list(self.policy),
            policy_hash=pc.policy_hash(self.policy),
            admission_paused=False,
            active_requests=0,
        )

    def request_demote(self, **kwargs):
        self.requests.append(dict(kwargs))
        return {"status": "REQUESTED", **kwargs}

    def verify_runtime_txn(self, txn_id):
        if self.terminal_ack is None:
            raise grc.RuntimeControlError("runtime ACK not available")
        if self.terminal_ack.get("txn_id") != txn_id:
            raise grc.RuntimeControlError("wrong txn")
        return self.terminal_ack


class GpuObserverControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ControlStore(self.tmp.name, "modelrev", "mlx_metal")

    def tearDown(self):
        self.tmp.cleanup()

    def test_evidence_uses_runtime_policy_epoch_and_denominators(self):
        got = evidence(
            FAILED, 8,
            requests_completed=77,
            effective_attribution_checks=8,
            attribution_hits=1,
        )
        self.assertEqual(got.backend, "mlx_metal")
        self.assertEqual(got.policy_hash, pc.policy_hash(FAILED))
        self.assertEqual(got.weight_epoch, 8)
        self.assertEqual(got.requests_completed, 77)
        self.assertEqual(got.effective_attribution_checks, 8)

    def test_evidence_rejects_invalid_denominator_relationship(self):
        with self.assertRaises(ctl.GpuObserverControlError):
            evidence(
                FAILED, 8,
                reference_checks_attempted=0,
                effective_attribution_checks=1,
            )

    def test_non_regression_never_emits_demote(self):
        pre = evidence(BASE, 7, attribution_hits=2)
        post = evidence(FAILED, 8, attribution_hits=0)
        adapter = FakeAdapter(FAILED, 8)
        got = ctl.request_regression_rollback(
            adapter=adapter,
            store=self.store,
            baseline=pre,
            post=post,
            baseline_policy=BASE,
            failed_policy=FAILED,
            role="shared_down_proj",
            layer=4,
            n=6,
            txn_id="obs-1",
        )
        self.assertEqual(got["verdict"]["status"], "CANARY_PASS")
        self.assertEqual(got["action"], "NONE")
        self.assertEqual(adapter.requests, [])
        self.assertIsNone(self.store.desired())

    def test_regression_persists_quarantine_and_desired_before_ack(self):
        pre = evidence(BASE, 7, attribution_hits=1, median_margin=0.02)
        post = evidence(
            FAILED, 8, replay=False,
            attribution_hits=2, median_margin=0.01,
        )
        adapter = FakeAdapter(FAILED, 8)
        got = ctl.request_regression_rollback(
            adapter=adapter,
            store=self.store,
            baseline=pre,
            post=post,
            baseline_policy=BASE,
            failed_policy=FAILED,
            role="shared_down_proj",
            layer=4,
            n=6,
            txn_id="obs-2",
            evidence_id="ev-2",
        )
        self.assertEqual(got["action"], "ROLLBACK_REQUESTED")
        self.assertFalse(got["rollback_complete"])
        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(self.store.desired()["policy_hash"], pc.policy_hash(BASE))
        self.assertTrue(self.store.is_quarantined(
            context_hash="ctx",
            role="shared_down_proj",
            layer=4,
            n=6,
        ))
        self.assertIsNone(self.store.applied())

        restarted = ControlStore(self.tmp.name, "modelrev", "mlx_metal")
        self.assertEqual(restarted.desired()["policy_hash"], pc.policy_hash(BASE))
        self.assertTrue(restarted.is_quarantined(
            context_hash="ctx",
            role="shared_down_proj",
            layer=4,
            n=6,
        ))
        self.assertEqual(restarted.reconcile()["status"], "INCOMPLETE")

    def test_ack_completion_updates_applied_only_after_verified_restore(self):
        self.store.set_desired(
            context_hash="ctx", policy=BASE, reason="rollback"
        )
        adapter = FakeAdapter(FAILED, 8)
        adapter.terminal_ack = ack(
            BASE, 9, status="ROLLBACK_APPLIED", txn_id="obs-3"
        )
        got = ctl.complete_regression_rollback(
            adapter=adapter,
            store=self.store,
            txn_id="obs-3",
            context_hash="ctx",
            baseline_policy=BASE,
            failed_epoch=8,
        )
        self.assertTrue(got["rollback_complete"])
        self.assertEqual(got["status"], "ROLLBACK_APPLIED")
        self.assertEqual(self.store.applied()["policy_hash"], pc.policy_hash(BASE))
        self.assertEqual(self.store.reconcile()["status"], "IN_SYNC")
        self.assertEqual(len(self.store.applied()["ack_sha256"]), 64)

    def test_stale_command_ack_is_not_rollback_success(self):
        self.store.set_desired(
            context_hash="ctx", policy=BASE, reason="rollback"
        )
        adapter = FakeAdapter(FAILED, 8)
        adapter.terminal_ack = ack(
            FAILED, 8, status="STALE_COMMAND", txn_id="obs-4"
        )
        got = ctl.complete_regression_rollback(
            adapter=adapter,
            store=self.store,
            txn_id="obs-4",
            context_hash="ctx",
            baseline_policy=BASE,
            failed_epoch=8,
        )
        self.assertFalse(got["rollback_complete"])
        self.assertEqual(got["status"], "ROLLBACK_NOT_APPLIED")
        self.assertIsNone(self.store.applied())

    def test_wrong_policy_ack_is_rejected(self):
        self.store.set_desired(
            context_hash="ctx", policy=BASE, reason="rollback"
        )
        adapter = FakeAdapter(FAILED, 8)
        adapter.terminal_ack = ack(
            FAILED, 9, status="ROLLBACK_APPLIED", txn_id="obs-5"
        )
        with self.assertRaises(ctl.GpuObserverControlError):
            ctl.complete_regression_rollback(
                adapter=adapter,
                store=self.store,
                txn_id="obs-5",
                context_hash="ctx",
                baseline_policy=BASE,
                failed_epoch=8,
            )
        self.assertIsNone(self.store.applied())

    def test_multi_target_rollback_request_is_rejected(self):
        other = FAILED + [{"role": "shared_gate_proj", "layer": 14, "n": 7}]
        with self.assertRaises(ctl.GpuObserverControlError):
            ctl.require_single_target_rollback(
                baseline_policy=BASE,
                failed_policy=other,
                role="shared_down_proj",
                layer=4,
                n=6,
            )


if __name__ == "__main__":
    unittest.main()
