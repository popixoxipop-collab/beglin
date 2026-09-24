#!/usr/bin/env python3
import tempfile
import unittest

import precision_context as pc
from backend_adapters import MockBackendAdapter, MlxMetalBackendAdapter, BackendUnverified
from precision_transactions import TransactionJournal, execute_policy_transaction, reconcile


def h(ch):
    return ch * 64


def context(backend="mlx_metal", binary="a"):
    return pc.ExecutionContext(
        schema="precision-context-v3",
        model_id="deepseek-v2-lite",
        architecture="deepseek_v2_mla",
        checkpoint_sha256=h("1"),
        tokenizer_sha256=h("2"),
        base_artifact_sha256=h("3"),
        backend=backend,
        device_fingerprint="apple-m4-test",
        binary_sha256=h(binary),
        build_manifest_sha256=h("4"),
        kernel_revision="mlx-test",
        execution_mode="online",
        runtime_config_sha256=h("5"),
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        correction_mode="off",
    )


BASE = [{"role": "shared_up_proj", "layer": 3, "n": 5}]
CANDIDATE = BASE + [{"role": "shared_down_proj", "layer": 4, "n": 6}]


class ContextTests(unittest.TestCase):
    def test_backend_changes_context_hash(self):
        a = context("cpu")
        b = context("mlx_metal")
        self.assertNotEqual(a.context_hash, b.context_hash)

    def test_binary_changes_context_hash(self):
        self.assertNotEqual(context(binary="a").context_hash, context(binary="b").context_hash)

    def test_policy_hash_is_order_independent(self):
        x = [
            {"role": "b", "layer": 2, "n": 6},
            {"role": "a", "layer": 1, "n": 5},
        ]
        y = list(reversed(x))
        self.assertEqual(pc.policy_hash(x), pc.policy_hash(y))

    def test_mlx_adapter_is_fail_closed(self):
        with self.assertRaises(BackendUnverified):
            MlxMetalBackendAdapter().query_applied_state()


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.journal = TransactionJournal(self.tmp.name)
        self.ctx = context()
        self.adapter = MockBackendAdapter(
            backend="mlx_metal",
            policy=BASE,
            epoch=7,
            active_requests=3,
            context=self.ctx,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def run_txn(self, txn="t1", **kwargs):
        return execute_policy_transaction(
            self.adapter,
            self.journal,
            txn_id=txn,
            expected_context_hash=kwargs.get("context_hash", self.ctx.context_hash),
            expected_epoch=kwargs.get("epoch", 7),
            expected_policy_hash=kwargs.get("policy_hash", pc.policy_hash(BASE)),
            requested_policy=kwargs.get("policy", CANDIDATE),
        )

    def test_success_drains_before_apply_and_resumes_after_ack(self):
        got = self.run_txn()
        self.assertEqual(got["status"], "APPLIED")
        self.assertEqual(got["applied_policy_hash"], pc.policy_hash(CANDIDATE))
        state = self.adapter.query_applied_state()
        self.assertFalse(state.admission_paused)
        self.assertEqual(state.active_requests, 0)
        self.assertEqual(state.policy_hash, pc.policy_hash(CANDIDATE))
        order = [x["status"] for x in got["history"]]
        self.assertEqual(
            order,
            [
                "REQUESTED", "ADMISSION_PAUSED", "DRAINED", "GPU_SYNCED",
                "SNAPSHOT_DURABLE", "VERIFYING", "APPLIED",
            ],
        )

    def test_duplicate_terminal_txn_is_idempotent(self):
        first = self.run_txn()
        epoch = self.adapter.query_applied_state().epoch
        second = self.run_txn()
        self.assertEqual(first["status"], "APPLIED")
        self.assertEqual(second["status"], "APPLIED")
        self.assertEqual(self.adapter.query_applied_state().epoch, epoch)

    def test_stale_epoch_is_rejected_without_mutation(self):
        got = self.run_txn(epoch=6)
        self.assertEqual(got["status"], "STALE_COMMAND")
        state = self.adapter.query_applied_state()
        self.assertEqual(state.epoch, 7)
        self.assertEqual(state.policy_hash, pc.policy_hash(BASE))
        self.assertFalse(state.admission_paused)

    def test_context_mismatch_is_rejected(self):
        got = self.run_txn(context_hash=h("f"))
        self.assertEqual(got["status"], "STALE_COMMAND")
        self.assertEqual(self.adapter.query_applied_state().policy_hash, pc.policy_hash(BASE))

    def test_apply_failure_rolls_back_and_resumes(self):
        self.adapter.inject_failure(apply=True)
        got = self.run_txn()
        self.assertEqual(got["status"], "ROLLBACK_APPLIED")
        state = self.adapter.query_applied_state()
        self.assertEqual(state.policy_hash, pc.policy_hash(BASE))
        self.assertFalse(state.admission_paused)

    def test_restore_failure_leaves_worker_isolated(self):
        self.adapter.inject_failure(apply=True, restore=True)
        got = self.run_txn()
        self.assertEqual(got["status"], "FAILED_ISOLATED")
        self.assertTrue(self.adapter.query_applied_state().admission_paused)

    def test_sync_failure_never_mutates_policy(self):
        self.adapter.inject_failure(sync=True)
        got = self.run_txn()
        self.assertEqual(got["status"], "FAILED_ISOLATED")
        self.assertEqual(self.adapter.query_applied_state().policy_hash, pc.policy_hash(BASE))
        self.assertTrue(self.adapter.query_applied_state().admission_paused)

    def test_reconcile_reports_applied_state(self):
        self.run_txn()
        got = reconcile(self.adapter, self.journal, "t1")
        self.assertTrue(got["consistent"])
        self.assertEqual(got["actual_policy_hash"], pc.policy_hash(CANDIDATE))


if __name__ == "__main__":
    unittest.main()
