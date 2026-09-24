#!/usr/bin/env python3
"""Offline fail-closed drills for GPU precision control.

These are deterministic control-plane drills; they do not claim Apple/MLX
runtime certification.  The real-device versions remain required after the
vdsp_engine workspace/compile gate opens.
"""
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from backend_adapters import AppliedState, MockBackendAdapter
import gpu_isolated_preflight as gp
import gpu_observer_control as goc
import gpu_runtime_control as grc
import precision_context as pc
from precision_control_state import ControlStore
from precision_transactions import (
    TransactionJournal,
    execute_policy_transaction,
)


BASE = [{"role": "shared_up_proj", "layer": 3, "n": 5}]
CAND = BASE + [{"role": "shared_down_proj", "layer": 4, "n": 6}]


def h(ch):
    return ch * 64


def context():
    return pc.ExecutionContext(
        schema="precision-context-v3",
        model_id="deepseek-v2-lite",
        architecture="deepseek_v2_mla",
        checkpoint_sha256=h("1"),
        tokenizer_sha256=h("2"),
        base_artifact_sha256=h("3"),
        backend="mlx_metal",
        device_fingerprint="apple-test",
        binary_sha256=h("4"),
        build_manifest_sha256=h("5"),
        kernel_revision="mlx-test",
        execution_mode="online",
        runtime_config_sha256=h("6"),
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        correction_mode="off",
    )


class FakeProc:
    def __init__(self, rc=0, stdout="", stderr=""):
        self.returncode = rc
        self.stdout = stdout
        self.stderr = stderr


def runtime_ack(policy, epoch, *, status="PROMOTION_APPLIED", txn_id=None):
    return grc.normalize_runtime_ack({
        "schema": grc.ACK_SCHEMA,
        "status": status,
        "backend": "mlx_metal",
        "correction_mode": "off",
        "weight_epoch": int(epoch),
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


class AckAdapter:
    name = "mlx_metal"

    def __init__(self, policy, epoch, ack):
        self.policy = pc.normalize_policy(policy)
        self.epoch = int(epoch)
        self.ack = ack

    def query_applied_state(self):
        return AppliedState(
            backend=self.name,
            epoch=self.epoch,
            policy=list(self.policy),
            policy_hash=pc.policy_hash(self.policy),
            admission_paused=False,
            active_requests=0,
        )

    def verify_runtime_txn(self, txn_id):
        if self.ack.get("txn_id") != txn_id:
            raise grc.RuntimeControlError("wrong transaction ACK")
        return self.ack


class GpuFailureDrills(unittest.TestCase):
    def test_candidate_mismatch_blocks_preflight(self):
        with patch.object(gp, "run_isolated_worker") as worker:
            worker.side_effect = [
                {
                    "output": "[moe gpu cb online] req 0 x tokens: 111\n",
                    "worker": {"binary_sha256": "a" * 64},
                    "applied_policy_hash": pc.policy_hash(BASE),
                    "weight_epoch": len(BASE),
                },
                {
                    "output": "[moe gpu cb online] req 0 x tokens: 999\n",
                    "worker": {"binary_sha256": "a" * 64},
                    "applied_policy_hash": pc.policy_hash(CAND),
                    "weight_epoch": len(CAND),
                },
            ]
            with self.assertRaises(gp.GpuPreflightError):
                gp.run_ab_preflight(
                    host="xox",
                    cwd="/repo",
                    binary="/bin/q",
                    architecture="mla",
                    moe_base="/m",
                    manifest="/manifest",
                    safetensors="/st",
                    root="/tmp/preflight",
                    baseline_policy=BASE,
                    candidate_policy=CAND,
                    event={
                        "orig_token": 111,
                        "corrected_token": 222,
                        "pos": 0,
                    },
                    reference={"emitted_token": 222},
                    prompt_len=1,
                )

    def test_worker_crash_or_oom_exit_blocks_evidence(self):
        identity = {
            "host": "XOX.local",
            "arch": "arm64",
            "binary_path": "/bin/q",
            "binary_sha256": "a" * 64,
            "binary_size": 123,
        }
        with (
            patch.object(gp, "_write_text"),
            patch.object(gp, "_remove"),
            patch.object(gp, "_binary_identity", return_value=identity),
            patch.object(
                gp,
                "_require_binary_capability",
                return_value={"schema": "test-capability"},
            ),
            patch.object(
                gp,
                "_run",
                return_value=FakeProc(
                    rc=137,
                    stderr="simulated OOM/worker kill",
                ),
            ),
        ):
            with self.assertRaises(gp.GpuPreflightError):
                gp.run_isolated_worker(
                    host="xox",
                    cwd="/repo",
                    binary="/bin/q",
                    architecture="mla",
                    moe_base="/m",
                    manifest="/manifest",
                    safetensors="/st",
                    run_dir="/tmp/run",
                    policy=CAND,
                )

    def test_stale_epoch_never_publishes_runtime_txn(self):
        with tempfile.TemporaryDirectory() as td:
            ack_path = pathlib.Path(td) / "ack.json"
            txn_path = pathlib.Path(td) / "txn.txt"
            ack_path.write_text(json.dumps({
                "schema": grc.ACK_SCHEMA,
                "status": "PROMOTION_APPLIED",
                "backend": "mlx_metal",
                "correction_mode": "off",
                "weight_epoch": 8,
                "changed_targets": 2,
                "snapshot_count": 2,
                "txn_id": None,
                "expected_epoch": None,
                "expected_n": None,
                "expected_policy_hash": None,
                "target_role": None,
                "target_layer": None,
                "active_policy": CAND,
            }))
            with self.assertRaises(grc.StaleRuntimeState):
                grc.prepare_demote(
                    ack_path=ack_path,
                    txn_path=txn_path,
                    txn_id="stale-epoch",
                    expected_epoch=7,
                    expected_policy_hash=pc.policy_hash(CAND),
                    role="shared_down_proj",
                    layer=4,
                    expected_n=6,
                )
            self.assertFalse(txn_path.exists())

    def test_sync_failure_isolates_without_policy_mutation(self):
        ctx = context()
        adapter = MockBackendAdapter(
            backend="mlx_metal",
            policy=BASE,
            epoch=7,
            active_requests=2,
            context=ctx,
        )
        adapter.inject_failure(sync=True)
        with tempfile.TemporaryDirectory() as td:
            result = execute_policy_transaction(
                adapter,
                TransactionJournal(td),
                txn_id="sync-fail",
                expected_context_hash=ctx.context_hash,
                expected_epoch=7,
                expected_policy_hash=pc.policy_hash(BASE),
                requested_policy=CAND,
            )
        self.assertEqual(result["status"], "FAILED_ISOLATED")
        state = adapter.query_applied_state()
        self.assertEqual(state.policy_hash, pc.policy_hash(BASE))
        self.assertTrue(state.admission_paused)

    def test_restore_failure_leaves_worker_isolated(self):
        ctx = context()
        adapter = MockBackendAdapter(
            backend="mlx_metal",
            policy=BASE,
            epoch=7,
            active_requests=1,
            context=ctx,
        )
        adapter.inject_failure(apply=True, restore=True)
        with tempfile.TemporaryDirectory() as td:
            result = execute_policy_transaction(
                adapter,
                TransactionJournal(td),
                txn_id="restore-fail",
                expected_context_hash=ctx.context_hash,
                expected_epoch=7,
                expected_policy_hash=pc.policy_hash(BASE),
                requested_policy=CAND,
            )
        self.assertEqual(result["status"], "FAILED_ISOLATED")
        self.assertTrue(adapter.query_applied_state().admission_paused)

    def test_wrong_rollback_ack_cannot_be_recorded_as_success(self):
        with tempfile.TemporaryDirectory() as td:
            store = ControlStore(td, "model-rev", "mlx_metal")
            store.set_desired(
                context_hash="ctx",
                policy=BASE,
                reason="failure drill",
            )
            wrong_ack = runtime_ack(
                CAND,
                9,
                status="ROLLBACK_APPLIED",
                txn_id="wrong-ack",
            )
            adapter = AckAdapter(CAND, 8, wrong_ack)
            with self.assertRaises(goc.GpuObserverControlError):
                goc.complete_regression_rollback(
                    adapter=adapter,
                    store=store,
                    txn_id="wrong-ack",
                    context_hash="ctx",
                    baseline_policy=BASE,
                    failed_epoch=8,
                )
            self.assertIsNone(store.applied())

    def test_stale_ack_is_explicitly_not_rollback_success(self):
        with tempfile.TemporaryDirectory() as td:
            store = ControlStore(td, "model-rev", "mlx_metal")
            store.set_desired(context_hash="ctx", policy=BASE, reason="drill")
            stale = runtime_ack(
                CAND, 8, status="STALE_COMMAND", txn_id="stale-ack"
            )
            adapter = AckAdapter(CAND, 8, stale)
            got = goc.complete_regression_rollback(
                adapter=adapter, store=store, txn_id="stale-ack",
                context_hash="ctx", baseline_policy=BASE, failed_epoch=8,
            )
            self.assertFalse(got["rollback_complete"])
            self.assertIsNone(store.applied())


if __name__ == "__main__":
    unittest.main()
