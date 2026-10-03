#!/usr/bin/env python3
import json
import pathlib
import tempfile
import unittest

import precision_context as pc
from backend_adapters import (
    BackendUnverified,
    MlxMetalBackendAdapter,
)
import gpu_runtime_control as grc


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
        device_fingerprint="apple-m4-test",
        binary_sha256=h("4"),
        build_manifest_sha256=h("5"),
        kernel_revision="mlx-test",
        execution_mode="online",
        runtime_config_sha256=h("6"),
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        correction_mode="off",
    )


POLICY = [
    {"role": "shared_down_proj", "layer": 4, "n": 6},
    {"role": "shared_up_proj", "layer": 3, "n": 5},
]


class MlxRuntimeAdapterTests(unittest.TestCase):
    def write_ack(self, root, **overrides):
        value = {
            "schema": grc.ACK_SCHEMA,
            "status": "PROMOTION_APPLIED",
            "backend": "mlx_metal",
            "correction_mode": "off",
            "weight_epoch": 7,
            "changed_targets": 2,
            "snapshot_count": 2,
            "txn_id": None,
            "expected_epoch": None,
            "expected_n": None,
            "expected_policy_hash": None,
            "target_role": None,
            "target_layer": None,
            "active_policy": POLICY,
        }
        value.update(overrides)
        path = pathlib.Path(root) / "ack.json"
        path.write_text(json.dumps(value))
        return path

    def test_unverified_adapter_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            ack = self.write_ack(td)
            adapter = MlxMetalBackendAdapter(
                context=context(),
                ack_path=str(ack),
            )
            with self.assertRaises(BackendUnverified):
                adapter.query_applied_state()

    def test_verified_adapter_reads_runtime_ack(self):
        with tempfile.TemporaryDirectory() as td:
            ack = self.write_ack(td)
            adapter = MlxMetalBackendAdapter(
                verified=True,
                context=context(),
                ack_path=str(ack),
            )
            state = adapter.query_applied_state()
            self.assertEqual(state.backend, "mlx_metal")
            self.assertEqual(state.epoch, 7)
            self.assertEqual(state.policy_hash, pc.policy_hash(POLICY))
            self.assertEqual(state.policy, pc.normalize_policy(POLICY))
            self.assertEqual(adapter.collect_context().context_hash, context().context_hash)

    def test_verified_adapter_still_blocks_generic_hot_promotion(self):
        with tempfile.TemporaryDirectory() as td:
            ack = self.write_ack(td)
            adapter = MlxMetalBackendAdapter(
                verified=True,
                context=context(),
                ack_path=str(ack),
            )
            with self.assertRaises(BackendUnverified):
                adapter.apply_policy(POLICY)

    def test_verified_adapter_emits_exact_demote_command(self):
        with tempfile.TemporaryDirectory() as td:
            ack = self.write_ack(td)
            txn = pathlib.Path(td) / "txn.txt"
            adapter = MlxMetalBackendAdapter(
                verified=True,
                context=context(),
                ack_path=str(ack),
                txn_path=str(txn),
            )
            ph = pc.policy_hash(POLICY)
            got = adapter.request_demote(
                txn_id="demote-1",
                expected_epoch=7,
                expected_policy_hash=ph,
                role="shared_down_proj",
                layer=4,
                expected_n=6,
            )
            self.assertEqual(got["status"], "REQUESTED")
            self.assertEqual(
                txn.read_text(),
                f"DEMOTE demote-1 7 6 shared_down_proj 4 {ph}\n",
            )

    def test_verified_adapter_emits_exact_rebind_command(self):
        with tempfile.TemporaryDirectory() as td:
            ack = self.write_ack(td)
            txn = pathlib.Path(td) / "txn.txt"
            adapter = MlxMetalBackendAdapter(
                verified=True,
                context=context(),
                ack_path=str(ack),
                txn_path=str(txn),
            )
            ph = pc.policy_hash(POLICY)
            got = adapter.request_rebind(
                txn_id="rebind-2",
                expected_epoch=7,
                expected_policy_hash=ph,
                role="shared_up_proj",
                layer=3,
                expected_n=5,
                target_n=9,
            )
            self.assertEqual(got["status"], "REQUESTED")
            self.assertEqual(
                txn.read_text(),
                f"REBIND rebind-2 7 5 9 shared_up_proj 3 {ph}\n",
            )

    def test_adapter_preserves_runtime_stale_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            ack = self.write_ack(td)
            txn = pathlib.Path(td) / "txn.txt"
            adapter = MlxMetalBackendAdapter(
                verified=True,
                context=context(),
                ack_path=str(ack),
                txn_path=str(txn),
            )
            with self.assertRaises(grc.StaleRuntimeState):
                adapter.request_demote(
                    txn_id="demote-stale",
                    expected_epoch=6,
                    expected_policy_hash=pc.policy_hash(POLICY),
                    role="shared_down_proj",
                    layer=4,
                    expected_n=6,
                )
            self.assertFalse(txn.exists())

    def test_adapter_verifies_terminal_runtime_ack(self):
        with tempfile.TemporaryDirectory() as td:
            ack = self.write_ack(
                td,
                status="ROLLBACK_APPLIED",
                txn_id="demote-2",
                weight_epoch=8,
                active_policy=[POLICY[1]],
            )
            adapter = MlxMetalBackendAdapter(
                verified=True,
                context=context(),
                ack_path=str(ack),
            )
            got = adapter.verify_runtime_txn("demote-2")
            self.assertEqual(got["status"], "ROLLBACK_APPLIED")
            self.assertEqual(got["weight_epoch"], 8)


if __name__ == "__main__":
    unittest.main()
