#!/usr/bin/env python3
import json
import pathlib
import tempfile
import unittest

import gpu_runtime_control as ctl
import precision_context as pc


POLICY = [
    {"role": "shared_down_proj", "layer": 4, "n": 6},
    {"role": "shared_up_proj", "layer": 3, "n": 5},
]


class RuntimeControlTests(unittest.TestCase):
    def write_ack(self, root, **overrides):
        value = {
            "schema": ctl.ACK_SCHEMA,
            "status": "PROMOTION_APPLIED",
            "backend": "mlx_metal",
            "weight_epoch": 7,
            "changed_targets": 1,
            "snapshot_count": 2,
            "txn_id": None,
            "expected_epoch": None,
            "expected_n": None,
            "expected_policy_hash": None,
            "target_role": None,
            "target_layer": None,
            "active_policy": list(reversed(POLICY)),
        }
        value.update(overrides)
        path = pathlib.Path(root) / "ack.json"
        path.write_text(json.dumps(value))
        return path

    def test_ack_normalizes_policy_and_computes_hash(self):
        with tempfile.TemporaryDirectory() as td:
            ack = ctl.read_runtime_ack(self.write_ack(td))
            self.assertEqual(ack["active_policy"], pc.normalize_policy(POLICY))
            self.assertEqual(ack["active_policy_hash"], pc.policy_hash(POLICY))

    def test_prepare_demote_writes_exact_runtime_command(self):
        with tempfile.TemporaryDirectory() as td:
            ack_path = self.write_ack(td)
            txn_path = pathlib.Path(td) / "txn.txt"
            ph = pc.policy_hash(POLICY)
            got = ctl.prepare_demote(
                ack_path=ack_path,
                txn_path=txn_path,
                txn_id="demote-001",
                expected_epoch=7,
                expected_policy_hash=ph,
                role="shared_down_proj",
                layer=4,
                expected_n=6,
            )
            self.assertEqual(got["status"], "REQUESTED")
            self.assertEqual(
                txn_path.read_text(),
                f"DEMOTE demote-001 7 6 shared_down_proj 4 {ph}\n",
            )

    def test_prepare_rejects_stale_epoch_without_writing(self):
        with tempfile.TemporaryDirectory() as td:
            ack_path = self.write_ack(td)
            txn_path = pathlib.Path(td) / "txn.txt"
            with self.assertRaises(ctl.StaleRuntimeState):
                ctl.prepare_demote(
                    ack_path=ack_path,
                    txn_path=txn_path,
                    txn_id="t1",
                    expected_epoch=6,
                    expected_policy_hash=pc.policy_hash(POLICY),
                    role="shared_down_proj",
                    layer=4,
                    expected_n=6,
                )
            self.assertFalse(txn_path.exists())

    def test_prepare_rejects_stale_policy_without_writing(self):
        with tempfile.TemporaryDirectory() as td:
            ack_path = self.write_ack(td)
            txn_path = pathlib.Path(td) / "txn.txt"
            with self.assertRaises(ctl.StaleRuntimeState):
                ctl.prepare_demote(
                    ack_path=ack_path,
                    txn_path=txn_path,
                    txn_id="t2",
                    expected_epoch=7,
                    expected_policy_hash="a" * 64,
                    role="shared_down_proj",
                    layer=4,
                    expected_n=6,
                )
            self.assertFalse(txn_path.exists())

    def test_prepare_rejects_stale_target_without_writing(self):
        with tempfile.TemporaryDirectory() as td:
            ack_path = self.write_ack(td)
            txn_path = pathlib.Path(td) / "txn.txt"
            with self.assertRaises(ctl.StaleRuntimeState):
                ctl.prepare_demote(
                    ack_path=ack_path,
                    txn_path=txn_path,
                    txn_id="t3",
                    expected_epoch=7,
                    expected_policy_hash=pc.policy_hash(POLICY),
                    role="shared_down_proj",
                    layer=4,
                    expected_n=7,
                )
            self.assertFalse(txn_path.exists())

    def test_duplicate_role_layer_in_ack_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            ack_path = self.write_ack(
                td,
                active_policy=[
                    {"role": "shared_down_proj", "layer": 4, "n": 6},
                    {"role": "shared_down_proj", "layer": 4, "n": 7},
                ],
            )
            with self.assertRaises(ctl.RuntimeControlError):
                ctl.read_runtime_ack(ack_path)

    def test_verify_terminal_ack_matches_txn_and_status(self):
        with tempfile.TemporaryDirectory() as td:
            ack_path = self.write_ack(
                td,
                status="ROLLBACK_APPLIED",
                txn_id="demote-001",
                weight_epoch=8,
                active_policy=[POLICY[1]],
            )
            got = ctl.verify_terminal_ack(
                ack_path=ack_path,
                txn_id="demote-001",
            )
            self.assertEqual(got["status"], "ROLLBACK_APPLIED")
            self.assertEqual(got["weight_epoch"], 8)

    def test_verify_terminal_ack_rejects_wrong_txn(self):
        with tempfile.TemporaryDirectory() as td:
            ack_path = self.write_ack(td, status="STALE_COMMAND", txn_id="old")
            with self.assertRaises(ctl.RuntimeControlError):
                ctl.verify_terminal_ack(ack_path=ack_path, txn_id="new")


if __name__ == "__main__":
    unittest.main()

