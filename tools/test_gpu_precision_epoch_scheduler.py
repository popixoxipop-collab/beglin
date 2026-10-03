#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import gpu_runtime_control as grc
import precision_context as pc
import precision_epoch_scheduler as pes


BASE = [
    {"role": "shared_down_proj", "layer": 26, "n": 5},
    {"role": "shared_up_proj", "layer": 3, "n": 6},
]
TARGET = [
    {"role": "shared_down_proj", "layer": 26, "n": 6},
    {"role": "shared_up_proj", "layer": 3, "n": 5},
]


def write_ack(path, *, epoch=7, policy=BASE, status="PROMOTION_APPLIED",
              txn_id=None, changed_targets=0, **overrides):
    value = {
        "schema": grc.ACK_SCHEMA,
        "status": status,
        "backend": grc.BACKEND,
        "correction_mode": "off",
        "weight_epoch": epoch,
        "changed_targets": changed_targets,
        "snapshot_count": 2,
        "txn_id": txn_id,
        "expected_epoch": None,
        "expected_n": None,
        "expected_policy_hash": None,
        "target_role": None,
        "target_layer": None,
        "active_policy": policy,
    }
    value.update(overrides)
    Path(path).write_text(json.dumps(value))


class PlanTests(unittest.TestCase):
    def test_multi_target_plan_is_deterministic(self):
        self.assertEqual(
            pes.plan_transition(BASE, TARGET),
            [
                {"role": "shared_down_proj", "layer": 26, "expected_n": 5, "target_n": 6},
                {"role": "shared_up_proj", "layer": 3, "expected_n": 6, "target_n": 5},
            ],
        )

    def test_shape_change_requires_restart(self):
        with self.assertRaises(pes.PrecisionEpochSchedulerError):
            pes.plan_transition(
                BASE,
                BASE + [{"role": "shared_gate_proj", "layer": 6, "n": 5}],
            )


class SchedulerTests(unittest.TestCase):
    def test_external_admission_lock_covers_preimage_through_submit(self):
        with tempfile.TemporaryDirectory() as td:
            ack = Path(td) / "ack.json"
            txn = Path(td) / "txn.cmd"
            write_ack(ack)
            shared = threading.RLock()
            events = []

            def submit(parsed):
                events.append("scheduler-submit")
                command = txn.read_text().strip().split()
                write_ack(
                    ack,
                    epoch=8,
                    policy=TARGET,
                    status="REBIND_SET_APPLIED",
                    txn_id=command[1],
                    changed_targets=2,
                )
                return {"finite_logits": True}

            scheduler = pes.PrecisionEpochScheduler(
                ack_path=ack,
                txn_path=txn,
                submit_fn=submit,
                admission_lock=shared,
            )

            def competing_direct_submit():
                time.sleep(0.01)
                with shared:
                    events.append("direct-submit")

            other = threading.Thread(target=competing_direct_submit)
            other.start()
            got = scheduler.run(
                [([1], 1)],
                target_policy=TARGET,
                admission_id="shared-lock",
            )
            other.join()
            self.assertTrue(got["precision_epoch"]["transitioned"])
            self.assertEqual(events, ["scheduler-submit", "direct-submit"])

    def test_two_targets_advance_one_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            ack = Path(td) / "ack.json"
            txn = Path(td) / "txn.cmd"
            write_ack(ack)

            def submit(parsed):
                command = txn.read_text().strip().split()
                self.assertEqual(command[0], "REBIND_SET")
                txn_id = command[1]
                write_ack(
                    ack,
                    epoch=8,
                    policy=TARGET,
                    status="REBIND_SET_APPLIED",
                    txn_id=txn_id,
                    changed_targets=2,
                    transition_wall_ms=7.25,
                    transition_cache_hits=1,
                    transition_cache_misses=1,
                    transition_cache_bytes_added=4096,
                    resident_qng64_cache_bytes=8192,
                )
                return {
                    "finite_logits": True,
                    "responses": [[1]],
                    "engine_wall_ms": 20.0,
                    "roundtrip_ms": 30.0,
                }

            scheduler = pes.PrecisionEpochScheduler(
                ack_path=ack, txn_path=txn, submit_fn=submit
            )
            got = scheduler.run(
                [([1, 2], 1)],
                target_policy=TARGET,
                admission_id="admit-a",
            )
            meta = got["precision_epoch"]
            self.assertTrue(meta["transitioned"])
            self.assertEqual(meta["before_epoch"], 7)
            self.assertEqual(meta["after_epoch"], 8)
            self.assertEqual(len(meta["changed_targets"]), 2)
            self.assertEqual(meta["after_policy_hash"], pc.policy_hash(TARGET))
            self.assertEqual(meta["transition_cost"]["transition_wall_ms"], 7.25)
            self.assertEqual(meta["transition_cost"]["cache_hits"], 1)
            self.assertEqual(meta["transition_cost"]["cache_misses"], 1)
            self.assertEqual(meta["transition_cost"]["cache_bytes_added"], 4096)
            self.assertEqual(meta["transition_cost"]["resident_cache_bytes"], 8192)
            self.assertEqual(meta["inference_passes"], 1)
            self.assertEqual(meta["engine_wall_ms"], 20.0)
            self.assertEqual(meta["roundtrip_ms"], 30.0)

    def test_noop_policy_does_not_advance_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            ack = Path(td) / "ack.json"
            txn = Path(td) / "txn.cmd"
            write_ack(ack)
            scheduler = pes.PrecisionEpochScheduler(
                ack_path=ack,
                txn_path=txn,
                submit_fn=lambda parsed: {"finite_logits": True},
            )
            got = scheduler.run([([1], 1)], target_policy=BASE, admission_id="noop")
            self.assertFalse(got["precision_epoch"]["transitioned"])
            self.assertEqual(got["precision_epoch"]["before_epoch"], 7)
            self.assertEqual(got["precision_epoch"]["after_epoch"], 7)
            self.assertEqual(
                got["precision_epoch"]["transition_cost"]["transition_wall_ms"],
                0.0,
            )
            self.assertFalse(txn.exists())

    def test_concurrent_admissions_do_not_interleave(self):
        with tempfile.TemporaryDirectory() as td:
            ack = Path(td) / "ack.json"
            txn = Path(td) / "txn.cmd"
            write_ack(ack)
            order = []
            call = 0
            call_lock = threading.Lock()

            def submit(parsed):
                nonlocal call
                with call_lock:
                    call += 1
                    idx = call
                order.append(f"start-{idx}")
                time.sleep(0.05)
                command = txn.read_text().strip().split()
                txn_id = command[1]
                target = TARGET if idx == 1 else BASE
                write_ack(
                    ack,
                    epoch=7 + idx,
                    policy=target,
                    status="REBIND_SET_APPLIED",
                    txn_id=txn_id,
                    changed_targets=2,
                )
                order.append(f"end-{idx}")
                return {"finite_logits": True}

            scheduler = pes.PrecisionEpochScheduler(
                ack_path=ack, txn_path=txn, submit_fn=submit
            )
            errors = []

            def run(policy, name):
                try:
                    scheduler.run([([1], 1)], target_policy=policy, admission_id=name)
                except Exception as exc:
                    errors.append(exc)

            a = threading.Thread(target=run, args=(TARGET, "a"))
            b = threading.Thread(target=run, args=(BASE, "b"))
            a.start(); time.sleep(0.01); b.start()
            a.join(); b.join()
            self.assertEqual(errors, [])
            self.assertEqual(order, ["start-1", "end-1", "start-2", "end-2"])


if __name__ == "__main__":
    unittest.main()
