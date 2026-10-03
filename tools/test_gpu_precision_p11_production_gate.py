#!/usr/bin/env python3
import copy
import unittest

import precision_p11_production_gate as p11


class P11GateTests(unittest.TestCase):
    def fixture(self):
        active = {
            "schema": "beglin-serving-route-v1",
            "route_id": p11.ACTIVE_ROUTE_ID,
            "endpoint": "local-supervisor://candidate",
            "role": "shared_up_proj",
            "layer": 3,
            "n": 6,
            "policy_hash": "0" * 64,
            "source_commit": "deadbeef",
            "binary_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "worker_instance_id": "spawn-per-admission",
        }
        manifest = {
            "schema": "beglin-serving-route-manifest/1",
            "generation": 6,
            "manifest_sha256": "3" * 64,
            "active_route": active,
        }
        ack = {
            "status": "REBIND_APPLIED",
            "txn_id": "txn-1",
            "weight_epoch": 8,
            "active_policy": copy.deepcopy(p11.BASELINE_POLICY),
            "active_policy_hash": p11.BASELINE_POLICY_HASH,
            "ack_sha256": "4" * 64,
        }
        worker = {
            "alive": True,
            "pid": 35493,
            "weight_epoch": 8,
            "runtime_policy": copy.deepcopy(p11.BASELINE_POLICY),
            "runtime_policy_hash": p11.BASELINE_POLICY_HASH,
            "ack_sha256": "4" * 64,
        }
        health = {
            "schema": "beglin-supervisor-persistent-health-v1",
            "status": "ok",
            "router_id": p11.ROUTER_ID,
            "listen_host": "127.0.0.1",
            "external_network_exposed": False,
            "production_write_allowed": False,
            "auto_promotion_enabled": False,
            "route_generation": 6,
            "route_manifest_sha256": "3" * 64,
            "active_route": active,
            "workers": {p11.ACTIVE_ROUTE_ID: worker},
        }
        evidence = {
            "p10_result_sha256": p11.P10_RESULT_SHA256,
            "p9_bundle_sha256": "5" * 64,
            "canary_pass_sha256": "6" * 64,
            "rollback_drill_sha256": "7" * 64,
            "raw_token_sha256": "8" * 64,
        }
        return health, manifest, ack, evidence

    def build(self):
        health, manifest, ack, evidence = self.fixture()
        return p11.build_preimage(
            captured_at="2026-10-03T17:30:00Z",
            health=health,
            manifest=manifest,
            manifest_file_sha256="9" * 64,
            ack=ack,
            ack_file_sha256="a" * 64,
            txn_text="REBIND txn-1 7 6 5 shared_down_proj 26 deadbeef",
            txn_file_sha256="b" * 64,
            persistent_binary_sha256="c" * 64,
            p10_evidence=evidence,
        )

    def test_exact_preimage_builds_fail_closed_plan(self):
        preimage = self.build()
        plan = p11.build_plan(preimage, executor_source_sha256="d" * 64, production_binary_rebind_compat_sha256="e" * 64)
        self.assertEqual(plan["status"], "AWAITING_TRUSTED_PRODUCTION_APPROVAL")
        self.assertFalse(plan["production_cutover_allowed"])
        self.assertFalse(plan["production_write_allowed"])
        self.assertFalse(plan["automatic_live_promotion"])
        self.assertEqual(plan["expected_live_preimage"]["worker_pid"], 35493)
        self.assertEqual(plan["expected_live_preimage"]["weight_epoch"], 8)
        self.assertEqual(plan["target"]["expected_after_epoch"], 9)
        self.assertEqual(plan["target"]["runtime_policy_hash"], p11.TARGET_POLICY_HASH)
        check = dict(plan)
        digest = check.pop("cutover_plan_sha256")
        self.assertEqual(digest, p11.sha256_json(check))

    def test_epoch_drift_is_rejected(self):
        health, manifest, ack, evidence = self.fixture()
        health["workers"][p11.ACTIVE_ROUTE_ID]["weight_epoch"] = 9
        with self.assertRaisesRegex(p11.P11Error, "epoch mismatch"):
            p11.build_preimage(
                captured_at="2026-10-03T17:30:00Z",
                health=health,
                manifest=manifest,
                manifest_file_sha256="9" * 64,
                ack=ack,
                ack_file_sha256="a" * 64,
                txn_text="REBIND txn-1 x",
                txn_file_sha256="b" * 64,
                persistent_binary_sha256="c" * 64,
                p10_evidence=evidence,
            )

    def test_policy_drift_is_rejected(self):
        health, manifest, ack, evidence = self.fixture()
        health["workers"][p11.ACTIVE_ROUTE_ID]["runtime_policy"][1]["n"] = 4
        with self.assertRaisesRegex(p11.P11Error, "policy"):
            p11.build_preimage(
                captured_at="2026-10-03T17:30:00Z",
                health=health,
                manifest=manifest,
                manifest_file_sha256="9" * 64,
                ack=ack,
                ack_file_sha256="a" * 64,
                txn_text="REBIND txn-1 x",
                txn_file_sha256="b" * 64,
                persistent_binary_sha256="c" * 64,
                p10_evidence=evidence,
            )

    def test_approval_request_does_not_enable_cutover(self):
        plan = p11.build_plan(self.build(), executor_source_sha256="d" * 64, production_binary_rebind_compat_sha256="e" * 64)
        req = p11.build_approval_request(plan)
        self.assertEqual(req["status"], "AWAITING_TRUSTED_PRODUCTION_APPROVAL")
        self.assertFalse(req["trusted_production_approval_present"])
        self.assertFalse(req["production_cutover_allowed"])
        self.assertFalse(req["automatic_live_promotion"])
        self.assertEqual(req["worker_pid"], 35493)
        self.assertEqual(req["weight_epoch"], 8)
        self.assertEqual(req["candidate_policy_hash"], p11.TARGET_POLICY_HASH)


if __name__ == "__main__":
    unittest.main()
