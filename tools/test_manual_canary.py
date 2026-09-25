#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
import tempfile
import unittest

import manual_canary_contract as mc
import manual_canary_controller as ctl


def h(ch):
    return ch * 64


BASELINE = [{"role": "shared_down_proj", "layer": 4, "n": 5}]
CANDIDATE = [{"role": "shared_down_proj", "layer": 4, "n": 6}]
NOW = datetime(2026, 9, 26, 0, 1, tzinfo=timezone.utc)


def base_proposal(**overrides):
    value = {
        "mode": "dry_run",
        "production_write_allowed": False,
        "proposal_id": "p-1",
        "proposer": "planner-agent",
        "environment_id": "fixture-env",
        "model_revision": "fixture-model",
        "backend": "mlx_metal",
        "architecture": "deepseek-v2-lite",
        "source_commit": "deadbeef",
        "binary_sha256": h("a"),
        "checkpoint_sha256": h("b"),
        "baseline_policy_hash": mc.sha256_json(BASELINE),
        "candidate_policy_hash": mc.sha256_json(CANDIDATE),
        "single_target": {
            "role": "shared_down_proj",
            "layer": 4,
            "before_n": 5,
            "after_n": 6,
        },
        "evidence_refs": [
            {"kind": "G4_A_B_R", "run_id": "g4-1", "sha256": h("c")},
            {"kind": "G6_RESTART_CANARY", "run_id": "g6-1", "sha256": h("d")},
        ],
        "budget": {
            "max_requests": 50,
            "max_tokens": 1000,
            "max_duration_ms": 10000,
            "max_memory_bytes": 1024 * 1024,
        },
        "expected_epoch": 7,
        "restart_instance_id": "worker-pre",
        "kill_switch_scope": "single-target",
        "rollback_plan": "restore exact baseline preimage",
    }
    value.update(overrides)
    return value


def approval(proposal, *, now=None, **overrides):
    now = now or datetime(2026, 9, 26, tzinfo=timezone.utc)
    value = {
        "approval_id": "a-1",
        "proposal_digest": mc.proposal_digest(proposal),
        "issuer": "human-reviewer",
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=10)).isoformat(),
        "nonce": "nonce-1",
        "mode": "dry_run",
        "signature_status": "DRY_RUN_TEST_ONLY",
        "production_write_allowed": False,
    }
    value.update(overrides)
    return value


def evidence(proposal):
    p = mc.normalize_proposal(proposal)
    return {
        "production_write_allowed": False,
        "g4_status": "PASS",
        "g6_status": "PASS",
        "refs": p["evidence_refs"],
    }


class ContractTests(unittest.TestCase):
    def test_production_mode_is_rejected(self):
        with self.assertRaises(mc.ManualCanaryContractError):
            mc.normalize_proposal(base_proposal(mode="production"))

    def test_production_write_flag_is_rejected(self):
        with self.assertRaises(mc.ManualCanaryContractError):
            mc.normalize_proposal(base_proposal(production_write_allowed=True))

    def test_single_target_requires_real_change(self):
        p = base_proposal()
        p["single_target"]["after_n"] = 5
        with self.assertRaises(mc.ManualCanaryContractError):
            mc.normalize_proposal(p)

    def test_both_g4_and_g6_evidence_are_required(self):
        p = base_proposal()
        p["evidence_refs"] = p["evidence_refs"][:1]
        with self.assertRaises(mc.ManualCanaryContractError):
            mc.normalize_proposal(p)

    def test_self_approval_is_rejected(self):
        p = base_proposal()
        with self.assertRaises(mc.ManualCanaryContractError):
            mc.validate_approval(proposal=p, approval=approval(p, issuer="planner-agent"), now=NOW)

    def test_forged_proposal_digest_is_rejected(self):
        p = base_proposal()
        with self.assertRaises(mc.ManualCanaryContractError):
            mc.validate_approval(proposal=p, approval=approval(p, proposal_digest=h("f")), now=NOW)

    def test_expired_approval_is_rejected_at_boundary(self):
        p = base_proposal()
        issued = datetime(2026, 9, 26, tzinfo=timezone.utc)
        a = approval(p, now=issued, expires_at=(issued + timedelta(seconds=1)).isoformat())
        with self.assertRaises(mc.ManualCanaryContractError):
            mc.validate_approval(proposal=p, approval=a, now=issued + timedelta(seconds=1))

    def test_untrusted_signature_status_is_rejected(self):
        p = base_proposal()
        with self.assertRaises(mc.ManualCanaryContractError):
            mc.validate_approval(
                proposal=p,
                approval=approval(p, signature_status="TRUSTED_PRODUCTION_SIGNATURE"),
                now=NOW,
            )

    def test_nonce_reuse_is_rejected(self):
        p = base_proposal()
        with self.assertRaises(mc.ManualCanaryContractError):
            mc.validate_approval(
                proposal=p,
                approval=approval(p),
                now=NOW,
                consumed_nonces={"nonce-1"},
            )


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = base_proposal()
        self.adapter = ctl.DryRunAdapter(
            policy=BASELINE.copy(),
            epoch=7,
            policy_hash=mc.sha256_json(BASELINE),
        )
        self.store = ctl.ManualCanaryStore(self.tmp.name, "run-1")
        self.controller = ctl.ManualCanaryController(
            store=self.store,
            adapter=self.adapter,
            proposal=self.p,
            baseline_policy=BASELINE,
            candidate_policy=CANDIDATE,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def approve(self, *, nonce="nonce-1"):
        self.controller.initialize()
        self.controller.verify_shadow_evidence(evidence(self.p))
        self.controller.await_approval()
        return self.controller.validate_manual_approval(
            approval(self.p, nonce=nonce),
            now=NOW,
        )

    def start(self):
        self.approve()
        self.controller.prepare_canary()
        return self.controller.start_canary()

    def test_start_before_approval_is_rejected(self):
        self.controller.initialize()
        with self.assertRaises(ctl.ManualCanaryControllerError):
            self.controller.start_canary()

    def test_verify_before_initialize_is_rejected(self):
        with self.assertRaises(ctl.ManualCanaryControllerError):
            self.controller.verify_shadow_evidence(evidence(self.p))

    def test_reinitialize_same_run_is_rejected(self):
        self.controller.initialize()
        with self.assertRaises(ctl.ManualCanaryControllerError):
            self.controller.initialize()

    def test_duplicate_evidence_rows_are_rejected(self):
        self.controller.initialize()
        ev = evidence(self.p)
        ev["refs"].append(dict(ev["refs"][0]))
        with self.assertRaises(ctl.ManualCanaryControllerError):
            self.controller.verify_shadow_evidence(ev)

    def test_happy_dry_run_ends_review_required_not_auto_promoted(self):
        self.start()
        state = self.controller.observe(
            requests=20, tokens=500, duration_ms=5000, memory_bytes=1000,
            regression=False,
        )
        self.assertEqual(state["state"], "CANARY_PASS_REVIEW_REQUIRED")
        self.assertFalse(state["production_write_allowed"])

    def test_multitarget_policy_is_rejected(self):
        candidate = CANDIDATE + [{"role": "shared_up_proj", "layer": 3, "n": 6}]
        p = base_proposal(candidate_policy_hash=mc.sha256_json(candidate))
        with self.assertRaises(ctl.ManualCanaryControllerError):
            ctl.ManualCanaryController(
                store=self.store, adapter=self.adapter, proposal=p,
                baseline_policy=BASELINE, candidate_policy=candidate,
            )

    def test_baseline_epoch_drift_is_rejected(self):
        self.approve()
        self.adapter.epoch = 8
        with self.assertRaises(ctl.ManualCanaryControllerError):
            self.controller.prepare_canary()

    def test_baseline_policy_drift_is_rejected(self):
        self.approve()
        self.adapter.policy_hash = h("e")
        with self.assertRaises(ctl.ManualCanaryControllerError):
            self.controller.prepare_canary()

    def test_kill_before_apply_is_isolated(self):
        self.approve()
        self.controller.prepare_canary()
        self.store.request_kill(reason="operator stop", requested_by="human-reviewer")
        state = self.controller.start_canary()
        self.assertEqual(state["state"], "ISOLATED")
        self.assertEqual(self.adapter.policy_hash, mc.sha256_json(BASELINE))

    def test_kill_during_run_requires_rollback_then_exact_baseline(self):
        self.start()
        self.store.request_kill(reason="operator stop", requested_by="human-reviewer")
        state = self.controller.observe(
            requests=1, tokens=1, duration_ms=1, memory_bytes=1,
            regression=False,
        )
        self.assertEqual(state["state"], "ROLLBACK_REQUIRED")
        restored = self.controller.rollback(txn_id="txn-kill")
        self.assertEqual(restored["state"], "ROLLBACK_VERIFIED")
        self.assertEqual(restored["restored_policy_hash"], mc.sha256_json(BASELINE))

    def test_each_budget_axis_requires_rollback(self):
        cases = [
            dict(requests=51, tokens=1, duration_ms=1, memory_bytes=1),
            dict(requests=1, tokens=1001, duration_ms=1, memory_bytes=1),
            dict(requests=1, tokens=1, duration_ms=10001, memory_bytes=1),
            dict(requests=1, tokens=1, duration_ms=1, memory_bytes=1024*1024+1),
        ]
        for idx, metrics in enumerate(cases):
            with self.subTest(idx=idx):
                tmp = tempfile.TemporaryDirectory()
                try:
                    adapter = ctl.DryRunAdapter(
                        policy=BASELINE.copy(), epoch=7,
                        policy_hash=mc.sha256_json(BASELINE),
                    )
                    controller = ctl.ManualCanaryController(
                        store=ctl.ManualCanaryStore(tmp.name, f"budget-{idx}"),
                        adapter=adapter, proposal=self.p,
                        baseline_policy=BASELINE, candidate_policy=CANDIDATE,
                    )
                    controller.initialize()
                    controller.verify_shadow_evidence(evidence(self.p))
                    controller.await_approval()
                    controller.validate_manual_approval(
                        approval(self.p, nonce=f"nonce-{idx}"), now=NOW
                    )
                    controller.prepare_canary()
                    controller.start_canary()
                    state = controller.observe(regression=False, **metrics)
                    self.assertEqual(state["state"], "ROLLBACK_REQUIRED")
                    self.assertIn("budget exceeded", state["reason"])
                finally:
                    tmp.cleanup()

    def test_negative_metric_is_rejected(self):
        self.start()
        with self.assertRaises(ctl.ManualCanaryControllerError):
            self.controller.observe(
                requests=-1,tokens=1,duration_ms=1,memory_bytes=1,regression=False
            )

    def test_regression_requires_rollback(self):
        self.start()
        state = self.controller.observe(
            requests=1,tokens=1,duration_ms=1,memory_bytes=1,regression=True
        )
        self.assertEqual(state["state"], "ROLLBACK_REQUIRED")

    def test_inconclusive_requires_rollback(self):
        self.start()
        state = self.controller.observe(
            requests=1,tokens=1,duration_ms=1,memory_bytes=1,
            regression=False,inconclusive=True,
        )
        self.assertEqual(state["state"], "ROLLBACK_REQUIRED")

    def test_apply_failure_is_isolated(self):
        self.approve()
        self.controller.prepare_canary()
        self.adapter.fail_apply = True
        self.assertEqual(self.controller.start_canary()["state"], "ISOLATED")

    def test_sync_failure_is_isolated(self):
        self.start()
        self.controller.observe(requests=1,tokens=1,duration_ms=1,memory_bytes=1,regression=True)
        self.adapter.fail_sync = True
        self.assertEqual(self.controller.rollback(txn_id="txn-sync")["state"], "ISOLATED")

    def test_restore_failure_is_isolated(self):
        self.start()
        self.controller.observe(requests=1,tokens=1,duration_ms=1,memory_bytes=1,regression=True)
        self.adapter.fail_restore = True
        self.assertEqual(self.controller.rollback(txn_id="txn-restore")["state"], "ISOLATED")

    def test_nonce_is_durable_across_store_restart(self):
        self.approve()
        restarted = ctl.ManualCanaryStore(self.tmp.name, "run-2")
        self.assertIn("nonce-1", restarted.consumed_nonces())

    def test_restart_before_mutation_preserves_approval_state(self):
        self.approve()
        state = self.controller.reconcile_after_restart()
        self.assertEqual(state["state"], "APPROVAL_VALIDATED")

    def test_restart_pending_baseline_becomes_verified(self):
        self.start()
        self.controller.observe(requests=1,tokens=1,duration_ms=1,memory_bytes=1,regression=True)
        self.store.transition("ROLLBACK_PENDING", txn_id="txn-r")
        self.adapter.policy = BASELINE.copy()
        self.adapter.policy_hash = mc.sha256_json(BASELINE)
        self.adapter.epoch = 9
        self.assertEqual(self.controller.reconcile_after_restart()["state"], "ROLLBACK_VERIFIED")

    def test_restart_candidate_requires_rollback(self):
        self.start()
        self.assertEqual(self.controller.reconcile_after_restart()["state"], "ROLLBACK_REQUIRED")

    def test_restart_unknown_policy_is_isolated(self):
        self.start()
        self.adapter.policy_hash = h("f")
        self.assertEqual(self.controller.reconcile_after_restart()["state"], "ISOLATED")

    def _rollback_with_adapter(self, adapter):
        store = ctl.ManualCanaryStore(self.tmp.name, "ack-fixture-"+str(id(adapter)))
        controller = ctl.ManualCanaryController(
            store=store,adapter=adapter,proposal=self.p,
            baseline_policy=BASELINE,candidate_policy=CANDIDATE,
        )
        controller.initialize()
        controller.verify_shadow_evidence(evidence(self.p))
        controller.await_approval()
        controller.validate_manual_approval(
            approval(self.p, nonce="nonce-"+str(id(adapter))),now=NOW
        )
        controller.prepare_canary()
        controller.start_canary()
        controller.observe(requests=1,tokens=1,duration_ms=1,memory_bytes=1,regression=True)
        return controller.rollback(txn_id="txn-expected")

    def test_wrong_rollback_policy_report_is_isolated(self):
        class WrongPolicy(ctl.DryRunAdapter):
            def restore_baseline(self, *, policy, policy_hash, txn_id):
                before=self.epoch
                self.epoch += 1
                self.policy_hash=h("f")
                return {**self.query(),"txn_id":txn_id,"previous_epoch":before}
        adapter=WrongPolicy(policy=BASELINE.copy(),epoch=7,policy_hash=mc.sha256_json(BASELINE))
        self.assertEqual(self._rollback_with_adapter(adapter)["state"],"ISOLATED")

    def test_wrong_rollback_txn_is_isolated(self):
        class WrongTxn(ctl.DryRunAdapter):
            def restore_baseline(self, *, policy, policy_hash, txn_id):
                row=super().restore_baseline(policy=policy,policy_hash=policy_hash,txn_id=txn_id)
                row["txn_id"]="other"
                return row
        adapter=WrongTxn(policy=BASELINE.copy(),epoch=7,policy_hash=mc.sha256_json(BASELINE))
        self.assertEqual(self._rollback_with_adapter(adapter)["state"],"ISOLATED")

    def test_wrong_previous_epoch_is_isolated(self):
        class WrongPrevious(ctl.DryRunAdapter):
            def restore_baseline(self, *, policy, policy_hash, txn_id):
                row=super().restore_baseline(policy=policy,policy_hash=policy_hash,txn_id=txn_id)
                row["previous_epoch"] -= 1
                return row
        adapter=WrongPrevious(policy=BASELINE.copy(),epoch=7,policy_hash=mc.sha256_json(BASELINE))
        self.assertEqual(self._rollback_with_adapter(adapter)["state"],"ISOLATED")

    def test_nonadvancing_restore_epoch_is_isolated(self):
        class NoAdvance(ctl.DryRunAdapter):
            def restore_baseline(self, *, policy, policy_hash, txn_id):
                before=self.epoch
                self.policy=list(policy)
                self.policy_hash=policy_hash
                return {**self.query(),"txn_id":txn_id,"previous_epoch":before}
        adapter=NoAdvance(policy=BASELINE.copy(),epoch=7,policy_hash=mc.sha256_json(BASELINE))
        self.assertEqual(self._rollback_with_adapter(adapter)["state"],"ISOLATED")


if __name__ == "__main__":
    unittest.main()
