#!/usr/bin/env python3
from pathlib import Path
import tempfile
import unittest

import production_routing_cutover as cutover


H=lambda c:c*64


def bridge_evidence():
    return {
        "schema":"beglin-production-adapter-bridge-execution/1",
        "plan":{"plan_sha256":"d26bcc0273a887f4341581536e41dcb4398c319138107d12c9061aa3d07bc1dc"},
        "result":{
            "schema":"mlx-isolated-restart-canary-result-v1",
            "source_commit":"330954b27f146b8a17db2cb353c3e620968bad5e",
            "binary_sha256":"6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392",
            "checkpoint_sha256":"1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898",
            "candidate_policy_hash":"0e048b8c5c50a1c005ea573caadd6ccdf99797c24b9c443e993fe0d7b0b27141",
            "requested_policy_applied":True,
            "finite_logits":True,
            "reference_token_match":True,
            "candidate_worker_exited":True,
            "baseline_worker_mutated":False,
            "production_routing_switched":False,
            "result_sha256":"37227ea66f6dc5766e879eca383e41494860bd7d5496334de5bc8ecd90f3202e",
        },
        "verdict":{
            "status":"CANARY_PASS_REVIEW_REQUIRED",
            "decision":"NO_AUTO_PROMOTION",
            "production_routing_switch_allowed":False,
            "verdict_sha256":"1cc814c61e886befd30d1b9549d47ab4a53186a16d4f81ce82f609e9ccf8a1ea",
        },
    }


def baseline_route():
    return {
        "route_id":"baseline-330954",
        "worker_instance_id":"pid-baseline",
        "endpoint":"shadow://baseline",
        "source_commit":"330954b27f146b8a17db2cb353c3e620968bad5e",
        "binary_sha256":"6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392",
        "checkpoint_sha256":"1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898",
        "policy_hash":"4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "role":"shared_up_proj","layer":3,"n":None,
    }


def candidate_route():
    return {
        "route_id":"candidate-shared-up-l3-n6",
        "worker_instance_id":"pid-candidate",
        "endpoint":"shadow://candidate",
        "source_commit":"330954b27f146b8a17db2cb353c3e620968bad5e",
        "binary_sha256":"6e6258d6d4e402033c303c8c4b622db5a088354a0de79e687eb0fad4190c4392",
        "checkpoint_sha256":"1ea6be7e3e5236d6d6f7c265f725f703db1b64170faf672ee9b98352b131e898",
        "policy_hash":"0e048b8c5c50a1c005ea573caadd6ccdf99797c24b9c443e993fe0d7b0b27141",
        "role":"shared_up_proj","layer":3,"n":6,
    }


def no_router():
    return {
        "schema":"beglin-serving-router-capability/1",
        "status":"ABSENT",
        "router_id":"",
        "implementation_verified":False,
        "atomic_cas":False,
        "health_observation":False,
        "rollback":False,
        "request_drain":False,
        "route_store_kind":"",
    }


def verified_router():
    return {
        "schema":"beglin-serving-router-capability/1",
        "status":"VERIFIED",
        "router_id":"shadow-supervisor-v1",
        "implementation_verified":True,
        "atomic_cas":True,
        "health_observation":True,
        "rollback":True,
        "request_drain":True,
        "route_store_kind":"atomic-json-manifest",
    }


def plan(cap=None):
    return cutover.build_cutover_plan(
        bridge_status=bridge_evidence(),
        baseline_route=baseline_route(),
        candidate_route=candidate_route(),
        router_capability=cap or verified_router(),
    )


def approval(p):
    return {
        "schema":"beglin-production-routing-cutover-approval-v1",
        "status":"VERIFIED_GITHUB_OIDC",
        "cutover_plan_sha256":p["plan_sha256"],
        "candidate_route_id":p["candidate_route_id"],
        "baseline_route_id":p["rollback_route_id"],
        "automatic_cutover_allowed":False,
        "production_routing_switch_allowed":True,
    }


class PlanTests(unittest.TestCase):
    def test_no_router_blocks_cutover(self):
        p=plan(no_router())
        self.assertEqual(p["status"],"BLOCKED_NO_VERIFIED_SERVING_ROUTER")
        self.assertFalse(p["router_verified"])
        self.assertFalse(p["production_routing_switch_allowed"])

    def test_verified_router_requires_explicit_cutover_approval(self):
        p=plan()
        self.assertEqual(p["status"],"READY_FOR_EXPLICIT_CUTOVER_APPROVAL")
        self.assertTrue(p["router_verified"])
        self.assertTrue(p["cutover_authorization_required"])
        self.assertFalse(p["automatic_cutover_allowed"])
        self.assertFalse(p["production_routing_switch_allowed"])

    def test_bad_bridge_verdict_is_rejected(self):
        b=bridge_evidence()
        b["verdict"]["status"]="ROLLBACK_REQUIRED"
        with self.assertRaises(cutover.RoutingCutoverError):
            cutover.build_cutover_plan(
                bridge_status=b,
                baseline_route=baseline_route(),
                candidate_route=candidate_route(),
                router_capability=verified_router(),
            )


class RouteStoreTests(unittest.TestCase):
    def test_atomic_cutover_and_rollback(self):
        with tempfile.TemporaryDirectory() as td:
            store=cutover.AtomicRouteManifestStore(Path(td)/"active-route.json")
            initial=store.initialize(baseline_route())
            self.assertEqual(initial["generation"],1)
            p=plan()
            result=cutover.execute_cutover(
                store=store,cutover_plan=p,approval=approval(p)
            )
            self.assertEqual(result["status"],"CUTOVER_COMMITTED_AWAITING_HEALTH")
            self.assertEqual(result["generation"],2)
            self.assertEqual(result["active_route"]["route_id"],candidate_route()["route_id"])
            health=cutover.evaluate_health(
                cutover_plan=p,
                metrics={"requests":12,"errors":1,"duration_ms":5000,"reference_parity":False},
            )
            self.assertEqual(health["status"],"ROLLBACK_REQUIRED")
            rb=cutover.rollback(
                store=store,cutover_plan=p,cutover_result=result,reason="health failure"
            )
            self.assertEqual(rb["status"],"ROLLBACK_COMMITTED")
            self.assertEqual(rb["generation"],3)
            self.assertEqual(rb["active_route"]["route_id"],baseline_route()["route_id"])

    def test_stale_generation_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store=cutover.AtomicRouteManifestStore(Path(td)/"active-route.json")
            store.initialize(baseline_route())
            with self.assertRaises(cutover.StaleRoute):
                store.compare_and_swap(
                    expected_generation=0,
                    expected_route_id=baseline_route()["route_id"],
                    new_route=candidate_route(),
                    reason="stale",
                    authorization_digest=H("a"),
                )

    def test_missing_explicit_approval_blocks_switch(self):
        with tempfile.TemporaryDirectory() as td:
            store=cutover.AtomicRouteManifestStore(Path(td)/"active-route.json")
            store.initialize(baseline_route())
            p=plan()
            bad=approval(p)
            bad["production_routing_switch_allowed"]=False
            with self.assertRaises(cutover.CutoverAuthorizationError):
                cutover.execute_cutover(store=store,cutover_plan=p,approval=bad)


class HealthTests(unittest.TestCase):
    def test_clean_health_still_requires_review(self):
        got=cutover.evaluate_health(
            cutover_plan=plan(),
            metrics={"requests":12,"errors":0,"duration_ms":6000,"reference_parity":True},
        )
        self.assertEqual(got["status"],"CUTOVER_HEALTH_PASS_REVIEW_REQUIRED")
        self.assertEqual(got["error_rate"],0.0)

    def test_reference_mismatch_forces_rollback(self):
        got=cutover.evaluate_health(
            cutover_plan=plan(),
            metrics={"requests":12,"errors":0,"duration_ms":6000,"reference_parity":False},
        )
        self.assertEqual(got["status"],"ROLLBACK_REQUIRED")
        self.assertIn("reference_parity",got["reasons"])


if __name__=="__main__":
    unittest.main()
