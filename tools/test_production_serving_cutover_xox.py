#!/usr/bin/env python3
import unittest

import production_serving_cutover_xox as cut


class PlanTests(unittest.TestCase):
    def test_exact_plan_sha_is_stable(self):
        plan=cut.build_exact_plan()
        self.assertEqual(
            plan["plan_sha256"],
            "d398f2c56181f7ef819bbdce7418e00a73571687027bbc00e67e477246585c3d",
        )
        self.assertEqual(
            plan["candidate_route_id"],
            "beglin-candidate-shared-up-l3-n6",
        )
        self.assertEqual(
            plan["rollback_route_id"],
            "beglin-baseline-q4g64",
        )
        self.assertFalse(plan["automatic_cutover_allowed"])
        self.assertFalse(plan["production_routing_switch_allowed"])


class HealthTests(unittest.TestCase):
    def _response(self, token=1224, count=12, duration_ms=6000):
        return {
            "finite_logits": True,
            "duration_ms": duration_ms,
            "peak_child_rss_bytes": 6300000000,
            "worker_instance_id": "pid-test",
            "route": {
                "route_id": "beglin-candidate-shared-up-l3-n6",
                "policy_hash": "0e048b8c5c50a1c005ea573caadd6ccdf99797c24b9c443e993fe0d7b0b27141",
            },
            "responses": [
                {"generated_tokens": [0]*8+[token,1]}
                for _ in range(count)
            ],
        }

    def test_good_candidate_health_is_exact_12_of_12(self):
        metrics, detail=cut.health_from_response(self._response())
        self.assertEqual(metrics["requests"],12)
        self.assertEqual(metrics["errors"],0)
        self.assertTrue(metrics["reference_parity"])
        self.assertEqual(detail["reference_hits"],12)
        self.assertEqual(detail["distinct_reference_tokens"],[1224])

    def test_reference_mismatch_marks_health_failure(self):
        metrics, detail=cut.health_from_response(self._response(token=3268))
        self.assertEqual(metrics["errors"],1)
        self.assertFalse(metrics["reference_parity"])
        self.assertEqual(detail["reference_hits"],0)

    def test_route_mismatch_marks_health_failure(self):
        response=self._response()
        response["route"]["route_id"]="beglin-baseline-q4g64"
        metrics,_=cut.health_from_response(response)
        self.assertEqual(metrics["errors"],1)
        self.assertFalse(metrics["reference_parity"])


if __name__=="__main__":
    unittest.main()
