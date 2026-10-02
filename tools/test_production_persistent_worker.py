#!/usr/bin/env python3
from pathlib import Path
import tempfile
import unittest

import production_persistent_worker as pw
import production_serving_supervisor as legacy
import production_serving_supervisor_persistent as ps


class PolicyTests(unittest.TestCase):
    def test_baseline_expected_policy_is_empty(self):
        self.assertEqual(pw._expected_active_policy(legacy.baseline_route()), [])

    def test_candidate_expected_policy_is_exact_target(self):
        self.assertEqual(
            pw._expected_active_policy(legacy.candidate_route()),
            [{"role":"shared_up_proj","layer":3,"n":6}],
        )


class PoolTests(unittest.TestCase):
    def test_pool_maps_both_certified_routes(self):
        with tempfile.TemporaryDirectory() as td:
            pool=pw.PersistentWorkerPool(root=td,binary="/tmp/not-used")
            pool.add(name="baseline",route=legacy.baseline_route())
            pool.add(name="candidate",route=legacy.candidate_route())
            self.assertEqual(
                set(pool.workers),
                {
                    "beglin-baseline-q4g64",
                    "beglin-candidate-shared-up-l3-n6",
                },
            )

    def test_persistent_supervisor_build_pool_has_two_workers(self):
        with tempfile.TemporaryDirectory() as td:
            old=ps.WORKER_ROOT
            try:
                ps.WORKER_ROOT=Path(td)
                pool=ps.build_pool(binary="/tmp/not-used")
                self.assertEqual(len(pool.workers),2)
            finally:
                ps.WORKER_ROOT=old


class RouteTests(unittest.TestCase):
    def test_current_route_contract_is_compatible(self):
        for route in (legacy.baseline_route(),legacy.candidate_route()):
            normalized=legacy.routing.normalize_route(route)
            legacy._route_policy_line(normalized)

    def test_nonloopback_contract_remains_forbidden(self):
        self.assertEqual("127.0.0.1", legacy.router_capability(
            route_manifest="/tmp/route.json",port=18765
        )["listen_host"])


if __name__=="__main__":
    unittest.main()
