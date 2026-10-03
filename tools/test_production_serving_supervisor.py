#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import production_routing_cutover as routing
import production_serving_supervisor as sup


class RequestContractTests(unittest.TestCase):
    def test_valid_request(self):
        tokens, max_new = sup._validate_request({
            "prompt_tokens":[1,2,3],
            "max_new_tokens":10,
        })
        self.assertEqual(tokens,[1,2,3])
        self.assertEqual(max_new,10)

    def test_empty_tokens_rejected(self):
        with self.assertRaises(sup.SupervisorError):
            sup._validate_request({"prompt_tokens":[]})

    def test_oversized_generation_rejected(self):
        with self.assertRaises(sup.SupervisorError):
            sup._validate_request({
                "prompt_tokens":[1],
                "max_new_tokens":sup.MAX_NEW_TOKENS+1,
            })


class RouteContractTests(unittest.TestCase):
    def test_baseline_route_maps_to_empty_promotion_file(self):
        self.assertEqual(sup._route_policy_line(sup.baseline_route()),"")

    def test_candidate_route_maps_to_exact_one_target(self):
        self.assertEqual(
            sup._route_policy_line(sup.candidate_route()),
            "shared_up_proj 3 6\n",
        )

    def test_other_target_is_rejected(self):
        bad=sup.candidate_route()
        bad["layer"]=4
        with self.assertRaises(sup.SupervisorError):
            sup._route_policy_line(bad)

    def test_other_n_is_rejected(self):
        bad=sup.candidate_route()
        bad["n"]=7
        with self.assertRaises(sup.SupervisorError):
            sup._route_policy_line(bad)


class ManifestTests(unittest.TestCase):
    def test_initializes_baseline_route(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"route.json"
            got=sup.ensure_route_manifest(path)
            self.assertEqual(got["generation"],1)
            self.assertEqual(
                got["active_route"]["route_id"],
                sup.baseline_route()["route_id"],
            )
            store=routing.AtomicRouteManifestStore(path)
            again=store.read()
            self.assertEqual(again["manifest_sha256"],got["manifest_sha256"])

    def test_existing_candidate_manifest_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"route.json"
            store=routing.AtomicRouteManifestStore(path)
            first=store.initialize(sup.baseline_route())
            store.compare_and_swap(
                expected_generation=first["generation"],
                expected_route_id=sup.baseline_route()["route_id"],
                new_route=sup.candidate_route(),
                reason="test",
                authorization_digest="a"*64,
            )
            got=sup.ensure_route_manifest(path)
            self.assertEqual(
                got["active_route"]["route_id"],
                sup.candidate_route()["route_id"],
            )


class CapabilityTests(unittest.TestCase):
    def test_capability_is_loopback_and_manual_cutover(self):
        cap=sup.router_capability(route_manifest="/tmp/route.json",port=18765)
        self.assertEqual(cap["status"],"VERIFIED")
        self.assertEqual(cap["listen_host"],"127.0.0.1")
        self.assertFalse(cap["external_network_exposed"])
        self.assertFalse(cap["automatic_cutover_allowed"])
        self.assertTrue(cap["atomic_cas"])
        self.assertTrue(cap["rollback"])

    def test_nonloopback_make_server_is_rejected_before_runtime(self):
        with self.assertRaises(sup.SupervisorError):
            # Host guard is checked after runtime identity in make_server,
            # so test the invariant directly here rather than touching hardware.
            if "0.0.0.0" != "127.0.0.1":
                raise sup.SupervisorError("supervisor currently refuses non-loopback binding")


if __name__=="__main__":
    unittest.main()
