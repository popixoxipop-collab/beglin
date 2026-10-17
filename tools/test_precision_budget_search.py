#!/usr/bin/env python3
"""Offline verification of qNg64 storage and exact mixed-policy search."""
import copy
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qng64_storage import (effective_bits_per_weight, packed_weight_bytes,
    check_backend_n, QuantizationConfigurationError)
from precision_budget_search import (SCHEMA, PlanError, InfeasibleBudget,
    StateLimitExceeded, solve)
from precision_budget_fixture import generate, quant_sim_error, bf16_truncate


def input_profile(n_tensors=2, backend="cpu", budget=2200):
    return {"schema": SCHEMA, "backend": backend, "model_id": "synthetic-fixture",
            "checkpoint_sha256": "a" * 64, "budget_bytes": budget,
            "tensors": [{"name": f"synthetic.tensor.{i}", "role": "q_proj" if i%2==0 else "down_proj",
                        "layer": i//2, "expert_id": i%2, "elements": 128,
                        "importance": 1.0+i/2, "candidates": [
                            {"n": 2, "rel_l2": .27+i*.03},
                            {"n": 5, "rel_l2": .09+i*.01},
                            {"n": 7, "rel_l2": .01+i*.005},
                        ]} for i in range(n_tensors)]}


class TestQng64Storage(unittest.TestCase):
    def test_exact_packed_code_plus_scale_bytes(self):
        for n in range(2, 16):
            if n == 4:  # separate q4g64 format, never qNg64
                continue
            self.assertEqual(packed_weight_bytes(64, n), n*8+4)
            self.assertEqual(packed_weight_bytes(128, n), 2*(n*8+4))
            self.assertEqual(effective_bits_per_weight(n), n + 0.5)
        with self.assertRaises(QuantizationConfigurationError):
            packed_weight_bytes(64, 4)
        with self.assertRaises(QuantizationConfigurationError):
            effective_bits_per_weight(4)
        self.assertEqual(effective_bits_per_weight(5), 5.5)
        self.assertEqual(effective_bits_per_weight(7), 7.5)

    def test_gpu_supported_bits(self):
        for n in (2,3,5,6,7,9,10,11,12,13,14,15):
            self.assertEqual(check_backend_n("mlx_metal", n), n)
        for n in (4,8):
            with self.assertRaises(QuantizationConfigurationError):
                check_backend_n("mlx_metal",n)
        self.assertEqual(check_backend_n("cpu", 8), 8)  # distinct CPU qNg64 n8
        with self.assertRaises(QuantizationConfigurationError):
            check_backend_n("cpu", 4)  # q4g64 is not qNg64

    def test_bad_n_and_shape(self):
        for n in (0,1,4,16,32,True,"5",5.0):
            with self.assertRaises(QuantizationConfigurationError):
                packed_weight_bytes(64,n)
        for elements in (0,-64,3,65,128.0,True):
            with self.assertRaises(QuantizationConfigurationError):
                packed_weight_bytes(elements, 5)
        with self.assertRaises(QuantizationConfigurationError):
            packed_weight_bytes(64,5,group=32)


class TestBudgetPlanner(unittest.TestCase):
    def test_exact_optimum_for_two_tensors(self):
        p=input_profile(budget=192)
        plan=solve(p)
        self.assertLessEqual(plan["serialized_qng64_weight_bytes"],192)
        self.assertEqual(plan["mode"],"OFFLINE_RESEARCH_ONLY_NOT_QUALIFIED_FOR_LIVE")
        self.assertFalse(plan["automatic_live_promotion"])
        choices={row["n"] for row in plan["selections"]}
        self.assertTrue(choices.issubset({2,5,7}))

    def test_gpu_cannot_select_n8_qng64_even_if_requested(self):
        p=input_profile(backend="mlx_metal")
        p["tensors"][0]["candidates"].append({"n":8,"rel_l2":0})
        with self.assertRaisesRegex(PlanError,"does not have a declared"):
            solve(p)

    def test_error_nonmonotone_exact_not_binary_search(self):
        p=input_profile(1, budget=10000)
        p["tensors"][0]["candidates"]=[
            {"n":2,"rel_l2":.2}, {"n":5,"rel_l2":.25}, {"n":7,"rel_l2":.05}]
        plan=solve(p)
        self.assertEqual(plan["selections"][0]["n"],7)
        p["budget_bytes"]=packed_weight_bytes(128,5)
        self.assertEqual(solve(p)["selections"][0]["n"],2)

    def test_impossible_budget(self):
        p=input_profile(2,budget=10)
        with self.assertRaises(InfeasibleBudget): solve(p)

    def test_duplicate_target_disallowed(self):
        p=input_profile(2)
        p["tensors"][1]["role"]=p["tensors"][0]["role"]
        p["tensors"][1]["layer"]=p["tensors"][0]["layer"]
        p["tensors"][1]["expert_id"]=p["tensors"][0]["expert_id"]
        with self.assertRaises(PlanError): solve(p)

    def test_same_name_repeated_not_valid_identity(self):
        p=input_profile(2)
        p["tensors"][1]["name"]=p["tensors"][0]["name"]
        with self.assertRaises(PlanError): solve(p)

    def test_profile_schema_and_values(self):
        p=input_profile()
        for key,value in (("schema","bad"),("backend","cuda"),("checkpoint_sha256","bad"),
                          ("budget_bytes",0),("tensors",[])):
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises(PlanError,msg=key):solve(q)
        q=copy.deepcopy(p);q["tensors"][0]["candidates"][0]["rel_l2"]=float("nan")
        with self.assertRaises(PlanError):solve(q)
        q=copy.deepcopy(p);q["tensors"][0]["candidates"][0]["n"]=True
        with self.assertRaises(PlanError):solve(q)
        q=copy.deepcopy(p);q["tensors"][0]["candidates"].append(q["tensors"][0]["candidates"][0])
        with self.assertRaises(PlanError):solve(q)

    def test_max_rel_l2_bound(self):
        p=input_profile(1,budget=80)
        p["tensors"][0]["max_rel_l2"]=.12
        with self.assertRaises(InfeasibleBudget):solve(p)
        p["budget_bytes"]=packed_weight_bytes(128,5)
        self.assertEqual(solve(p)["selections"][0]["n"],5)

    def test_frontier_limit_fails_no_approximate_claim(self):
        p=input_profile(2,budget=5000)
        with self.assertRaises(StateLimitExceeded):solve(p,max_states=1)

    def test_solver_matches_bruteforce_random_small_nonmonotone(self):
        rng=random.Random(219840)
        for case in range(75):
            q=input_profile(4,budget=rng.choice((500,600,700,950)))
            for t in q["tensors"]:
                for c in t["candidates"]:
                    c["rel_l2"]=rng.random()*0.2
            possible=[]
            for cand in itertools.product(*(t["candidates"] for t in q["tensors"])):
                size=sum(packed_weight_bytes(t["elements"],c["n"]) for t,c in zip(q["tensors"],cand))
                loss=sum(t["importance"]*c["rel_l2"]**2 for t,c in zip(q["tensors"],cand))
                if size <= q["budget_bytes"]:
                    possible.append((loss,size))
            if not possible:
                with self.assertRaises(InfeasibleBudget):solve(q)
            else:
                actual=solve(q)
                wanted=min(possible)
                self.assertAlmostEqual(actual["weighted_relative_l2_squared_proxy"], wanted[0], places=12,
                                       msg=f"case {case}")
                self.assertLessEqual(actual["serialized_qng64_weight_bytes"],q["budget_bytes"])

    def test_policy_digest_stable(self):
        p=input_profile(2)
        self.assertEqual(solve(p)["policy_sha256"],solve(p)["policy_sha256"])
        q=copy.deepcopy(p);q["backend"]="mlx_metal"
        self.assertNotEqual(solve(p)["policy_sha256"],solve(q)["policy_sha256"])

    def test_cli_and_synthetic_reference_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/"profile.json"
            dst=Path(td)/"plan.json"
            profile=generate("mlx_metal",seed=20261017,budget_frac=.5)
            src.write_text(json.dumps(profile))
            r=subprocess.run([sys.executable,str(HERE/"precision_budget_search.py"),"--input",str(src),"--output",str(dst)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            j=json.loads(dst.read_text())
            self.assertEqual(j["backend"],"mlx_metal")
            self.assertLessEqual(j["serialized_qng64_weight_bytes"],j["budget_bytes"])
            self.assertFalse(j["quality_validated"])
            self.assertFalse(j["latency_measured"])
            self.assertFalse(j["gpu_resident_memory_measured"])
            self.assertTrue(all(x["n"] not in {4,8} for x in j["selections"]))


class TestSyntheticAccuracy(unittest.TestCase):
    def test_zero_weights_have_zero_error(self):
        self.assertEqual(quant_sim_error([0.0]*64,5),0)
    def test_error_reduces_on_representative_vector(self):
        w=[bf16_truncate((i-32)*.006) for i in range(64)]
        self.assertLess(quant_sim_error(w,15), quant_sim_error(w,5))


if __name__=="__main__":
    unittest.main(verbosity=2)
