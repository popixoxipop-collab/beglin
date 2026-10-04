#!/usr/bin/env python3
import array
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import backend_adapters_v2 as planv2
import model_capability as mc
import p12_cpu_qng64_restart_canary as canary
import precision_context as pc


def write_fixture(path: Path) -> None:
    vals = array.array("f", [(i - 32) / 17.0 for i in range(128)])
    raw = vals.tobytes()
    header = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32", "shape": [2, 64], "data_offsets": [0, len(raw)]
        }
    }
    hb = json.dumps(header, separators=(",", ":")).encode()
    pad = (-(len(hb)) - 8) % 8
    hb += b" " * pad
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + raw)


def bundle(target_key: str) -> dict:
    evidence = "a" * 64
    out = {
        "schema": "beglin-model-capability-bundle-v1",
        "model_id": "qwen2.5-fixture",
        "checkpoint_identity": "b" * 64,
        "checkpoint_identity_sha256": "b" * 64,
        "weight_checkpoint_identity_sha256": "c" * 64,
        "skeleton_sha256": "d" * 64,
        "tensor_role_graph": {
            "nodes": [{
                "canonical_target_key": target_key,
                "role": "q_proj", "layer": 0, "expert_id": None,
            }]
        },
        "backend_capability_matrix": {"rows": [{
            "target_key": target_key, "backend": "cpu",
            "inference_status": "VERIFIED",
            "quant_formats": ["qNg64"],
            "supported_n": [5, 6],
            "mutable": True,
            "mutation_mode": "RESTART_REQUIRED",
            "transition_limit": 1,
            "validation_required": True,
            "evidence_refs": [evidence],
            "reason_code": "CPU_HOT_MUTATION_NOT_CERTIFIED",
        }],
        "quant_matrix": [{
            "target_key": target_key, "backend": "cpu",
            "status": "VERIFIED", "supported_n": [5, 6],
            "evidence_refs": [evidence],
        }],
        "mutation_matrix": [{
            "target_key": target_key, "backend": "cpu",
            "status": "VERIFIED",
            "mutation_mode": "RESTART_REQUIRED",
            "allowed_target_precisions": [5, 6],
            "max_atomic_targets": 1,
            "requires_quiesce": True,
            "requires_snapshot": False,
            "epoch_increment": True,
            "rollback_supported": True,
            "policy_shape_change_allowed": False,
            "evidence_refs": [evidence],
        }],
        "precision_search_targets": [target_key],
        "p8_p11_eligibility": "PARTIAL",
    }
    out["bundle_sha256"] = mc.stable_identity_sha256(out)
    return out


class CanaryTests(unittest.TestCase):
    def test_materialization_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "model.safetensors"
            a = root / "a.safetensors"
            b = root / "b.safetensors"
            write_fixture(src)
            ra = canary.materialize_tensor(
                checkpoint=src,
                tensor_name="model.layers.0.self_attn.q_proj.weight",
                n=5, output=a,
            )
            rb = canary.materialize_tensor(
                checkpoint=src,
                tensor_name="model.layers.0.self_attn.q_proj.weight",
                n=5, output=b,
            )
            self.assertEqual(ra["artifact_sha256"], rb["artifact_sha256"])
            self.assertEqual(ra["shape"], [2, 64])
            self.assertEqual(ra["group_size"], 64)
            self.assertGreater(ra["rel_l2"], 0.0)

    def test_restart_canary_passes_common_cpu_contract(self):
        key = "qwen2.5-fixture/L0/q_proj"
        cap = bundle(key)
        state = {
            "epoch": 4,
            "policy": [{"role": "q_proj", "layer": 0, "n": 6}],
        }

        def query():
            return copy.deepcopy(state)

        def restart(policy):
            state["epoch"] += 1
            state["policy"] = copy.deepcopy(policy)
            return query()

        def validate():
            return {"status": "PASS", "reference_match": True}

        def rollback(policy):
            state["epoch"] += 1
            state["policy"] = copy.deepcopy(policy)
            return query()

        runner = canary.IsolatedCpuQng64RestartCanary(
            bundle=cap, target_key=key, role="q_proj", layer=0,
            baseline_n=6, candidate_n=5,
            query_state=query, restart=restart,
            validate=validate, rollback=rollback,
        )
        result = runner.run()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["transition"]["status"], "RESTART_VERIFIED")
        self.assertEqual(result["transition"]["before"]["epoch"], 4)
        self.assertEqual(result["transition"]["after"]["epoch"], 5)
        self.assertFalse(result["production_touched"])

    def test_validation_failure_rolls_back(self):
        key = "qwen2.5-fixture/L0/q_proj"
        cap = bundle(key)
        state = {
            "epoch": 2,
            "policy": [{"role": "q_proj", "layer": 0, "n": 6}],
        }
        rollbacks = []

        def query():
            return copy.deepcopy(state)

        def restart(policy):
            state["epoch"] += 1
            state["policy"] = copy.deepcopy(policy)
            return query()

        def validate():
            raise RuntimeError("reference mismatch")

        def rollback(policy):
            rollbacks.append(copy.deepcopy(policy))
            state["epoch"] += 1
            state["policy"] = copy.deepcopy(policy)
            return query()

        runner = canary.IsolatedCpuQng64RestartCanary(
            bundle=cap, target_key=key, role="q_proj", layer=0,
            baseline_n=6, candidate_n=5,
            query_state=query, restart=restart,
            validate=validate, rollback=rollback,
        )
        with self.assertRaises(Exception):
            runner.run()
        self.assertEqual(len(rollbacks), 1)
        self.assertEqual(pc.normalize_policy(rollbacks[0]),
                         pc.normalize_policy([{"role":"q_proj","layer":0,"n":6}]))

    def test_production_path_refused(self):
        with self.assertRaises(canary.CpuQng64CanaryError):
            canary._refuse_production_path(
                "/tmp/vdsp_serving/persistent-workers/candidate/model.safetensors"
            )


if __name__ == "__main__":
    unittest.main()
