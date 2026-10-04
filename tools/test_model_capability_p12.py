#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import backend_adapter_v2 as bav2
import model_capability as mc
import model_source_inspector as inspector


class IdentityTests(unittest.TestCase):
    def test_source_identity_ignores_root_path_and_discovery_time(self):
        files = [{"logical_path":"config.json","sha256":"a"*64,"size_bytes":12,"kind":"config"}]
        a = mc.build_model_source_manifest(
            model_id="m", model_revision="r", source_format="SAFETENSORS_SINGLE",
            files=files, root_path="/one", discovered_at="t1",
        )
        b = mc.build_model_source_manifest(
            model_id="m", model_revision="r", source_format="SAFETENSORS_SINGLE",
            files=files, root_path="/two", discovered_at="t2",
        )
        self.assertEqual(a["checkpoint_identity_sha256"], b["checkpoint_identity_sha256"])

    def test_architecture_unknown_fails_closed(self):
        got = mc.identify_architecture("brand_new_arch")
        self.assertEqual(got["status"], "UNKNOWN_ARCHITECTURE")
        self.assertFalse(got["inference_allowed"])

    def test_architecture_known_is_deterministic(self):
        a = mc.identify_architecture("deepseek_v2", facts={"hidden_size":2048})
        b = mc.identify_architecture("deepseek-v2", facts={"hidden_size":2048})
        self.assertEqual(a["descriptor_sha256"], b["descriptor_sha256"])
        self.assertEqual(a["attention_kind"], "MLA")


class TensorGraphTests(unittest.TestCase):
    def test_duplicate_source_is_rejected(self):
        nodes = [
            {"source_tensor_name":"x","role":"Q_PROJ","layer":0},
            {"source_tensor_name":"x","role":"K_PROJ","layer":0},
        ]
        with self.assertRaises(mc.ModelCapabilityError):
            mc.build_tensor_role_graph(model_id="m", nodes=nodes)

    def test_graph_target_is_canonical(self):
        graph = mc.build_tensor_role_graph(model_id="m", nodes=[
            {"source_tensor_name":"layers.3.shared.up","role":"SHARED_UP","layer":3,"shape":[8,8]}
        ])
        self.assertEqual(graph["nodes"][0]["canonical_target_key"], "m/L3/shared_up")
        self.assertEqual(graph["unclaimed_tensor_count"], 0)


class CapabilityTests(unittest.TestCase):
    def evidence(self):
        return [{"kind":"REAL_GPU","sha256":"e"*64,"status":"VERIFIED","ref":"fixture"}]

    def test_verified_requires_evidence(self):
        with self.assertRaises(mc.ModelCapabilityError):
            mc.validate_capability_cell({
                "target_key":"m/L3/shared_up","backend":"mlx_metal",
                "inference_status":"VERIFIED","mutation_mode":"HOT_REBIND_SINGLE",
            })

    def test_bundle_full_and_search_targets(self):
        bundle = mc.build_model_capability_bundle(
            model_id="m", checkpoint_identity="a"*64, skeleton_sha256="b"*64,
            architecture_status="KNOWN", tokenizer_status="IN_ENGINE_VERIFIED", loader_status="VERIFIED",
            backend_matrix=[{
                "target_key":"m/L3/shared_up","backend":"mlx_metal",
                "inference_status":"VERIFIED","mutation_mode":"HOT_REBIND_SINGLE",
                "supported_n":[5,6],"quant_formats":["qNg64"],"evidence_refs":self.evidence(),
            }],
        )
        self.assertEqual(bundle["p8_p11_eligibility"], "FULL")
        self.assertEqual(bundle["precision_search_targets"], ["m/L3/shared_up"])
        self.assertEqual(bundle["hot_rebind_targets"], ["m/L3/shared_up"])

    def test_bundle_partial_when_restart_only_target_exists(self):
        bundle = mc.build_model_capability_bundle(
            model_id="m", checkpoint_identity="a"*64, skeleton_sha256="b"*64,
            architecture_status="KNOWN", tokenizer_status="EXTERNAL_VERIFIED", loader_status="VERIFIED",
            backend_matrix=[{
                "target_key":"m/embed","backend":"cpu",
                "inference_status":"VERIFIED","mutation_mode":"RESTART_REQUIRED",
                "supported_n":[4,8],"evidence_refs":self.evidence(),
            }],
        )
        self.assertEqual(bundle["p8_p11_eligibility"], "PARTIAL")
        self.assertEqual(bundle["restart_only_targets"], ["m/embed"])

    def test_bundle_denied_for_unknown_architecture(self):
        bundle = mc.build_model_capability_bundle(
            model_id="m", checkpoint_identity="a"*64, skeleton_sha256="b"*64,
            architecture_status="UNKNOWN_ARCHITECTURE", tokenizer_status="UNSUPPORTED", loader_status="UNSUPPORTED",
            backend_matrix=[],
        )
        self.assertEqual(bundle["p8_p11_eligibility"], "DENIED")


class BackendSymmetryTests(unittest.TestCase):
    def state(self, backend, model_id=None):
        policy=[{"role":"shared_down_proj","layer":26,"n":5},{"role":"shared_up_proj","layer":3,"n":6}]
        return bav2.BackendState(
            backend=backend,
            epoch=11,
            policy=policy,
            policy_hash=bav2.policy_hash(policy),
            model_id=model_id,
        )

    def target(self):
        return [{"role":"shared_down_proj","layer":26,"n":5},{"role":"shared_up_proj","layer":3,"n":5}]

    def test_cpu_and_mlx_return_same_schema(self):
        cpu = bav2.CpuBackendAdapterV2().plan_transition(state=self.state("cpu"), target_policy=self.target())
        mlx = bav2.MlxMetalBackendAdapterV2(
            verified_hot_targets={"L3/shared_up_proj"}
        ).plan_transition(state=self.state("mlx_metal"), target_policy=self.target())
        self.assertEqual(cpu["schema"], mlx["schema"])
        self.assertEqual(cpu["target_policy_hash"], mlx["target_policy_hash"])
        self.assertEqual(cpu["action"], "RESTART_REQUIRED")
        self.assertEqual(mlx["action"], "HOT_REBIND_SINGLE")
        self.assertFalse(cpu["production_write_allowed"])
        self.assertFalse(mlx["production_write_allowed"])


    def test_model_scoped_hot_target_uses_canonical_key(self):
        state = self.state("mlx_metal", model_id="deepseek-v2-lite")
        plan = bav2.MlxMetalBackendAdapterV2(
            verified_hot_targets={"deepseek-v2-lite/L3/shared_up_proj"}
        ).plan_transition(state=state, target_policy=self.target())
        self.assertEqual(plan["action"], "HOT_REBIND_SINGLE")

    def test_tampered_state_policy_hash_is_rejected(self):
        policy=[{"role":"shared_up_proj","layer":3,"n":6}]
        state=bav2.BackendState(
            backend="cpu",
            epoch=1,
            policy=policy,
            policy_hash="0"*64,
        )
        with self.assertRaisesRegex(bav2.BackendPlanError, "policy hash mismatch"):
            bav2.CpuBackendAdapterV2().plan_transition(
                state=state,
                target_policy=[{"role":"shared_up_proj","layer":3,"n":5}],
            )

    def test_policy_shape_change_requires_restart(self):
        state = self.state("mlx_metal")
        target = self.target() + [{"role":"q_proj","layer":0,"n":5}]
        plan = bav2.MlxMetalBackendAdapterV2(
            verified_hot_targets={"L3/shared_up_proj","L0/q_proj"}
        ).plan_transition(state=state, target_policy=target)
        self.assertEqual(plan["action"], "RESTART_REQUIRED")
        self.assertTrue(plan["policy_shape_change"])


class SourceInspectorTests(unittest.TestCase):
    def test_single_safetensors_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/"config.json").write_text(json.dumps({
                "model_type":"llama","hidden_size":16,"num_hidden_layers":2
            }))
            (root/"model.safetensors").write_bytes(b"weights")
            first=inspector.inspect_model_source(root)
            second=inspector.inspect_model_source(root)
            self.assertEqual(first["manifest"]["source_format"], "SAFETENSORS_SINGLE")
            self.assertEqual(first["manifest"]["checkpoint_identity_sha256"], second["manifest"]["checkpoint_identity_sha256"])
            self.assertEqual(first["architecture_source_name"], "llama")

    def test_missing_index_shard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/"model.safetensors.index.json").write_text(json.dumps({
                "weight_map":{"a":"missing-00001-of-00002.safetensors"}
            }))
            with self.assertRaisesRegex(inspector.SourceInspectionError, "missing safetensors shard"):
                inspector.inspect_model_source(root)


class ContractFilesTests(unittest.TestCase):
    def test_contract_files_have_unique_ids_and_schema_constants(self):
        root=Path(__file__).resolve().parents[1]/"contracts"/"model"
        files=sorted(root.glob("*.schema.json"))
        self.assertGreaterEqual(len(files), 13)
        ids=set()
        for path in files:
            obj=json.loads(path.read_text())
            self.assertEqual(obj["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertNotIn(obj["$id"], ids)
            ids.add(obj["$id"])
            self.assertIn("schema", obj["properties"])
            self.assertIn("const", obj["properties"]["schema"])


if __name__ == "__main__":
    unittest.main()
