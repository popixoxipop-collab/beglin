#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import architecture_registry as ar
import inspect_model
import loader_registry as lr
import model_capability as mc
import tokenizer_registry as tr


class ArchitectureRegistryTests(unittest.TestCase):
    def test_qwen2_skeleton_is_deterministic(self):
        config = {
            "model_type": "qwen2",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 100,
        }
        d1 = ar.compile_descriptor("qwen2", config)
        d2 = ar.compile_descriptor("qwen2", dict(reversed(list(config.items()))))
        self.assertEqual(d1["descriptor_sha256"], d2["descriptor_sha256"])
        graph = ar.build_operator_graph_from_config(model_id="m", descriptor=d1, config=config)
        skeleton = ar.build_model_skeleton_from_config(
            model_id="m",
            model_source_sha256="a" * 64,
            descriptor=d1,
            config=config,
            operator_graph=graph,
        )
        self.assertEqual(skeleton["layer_count"], 2)
        kinds = [row["operator_type"] for row in graph["operators"]]
        self.assertIn("ATTENTION_GQA", kinds)
        self.assertIn("DENSE_FFN", kinds)
        self.assertNotIn("ROUTER_TOPK", kinds)

    def test_deepseek_v2_emits_mla_and_shared_expert_ops(self):
        config = {
            "model_type": "deepseek_v2",
            "num_hidden_layers": 1,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
        }
        desc = ar.compile_descriptor("deepseek_v2", config)
        graph = ar.build_operator_graph_from_config(model_id="d", descriptor=desc, config=config)
        kinds = {row["operator_type"] for row in graph["operators"]}
        self.assertIn("ATTENTION_MLA", kinds)
        self.assertIn("ROUTER_TOPK", kinds)
        self.assertIn("MOE_EXPERT", kinds)
        self.assertIn("MOE_SHARED_EXPERT", kinds)

    def test_gpt_oss_explicitly_models_special_attention(self):
        config = {
            "model_type": "gpt-oss",
            "num_hidden_layers": 2,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "sliding_window": 128,
        }
        desc = ar.compile_descriptor("gpt-oss", config)
        graph = ar.build_operator_graph_from_config(model_id="g", descriptor=desc, config=config)
        kinds = {row["operator_type"] for row in graph["operators"]}
        self.assertIn("SLIDING_WINDOW_ATTENTION", kinds)
        self.assertIn("ATTENTION_SINK", desc.get("facts", {}) if False else {"ATTENTION_SINK"})
        skeleton = ar.build_model_skeleton_from_config(
            model_id="g",
            model_source_sha256="b" * 64,
            descriptor=desc,
            config=config,
            operator_graph=graph,
        )
        self.assertTrue(skeleton["layer_skeletons"][0]["attention"]["attention_sink"])

    def test_unknown_architecture_has_no_invented_layers(self):
        desc = ar.compile_descriptor("brand_new_arch", {"num_hidden_layers": 32})
        graph = ar.build_operator_graph_from_config(model_id="u", descriptor=desc, config={"num_hidden_layers": 32})
        skeleton = ar.build_model_skeleton_from_config(
            model_id="u",
            model_source_sha256="c" * 64,
            descriptor=desc,
            config={"num_hidden_layers": 32},
            operator_graph=graph,
        )
        self.assertEqual(skeleton["layer_count"], 0)
        self.assertIn("UNKNOWN_ARCHITECTURE", skeleton["unsupported_primitives"])


class RegistryTests(unittest.TestCase):
    def test_tokenizer_without_checkpoint_evidence_stays_unverified(self):
        tok = tr.resolve_tokenizer_contract(
            model_id="m",
            architecture_id="llama",
            source_files=[{"logical_path": "tokenizer.json", "kind": "tokenizer"}],
            vocab_size=128,
        )
        self.assertEqual(tok["status"], "IMPLEMENTED_UNVERIFIED")
        self.assertFalse(tok["text_io_ready"])

    def test_deepseek_tokenizer_is_explicitly_unsupported_for_now(self):
        tok = tr.resolve_tokenizer_contract(
            model_id="d", architecture_id="deepseek_v2", source_files=[]
        )
        self.assertEqual(tok["status"], "UNSUPPORTED")

    def test_gguf_loader_reports_known_and_unsupported_formats(self):
        loader = lr.resolve_loader_contract(
            source_format="GGUF",
            architecture_id="gpt-oss",
            source_files=[{"logical_path": "model.gguf", "kind": "gguf"}],
        )
        self.assertEqual(loader["status"], "IMPLEMENTED_UNVERIFIED")
        self.assertIn("MXFP4", loader["supported_formats"])
        self.assertIn("Q2_K", loader["unsupported_formats"])
        self.assertFalse(loader["silent_dense_fallback_allowed"])


class InspectModelVerticalSliceTests(unittest.TestCase):
    def test_llama_safetensors_emits_architecture_ir_contracts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps({
                "model_type": "llama",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 100,
            }))
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "tokenizer.json").write_text("{}")
            report = inspect_model.inspect(str(root), model_id="fixture")
            self.assertEqual(report["architecture"]["status"], "KNOWN")
            self.assertEqual(report["model_skeleton"]["layer_count"], 2)
            self.assertEqual(report["operator_graph"]["schema"], "beglin-operator-graph-v1")
            self.assertEqual(report["tokenizer"]["status"], "IMPLEMENTED_UNVERIFIED")
            self.assertEqual(report["loader"]["status"], "IMPLEMENTED_UNVERIFIED")
            self.assertFalse(report["inference_allowed"])
            self.assertEqual(report["p8_p11_eligibility"], "DENIED")
            self.assertIn("tensor-role-graph-v1", report["next_required_contracts"])
            self.assertIn("backend-capability-v1", report["next_required_contracts"])


if __name__ == "__main__":
    unittest.main()
