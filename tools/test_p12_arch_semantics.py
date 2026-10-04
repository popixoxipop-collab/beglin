#!/usr/bin/env python3
import json
import struct
import tempfile
import unittest
from pathlib import Path

import architecture_registry as ar
import inspect_model
import tensor_role_mapper as trm


def write_safetensors(path: Path, tensors: dict[str, dict]) -> None:
    header = {}
    cursor = 0
    payload = bytearray()
    for name, meta in tensors.items():
        nbytes = int(meta.get("nbytes", 8))
        header[name] = {
            "dtype": meta.get("dtype", "F16"),
            "shape": list(meta.get("shape", [1])),
            "data_offsets": [cursor, cursor + nbytes],
        }
        payload.extend(b"\0" * nbytes)
        cursor += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class ArchitectureSemanticTests(unittest.TestCase):
    def test_qwen3_and_olmoe_qk_norm_semantics_are_distinct(self):
        qwen = ar.compile_descriptor("qwen3_moe", {"num_hidden_layers": 1})
        olmo = ar.compile_descriptor("olmoe", {"num_hidden_layers": 1})
        self.assertEqual(qwen["facts"]["qk_norm_kind"], "PER_HEAD")
        self.assertEqual(olmo["facts"]["qk_norm_kind"], "WHOLE_VECTOR")

    def test_deepseek_initial_dense_layer_is_not_moe(self):
        config = {
            "num_hidden_layers": 3,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
        }
        desc = ar.compile_descriptor("deepseek_v2", config)
        graph = ar.build_operator_graph_from_config(
            model_id="d", descriptor=desc, config=config
        )
        l0 = {
            row["operator_type"]
            for row in graph["operators"]
            if row.get("layer") == 0
        }
        l1 = {
            row["operator_type"]
            for row in graph["operators"]
            if row.get("layer") == 1
        }
        self.assertIn("DENSE_FFN", l0)
        self.assertNotIn("ROUTER_TOPK", l0)
        self.assertIn("ROUTER_TOPK", l1)
        self.assertIn("MOE_SHARED_EXPERT", l1)

    def test_safetensors_offsets_must_fit_payload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors"
            header = {
                "x": {
                    "dtype": "F16",
                    "shape": [1],
                    "data_offsets": [0, 999],
                }
            }
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw)
            with self.assertRaisesRegex(
                trm.TensorRoleMappingError, "exceed file"
            ):
                trm.read_safetensors_header(path)

    def test_structural_status_complete_vs_partial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps({
                "model_type": "llama",
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 32,
            }))
            tensors = {
                "model.embed_tokens.weight": {"shape": [32, 16]},
                "model.layers.0.self_attn.q_proj.weight": {"shape": [16, 16]},
                "model.layers.0.self_attn.k_proj.weight": {"shape": [8, 16]},
                "model.layers.0.self_attn.v_proj.weight": {"shape": [8, 16]},
                "model.layers.0.self_attn.o_proj.weight": {"shape": [16, 16]},
                "model.layers.0.mlp.gate_proj.weight": {"shape": [32, 16]},
                "model.layers.0.mlp.up_proj.weight": {"shape": [32, 16]},
                "model.layers.0.mlp.down_proj.weight": {"shape": [16, 32]},
                "model.norm.weight": {"shape": [16]},
                "lm_head.weight": {"shape": [32, 16]},
            }
            write_safetensors(root / "model.safetensors", tensors)
            report = inspect_model.inspect(str(root))
            self.assertEqual(report["structural_status"], "COMPLETE")

            tensors["model.layers.0.new_unknown.weight"] = {"shape": [1]}
            write_safetensors(root / "model.safetensors", tensors)
            report = inspect_model.inspect(str(root))
            self.assertEqual(report["structural_status"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()
