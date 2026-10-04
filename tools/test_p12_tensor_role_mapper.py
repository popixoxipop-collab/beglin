#!/usr/bin/env python3
import json
import struct
import tempfile
import unittest
from pathlib import Path

import tensor_role_mapper as trm


def write_safetensors(path: Path, tensors: dict) -> None:
    header = {}
    cursor = 0
    for name, meta in tensors.items():
        nbytes = int(meta.get("nbytes", 16))
        header[name] = {
            "dtype": meta.get("dtype", "F16"),
            "shape": list(meta.get("shape", [2, 4])),
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + (b"\0" * cursor))


class SafetensorsInventoryTests(unittest.TestCase):
    def test_header_inventory_does_not_load_payload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.safetensors"
            write_safetensors(path, {
                "model.embed_tokens.weight": {"shape": [32, 16], "nbytes": 1024},
                "lm_head.weight": {"shape": [32, 16], "nbytes": 1024},
            })
            got = trm.read_safetensors_header(path)
            self.assertEqual(got["model.embed_tokens.weight"]["shape"], [32, 16])
            self.assertEqual(got["lm_head.weight"]["dtype"], "F16")

    def test_index_header_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_safetensors(root / "model-00001-of-00001.safetensors", {
                "actual.weight": {"shape": [2, 2]},
            })
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"missing.weight": "model-00001-of-00001.safetensors"}
            }))
            with self.assertRaisesRegex(trm.TensorRoleMappingError, "index/header mismatch"):
                trm.inventory_safetensors(root)


class TensorRoleMapperTests(unittest.TestCase):
    def test_llama_core_roles_map_without_unclaimed_tensors(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_safetensors(root / "model.safetensors", {
                "model.embed_tokens.weight": {"shape": [32, 16]},
                "model.layers.0.input_layernorm.weight": {"shape": [16]},
                "model.layers.0.self_attn.q_proj.weight": {"shape": [16, 16]},
                "model.layers.0.self_attn.k_proj.weight": {"shape": [8, 16]},
                "model.layers.0.self_attn.v_proj.weight": {"shape": [8, 16]},
                "model.layers.0.self_attn.o_proj.weight": {"shape": [16, 16]},
                "model.layers.0.post_attention_layernorm.weight": {"shape": [16]},
                "model.layers.0.mlp.gate_proj.weight": {"shape": [32, 16]},
                "model.layers.0.mlp.up_proj.weight": {"shape": [32, 16]},
                "model.layers.0.mlp.down_proj.weight": {"shape": [16, 32]},
                "model.norm.weight": {"shape": [16]},
                "lm_head.weight": {"shape": [32, 16]},
            })
            graph = trm.build_tensor_role_graph_from_safetensors(
                model_id="llama-fixture",
                architecture_id="llama",
                path=root,
            )
            self.assertEqual(graph["unclaimed_tensor_count"], 0)
            roles = {row["role"] for row in graph["nodes"]}
            self.assertIn("Q_PROJ", roles)
            self.assertIn("DENSE_GATE", roles)
            self.assertIn("EMBEDDING", roles)
            self.assertIn("LM_HEAD", roles)

    def test_deepseek_mla_and_expert_roles_map(self):
        names = {
            "model.layers.3.self_attn.q_a_proj.weight": "Q_A_PROJ",
            "model.layers.3.self_attn.kv_a_proj_with_mqa.weight": "KV_A_PROJ",
            "model.layers.3.self_attn.kv_b_proj.weight": "KV_B_PROJ",
            "model.layers.3.mlp.gate.weight": "ROUTER",
            "model.layers.3.mlp.experts.7.up_proj.weight": "EXPERT_UP",
            "model.layers.3.mlp.shared_experts.down_proj.weight": "SHARED_DOWN",
        }
        for name, role in names.items():
            with self.subTest(name=name):
                got = trm.map_tensor_name("deepseek_v2", name)
                self.assertEqual(got["mapping_status"], "MAPPED")
                self.assertEqual(got["role"], role)

    def test_unknown_tensor_is_explicitly_unsupported(self):
        got = trm.map_tensor_name("llama", "model.layers.0.some_new_block.weight")
        self.assertEqual(got["mapping_status"], "UNSUPPORTED")
        self.assertTrue(got["role"].startswith("UNMAPPED_"))


if __name__ == "__main__":
    unittest.main()
