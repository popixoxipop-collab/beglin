#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend_adapters_v2 as bav2
import model_capability as mc
import verify_model_contracts as vmc


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header = {}
    offset = 0
    payload = bytearray()
    width = {"F16": 2, "BF16": 2, "F32": 4}
    for name, (dtype, shape) in tensors.items():
        n = 1
        for dim in shape:
            n *= dim
        size = n * width[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        payload.extend(b"\0" * size)
        offset += size
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def qwen_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "_name_or_path": "acme/qwen2-p12-fixture",
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "max_position_embeddings": 256,
    }
    (root / "config.json").write_text(json.dumps(config, sort_keys=True))
    (root / "tokenizer.json").write_text('{"version":"1.0"}')
    tensors = {
        "model.embed_tokens.weight": ("F16", [128, 64]),
        "model.norm.weight": ("F16", [64]),
        "lm_head.weight": ("F16", [128, 64]),
    }
    for layer in range(2):
        for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"model.layers.{layer}.self_attn.{role}.weight"] = ("F16", [64, 64])
        for role in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"model.layers.{layer}.mlp.{role}.weight"] = ("F16", [128, 64])
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = ("F16", [64])
        tensors[f"model.layers.{layer}.post_attention_layernorm.weight"] = ("F16", [64])
    write_safetensors(root / "model.safetensors", tensors)
    return root


def gguf_string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_minimal_gguf(path: Path, *, architecture: str = "qwen2") -> None:
    kv = [
        ("general.architecture", 8, architecture),
        (f"{architecture}.block_count", 4, 1),
        (f"{architecture}.embedding_length", 4, 64),
        (f"{architecture}.attention.head_count", 4, 8),
        (f"{architecture}.attention.head_count_kv", 4, 2),
        (f"{architecture}.feed_forward_length", 4, 128),
        (f"{architecture}.context_length", 4, 256),
    ]
    tensors = [
        ("token_embd.weight", [64, 128], 1, 0),
        ("blk.0.attn_q.weight", [64, 64], 12, 0),
        ("blk.0.ffn_up.weight", [64, 128], 12, 0),
        ("output.weight", [64, 128], 1, 0),
    ]
    out = bytearray(b"GGUF")
    out.extend(struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for key, type_id, value in kv:
        out.extend(gguf_string(key))
        out.extend(struct.pack("<I", type_id))
        if type_id == 8:
            out.extend(gguf_string(value))
        elif type_id == 4:
            out.extend(struct.pack("<I", int(value)))
        else:
            raise AssertionError(type_id)
    for name, dims, type_id, offset in tensors:
        out.extend(gguf_string(name))
        out.extend(struct.pack("<I", len(dims)))
        for dim in dims:
            out.extend(struct.pack("<Q", dim))
        out.extend(struct.pack("<IQ", type_id, offset))
    path.write_bytes(out)


class SourceAndCompilerTests(unittest.TestCase):
    def test_safetensors_model_compiles_and_is_path_independent(self):
        with tempfile.TemporaryDirectory() as td:
            a = qwen_fixture(Path(td) / "a")
            b = Path(td) / "b"
            shutil.copytree(a, b)
            first = mc.compile_model_capabilities(a)
            second = mc.compile_model_capabilities(b)
            self.assertEqual(first["bundle_sha256"], second["bundle_sha256"])
            self.assertEqual(first["architecture_descriptor"]["architecture_id"], "qwen2")
            self.assertEqual(first["architecture_descriptor"]["attention_kind"], "GQA")
            self.assertEqual(first["model_skeleton"]["layer_count"], 2)
            self.assertEqual(first["tokenizer_contract"]["status"], "IN_ENGINE_VERIFIED")
            self.assertGreater(len(first["precision_search_targets"]), 0)
            self.assertEqual(first["p8_p11_eligibility"]["status"], "PARTIAL")

    def test_content_change_changes_bundle_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            first = mc.compile_model_capabilities(root)
            with (root / "model.safetensors").open("ab") as f:
                f.write(b"x")
            second = mc.compile_model_capabilities(root)
            self.assertNotEqual(first["checkpoint_identity_sha256"], second["checkpoint_identity_sha256"])
            self.assertNotEqual(first["bundle_sha256"], second["bundle_sha256"])

    def test_unknown_architecture_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps({
                "_name_or_path": "acme/mystery",
                "model_type": "mystery_arch",
                "hidden_size": 64,
                "num_hidden_layers": 1,
            }))
            write_safetensors(
                root / "model.safetensors",
                {"model.embed_tokens.weight": ("F16", [16, 64])},
            )
            bundle = mc.compile_model_capabilities(root)
            self.assertEqual(
                bundle["architecture_descriptor"]["status"],
                "UNKNOWN_ARCHITECTURE",
            )
            self.assertEqual(bundle["p8_p11_eligibility"]["status"], "DENIED")
            self.assertFalse(bundle["p8_p11_eligibility"]["p10_allowed"])

    def test_minimal_gguf_is_parsed_without_external_gguf_package(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "tiny.gguf"
            write_minimal_gguf(path)
            bundle = mc.compile_model_capabilities(path, backend="cpu")
            self.assertEqual(bundle["source_manifest"]["source_format"], "GGUF")
            self.assertEqual(bundle["architecture_descriptor"]["architecture_id"], "qwen2")
            self.assertEqual(bundle["source_manifest"]["tensor_count"], 4)
            roles = {
                row["role"] for row in bundle["tensor_role_graph"]["nodes"]
            }
            self.assertIn("Q_PROJ", roles)
            self.assertIn("DENSE_UP", roles)

    def test_node_cli_inspect_model_front_door(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            repo = Path(__file__).resolve().parents[1]
            proc = subprocess.run(
                ["node", str(repo / "bin" / "beglin.js"), "inspect-model", str(root)],
                cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary = json.loads(proc.stdout)
            self.assertEqual(summary["architecture"], "qwen2")
            self.assertIn(summary["p8_p11_eligibility"]["status"], {"PARTIAL", "FULL"})

    def test_gguf_parser_is_streaming_and_does_not_use_path_read_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.gguf"
            write_minimal_gguf(path)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("read_bytes forbidden")):
                parsed = mc.parse_gguf(path)
            self.assertEqual(parsed["tensor_count"], 4)
            self.assertEqual(parsed["metadata"]["general.architecture"], "qwen2")

    def test_cpu_runtime_is_not_verified_by_inspection_alone(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            bundle = mc.compile_model_capabilities(root, backend="cpu")
            rows = bundle["backend_capability_matrix"]["rows"]
            self.assertTrue(rows)
            self.assertTrue(all(r["inference_status"] == "IMPLEMENTED_UNVERIFIED" for r in rows))
            precision = [r for r in rows if r["supported_n"]]
            self.assertTrue(precision)
            self.assertTrue(all(r["qng64_status"] == "IMPLEMENTED_UNVERIFIED" for r in precision))
            self.assertTrue(all(r["verification_source"] == "inspection_only" for r in rows))

    def test_cpu_runtime_verified_requires_explicit_evidence_flag(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            bundle = mc.compile_model_capabilities(
                root, backend="cpu", cpu_runtime_verified=True
            )
            rows = bundle["backend_capability_matrix"]["rows"]
            self.assertTrue(all(r["inference_status"] == "VERIFIED" for r in rows))
            precision = [r for r in rows if r["supported_n"]]
            self.assertTrue(all(r["qng64_status"] == "VERIFIED" for r in precision))
            self.assertTrue(all(r["verification_source"] == "explicit_runtime_evidence" for r in rows))

    def test_olmoe_gguf_is_not_overclaimed_as_supported_loader(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "olmoe.gguf"
            write_minimal_gguf(path, architecture="olmoe")
            bundle = mc.compile_model_capabilities(path)
            self.assertEqual(bundle["architecture_descriptor"]["architecture_id"], "olmoe")
            self.assertEqual(bundle["loader_contract"]["status"], "UNSUPPORTED")
            self.assertEqual(bundle["p8_p11_eligibility"]["status"], "DENIED")

    def test_missing_safetensors_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps({
                "_name_or_path": "acme/qwen2-sharded",
                "model_type": "qwen2",
            }))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.embed_tokens.weight": "missing-00001.safetensors"}
            }))
            with self.assertRaisesRegex(mc.ModelCapabilityError, "missing safetensors shard"):
                mc.inspect_model_source(root)


class BackendSymmetryTests(unittest.TestCase):
    def _bundle(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = qwen_fixture(Path(td.name) / "m")
        return mc.compile_model_capabilities(root, mlx_runtime_verified=True)

    def test_cpu_and_mlx_use_same_plan_schema_with_different_actions(self):
        bundle = self._bundle()
        q = next(
            row for row in bundle["tensor_role_graph"]["nodes"]
            if row["role"] == "Q_PROJ" and row["layer"] == 0
        )
        before = [{"role": "Q_PROJ", "layer": 0, "n": 6}]
        after = [{"role": "Q_PROJ", "layer": 0, "n": 5}]
        keys = {("Q_PROJ", 0): q["canonical_target_key"]}
        cpu_state = bav2.BackendStateV2.build(
            backend="cpu", epoch=2, policy=before,
            model_capability_bundle_sha256=bundle["bundle_sha256"],
        )
        mlx_state = bav2.BackendStateV2.build(
            backend="mlx_metal", epoch=2, policy=before,
            model_capability_bundle_sha256=bundle["bundle_sha256"],
        )
        cpu = bav2.CpuBackendAdapterV2(bundle).plan_transition(
            state=cpu_state, target_policy=after, target_keys=keys
        )
        mlx = bav2.MlxMetalBackendAdapterV2(bundle).plan_transition(
            state=mlx_state, target_policy=after, target_keys=keys
        )
        self.assertEqual(cpu["schema"], mlx["schema"])
        self.assertEqual(cpu["action"], "RESTART_REQUIRED")
        self.assertEqual(mlx["action"], "VALIDATION_REQUIRED")
        self.assertEqual(cpu["target_policy_hash"], mlx["target_policy_hash"])

    def test_policy_shape_change_requires_restart_on_both_backends(self):
        bundle = self._bundle()
        before = [{"role": "Q_PROJ", "layer": 0, "n": 6}]
        after = [
            {"role": "Q_PROJ", "layer": 0, "n": 6},
            {"role": "K_PROJ", "layer": 0, "n": 5},
        ]
        for backend, adapter_cls in [
            ("cpu", bav2.CpuBackendAdapterV2),
            ("mlx_metal", bav2.MlxMetalBackendAdapterV2),
        ]:
            state = bav2.BackendStateV2.build(
                backend=backend, epoch=1, policy=before,
                model_capability_bundle_sha256=bundle["bundle_sha256"],
            )
            plan = adapter_cls(bundle).plan_transition(
                state=state, target_policy=after, target_keys={}
            )
            self.assertEqual(plan["action"], "RESTART_REQUIRED")
            self.assertEqual(plan["reason"]["code"], "POLICY_SHAPE_CHANGE")


class ContractTests(unittest.TestCase):
    def test_bskel_contracts_verify(self):
        root = Path(__file__).resolve().parents[1] / "schemas" / "model"
        result = vmc.verify(root)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["schema_count"], 13)


if __name__ == "__main__":
    unittest.main()
