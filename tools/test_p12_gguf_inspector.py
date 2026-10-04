#!/usr/bin/env python3
import struct
import tempfile
import unittest
from pathlib import Path

import gguf_metadata_inspector as gguf
import inspect_model
import model_source_inspector as source_inspector


def gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def scalar_kv(key: str, type_id: int, payload: bytes) -> bytes:
    return gguf_string(key) + struct.pack("<I", type_id) + payload


def string_kv(key: str, value: str) -> bytes:
    return scalar_kv(key, 8, gguf_string(value))


def u32_kv(key: str, value: int) -> bytes:
    return scalar_kv(key, 4, struct.pack("<I", value))


def string_array_kv(key: str, values: list[str]) -> bytes:
    body = struct.pack("<I", 8) + struct.pack("<Q", len(values))
    body += b"".join(gguf_string(v) for v in values)
    return gguf_string(key) + struct.pack("<I", 9) + body


def tensor_info(name: str, shape: list[int], type_id: int, offset: int = 0) -> bytes:
    raw = gguf_string(name)
    raw += struct.pack("<I", len(shape))
    raw += b"".join(struct.pack("<Q", int(v)) for v in shape)
    raw += struct.pack("<I", int(type_id))
    raw += struct.pack("<Q", int(offset))
    return raw


def write_llama_gguf(path: Path) -> None:
    kv = [
        string_kv("general.architecture", "llama"),
        string_kv("general.name", "fixture-llama"),
        u32_kv("llama.embedding_length", 16),
        u32_kv("llama.block_count", 2),
        u32_kv("llama.attention.head_count", 4),
        u32_kv("llama.attention.head_count_kv", 2),
        u32_kv("llama.context_length", 1024),
        string_array_kv("tokenizer.ggml.tokens", ["a", "b", "c"]),
    ]
    tensors = [
        tensor_info("token_embd.weight", [16, 3], 1, 0),
        tensor_info("blk.0.attn_q.weight", [16, 16], 12, 64),
    ]
    header = b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(kv))
    header += b"".join(kv)
    header += b"".join(tensors)
    path.write_bytes(header)


class GgufMetadataInspectorTests(unittest.TestCase):
    def test_reads_architecture_facts_and_tensor_descriptors(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "arbitrary-name.gguf"
            write_llama_gguf(path)
            inv = gguf.inspect_gguf_header(path)
            self.assertEqual(inv["architecture"], "llama")
            self.assertEqual(inv["version"], 3)
            self.assertEqual(inv["tensor_count"], 2)
            self.assertEqual(inv["tensors"][1]["ggml_type"], "Q4_K")
            facts = gguf.architecture_facts_from_gguf(inv)
            self.assertEqual(facts["hidden_size"], 16)
            self.assertEqual(facts["num_hidden_layers"], 2)
            self.assertEqual(facts["num_attention_heads"], 4)
            self.assertEqual(facts["num_key_value_heads"], 2)
            self.assertEqual(facts["max_position_embeddings"], 1024)
            self.assertEqual(facts["vocab_size"], 3)

    def test_source_inspector_does_not_use_filename_for_architecture(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "definitely-not-llama-by-name.gguf"
            write_llama_gguf(path)
            got = source_inspector.inspect_model_source(path)
            self.assertEqual(got["architecture_source_name"], "llama")
            self.assertEqual(got["config"]["num_hidden_layers"], 2)
            self.assertEqual(got["gguf_inventory"]["name"], "fixture-llama")

    def test_inspect_model_builds_known_gguf_skeleton_but_denies_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.gguf"
            write_llama_gguf(path)
            report = inspect_model.inspect(str(path), model_id="gguf-fixture")
            self.assertEqual(report["architecture"]["status"], "KNOWN")
            self.assertEqual(report["architecture"]["architecture_id"], "llama")
            self.assertEqual(report["model_skeleton"]["layer_count"], 2)
            self.assertIsNotNone(report["tensor_role_graph"])
            self.assertEqual(report["tensor_role_graph"]["unclaimed_tensor_count"], 0)
            roles = {row["role"] for row in report["tensor_role_graph"]["nodes"]}
            self.assertIn("EMBEDDING", roles)
            self.assertIn("Q_PROJ", roles)
            self.assertEqual(
                report["model_skeleton"]["tensor_role_graph_ref"],
                report["tensor_role_graph"]["graph_sha256"],
            )
            self.assertIsNotNone(report["model_capability_bundle"])
            self.assertFalse(report["inference_allowed"])
            self.assertEqual(report["p8_p11_eligibility"], "DENIED")
            self.assertNotIn("tensor-role-graph-v1", report["next_required_contracts"])
            self.assertIn(
                "checkpoint-bound-capability-evidence",
                report["next_required_contracts"],
            )

    def test_bad_magic_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.gguf"
            path.write_bytes(b"NOPE" + b"\0" * 64)
            with self.assertRaisesRegex(gguf.GgufInspectionError, "bad GGUF magic"):
                gguf.inspect_gguf_header(path)


if __name__ == "__main__":
    unittest.main()
