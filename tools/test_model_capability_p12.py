#!/usr/bin/env python3
from __future__ import annotations

import copy
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
    width = {"F16": 2, "BF16": 2, "F32": 4, "U32": 4}
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


def verification_evidence(
    root: Path,
    *,
    component: str,
    backend: str | None = None,
    evidence_byte: str = "e",
    target_key: str | None = None,
    supported_n: list[int] | None = None,
    mutation_mode: str | None = None,
) -> dict:
    source = mc.inspect_model_source(root)
    descriptor = mc.build_architecture_descriptor(source)
    row = {
        "schema": "beglin-verification-evidence-v1",
        "status": "VERIFIED",
        "component": component,
        "architecture_id": descriptor["architecture_id"],
        "checkpoint_identity_sha256": source["checkpoint_identity_sha256"],
        "evidence_sha256": evidence_byte * 64,
        "run_id": f"fixture-{component}-{backend or 'none'}",
        "kind": "UNIT_TEST_FIXTURE",
    }
    if backend is not None:
        row["backend"] = backend
    if target_key is not None:
        row["target_key"] = target_key
    if supported_n is not None:
        row["supported_n"] = list(supported_n)
    if mutation_mode is not None:
        row["mutation_mode"] = mutation_mode
    return row


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
            self.assertEqual(first["tokenizer_contract"]["status"], "IMPLEMENTED_UNVERIFIED")
            self.assertGreater(len(first["precision_search_targets"]), 0)
            self.assertEqual(first["p8_p11_eligibility"]["status"], "PARTIAL")

    def test_loader_memory_preflight_is_deterministic_and_non_claiming(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            bundle = mc.compile_model_capabilities(root)
            preflight = bundle["loader_contract"]["memory_preflight"]
            self.assertEqual(
                preflight["weight_storage_bytes"],
                (root / "model.safetensors").stat().st_size,
            )
            self.assertEqual(
                preflight["storage_lower_bound_bytes"],
                preflight["weight_storage_bytes"],
            )
            self.assertEqual(
                preflight["estimate_kind"], "STORAGE_LOWER_BOUND_ONLY"
            )
            self.assertIsNone(preflight["peak_resident_bytes_estimate"])
            self.assertIsNone(preflight["workspace_bytes_estimate"])
            self.assertTrue(preflight["requires_runtime_probe"])

    def test_sentencepiece_source_is_explicit_and_not_silently_bpe(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            (root / "tokenizer.json").unlink()
            (root / "tokenizer.model").write_bytes(b"sentencepiece-fixture")
            bundle = mc.compile_model_capabilities(root)
            tok = bundle["tokenizer_contract"]
            self.assertEqual(tok["artifact_kind"], "SENTENCEPIECE_MODEL")
            self.assertEqual(tok["tokenizer_family"], "SENTENCEPIECE")
            self.assertEqual(tok["status"], "UNSUPPORTED")
            self.assertIsNone(tok["encode_backend"])
            self.assertIn("SENTENCEPIECE_IN_ENGINE", tok["missing_primitives"])
            self.assertFalse(tok["text_io_supported"])
            self.assertFalse(tok["silent_fallback_allowed"])

    def test_deepseek_external_tokenizer_evidence_does_not_claim_text_io(self):
        source = {
            "checkpoint_identity_sha256": "a" * 64,
            "source_format": "SAFETENSORS_SHARDED",
            "tokenizer_paths": ["/tmp/tokenizer.json"],
        }
        descriptor = {
            "architecture_id": "deepseek_v2",
        }
        evidence = {
            "schema": "beglin-verification-evidence-v1",
            "status": "VERIFIED",
            "component": "tokenizer",
            "architecture_id": "deepseek_v2",
            "checkpoint_identity_sha256": "a" * 64,
            "evidence_sha256": "b" * 64,
            "run_id": "deepseek-tokenizer-fixture",
            "kind": "UNIT_TEST_FIXTURE",
        }
        tok = mc.build_tokenizer_contract(
            source, descriptor, verification_evidence=evidence
        )
        self.assertEqual(tok["tokenizer_family"], "DEEPSEEK_BPE")
        self.assertEqual(tok["status"], "EXTERNAL_VERIFIED")
        self.assertEqual(tok["encode_backend"], "external_deepseek_reference")
        self.assertIn(
            "DEEPSEEK_PRETOKENIZER_IN_ENGINE", tok["missing_primitives"]
        )
        self.assertFalse(tok["text_io_supported"])
        self.assertEqual(tok["text_io_mode"], "NOT_WIRED")

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

    def test_mlx_affine_quantized_safetensors_auxiliaries_are_structurally_grouped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mlx-q4"
            root.mkdir(parents=True)
            (root / "config.json").write_text(json.dumps({
                "_name_or_path": "acme/qwen2-mlx-q4",
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 256,
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
            }, sort_keys=True))
            (root / "tokenizer.json").write_text('{"version":"1.0"}')
            tensors = {
                "model.embed_tokens.weight": ("U32", [128, 8]),
                "model.embed_tokens.scales": ("BF16", [128, 1]),
                "model.embed_tokens.biases": ("BF16", [128, 1]),
                "model.norm.weight": ("BF16", [64]),
                "lm_head.weight": ("U32", [128, 8]),
                "lm_head.scales": ("BF16", [128, 1]),
                "lm_head.biases": ("BF16", [128, 1]),
                "model.layers.0.self_attn.q_proj.weight": ("U32", [64, 8]),
                "model.layers.0.self_attn.q_proj.scales": ("BF16", [64, 1]),
                "model.layers.0.self_attn.q_proj.biases": ("BF16", [64, 1]),
                "model.layers.0.self_attn.k_proj.weight": ("U32", [16, 8]),
                "model.layers.0.self_attn.k_proj.scales": ("BF16", [16, 1]),
                "model.layers.0.self_attn.k_proj.biases": ("BF16", [16, 1]),
                "model.layers.0.self_attn.v_proj.weight": ("U32", [16, 8]),
                "model.layers.0.self_attn.v_proj.scales": ("BF16", [16, 1]),
                "model.layers.0.self_attn.v_proj.biases": ("BF16", [16, 1]),
                "model.layers.0.self_attn.o_proj.weight": ("U32", [64, 8]),
                "model.layers.0.self_attn.o_proj.scales": ("BF16", [64, 1]),
                "model.layers.0.self_attn.o_proj.biases": ("BF16", [64, 1]),
                "model.layers.0.mlp.gate_proj.weight": ("U32", [128, 8]),
                "model.layers.0.mlp.gate_proj.scales": ("BF16", [128, 1]),
                "model.layers.0.mlp.gate_proj.biases": ("BF16", [128, 1]),
                "model.layers.0.mlp.up_proj.weight": ("U32", [128, 8]),
                "model.layers.0.mlp.up_proj.scales": ("BF16", [128, 1]),
                "model.layers.0.mlp.up_proj.biases": ("BF16", [128, 1]),
                "model.layers.0.mlp.down_proj.weight": ("U32", [64, 16]),
                "model.layers.0.mlp.down_proj.scales": ("BF16", [64, 2]),
                "model.layers.0.mlp.down_proj.biases": ("BF16", [64, 2]),
            }
            write_safetensors(root / "model.safetensors", tensors)
            bundle = mc.compile_model_capabilities(root)
            graph = bundle["tensor_role_graph"]
            self.assertEqual(graph["unmapped_tensor_count"], 0)
            self.assertEqual(graph["mapping_coverage"], 1.0)
            aux = [row for row in graph["nodes"] if row["role"] == "QUANT_AUX"]
            self.assertTrue(aux)
            self.assertTrue(all(row["parent_target_key"] for row in aux))
            loader = bundle["loader_contract"]
            self.assertEqual(loader["status"], "IMPLEMENTED_UNVERIFIED")
            self.assertEqual(
                loader["source_quantization"]["scheme"], "MLX_AFFINE"
            )
            self.assertEqual(
                loader["source_quantization"]["status"], "IMPLEMENTED_UNVERIFIED"
            )
            self.assertEqual(
                loader["unsupported_reason_codes"],
                ["SOURCE_QUANTIZATION_MLX_AFFINE_REQUIRES_VERIFICATION"],
            )
            loader_evidence = verification_evidence(
                root, component="loader", evidence_byte="6"
            )
            verified = mc.compile_model_capabilities(
                root, loader_evidence=loader_evidence
            )["loader_contract"]
            self.assertEqual(verified["status"], "VERIFIED")
            self.assertEqual(
                verified["source_quantization"]["status"], "VERIFIED"
            )
            self.assertEqual(verified["unsupported_reason_codes"], [])
            backend_roles = {
                row["role"] for row in bundle["backend_capability_matrix"]["rows"]
            }
            self.assertNotIn("QUANT_AUX", backend_roles)

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

    def test_mutation_precisions_are_intersected_with_target_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            provisional = mc.compile_model_capabilities(root)
            target = next(
                row["canonical_target_key"]
                for row in provisional["tensor_role_graph"]["nodes"]
                if row["role"] == "Q_PROJ" and row["layer"] == 0
            )
            runtime = verification_evidence(
                root, component="backend_runtime", backend="mlx_metal",
                evidence_byte="1",
            )
            qng = verification_evidence(
                root, component="qng64_runtime", backend="mlx_metal",
                evidence_byte="2", target_key=target, supported_n=[5, 6],
            )
            mutation = verification_evidence(
                root, component="mutation_runtime", backend="mlx_metal",
                evidence_byte="3", target_key=target, supported_n=[5],
                mutation_mode="HOT_REBIND_SINGLE",
            )
            bundle = mc.compile_model_capabilities(
                root,
                mlx_runtime_evidence=runtime,
                mlx_qng64_evidence=[qng],
                mlx_mutation_evidence=[mutation],
            )
            row = next(
                r for r in bundle["runtime_mutation_matrix"]["rows"]
                if r["target_key"] == target and r["backend"] == "mlx_metal"
            )
            self.assertEqual(row["mutation_mode"], "HOT_REBIND_SINGLE")
            self.assertEqual(row["allowed_target_precisions"], [5])
            search = next(
                r for r in bundle["precision_search_targets"]
                if r["target_key"] == target and r["backend"] == "mlx_metal"
            )
            self.assertEqual(search["supported_n"], [5])
            self.assertFalse(search["requires_validation"])

    def test_backend_adapter_rejects_target_key_for_wrong_policy_role(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            provisional = mc.compile_model_capabilities(root)
            q_target = next(
                row["canonical_target_key"]
                for row in provisional["tensor_role_graph"]["nodes"]
                if row["role"] == "Q_PROJ" and row["layer"] == 0
            )
            runtime = verification_evidence(
                root, component="backend_runtime", backend="mlx_metal",
                evidence_byte="4",
            )
            qng = verification_evidence(
                root, component="qng64_runtime", backend="mlx_metal",
                evidence_byte="5", target_key=q_target, supported_n=[5, 6],
            )
            mutation = verification_evidence(
                root, component="mutation_runtime", backend="mlx_metal",
                evidence_byte="6", target_key=q_target, supported_n=[5, 6],
                mutation_mode="HOT_REBIND_SINGLE",
            )
            bundle = mc.compile_model_capabilities(
                root,
                mlx_runtime_evidence=runtime,
                mlx_qng64_evidence=[qng],
                mlx_mutation_evidence=[mutation],
            )
            adapter = bav2.MlxMetalBackendAdapterV2(bundle)
            before = [{"role":"K_PROJ","layer":0,"n":6}]
            after = [{"role":"K_PROJ","layer":0,"n":5}]
            state = bav2.BackendStateV2.build(
                backend="mlx_metal",
                epoch=1,
                policy=before,
                model_capability_bundle_sha256=bundle["bundle_sha256"],
            )
            with self.assertRaisesRegex(
                bav2.BackendV2Error, "target-key binding does not match policy entry"
            ):
                adapter.plan_transition(
                    state=state,
                    target_policy=after,
                    target_keys={("K_PROJ",0): q_target},
                )

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

    def test_cpu_runtime_verified_requires_explicit_evidence_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            with self.assertRaisesRegex(
                mc.ModelCapabilityError, "requires explicit cpu_runtime_evidence"
            ):
                mc.compile_model_capabilities(
                    root, backend="cpu", cpu_runtime_verified=True
                )
            evidence = verification_evidence(
                root, component="backend_runtime", backend="cpu"
            )
            bundle = mc.compile_model_capabilities(
                root,
                backend="cpu",
                cpu_runtime_verified=True,
                cpu_runtime_evidence=evidence,
            )
            rows = bundle["backend_capability_matrix"]["rows"]
            self.assertTrue(all(r["inference_status"] == "VERIFIED" for r in rows))
            precision = [r for r in rows if r["supported_n"]]
            self.assertTrue(all(r["qng64_status"] == "IMPLEMENTED_UNVERIFIED" for r in precision))
            self.assertTrue(
                all(r["verification_source"] == "explicit_runtime_evidence" for r in rows)
            )
            self.assertTrue(all(r["evidence_refs"] for r in rows))
            self.assertTrue(
                all(
                    r["evidence_refs"][0]["evidence_sha256"] == "e" * 64
                    for r in rows
                )
            )

    def test_sentencepiece_is_explicit_external_gap_not_bpe_fallback(self):
        source = {
            "checkpoint_identity_sha256": "a" * 64,
            "source_format": "SAFETENSORS_SINGLE",
            "tokenizer_paths": ["/model/tokenizer.model"],
            "metadata": {},
        }
        descriptor = {"architecture_id": "llama"}
        missing = mc.build_tokenizer_contract(source, descriptor)
        self.assertEqual(missing["tokenizer_family"], "SENTENCEPIECE")
        self.assertEqual(missing["artifact_kind"], "SENTENCEPIECE_MODEL")
        self.assertEqual(missing["status"], "UNSUPPORTED")
        self.assertIsNone(missing["encode_backend"])
        self.assertIn("SENTENCEPIECE_IN_ENGINE", missing["missing_primitives"])

        evidence = {
            "schema": "beglin-verification-evidence-v1",
            "status": "VERIFIED",
            "component": "tokenizer",
            "architecture_id": "llama",
            "checkpoint_identity_sha256": "a" * 64,
            "evidence_sha256": "b" * 64,
            "run_id": "sentencepiece-reference",
            "kind": "TOKENIZER_REFERENCE",
        }
        external = mc.build_tokenizer_contract(
            source, descriptor, verification_evidence=evidence
        )
        self.assertEqual(external["status"], "EXTERNAL_VERIFIED")
        self.assertEqual(external["encode_backend"], "sentencepiece_external")
        self.assertFalse(external["text_io_supported"])

    def test_deepseek_tokenizer_requires_external_evidence_and_keeps_text_io_off(self):
        source = {
            "checkpoint_identity_sha256": "c" * 64,
            "source_format": "SAFETENSORS_SHARDED",
            "tokenizer_paths": ["/model/tokenizer.json"],
            "metadata": {},
        }
        descriptor = {"architecture_id": "deepseek_v2"}
        missing = mc.build_tokenizer_contract(source, descriptor)
        self.assertEqual(missing["status"], "UNSUPPORTED")
        self.assertIn(
            "DEEPSEEK_PRETOKENIZER_IN_ENGINE", missing["missing_primitives"]
        )

        evidence = {
            "schema": "beglin-verification-evidence-v1",
            "status": "VERIFIED",
            "component": "tokenizer",
            "architecture_id": "deepseek_v2",
            "checkpoint_identity_sha256": "c" * 64,
            "evidence_sha256": "d" * 64,
            "run_id": "deepseek-reference",
            "kind": "TOKENIZER_REFERENCE",
        }
        external = mc.build_tokenizer_contract(
            source, descriptor, verification_evidence=evidence
        )
        self.assertEqual(external["status"], "EXTERNAL_VERIFIED")
        self.assertEqual(
            external["encode_backend"], "external_deepseek_reference"
        )
        self.assertFalse(external["text_io_supported"])
        self.assertIn("MODEL_TEXT_IO_WIRING", external["missing_primitives"])

    def test_loader_memory_preflight_is_storage_bound_not_peak_rss_guess(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            bundle = mc.compile_model_capabilities(root)
            memory = bundle["loader_contract"]["memory_preflight"]
            self.assertEqual(memory["status"], "REQUIRES_RUNTIME_PROBE")
            self.assertGreater(memory["source_total_bytes"], 0)
            self.assertGreater(memory["weight_storage_bytes"], 0)
            self.assertEqual(
                memory["storage_lower_bound_bytes"],
                memory["weight_storage_bytes"],
            )
            self.assertIsNone(memory["peak_resident_bytes_estimate"])
            self.assertIsNone(memory["workspace_bytes_estimate"])
            self.assertTrue(memory["requires_runtime_probe"])

    def test_tokenizer_verification_requires_matching_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            evidence = verification_evidence(
                root, component="tokenizer", evidence_byte="a"
            )
            bundle = mc.compile_model_capabilities(
                root, tokenizer_evidence=evidence
            )
            self.assertEqual(
                bundle["tokenizer_contract"]["status"], "IN_ENGINE_VERIFIED"
            )
            self.assertFalse(bundle["tokenizer_contract"]["text_io_supported"])
            self.assertEqual(bundle["tokenizer_contract"]["text_io_mode"], "NOT_WIRED")
            self.assertEqual(
                bundle["tokenizer_contract"]["verification_evidence"]["evidence_sha256"],
                "a" * 64,
            )
            stale = dict(evidence)
            stale["checkpoint_identity_sha256"] = "b" * 64
            with self.assertRaisesRegex(
                mc.ModelCapabilityError, "checkpoint identity mismatch"
            ):
                mc.compile_model_capabilities(
                    root, tokenizer_evidence=stale
                )

    def test_runtime_evidence_accepts_and_binds_binary_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            evidence = verification_evidence(
                root, component="backend_runtime", backend="cpu", evidence_byte="c"
            )
            evidence["binary_sha256"] = "b" * 64
            bundle = mc.compile_model_capabilities(
                root, backend="cpu", cpu_runtime_evidence=evidence
            )
            rows = bundle["backend_capability_matrix"]["rows"]
            self.assertTrue(rows)
            self.assertTrue(all(r["inference_status"] == "VERIFIED" for r in rows))
            self.assertTrue(
                all(r["evidence_refs"][0]["binary_sha256"] == "b" * 64 for r in rows)
            )

    def test_runtime_evidence_must_match_backend_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            evidence = verification_evidence(
                root, component="backend_runtime", backend="cpu"
            )
            wrong_backend = dict(evidence)
            wrong_backend["backend"] = "mlx_metal"
            with self.assertRaisesRegex(
                mc.ModelCapabilityError, "backend mismatch"
            ):
                mc.compile_model_capabilities(
                    root, backend="cpu", cpu_runtime_evidence=wrong_backend
                )

    def test_full_eligibility_requires_matching_backend_tokenizer_and_loader_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            runtime = verification_evidence(
                root, component="backend_runtime", backend="mlx_metal", evidence_byte="7"
            )
            tokenizer = verification_evidence(
                root, component="tokenizer", evidence_byte="8"
            )
            loader = verification_evidence(
                root, component="loader", evidence_byte="9"
            )
            inspected = mc.compile_model_capabilities(root, backend="mlx_metal")
            qng64 = [
                verification_evidence(
                    root, component="qng64_runtime", backend="mlx_metal",
                    evidence_byte="a", target_key=row["target_key"],
                    supported_n=row["supported_n"],
                )
                for row in inspected["precision_search_targets"]
            ]
            mutation = [
                verification_evidence(
                    root, component="mutation_runtime", backend="mlx_metal",
                    evidence_byte="b", target_key=row["target_key"],
                    supported_n=row["supported_n"],
                    mutation_mode="HOT_REBIND_SINGLE",
                )
                for row in inspected["precision_search_targets"]
            ]
            bundle = mc.compile_model_capabilities(
                root,
                backend="mlx_metal",
                mlx_runtime_evidence=runtime,
                mlx_qng64_evidence=qng64,
                mlx_mutation_evidence=mutation,
                tokenizer_evidence=tokenizer,
                loader_evidence=loader,
            )
            self.assertEqual(bundle["tokenizer_contract"]["status"], "IN_ENGINE_VERIFIED")
            self.assertEqual(bundle["loader_contract"]["status"], "VERIFIED")
            self.assertEqual(bundle["p8_p11_eligibility"]["status"], "PARTIAL")
            self.assertIn("TEXT_IO_NOT_WIRED", bundle["p8_p11_eligibility"]["reasons"])
            self.assertTrue(bundle["p8_p11_eligibility"]["p11_allowed"])
            self.assertTrue(bundle["p8_p11_eligibility"]["p11_allowed"])
            self.assertTrue(bundle["precision_search_targets"])
            self.assertTrue(
                all(not row["requires_validation"] for row in bundle["precision_search_targets"])
            )

    def test_loader_verification_rejects_stale_checkpoint_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "m")
            evidence = verification_evidence(
                root, component="loader", evidence_byte="d"
            )
            stale = dict(evidence)
            stale["checkpoint_identity_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                mc.ModelCapabilityError, "checkpoint identity mismatch"
            ):
                mc.compile_model_capabilities(
                    root, loader_evidence=stale
                )

    def test_olmoe_gguf_is_not_overclaimed_as_supported_loader(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "olmoe.gguf"
            write_minimal_gguf(path, architecture="olmoe")
            bundle = mc.compile_model_capabilities(path)
            self.assertEqual(bundle["architecture_descriptor"]["architecture_id"], "olmoe")
            self.assertEqual(bundle["loader_contract"]["status"], "UNSUPPORTED")
            self.assertEqual(bundle["p8_p11_eligibility"]["status"], "DENIED")

    def test_unnamed_gguf_model_identity_is_directory_independent(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/"a"; b=Path(td)/"b"
            a.mkdir(); b.mkdir()
            pa=a/"model.gguf"; pb=b/"model.gguf"
            write_minimal_gguf(pa)
            write_minimal_gguf(pb)
            first=mc.compile_model_capabilities(pa)
            second=mc.compile_model_capabilities(pb)
            self.assertTrue(
                first["model_id"].startswith("checkpoint-")
            )
            self.assertEqual(first["model_id"],second["model_id"])
            self.assertEqual(
                first["checkpoint_identity_sha256"],
                second["checkpoint_identity_sha256"],
            )
            self.assertEqual(first["bundle_sha256"],second["bundle_sha256"])

    def test_sharded_weight_map_must_match_tensor_membership(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps({
                "_name_or_path": "acme/qwen2-sharded-membership",
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 256,
            }))
            write_safetensors(
                root / "model-00001-of-00002.safetensors",
                {
                    "model.embed_tokens.weight": ("F16", [128, 64]),
                    "lm_head.weight": ("F16", [128, 64]),
                },
            )
            write_safetensors(
                root / "model-00002-of-00002.safetensors",
                {"model.norm.weight": ("F16", [64])},
            )
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                }
            }))
            with self.assertRaisesRegex(
                mc.ModelCapabilityError, "membership disagrees with weight_map"
            ):
                mc.inspect_model_source(root)

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
        evidence = verification_evidence(
            root, component="backend_runtime", backend="mlx_metal", evidence_byte="f"
        )
        inspected = mc.compile_model_capabilities(root)
        target = next(
            row["canonical_target_key"]
            for row in inspected["tensor_role_graph"]["nodes"]
            if row["role"] == "Q_PROJ" and row["layer"] == 0
        )
        qng = verification_evidence(
            root, component="qng64_runtime", backend="mlx_metal",
            evidence_byte="a", target_key=target, supported_n=[5, 6],
        )
        mutation = verification_evidence(
            root, component="mutation_runtime", backend="mlx_metal",
            evidence_byte="b", target_key=target, supported_n=[5, 6],
            mutation_mode="HOT_REBIND_SINGLE",
        )
        return mc.compile_model_capabilities(
            root,
            mlx_runtime_verified=True,
            mlx_runtime_evidence=evidence,
            mlx_qng64_evidence=[qng],
            mlx_mutation_evidence=[mutation],
        )

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
        self.assertEqual(mlx["action"], "HOT_REBIND_SINGLE")
        self.assertEqual(cpu["target_policy_hash"], mlx["target_policy_hash"])

    def test_backend_adapter_rejects_tampered_bundle_and_snapshots_input(self):
        bundle = self._bundle()
        tampered = copy.deepcopy(bundle)
        tampered["backend_capability_matrix"]["rows"][0]["inference_status"] = "VERIFIED"
        with self.assertRaisesRegex(
            bav2.BackendV2Error, "capability bundle hash mismatch"
        ):
            bav2.MlxMetalBackendAdapterV2(tampered)

        adapter = bav2.MlxMetalBackendAdapterV2(bundle)
        original = adapter.probe_capabilities()["rows"][0]["inference_status"]
        bundle["backend_capability_matrix"]["rows"][0]["inference_status"] = "CORRUPTED"
        self.assertEqual(
            adapter.probe_capabilities()["rows"][0]["inference_status"], original
        )

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
        self.assertEqual(result["schema_count"], len(list(root.glob("*.schema.json"))))


if __name__ == "__main__":
    unittest.main()
