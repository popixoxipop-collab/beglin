#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import gpu_shadow_materialize as gm


READY = {
    "candidate_id": "candidate-1",
    "status": "READY",
    "role": "shared_down_proj",
    "layer": 4,
    "n": 6,
    "event": {
        "orig_token": 3268,
        "corrected_token": 1224,
        "pos": 16,
        "req": 3,
    },
    "reference": {"emitted_token": 1224},
    "provenance": {
        "manifest": "/prod/discovery/manifest.txt",
        "source_jsonl": "/prod/discovery/events.jsonl",
    },
}


def discovery(rows=None):
    ready = [READY] if rows is None else rows
    return {
        "schema": "gpu-shadow-discovery-v1",
        "mode": "read_only",
        "production_write_allowed": False,
        "payload_sha256": "c" * 64,
        "ready": ready,
    }


class ShadowMaterializeTests(unittest.TestCase):
    def test_parse_maps_prefers_longest_prefix(self):
        got = gm.parse_maps([
            "/prod=/mirror/root",
            "/prod/discovery=/mirror/discovery",
        ])
        self.assertEqual(str(got[0][0]), "/prod/discovery")
        self.assertEqual(
            gm.map_path("/prod/discovery/a.txt", got),
            Path("/mirror/discovery/a.txt"),
        )


    def test_map_path_rejects_parent_escape(self):
        with tempfile.TemporaryDirectory() as td:
            mirror = Path(td) / "mirror"
            mirror.mkdir()
            maps = gm.parse_maps([f"/prod={mirror}"])
            with self.assertRaises(gm.ShadowMaterializeError):
                gm.map_path("/prod/../outside.txt", maps)

    def test_map_path_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mirror = root / "mirror"
            outside = root / "outside"
            mirror.mkdir()
            outside.mkdir()
            (outside / "secret.txt").write_text("x")
            (mirror / "link").symlink_to(outside, target_is_directory=True)
            maps = gm.parse_maps([f"/prod={mirror}"])
            with self.assertRaises(gm.ShadowMaterializeError):
                gm.map_path("/prod/link/secret.txt", maps)

    def test_relative_token_path_is_resolved_from_source_manifest_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            mirror = root / "mirror"
            (mirror / "discovery" / "tokens").mkdir(parents=True)
            token = mirror / "discovery" / "tokens" / "prompt.i32"
            token.write_bytes(b"\0" * 8)
            manifest = mirror / "discovery" / "manifest.txt"
            manifest.write_text("tokens/prompt.i32 2\n")
            binary = root / "bin"
            binary.write_bytes(b"x")
            got = gm.materialize(
                discovery(),
                candidate_id="candidate-1",
                path_mappings=gm.parse_maps([f"/prod={mirror}"]),
                output_dir=str(root / "shadow"),
                cwd=str(repo),
                binary=str(binary),
                checkpoint_sha256="d" * 64,
                moe_base=str(root / "moe"),
                safetensors=str(root / "st"),
                g6_repeats=1,
            )
            spec = json.loads(Path(got["candidate_spec"]).read_text())
            self.assertEqual(
                spec["source"]["mapped_raw_token_file"],
                str(token.resolve()),
            )
            self.assertEqual(spec["prompt_len"], 2)

    def test_requires_candidate_id_when_multiple_ready(self):
        rows = [dict(READY), {**READY, "candidate_id": "candidate-2"}]
        with self.assertRaises(gm.ShadowMaterializeError):
            gm.select_ready(discovery(rows))

    def test_materialize_builds_local_g4_g6_spec_and_hashes_binary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            mirror = root / "mirror"
            (mirror / "discovery").mkdir(parents=True)
            (mirror / "tokens").mkdir(parents=True)

            token = mirror / "tokens" / "prompt.i32"
            token.write_bytes(b"\x00\x00\x00\x00" * 9)
            manifest = mirror / "discovery" / "manifest.txt"
            manifest.write_text("/prod/tokens/prompt.i32 10\n")

            binary = root / "qwen_infer_gpu"
            binary.write_bytes(b"gpu-binary")
            out = root / "shadow-inputs"

            got = gm.materialize(
                discovery(),
                candidate_id="candidate-1",
                path_mappings=gm.parse_maps([
                    f"/prod={mirror}",
                ]),
                output_dir=str(out),
                cwd=str(repo),
                binary=str(binary),
                checkpoint_sha256="d" * 64,
                moe_base=str(root / "moe"),
                safetensors=str(root / "model.safetensors.index.json"),
                g6_repeats=3,
            )

            self.assertEqual(got["status"], "SPEC_READY")
            self.assertFalse(got["production_write_allowed"])
            self.assertEqual(got["prompt_len"], 9)
            self.assertEqual(
                got["binary_sha256"],
                hashlib.sha256(b"gpu-binary").hexdigest(),
            )

            spec = json.loads(Path(got["candidate_spec"]).read_text())
            self.assertFalse(spec["production_write_allowed"])
            self.assertEqual(spec["role"], "shared_down_proj")
            self.assertEqual(spec["layer"], 4)
            self.assertEqual(spec["n"], 6)
            self.assertEqual(spec["prompt_len"], 9)
            self.assertEqual(spec["reference"]["emitted_token"], 1224)
            g4 = Path(spec["g4_manifest"]).read_text()
            g6 = Path(spec["g6_manifest"]).read_text()
            expected = f"{token} 10\n"
            self.assertEqual(g4, expected)
            self.assertEqual(g6, expected * 3)
            self.assertEqual(
                spec["source"]["raw_token_sha256"],
                hashlib.sha256(token.read_bytes()).hexdigest(),
            )

    def test_output_root_inside_or_parent_of_repo_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            with self.assertRaises(gm.ShadowMaterializeError):
                gm.materialize(
                    discovery(),
                    candidate_id="candidate-1",
                    path_mappings=[],
                    output_dir=str(repo / "nested"),
                    cwd=str(repo),
                    binary=str(root / "missing"),
                    checkpoint_sha256="d" * 64,
                    moe_base=str(root / "moe"),
                    safetensors=str(root / "st"),
                )
            with self.assertRaises(gm.ShadowMaterializeError):
                gm.materialize(
                    discovery(),
                    candidate_id="candidate-1",
                    path_mappings=[],
                    output_dir=str(root),
                    cwd=str(repo),
                    binary=str(root / "missing"),
                    checkpoint_sha256="d" * 64,
                    moe_base=str(root / "moe"),
                    safetensors=str(root / "st"),
                )

    def test_unmapped_source_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            with self.assertRaises(gm.ShadowMaterializeError):
                gm.materialize(
                    discovery(),
                    candidate_id="candidate-1",
                    path_mappings=[],
                    output_dir=str(root / "shadow"),
                    cwd=str(repo),
                    binary=str(root / "missing"),
                    checkpoint_sha256="d" * 64,
                    moe_base=str(root / "moe"),
                    safetensors=str(root / "st"),
                )

    def test_g6_repeat_budget_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            mirror = root / "mirror"
            (mirror / "discovery").mkdir(parents=True)
            (mirror / "tokens").mkdir(parents=True)
            (mirror / "tokens" / "prompt.i32").write_bytes(b"\0" * 4)
            (mirror / "discovery" / "manifest.txt").write_text(
                "/prod/tokens/prompt.i32 1\n"
            )
            binary = root / "bin"
            binary.write_bytes(b"x")
            with self.assertRaises(gm.ShadowMaterializeError):
                gm.materialize(
                    discovery(),
                    candidate_id="candidate-1",
                    path_mappings=gm.parse_maps([f"/prod={mirror}"]),
                    output_dir=str(root / "shadow"),
                    cwd=str(repo),
                    binary=str(binary),
                    checkpoint_sha256="d" * 64,
                    moe_base=str(root / "moe"),
                    safetensors=str(root / "st"),
                    g6_repeats=101,
                )


if __name__ == "__main__":
    unittest.main()
