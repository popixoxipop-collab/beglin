#!/usr/bin/env python3
"""CPU-only fail-closed tests of BEGLIN v11 sealed prequantized mmap reader."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prequant_n7_sealed_v1 import PackReader, PackWriter, role_files

SOURCE_SHA = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
ROLE = "model.layers.0.self_attn.q_proj.weight"


class PrequantSealedTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(prefix="beglin_prequant_sealed_")
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name) / "packed"
        self.p = np.arange(128 * 8 * 56, dtype=np.uint32).astype(np.uint8)
        self.s = np.full((128 * 8,), 0.025, dtype=np.float32)

    def pack(self):
        w = PackWriter(self.root, SOURCE_SHA)
        self.assertEqual(w.add(ROLE, 128, 512, self.p, self.s), 61440)
        return w.seal()

    def reader(self, digest):
        return PackReader(self.root, digest, SOURCE_SHA)

    def test_reopen_readonly_mmap_no_source_weights(self):
        digest = self.pack()
        r = self.reader(digest)
        p = r.load(ROLE)
        self.assertEqual(p.n, 7)
        self.assertEqual(p.serialized_bytes, 61440)
        self.assertFalse(p.packed.flags.writeable)
        self.assertFalse(p.scales.flags.writeable)
        self.assertTrue(np.array_equal(p.packed, self.p))
        self.assertTrue(np.array_equal(p.scales, self.s))
        self.assertEqual(r.total_packed_bytes, 61440)

    def test_external_manifest_pin(self):
        self.pack()
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            self.reader("0" * 64)

    def test_external_source_index_pin(self):
        d = self.pack()
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            PackReader(self.root, d, "b" * 64)

    def test_plane_tamper(self):
        d = self.pack()
        pn, _ = role_files(ROLE)
        file = self.root / pn
        file.chmod(0o600)
        data = bytearray(file.read_bytes())
        data[11] ^= 0x40
        file.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "payload SHA"):
            self.reader(d).load(ROLE)

    def test_scale_tamper(self):
        d = self.pack()
        _, sn = role_files(ROLE)
        file = self.root / sn
        file.chmod(0o600)
        raw = bytearray(file.read_bytes())
        raw[1] ^= 1
        file.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "payload SHA"):
            self.reader(d).load(ROLE)

    def test_symlink_denied_even_same_bytes(self):
        d = self.pack()
        pn, _ = role_files(ROLE)
        file = self.root / pn
        target = Path(self.t.name) / "original.bin"
        target.write_bytes(file.read_bytes())
        file.unlink()
        file.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.reader(d).load(ROLE)

    def test_truncated_payload_denied(self):
        d = self.pack()
        pn, _ = role_files(ROLE)
        file = self.root / pn
        file.chmod(0o600)
        file.write_bytes(file.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "size/type"):
            self.reader(d).load(ROLE)

    def test_duplicate_writer_role_denied(self):
        w = PackWriter(self.root, SOURCE_SHA)
        w.add(ROLE, 128, 512, self.p, self.s)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            w.add(ROLE, 128, 512, self.p, self.s)

    def test_unsealed_artifact_denied(self):
        w = PackWriter(self.root, SOURCE_SHA)
        w.add(ROLE, 128, 512, self.p, self.s)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.reader("0" * 64)

    def test_wrong_width_or_geometry_denied(self):
        w = PackWriter(self.root, SOURCE_SHA)
        with self.assertRaisesRegex(ValueError, "geometry"):
            w.add(ROLE, 128, 513, self.p, self.s)

    def test_nonpositive_nan_scales_denied(self):
        w = PackWriter(self.root, SOURCE_SHA)
        bad = self.s.copy()
        bad[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "canonical"):
            w.add(ROLE, 128, 512, self.p, bad)
        bad[0] = 0
        with self.assertRaisesRegex(ValueError, "canonical"):
            w.add(ROLE, 128, 512, self.p, bad)

    def test_pathlike_role_denied(self):
        w = PackWriter(self.root, SOURCE_SHA)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            w.add("../../tmp/a.weight", 128, 512, self.p, self.s)

    def test_unlisted_role_denied(self):
        d = self.pack()
        with self.assertRaisesRegex(ValueError, "unapproved"):
            self.reader(d).load("model.layers.1.self_attn.q_proj.weight")

    def test_bogus_manifest_role_denied_even_if_rehashed(self):
        d = self.pack()
        file = self.root / "manifest.json"
        file.chmod(0o600)
        obj = json.loads(file.read_text())
        obj["roles"][0]["role"] = "../../untrusted.weight"
        raw = (json.dumps(obj, sort_keys=True, separators=(",", ":"))+"\n").encode("ascii")
        file.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            PackReader(self.root, hashlib.sha256(raw).hexdigest(), SOURCE_SHA)

    def test_manifest_duplicate_or_unsorted_denied(self):
        d = self.pack()
        file = self.root / "manifest.json"
        file.chmod(0o600)
        obj = json.loads(file.read_text())
        obj["roles"].append(dict(obj["roles"][0]))
        raw = (json.dumps(obj, sort_keys=True, separators=(",", ":"))+"\n").encode("ascii")
        file.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            PackReader(self.root, hashlib.sha256(raw).hexdigest(), SOURCE_SHA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
