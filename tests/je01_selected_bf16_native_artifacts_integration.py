#!/usr/bin/env python3
"""Standalone native-C quantization integration checks (requires NumPy + C compiler)."""
import hashlib
import importlib.util
import math
import json
import struct
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "je01_selected_bf16_native_artifacts", ROOT / "tools/je01_selected_bf16_native_artifacts.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class NativeCodecSelectedBF16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.lib = M.compile_original_quantizer(Path(cls.temp.name) / "libje01_test.so")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pinned_native_source_file_is_exact(self):
        raw = (ROOT / "gguf_transcode.c").read_bytes()
        self.assertEqual(M.git_blob_sha(raw), M.C_BLOB_SHA1)

    def test_bf16_endian_and_exact_float_conversion(self):
        x = np.array([-3.5, -1.0, 0.0, 1.0, 2.25, 16.0, 0.25, -0.25], dtype="<f4")
        little_bf16 = (x.view("<u4") >> 16).astype("<u2").tobytes()
        rebuilt = M.bf16_to_float(little_bf16, (2, 4))
        np.testing.assert_array_equal(rebuilt, x.reshape(2, 4))

    def test_native_tier_output_has_physical_byte_sizes(self):
        w = np.linspace(-1.2, 1.5, 128, dtype=np.float32).reshape(2, 64)
        for bits in M.TIERS:
            with self.subTest(bits=bits):
                packed, scales, rmse, relrmse = M.quantize_matrix(self.lib, w, bits)
                self.assertEqual(packed.nbytes, 128 * bits // 8)
                self.assertEqual(scales.nbytes, 8)
                self.assertTrue(np.isfinite(scales).all())
                self.assertTrue(math.isfinite(rmse))
                self.assertTrue(math.isfinite(relrmse))
                y = M.decode_native(packed, scales, w.shape, bits)
                self.assertEqual(y.shape, w.shape)
                self.assertTrue(np.isfinite(y).all())

    def test_native_n2_and_n6_zeros_have_expected_bias_code(self):
        w = np.zeros((2, 64), dtype=np.float32)
        for bits in (2, 6):
            with self.subTest(bits=bits):
                packed, scales, rmse, _ = M.quantize_matrix(self.lib, w, bits)
                self.assertEqual(rmse, 0.0)
                np.testing.assert_array_equal(scales, np.ones((2, 1), np.float32))
                decoded = M.decode_native(packed, scales, (2, 64), bits)
                np.testing.assert_array_equal(decoded, w)
                per_group = packed.reshape(2, bits, 8)
                for plane_id in range(bits):
                    expected_byte = 255 if plane_id == bits - 1 else 0
                    self.assertTrue(np.all(per_group[:, plane_id, :] == expected_byte))

    def test_q8_uses_signed_int8_raw_not_qng_bitplanes(self):
        w = np.zeros((2, 64), dtype=np.float32)
        p, s, rmse, _ = M.quantize_matrix(self.lib, w, 8)
        self.assertEqual(p.dtype, np.dtype("int8"))
        self.assertTrue(np.all(p == 0))
        self.assertEqual(rmse, 0)
        self.assertEqual(p.nbytes, 128)
        self.assertEqual(s.nbytes, 8)
        with self.assertRaisesRegex(ValueError, "JE01_DECODE_QNG_LAYOUT"):
            M.decode_native(p.view(np.uint8), s, w.shape, 6)

    def test_independent_handmade_plane_decoder(self):
        # For n2, first/last group code values are exactly [-2,-1,0,1].
        vals = np.array([-2, -1, 0, 1] * 16, dtype=np.int16)
        codes = vals + 2
        raw = np.zeros((1, 1, 2, 8), dtype=np.uint8)
        for p, code in enumerate(codes):
            for b in range(2):
                raw[0, 0, b, p // 8] |= np.uint8(((int(code) >> b) & 1) << (p % 8))
        scales = np.array([[0.25]], dtype=np.float32)
        decoded = M.decode_native(raw.reshape(-1), scales, (1, 64), 2)
        np.testing.assert_array_equal(decoded[0], vals.astype(np.float32) * 0.25)

    def test_mac_os_pressure_sensor_and_hard_35pct_stop(self):
        with patch.object(M.platform, "system", return_value="Darwin"):
            with patch.object(M.subprocess, "check_output",
                              return_value="System-wide memory free percentage: 45%"):
                self.assertEqual(M.enforce_safe_memory(), 45)
            with patch.object(M.subprocess, "check_output",
                              return_value="System-wide memory free percentage: 34%"):
                with self.assertRaisesRegex(ValueError, "BELOW_35PCT"):
                    M.enforce_safe_memory()
            with patch.object(M.subprocess, "check_output", return_value="No sensor"):
                with self.assertRaisesRegex(ValueError, "UNPARSEABLE"):
                    M.enforce_safe_memory()

    def test_attested_shard_handle_retained_until_context_exit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            idx = json.dumps({"metadata": {"total_size": 31412968448},
                              "weight_map": {"fake": "model-00001-of-000004.safetensors"}}).encode()
            cfg = b"{}"
            name = "model-00001-of-000004.safetensors"
            header = json.dumps({"padding": "test" * 20}).encode()
            raw = struct.pack("<Q", len(header)) + header + b"\\x00" * 64
            (root / "model.safetensors.index.json").write_bytes(idx)
            (root / "config.json").write_bytes(cfg)
            (root / name).write_bytes(raw)
            model = {"index_sha256": hashlib.sha256(idx).hexdigest(),
                     "config_sha256": hashlib.sha256(cfg).hexdigest(),
                     "shards": [{"name": name, "size_bytes": len(raw),
                                 "sha256": hashlib.sha256(raw).hexdigest()}]}
            with patch.object(M, "SOURCE_ROOT", root):
                with M.verify_source(model) as (weight_map, headers):
                    fh = headers[name][0]
                    self.assertFalse(fh.closed)
                    self.assertEqual(weight_map["fake"], name)
                self.assertTrue(fh.closed)
            with patch.object(M, "SOURCE_ROOT", root):
                altered = json.loads(json.dumps(model))
                altered["shards"][0]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "SHA_DRIFT"):
                    with M.verify_source(altered):
                        pass

    def test_failed_shapes_or_tiers_fail_closed(self):
        good = np.zeros((2, 64), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "JE01_QUANT_SHAPE_OR_TIER"):
            M.quantize_matrix(self.lib, good, 4)
        with self.assertRaisesRegex(ValueError, "JE01_QUANT_SHAPE_OR_TIER"):
            M.quantize_matrix(self.lib, np.zeros((2, 63), np.float32), 6)
        with self.assertRaisesRegex(ValueError, "JE01_BF16_DECODE_SIZE"):
            M.bf16_to_float(b"xx", (1, 64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
