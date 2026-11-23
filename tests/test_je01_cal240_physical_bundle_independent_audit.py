#!/usr/bin/env python3
"""Synthetic-only adversarial contract checks for read-only 43MB auditor.

This fixture contains no real model weights or NLL. It validates that the
independent byte auditor detects changes in the exact physical artifact.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/je01_cal240_physical_bundle_independent_audit.py"
spec = importlib.util.spec_from_file_location("je01_physical_audit", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class TestJE01PhysicalIndependentAuditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="je01-physical-fixture-")
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.ledger_path = cls.folder / "synthetic_CAL240_ledger.json"
        rows, ledger_rows, lines = [], [], []
        binary = cls.folder / mod.BINARY
        with binary.open("xb") as sink:
            for layer, expert in mod.EXPERT_ORDER:
                for role in mod.ROLE_ORDER:
                    tier = mod.tier(layer, expert, role)
                    bits = 5 if tier == "Q5" else 4
                    outdim, indim = ((2048, 1408) if role == "down_proj"
                                      else (1408, 2048))
                    sizes = {
                        "packed": outdim * indim * bits // 8,
                        "scale": outdim * (indim // 64) * 4,
                        "bias": outdim * (indim // 64) * 4,
                    }
                    names = ("packed", "scale", "bias") if tier == "AF4" else ("packed", "scale")
                    regs, digests = {}, {}
                    for kind in names:
                        raw = bytes([(layer + expert + len(kind)) % 256]) * sizes[kind]
                        start = sink.tell()
                        sink.write(raw)
                        end = sink.tell()
                        regs[kind] = [start, end]
                        digests[kind] = sha(raw)
                    source_bf16 = sha(f"{layer}-{expert}-{role}".encode())
                    ledger_rows.append({
                        "layer": layer, "expert": expert, "role": role,
                        "actual_original_BF16_tensor_sha256": source_bf16,
                        "actual_original_AF4_packed_sha256": digests.get(
                            "packed", sha(f"af4-{layer}-{expert}-{role}".encode())),
                    })
                    rows.append({
                        "layer": layer, "expert": expert, "role": role,
                        "tier": tier, "bits": bits, "output": outdim, "input": indim,
                        "regions": regs, "part_sha256": digests,
                        "source_original_bf16_tensor_sha256": source_bf16,
                        "source_original_affine4_packed_sha256":
                            ledger_rows[-1]["actual_original_AF4_packed_sha256"],
                        "physical_bytes": mod.COST[tier],
                    })
                    p, s, b = regs["packed"], regs["scale"], regs.get("bias", [0, 0])
                    lines.append("\t".join(map(str, (
                        layer, expert, role, bits, outdim, indim, *p, *s, *b, tier
                    ))))
        assert binary.stat().st_size == mod.BUDGET
        tsv = cls.folder / mod.TSV
        tsv.write_text("\n".join(lines) + "\n")
        ledger = {
            "per_role": ledger_rows,
            "training_unseen240_prompt_count": 240,
            "source_heldout_OOD_labels_used": False,
        }
        ledger["receipt_sha256"] = mod.canonical(ledger)
        cls.signed_ledger_sha = ledger["receipt_sha256"]
        cls.ledger_path.write_text(json.dumps(ledger, sort_keys=True))
        cls.ledger_file_sha = mod.file_sha(cls.ledger_path)
        cls.receipt_path = cls.folder / mod.RECEIPT
        receipt = {
            "status": "JE01_REAL_ORIGINAL_AF4_Q4Q5_CAL240_FROZEN_ROLEWISE_43MB_CREATED",
            "source_original_AF4_sha256": mod.ORIGINAL_AF4_SHA,
            "source_actual_cal240_SSE_DP_receipt_sha256": cls.signed_ledger_sha,
            "source_training_actual_h2_SwiGLU_receipt_sha256": mod.TRACE_RECEIPT_SHA,
            "first16_train_pilot_excluded": True,
            "calibration_training_prompt_count": 240,
            "calibration_routed_expert_records": 637,
            "heldout_OOD_gold_accessed": False, "production_modified": False,
            "true_full_model_gold_NLL": "NOT_MEASURED",
            "native_full_model_overlay": "NOT_YET_BOUND",
            "physical_file_size_bytes": mod.BUDGET,
            "role_count": 24, "counts": {"AF4": 12, "Q4": 6, "Q5": 6},
            "contiguous_file_spans": 60,
            "physical_file_sha256": mod.file_sha(binary),
            "role_manifest_sha256": mod.file_sha(tsv),
            "per_role": rows,
        }
        receipt["receipt_sha256"] = mod.canonical(receipt)
        cls.receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def call_auditor(self):
        with mock.patch.object(mod, "LEDGER_RECEIPT_SHA", self.signed_ledger_sha), \
             mock.patch.object(mod, "LEDGER_FILE_SHA", self.ledger_file_sha):
            return mod.audit(self.folder, self.ledger_path)

    def test_synthetic_43mb_exact_24role_and_60_regions_pass(self):
        result = self.call_auditor()
        self.assertEqual(result["status"], "JE01_CAL240_PHYSICAL_SOURCE_BYTE_AUDIT_PASS")
        self.assertEqual(result["bytes"], mod.BUDGET)
        self.assertEqual(result["tiers"], {"AF4": 12, "Q4": 6, "Q5": 6})
        self.assertEqual(result["native_27layer_NLL"], "NOT_MEASURED")

    def test_single_byte_corruption_fails(self):
        path = self.folder / mod.BINARY
        with path.open("r+b") as out:
            out.seek(mod.BUDGET // 2)
            original = out.read(1)
            out.seek(mod.BUDGET // 2)
            out.write(bytes([original[0] ^ 1]))
        try:
            with self.assertRaisesRegex(ValueError, "SHA_MISMATCH"):
                self.call_auditor()
        finally:
            with path.open("r+b") as out:
                out.seek(mod.BUDGET // 2)
                out.write(original)

    def test_role_tier_tamper_even_with_resigned_receipt_fails(self):
        original = self.receipt_path.read_bytes()
        altered = json.loads(original)
        altered["per_role"][0]["tier"] = "Q5"
        altered["receipt_sha256"] = mod.canonical(altered)
        self.receipt_path.write_text(json.dumps(altered, sort_keys=True))
        try:
            with self.assertRaisesRegex(ValueError, "TIER_POLICY_DRIFT"):
                self.call_auditor()
        finally:
            self.receipt_path.write_bytes(original)

    def test_heldout_scope_tamper_even_with_resigned_receipt_fails(self):
        original = self.receipt_path.read_bytes()
        altered = json.loads(original)
        altered["heldout_OOD_gold_accessed"] = True
        altered["receipt_sha256"] = mod.canonical(altered)
        self.receipt_path.write_text(json.dumps(altered, sort_keys=True))
        try:
            with self.assertRaisesRegex(ValueError, "SCOPE_CHANGED"):
                self.call_auditor()
        finally:
            self.receipt_path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
