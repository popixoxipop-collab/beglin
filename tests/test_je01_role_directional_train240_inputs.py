"""Synthetic tests of recovered EOE source + native preflight composition.

No real token file is opened, no input preparation or model execution runs.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import je01_role_directional_train240_inputs as source
import je01_role_directional_native_runner as native


def seal(path, data):
    value = copy.deepcopy(data)
    value["receipt_sha256"] = source.canonical(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return value["receipt_sha256"]


class SourceLinkageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.native_root = self.root / "native"
        (self.native_root / "calibration").mkdir(parents=True)
        self.cases, self.prompts, receipts = [], [], []
        for row in range(256):
            count = 41 if row < 16 else (91 if row == 255 else 43)
            prompt = [100000, 100 + row, 900]
            self.prompts.append(prompt)
            raw = struct.pack("<3i", *prompt)
            ident = hashlib.sha256(f"synthetic-{row}".encode()).hexdigest()
            checksum = hashlib.sha256(raw).hexdigest()
            (self.native_root / "calibration" / (ident + ".bin")).write_bytes(raw)
            receipts.append({"input_id": ident, "native_int32_bytes": len(raw),
                "native_int32_sha256": checksum, "token_count_with_bos": 3})
            self.cases.append({"row": row, "input_id": ident,
                "native_BOS_prompt_sha256": checksum,
                "original_BOS_prompt_positions": 3,
                "model_full_BOS_sequence_positions": 3 + count,
                "tokenizer_prefix_exact": True,
                "gold_continuation_token_ids": list(range(2000, 2000 + count)),
                "gold_token_count": count})
        native_pin = seal(self.native_root / source.SOURCE_RECEIPT_NAME,
            {"total_input_records": 512, "bos_id": 100000, "native_context_positions": 384,
             "source_manifest_sha256": source.FROZEN_SHA,
             "cohorts": {"calibration": {"file_receipts": receipts},
                         "validation": {"unopened_fixture_marker": True},
                         "ood": {"unopened_fixture_marker": True}}})
        self.gold_path = self.root / "train256-gold.json"
        self.gold = {"status": "TRAIN256_FULL_GOLD_SOURCE_READY",
            "original_256_source_rows": 256, "aligned_to_original_prompt_prefix_count": 256,
            "same_source_validation_and_ood_labels_used": False, "cases": self.cases}
        self.gold_pin = seal(self.gold_path, self.gold)
        self.baseline_path = self.root / "train256-baseline.json"
        self.chunks = [{"chunk": b, "gold_targets_scored": sum(
            c["gold_token_count"] for c in self.cases[b*16:(b+1)*16]),
            "result_sha256": hashlib.sha256(f"synthetic-chunk-{b}".encode()).hexdigest()}
            for b in range(16)]
        self.baseline = {"status": "REAL_ORIGINAL_AF_TRAIN_FULL_TARGET_NLL_CAL256_PASS",
            "training_only": True, "heldout_gold_labels_used": False,
            "gold_full_target_tokens_scored": 11024, "n_calibration_train_cases": 256,
            "mean_gold_token_NLL": source.BASELINE_NLL,
            "source_gold_sha256": self.gold_pin, "original_512_input_sha256": native_pin,
            "per_case_real_NLL": [{"row": c["row"], "source_id": c["input_id"],
                "tokens": c["gold_token_count"], "first_gold_nll": source.BASELINE_NLL,
                "nll_sum": source.BASELINE_NLL * c["gold_token_count"]} for c in self.cases],
            "per_chunk_native_output_sha256": self.chunks}
        self.baseline_pin = seal(self.baseline_path, self.baseline)
        # External pins are replaced only in synthetic fixtures, never via CLI.
        for name, value in (("GOLD_SHA", self.gold_pin), ("BASE_SHA", self.baseline_pin),
                            ("INPUT_SHA", native_pin)):
            p = patch.object(source, name, value)
            p.start()
            self.addCleanup(p.stop)

    def dataset(self, chunk=1):
        folder = self.root / f"chunk{chunk:02d}"
        folder.mkdir()
        inputs = self.root / "inputs"
        inputs.mkdir(exist_ok=True)
        lines, hashes, targets, sources = [], [], [], []
        meta = bytearray(source.META_MAGIC)
        for row in range(chunk * 16, (chunk + 1) * 16):
            case = self.cases[row]
            target = case["gold_continuation_token_ids"]
            tokens = self.prompts[row] + target[:-1]
            raw = struct.pack("<" + "i" * len(tokens), *tokens)
            path = inputs / (case["input_id"] + ".bin")
            path.write_bytes(raw)
            lines.append(str(path) + " 1\n")
            hashes.append(hashlib.sha256(raw).hexdigest())
            targets.append(target)
            meta.extend(struct.pack("<II384i", 3, len(target),
                *(target + [-1] * (384 - len(target)))))
            sources.append({"row": row, "input_id": case["input_id"],
                "original_prompt_path": str(self.native_root / "calibration" / (case["input_id"] + ".bin")),
                "original_prompt_sha256": case["native_BOS_prompt_sha256"],
                "original_prompt_positions": 3, "gold_tokens": len(target),
                "original_baseline_first_gold_nll": source.BASELINE_NLL,
                "original_baseline_nll_sum": source.BASELINE_NLL * len(target)})
        manifest, metadata = folder / "manifest.txt", folder / "gold_meta.bin"
        manifest.write_text("".join(lines))
        metadata.write_bytes(meta)
        return {"schema": source.SCHEMA, "scope": "TRAIN240", "chunk": chunk,
            "dataset_id": f"train240_chunk{chunk:02d}", "case_ids": list(range(chunk*16, (chunk+1)*16)),
            "independent_validation_ood_accessed": False,
            "manifest": str(manifest), "gold_metadata": str(metadata),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "gold_metadata_sha256": hashlib.sha256(meta).hexdigest(),
            "input_sha256": hashes, "target_token_ids": targets, "case_sources": sources,
            "gold_tokens": sum(map(len, targets)),
            "baseline_output_sha256": self.chunks[chunk]["result_sha256"],
            "source_linkage": source.source_linkage(self.native_root, self.gold_path, self.baseline_path)}

    def test_first_and_last_chunks_link_without_writes(self):
        for chunk in (1, 15):
            with self.subTest(chunk=chunk):
                data = self.dataset(chunk)
                before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
                rows = source.validate_source_linkage(data)
                self.assertEqual([r["row"] for r in rows], data["case_ids"])
                self.assertEqual(sum(r["gold_tokens"] for r in rows), data["gold_tokens"])
                native.validate_dataset(data)
                self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_pilot_and_noncontiguous_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "NOT_FROZEN_CHUNK"):
            source.validate_source_linkage(self.dataset(0))
        data = self.dataset(1)
        data["case_ids"][-1] = 32
        with self.assertRaisesRegex(ValueError, "NOT_FROZEN_CHUNK"):
            source.validate_source_linkage(data)

    def test_original_prompt_tampering_detected(self):
        data = self.dataset()
        path = Path(data["case_sources"][0]["original_prompt_path"])
        raw = bytearray(path.read_bytes())
        struct.pack_into("<i", raw, 4, 777)
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "ORIGINAL_CASE_IDENTITY_DRIFT"):
            native.validate_dataset(data)

    def test_rehashed_teacher_forcing_payload_cannot_override_source(self):
        data = self.dataset()
        path = Path(Path(data["manifest"]).read_text().splitlines()[0].rsplit(" ", 1)[0])
        raw = bytearray(path.read_bytes())
        struct.pack_into("<i", raw, 4, 777)
        path.write_bytes(raw)
        data["input_sha256"][0] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "DERIVED_DATA_NOT_FROM_PINNED_SOURCE"):
            native.validate_dataset(data)

    def test_actual_payload_and_metadata_checked_by_native_preflight(self):
        data = self.dataset()
        path = Path(data["gold_metadata"])
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "DATA_FILES_HASH_DRIFT"):
            native.validate_dataset(data)

    def test_other_chunk_baseline_rejected(self):
        data = self.dataset()
        data["baseline_output_sha256"] = self.chunks[2]["result_sha256"]
        with self.assertRaisesRegex(ValueError, "CHUNK_BASELINE_DRIFT"):
            native.validate_dataset(data)

    def test_resealed_source_rejected_by_external_pin(self):
        data = self.dataset()
        changed = copy.deepcopy(self.gold)
        changed["cases"][16]["original_BOS_prompt_positions"] += 1
        seal(self.gold_path, changed)
        with self.assertRaisesRegex(ValueError, "SOURCE_RECEIPT_HASH_MISMATCH"):
            native.validate_dataset(data)

    def test_heldout_scope_and_prompt_symlink_rejected(self):
        data = self.dataset()
        data["independent_validation_ood_accessed"] = True
        with self.assertRaisesRegex(ValueError, "DATASET_SCOPE_DRIFT"):
            native.validate_dataset(data)
        data["independent_validation_ood_accessed"] = False
        path = Path(data["case_sources"][0]["original_prompt_path"])
        renamed = path.with_suffix(".original")
        path.rename(renamed)
        path.symlink_to(renamed)
        with self.assertRaisesRegex(ValueError, "ORIGINAL_PROMPT_MISSING"):
            native.validate_dataset(data)


if __name__ == "__main__":
    unittest.main()
