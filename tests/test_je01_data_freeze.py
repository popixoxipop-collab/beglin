#!/usr/bin/env python3
"""Offline fail-closed JE01 real-cohort selection tests (no network/model)."""
import copy
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import je01_data_freeze as je


def _rows(source, offset, *, bad=False):
    name, ds, cfg, split, count, rev, blocks, target = source
    values = []
    for idx in range(offset, offset + 100):
        if ds == "Salesforce/wikitext":
            sentence = ("Original encyclopedia paragraph %s source %s includes "
                        "a discussion of chronology, evidence and mathematical "
                        "representations of independent research observations. "
                        "This passage is a fixed synthetic test record." % (idx, split))
            row = {"text": sentence}
        else:
            row = {"ctx": "A long real-world example scenario %d has been "
                   "described in detail by witnesses at a remote site." % idx,
                   "endings": ["An event with independent outcome %d option %d."
                               % (idx, k) for k in range(4)],
                   "label": idx % 4}
        if bad:
            row = {"text": ""}
        values.append({"row_idx": idx, "row": row, "truncated_cells": []})
    return {"offset": offset, "page_sha256": je.digest([ds, split, offset]),
            "row_records": values}


def pages_for(source, *, bad=False):
    return [_rows(source, off, bad=bad) for off in je.planned_offsets(source)]


def all_pages():
    return {source[0]: pages_for(source) for source in je.SOURCES}


class MockReply:
    def __init__(self, headers, data):
        self.headers = headers
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class PreregisterTests(unittest.TestCase):
    def test_offsets_frozen_and_distinct(self):
        for source in je.SOURCES:
            starts = je.planned_offsets(source)
            self.assertEqual(len(starts), source[-2])
            self.assertEqual(len(set(starts)), source[-2])
            self.assertTrue(all(0 <= i <= source[4]-100 for i in starts))
            # Source fetch windows may overlap (22 rows in real WikiText validation),
            # but the frozen selected prompt hashes and source IDs MUST NOT.
            self.assertTrue(all(b > a for a,b in zip(starts,starts[1:])))

    def test_full_frozen_512(self):
        manifest = je.freeze(all_pages())
        self.assertEqual(manifest["schema"], je.SCHEMA)
        self.assertEqual([manifest["cohorts"][x]["count"] for x in
                          ("calibration", "same_source_holdout", "ood_holdout")],
                         [256, 128, 128])
        rows = [r for grp in manifest["cohorts"].values() for r in grp["records"]]
        self.assertEqual(len({r["source_row_id_sha256"] for r in rows}), 512)
        self.assertEqual(len({r["prompt_sha256"] for r in rows}), 512)
        self.assertFalse(manifest["gold_labels_used_in_sampling"])
        self.assertFalse(manifest["native_model_forward_observed"])
        self.assertFalse(manifest["full_dataset_file_sha_verified"])
        self.assertNotIn('"label"', json.dumps(manifest))
        signature = manifest.pop("manifest_sha256")
        self.assertEqual(signature, je.digest(manifest))

    def test_deterministic_manifest_independent_page_order(self):
        pages = all_pages()
        baseline = je.freeze(pages)
        pages2 = copy.deepcopy(pages)
        for k in pages2:
            pages2[k].reverse()
        self.assertEqual(baseline, je.freeze(pages2))

    def test_label_change_cannot_change_selected_rows(self):
        pages = all_pages()
        before = je.freeze(pages)
        for page in pages["ood_holdout"]:
            for event in page["row_records"]:
                event["row"]["label"] = (event["row"]["label"] + 1) % 4
        after = je.freeze(pages)
        self.assertEqual(before, after)

    def test_missing_pages_or_candidate_insufficiency_fail(self):
        p = all_pages()
        p["calibration"] = p["calibration"][:-1]
        with self.assertRaisesRegex(ValueError, "PAGE_PLAN_TAMPER"):
            je.freeze(p)
        p = all_pages()
        p["ood_holdout"] = pages_for(je.SOURCES[-1], bad=True)
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_QUALIFYING_PROMPTS"):
            je.freeze(p)

    def test_source_page_revision_must_equal_frozen_commit(self):
        src = je.SOURCES[0]
        off = je.planned_offsets(src)[0]
        valid = {"rows": _rows(src, off)["row_records"],
                 "partial": False, "num_rows_total": src[4]}
        def wrong(*args, **kwargs):
            return MockReply({"x-revision": "1"*40}, json.dumps(valid).encode())
        with self.assertRaisesRegex(ValueError, "REVISION_DRIFT"):
            je.request_rows(src, off, opener=wrong)
        def correct(*args, **kwargs):
            return MockReply({"x-revision": src[5]}, json.dumps(valid).encode())
        self.assertEqual(je.request_rows(src,off,opener=correct)["offset"],off)

    def test_partial_and_row_index_drift_fail(self):
        src = je.SOURCES[1]
        off = je.planned_offsets(src)[0]
        page = {"rows": _rows(src, off)["row_records"],
                "partial": True, "num_rows_total": src[4]}
        def mock(*args, **kwargs):
            return MockReply({"x-revision": src[5]}, json.dumps(page).encode())
        with self.assertRaisesRegex(ValueError, "PARTIAL_OR_COUNT_DRIFT"):
            je.request_rows(src,off,opener=mock)
        page["partial"] = False
        page["rows"][0]["row_idx"] = -1
        with self.assertRaisesRegex(ValueError, "ROW_INDEX_DRIFT"):
            je.request_rows(src,off,opener=mock)


if __name__ == "__main__":
    unittest.main(verbosity=2)
