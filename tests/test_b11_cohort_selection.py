#!/usr/bin/env python3
"""B11 outcome-blind source/selection synthetic contracts (no dataset access)."""
import hashlib
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import b11_cohort_selection as coh


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def candidate(corpus, idx, length):
    key = "%s-%d-%d" % (corpus, idx, length)
    return {"source_id": key,
            "question_sha256": h("question:" + key),
            "full_prompt_sha256": h("prompt:" + key),
            "prefix_ids_sha256": h("prefix:" + key),
            "prefix_len": length}


def excluded(source, index, length):
    raw = candidate("old-" + source, index, length)
    return {k: raw[k] for k in ("source_id", "question_sha256")}


def fixtures(arc_n=210, ob_n=100):
    # Tests use synthetic IDs, not actual Qv21 score or heldout files.
    arc = [candidate("arc", i, 44) for i in range(arc_n)] + [
        candidate("arc", 1000 + i, 124) for i in range(arc_n)
    ]
    ob = [candidate("ob", i, 44) for i in range(ob_n)] + [
        candidate("ob", 1000 + i, 124) for i in range(ob_n)
    ]
    exclusions = {}
    for name, corpus, n in coh.EXCLUSION_SETS:
        exclusions[name] = [
            excluded(corpus, i + 10000 + (32 if name.startswith("qv21") else 0), 44)
            for i in range(n)
        ]
        # Avoid overlapping prior-exclusion qv21 split IDs.
        if name.startswith("qv21_"):
            offset = {"qv21_cal_arc_train96": 0,
                      "qv21_eval_arc_train48": 400,
                      "qv21_eval_ob_train48": 800}[name]
            exclusions[name] = [excluded(corpus, i + 10000 + offset, 44) for i in range(n)]
    return arc, ob, exclusions


class CohortContracts(unittest.TestCase):
    def test_400_unique_label_free_and_exact_split_strata(self):
        manifest = coh.select_cohorts(*fixtures())
        splits = manifest["selected"]
        self.assertEqual(manifest["public"]["split_counts"],
                         {"b11_fit_arc_train120": 120,
                          "b11_cal_arc_train80": 80,
                          "b11_eval_arc_train80": 80,
                          "b11_eval_ob_train120": 120})
        items = [i for cohort in splits.values() for i in cohort]
        self.assertEqual(len(items), 400)
        self.assertEqual(len(set(x["sample_id"] for x in items)), 400)
        self.assertEqual(len(set(x["question_sha256"] for x in items)), 400)
        for name, group in splits.items():
            self.assertEqual(sum(x["prefix_len"] < 85 for x in group), len(group) // 2)
            self.assertEqual(sum(x["prefix_len"] >= 85 for x in group), len(group) // 2)
            self.assertFalse(any("correct" in x or "observed_delta" in x for x in group))
        self.assertEqual(len(manifest["manifest_sha256"]), 64)

    def test_selection_order_invariant(self):
        arc, ob, exclusions = fixtures()
        one = coh.select_cohorts(arc, ob, exclusions)
        two = coh.select_cohorts(list(reversed(arc)), list(reversed(ob)), exclusions)
        self.assertEqual(one, two)

    def test_arc_short_or_long_shortage_fails(self):
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_FRESH_arc_train"):
            coh.select_cohorts(*fixtures(arc_n=139))
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_FRESH_ob_train"):
            coh.select_cohorts(*fixtures(ob_n=59))

    def test_duplicate_question_in_candidate_denied(self):
        arc, ob, exclusions = fixtures()
        arc[1]["question_sha256"] = arc[0]["question_sha256"]
        with self.assertRaisesRegex(ValueError, "SOURCE_DUPLICATE"):
            coh.select_cohorts(arc, ob, exclusions)

    def test_candidate_rejects_correct_or_gold(self):
        arc, ob, exclusions = fixtures()
        arc[0]["correct"] = "A"
        with self.assertRaisesRegex(ValueError, "LABEL_FREE"):
            coh.select_cohorts(arc, ob, exclusions)

    def test_exclusion_cohort_must_be_complete(self):
        arc, ob, exclusions = fixtures()
        exclusions["qv21_cal_arc_train96"].pop()
        with self.assertRaisesRegex(ValueError, "EXCLUSION_COUNT"):
            coh.select_cohorts(arc, ob, exclusions)

    def test_exclusion_rows_cannot_expose_outcome(self):
        arc, ob, exclusions = fixtures()
        exclusions["qv20_arc_train32"][0]["loss"] = 0.0
        with self.assertRaisesRegex(ValueError, "EXCLUSION_FIELDS"):
            coh.select_cohorts(arc, ob, exclusions)

    def test_bad_sha_or_prefix_rejected(self):
        arc, ob, exclusions = fixtures()
        arc[0]["prefix_ids_sha256"] = "z" * 64
        with self.assertRaisesRegex(ValueError, "SHA_INVALID"):
            coh.select_cohorts(arc, ob, exclusions)

    def test_exclusion_lists_cannot_overlap(self):
        arc, ob, exclusions = fixtures()
        exclusions["qv21_eval_arc_train48"][0] = exclusions["qv20_arc_train32"][0].copy()
        with self.assertRaisesRegex(ValueError, "PRIOR_EXCLUSION_OVERLAP"):
            coh.select_cohorts(arc, ob, exclusions)

    def test_prior_source_question_id_excluded(self):
        arc, ob, exclusions = fixtures()
        # Repeated prior question with a new source ID must not enter B11.
        arc.append({**arc[0], "source_id": "different"} )
        exclusions["qv20_arc_train32"][0]["question_sha256"] = arc[0]["question_sha256"]
        with self.assertRaisesRegex(ValueError, "SOURCE_DUPLICATE"):
            coh.select_cohorts(arc, ob, exclusions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
