#!/usr/bin/env python3
import unittest

import precision_risk_signals as prs

EVENTS=[{
    "req":0,"pos":9,"predicted_token":372,"competing_token":1,
    "margin":0.043699,"entropy":0.232097685,
    "routing_ambiguity_score":0.999847949,
    "routing_ambiguity_layer":24,
}]

class RiskSignalTests(unittest.TestCase):
    def test_realistic_event_activates_all_p5_triggers(self):
        got=prs.derive_signal(
            EVENTS,
            near_tie_margin_max=0.05,
            high_entropy_min=0.23,
            routing_ambiguity_min=0.999,
        )
        self.assertTrue(got["near_tie"])
        self.assertTrue(got["high_entropy"])
        self.assertTrue(got["routing_ambiguity"])
        self.assertEqual(got["risk_signal_summary"]["routing_ambiguity_layers"],[24])

    def test_stricter_buckets_hold_individual_signals(self):
        got=prs.derive_signal(
            EVENTS,
            near_tie_margin_max=0.04,
            high_entropy_min=0.24,
            routing_ambiguity_min=0.9999,
        )
        self.assertFalse(got["near_tie"])
        self.assertFalse(got["high_entropy"])
        self.assertFalse(got["routing_ambiguity"])
    def test_missing_optional_metrics_do_not_invent_signals(self):
        got=prs.derive_signal(
            [{"margin":0.01}],
            near_tie_margin_max=0.02,
            high_entropy_min=0.2,
            routing_ambiguity_min=0.9,
        )
        self.assertTrue(got["near_tie"])
        self.assertFalse(got["high_entropy"])
        self.assertFalse(got["routing_ambiguity"])
        self.assertIsNone(got["entropy"])
        self.assertIsNone(got["routing_ambiguity_score"])

    def test_invalid_normalized_values_fail_closed(self):
        with self.assertRaises(prs.RiskSignalError):
            prs.summarize_events([{"entropy":1.1}])
        with self.assertRaises(prs.RiskSignalError):
            prs.summarize_events([{"routing_ambiguity_score":-0.1}])

if __name__=="__main__":
    unittest.main()
