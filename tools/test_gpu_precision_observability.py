#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import precision_observability as po


def decision(triggered=True):
    signal={
        "active_triggers":["low_margin"] if triggered else [],
        "margin":0.01 if triggered else None,
    }
    return {
        "signal":signal,
        "evidence_snapshot_sha256":"a"*64,
        "allocation_sha256":"b"*64,
        "selection_sha256":"c"*64,
        "cost_evidence_sha256":"d"*64,
        "combined_policy_evidence":{"evidence_sha256":"e"*64},
        "selected_policy":[
            {"role":"shared_up_proj","layer":3,"n":5},
            {"role":"shared_down_proj","layer":26,"n":6},
        ],
        "changes":[
            {"role":"shared_down_proj","layer":26,"from_n":5,"to_n":6}
        ],
        "selection":{"targets":[{
            "role":"shared_down_proj","layer":26,
            "status":"TRIGGER_CONDITIONED_ALTERNATE","selected_n":6,
            "active_triggers":["low_margin"],
            "evidence":{"low_margin":[{
                "evidence_id":"ev-low","evidence_sha256":"f"*64,
                "trigger_type":"low_margin","from_n":5,"to_n":6,
                "requests":2,"signal_bucket":{"margin_max":0.02},
            }]},
        }]},
        "policy_cost_optimizer":{"selected_cost":{
            "expected_e2e_ms":100.0,"transition_p50_ms":20.0,
            "steady_roundtrip_p50_ms":80.0,"expected_inference_passes":1,
            "resident_cache_bytes_after":1200,"cache_state":"cold",
            "active_weight_bytes":500,
        }},
    }


def result(*, transitioned=True, passes=1):
    return {
        "finite_logits":True,
        "responses":[[1,2]],
        "engine_wall_ms":90.0,
        "roundtrip_ms":110.0,
        "precision_epoch":{
            "before_policy_hash":"1"*64,
            "after_policy_hash":"2"*64,
            "before_epoch":2,"after_epoch":3 if transitioned else 2,
            "transitioned":transitioned,
            "inference_passes":passes,
            "engine_wall_ms":90.0,
            "roundtrip_ms":110.0,
            "transition_cost":{
                "transition_wall_ms":20.0 if transitioned else 0.0,
                "cache_hits":0,"cache_misses":1 if transitioned else 0,
                "cache_bytes_added":200 if transitioned else 0,
                "resident_cache_bytes":1200,
            },
        },
    }


class ObservabilityTests(unittest.TestCase):
    def test_hash_chain_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            obs=po.PrecisionObservability(
                lineage_path=Path(td)/"lineage.jsonl",
                snapshot_path=Path(td)/"summary.json",
                strict=True,
            )
            a=obs.record(admission_id="a",worker_pid=7,request_count=1,
                         decision=decision(),result=result())
            b=obs.record(admission_id="b",worker_pid=7,request_count=2,
                         decision=decision(False),result=result(transitioned=False,passes=2))
            self.assertNotEqual(a["record_sha256"],b["record_sha256"])
            summary=obs.summary()
            self.assertEqual(summary["admissions"],2)
            self.assertEqual(summary["requests"],3)
            self.assertEqual(summary["trigger_counts"],{"low_margin":1})
            self.assertEqual(summary["trigger_rates"],{"low_margin":0.5})
            self.assertEqual(summary["cache_misses"],1)
            self.assertEqual(summary["inference_pass_histogram"],{"1":1,"2":1})
            self.assertEqual(summary["extra_pass_rate"],0.5)
            self.assertEqual(summary["evidence_use_counts"],{"ev-low":2})
            self.assertEqual(
                summary["precision_residency_admissions"],
                {"shared_down_proj/L26/n6":2,"shared_up_proj/L3/n5":2},
            )
            self.assertEqual(summary["e2e_error_ms_mean"],10.0)
            self.assertEqual(
                summary["precision_residency_admissions"],
                {"shared_down_proj/L26/n6":2,"shared_up_proj/L3/n5":2},
            )

    def test_multi_trigger_counts_are_independent(self):
        with tempfile.TemporaryDirectory() as td:
            obs=po.PrecisionObservability(lineage_path=Path(td)/"lineage.jsonl")
            d=decision()
            d["signal"]={
                "active_triggers":["near_tie","high_entropy","routing_ambiguity"],
                "margin":0.04,"entropy":0.23,"routing_ambiguity_score":0.99,
            }
            obs.record(admission_id="triad",worker_pid=1,request_count=1,
                       decision=d,result=result())
            self.assertEqual(
                obs.summary()["trigger_counts"],
                {"high_entropy":1,"near_tie":1,"routing_ambiguity":1},
            )

    def test_duplicate_record_is_rejected_before_append(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"lineage.jsonl"
            obs=po.PrecisionObservability(lineage_path=path,strict=True)
            obs.record(admission_id="same",worker_pid=1,request_count=1,
                       decision=decision(),result=result())
            before=path.read_bytes()
            with self.assertRaises(po.PrecisionObservabilityError):
                obs.record(admission_id="same",worker_pid=1,request_count=1,
                           decision=decision(),result=result())
            self.assertEqual(path.read_bytes(),before)

    def test_adaptive_decision_normalizes_low_margin_lineage(self):
        got=po.build_adaptive_decision(
            adaptive={
                "enabled":True,"action":"RECOVERY_N6",
                "role":"shared_down_proj","layer":26,
                "base_n":5,"recovery_n":6,"worker_left_at_n":6,
                "trigger_request_indices":[0],
                "base_events":[{"margin":0.010715}],
            },
            active_policy=[
                {"role":"shared_down_proj","layer":26,"n":6},
                {"role":"shared_up_proj","layer":3,"n":6},
            ],
            changes=[{"role":"shared_down_proj","layer":26,"from_n":5,"to_n":6}],
            evidence_sha256="e"*64,
        )
        self.assertEqual(got["signal"]["active_triggers"],["low_margin"])
        self.assertEqual(got["signal"]["margin"],0.010715)
        row=got["selection"]["targets"][0]
        self.assertEqual(row["status"],"TRIGGER_CONDITIONED_ALTERNATE")
        self.assertEqual(row["selected_n"],6)

    def test_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"lineage.jsonl"
            obs=po.PrecisionObservability(lineage_path=path,strict=True)
            obs.record(admission_id="a",worker_pid=1,request_count=1,
                       decision=decision(),result=result())
            row=json.loads(path.read_text())
            row["actual_cost"]["roundtrip_ms"]=999.0
            path.write_text(json.dumps(row)+"\n")
            with self.assertRaises(po.PrecisionObservabilityError):
                obs.summary()

    def test_duplicate_admission_is_rejected(self):
        rows=[
            po.build_record(admission_id="same",worker_pid=1,request_count=1,
                            decision=decision(),result=result(),prev_record_sha256=None),
        ]
        rows.append(po.build_record(
            admission_id="same",worker_pid=1,request_count=1,
            decision=decision(),result=result(),
            prev_record_sha256=rows[0]["record_sha256"],
        ))
        with self.assertRaises(po.PrecisionObservabilityError):
            po.verify_records(rows)


if __name__=="__main__":
    unittest.main()
