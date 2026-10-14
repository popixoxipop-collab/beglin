#!/usr/bin/env python3
"""P10 independent auditor tamper tests using sealed sanitized evidence only."""
import json,pathlib,shutil,tempfile,unittest
from hcp_p10_evidence_audit import check
REPO=pathlib.Path(__file__).resolve().parents[1]
EVID=REPO/"experiments/hcp_p10/evidence/native_xox_20261014"
NAMES=("new_holdout","restart","two_slots","steady_tail")

class P10EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=pathlib.Path(self.tmp.name)
        for name in NAMES:
            shutil.copyfile(EVID/(name+".json"),self.path/(name+".json"))
    def tearDown(self):
        self.tmp.cleanup()
    def modify(self,name,fn):
        p=self.path/(name+".json")
        j=json.loads(p.read_text());fn(j)
        p.write_text(json.dumps(j,sort_keys=True)+"\n")
    def verify_reject(self):
        with self.assertRaises((ValueError,AssertionError,KeyError)):
            check(self.path,REPO)
    def test_01_untampered_copy_passes(self):
        r=check(self.path,REPO)
        self.assertEqual(r["new_heldout_cache_parity_pairs"],24)
        self.assertEqual(r["steady_tail_requests"],96)
        self.assertFalse(r["automatic_live_promotion"])
    def test_02_fake_new_holdout_manifest_rejected(self):
        self.modify("new_holdout",lambda j:j.update(token_manifest_sha256="0"*64))
        self.verify_reject()
    def test_03_generated_output_mismatch_rejected(self):
        self.modify("new_holdout",lambda j:j["records"][25].update(output_sequence_sha256="f"*64))
        self.verify_reject()
    def test_04_missing_two_slot_sequence_rejected(self):
        self.modify("two_slots",lambda j:j.update(inferred_sequences=15))
        self.verify_reject()
    def test_05_worker_PID_reuse_claim_rejected(self):
        self.modify("restart",lambda j:j["restart_records"][0].update(new_process=False))
        self.verify_reject()
    def test_06_steady_missing_admission_rejected(self):
        self.modify("steady_tail",lambda j:j["rows"].pop())
        self.verify_reject()
    def test_07_steady_wrong_p95_rejected(self):
        self.modify("steady_tail",lambda j:j["summary"]["on"].update(p95_ms=999))
        self.verify_reject()
    def test_08_unmeasured_BF16_teacher_claim_rejected(self):
        self.modify("new_holdout",lambda j:j.update(BF16_reference_measured=True))
        self.verify_reject()
    def test_09_wrong_cluster_bootstrap_rejected(self):
        self.modify("steady_tail",lambda j:j["paired"].update(mean_on_minus_off_ms=100))
        self.verify_reject()
    def test_10_duplicate_holdout_respondent_rejected(self):
        self.modify("new_holdout",lambda j:j["records"].append(dict(j["records"][0])))
        self.verify_reject()

if __name__=="__main__":unittest.main()
