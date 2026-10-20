import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "oml_acs"))
from manifest import audit_rows, sha_text, stable_sha

def row(i,split="train",text=None,cluster=None):
    return dict(dataset="example/dataset", dataset_revision="abc123", source_id=str(i),
                cluster_id=cluster or f"c{i}", split=split, text=text or f"Public synthetic prompt {i}")

class ManifestTests(unittest.TestCase):
    def test_audited_not_frozen_when_v6_identities_known(self):
        r=audit_rows([row(1),row(2,"final_iid")],consumed_v6_text_sha256={sha_text(f"consumed historical document {i}") for i in range(24)})
        self.assertEqual(r["status"],"AUDITED_NOT_FROZEN")
        self.assertFalse(r["issues"])

    def test_v6_unknown_blocks_freeze(self):
        self.assertEqual(audit_rows([row(1)])["status"],"V6_EXCLUSION_UNVERIFIED")

    def test_empty_v6_hashes_do_not_certify(self):
        self.assertEqual(audit_rows([row(1)],consumed_v6_text_sha256=set())["status"],"V6_EXCLUSION_UNVERIFIED")

    def test_consumed_v6_is_blocked(self):
        t="this prompt was consumed"
        self.assertEqual(audit_rows([row(1,text=t)],consumed_v6_text_sha256={sha_text(t)})["status"],"INVALID")

    def test_cross_split_semantic_group_leakage(self):
        r=audit_rows([row(1,cluster="shared"),row(2,"final_iid",cluster="shared")],consumed_v6_text_sha256=set())
        self.assertEqual(r["status"],"INVALID")

    def test_duplicate_exact_text_leakage(self):
        r=audit_rows([row(1,text=" hi  there "),row(2,"final_iid",text="hi there")],consumed_v6_text_sha256=set())
        self.assertEqual(r["status"],"INVALID")

    def test_schema_fail_closed(self):
        with self.assertRaises(ValueError):
            audit_rows([{"split":"final_iid"}])
        with self.assertRaises(ValueError):
            audit_rows([row(1,split="frozen")])

    def test_hash_order_independent(self):
        r=[row(1),row(2)]
        a=audit_rows(r,consumed_v6_text_sha256=set())
        b=audit_rows(list(reversed(r)),consumed_v6_text_sha256=set())
        self.assertEqual(a["row_manifest_sha256"],b["row_manifest_sha256"])
if __name__=="__main__":
    unittest.main(verbosity=2)
