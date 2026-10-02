#!/usr/bin/env python3
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestPrecisionContextSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sql = (ROOT / "supabase_migration_precision_context_v3.sql").read_text()

    def test_schema_is_additive_and_has_no_destructive_drop(self):
        self.assertIn("create table if not exists moe_execution_contexts_v3", self.sql.lower())
        self.assertIn("create table if not exists moe_validation_runs_v3", self.sql.lower())
        self.assertIn("create table if not exists moe_precision_transactions_v3", self.sql.lower())
        self.assertNotIn("drop table", self.sql.lower())

    def test_backend_and_binary_are_part_of_context(self):
        self.assertIn("backend text not null", self.sql.lower())
        self.assertIn("binary_sha256 text not null", self.sql.lower())
        self.assertIn("build_manifest_sha256 text not null", self.sql.lower())
        self.assertIn("kernel_version text not null", self.sql.lower())

    def test_validation_run_is_context_scoped(self):
        self.assertIn("context_id text not null references moe_execution_contexts_v3(context_id)", self.sql.lower())
        self.assertIn("preimage_policy_sha256 text not null", self.sql.lower())
        self.assertIn("candidate_policy_sha256 text not null", self.sql.lower())
        self.assertIn("correction_enabled boolean not null", self.sql.lower())

    def test_rls_and_grants_are_not_guessed(self):
        self.assertNotIn("create policy", self.sql.lower())
        self.assertNotIn(" grant ", self.sql.lower())


if __name__ == "__main__":
    unittest.main()
