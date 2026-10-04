#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import textwrap
import unittest

import tokenizer_runtime_v1 as tr


def _contract(*, backend, status, evidence=None, arch="gpt-oss"):
    value = {
        "schema": "beglin-tokenizer-contract-v1",
        "architecture_id": arch,
        "tokenizer_family": "TEST",
        "artifact_kind": "TOKENIZER_JSON",
        "source_files": ["tokenizer.json"],
        "status": status,
        "encode_backend": backend,
        "decode_backend": backend,
        "text_io_supported": backend == "beglin_bpe",
        "text_io_mode": "DENSE_GGUF_GREEDY" if backend == "beglin_bpe" else "NOT_WIRED",
        "missing_primitives": [],
        "verification_evidence": evidence,
        "silent_fallback_allowed": False,
    }
    value["tokenizer_contract_sha256"] = tr.sha256_json(value)
    return value


class TokenizerRuntimePlanTests(unittest.TestCase):
    def test_in_engine_verified_plan_is_deterministic(self):
        contract = _contract(
            backend="beglin_bpe",
            status="IN_ENGINE_VERIFIED",
            evidence={"status": "VERIFIED"},
            arch="qwen2",
        )
        a = tr.compile_runtime_plan(contract)
        b = tr.compile_runtime_plan(contract)
        self.assertEqual(a["status"], "READY_IN_ENGINE")
        self.assertEqual(a["execution_mode"], "ENGINE_TEXT_PATH")
        self.assertEqual(a["runtime_plan_sha256"], b["runtime_plan_sha256"])
        self.assertFalse(a["silent_fallback_allowed"])

    def test_in_engine_unverified_requires_validation(self):
        contract = _contract(
            backend="beglin_bpe",
            status="IMPLEMENTED_UNVERIFIED",
            arch="llama",
        )
        got = tr.compile_runtime_plan(contract)
        self.assertEqual(got["status"], "VALIDATION_REQUIRED")

    def test_unsupported_contract_fails_closed(self):
        contract = _contract(
            backend="beglin_bpe",
            status="UNSUPPORTED",
        )
        with self.assertRaises(tr.TokenizerRuntimeError):
            tr.compile_runtime_plan(contract)

    def test_external_backend_requires_verified_evidence(self):
        contract = _contract(
            backend="tiktoken_o200k_harmony",
            status="IMPLEMENTED_UNVERIFIED",
        )
        with self.assertRaisesRegex(tr.TokenizerRuntimeError, "EXTERNAL_VERIFIED"):
            tr.compile_runtime_plan(
                contract,
                external_executable="/bin/echo",
                external_executable_sha256="0" * 64,
            )

    def test_external_plan_binds_executable_and_runs_json_protocol(self):
        contract = _contract(
            backend="external_deepseek_reference",
            status="EXTERNAL_VERIFIED",
            evidence={"schema": "fixture", "status": "VERIFIED"},
            arch="deepseek_v2",
        )
        with tempfile.TemporaryDirectory() as td:
            adapter = Path(td) / "adapter"
            adapter.write_text(textwrap.dedent("""\
                #!/usr/bin/env python3
                import json, sys
                req=json.loads(sys.stdin.read())
                if req["operation"]=="encode":
                    out={"schema":"beglin-tokenizer-response-v1","tokens":[7,11,13]}
                else:
                    out={"schema":"beglin-tokenizer-response-v1","text":"decoded"}
                print(json.dumps(out,sort_keys=True))
            """))
            adapter.chmod(adapter.stat().st_mode | stat.S_IXUSR)
            digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
            plan = tr.compile_runtime_plan(
                contract,
                external_executable=str(adapter),
                external_executable_sha256=digest,
            )
            self.assertEqual(plan["status"], "READY_EXTERNAL")
            enc = tr.run_external(plan, operation="encode", text="hello")
            self.assertEqual(enc["tokens"], [7, 11, 13])
            dec = tr.run_external(plan, operation="decode", tokens=[7, 11, 13])
            self.assertEqual(dec["text"], "decoded")

    def test_external_executable_drift_is_rejected(self):
        contract = _contract(
            backend="sentencepiece_external",
            status="EXTERNAL_VERIFIED",
            evidence={"schema": "fixture", "status": "VERIFIED"},
            arch="llama",
        )
        with tempfile.TemporaryDirectory() as td:
            adapter = Path(td) / "adapter"
            adapter.write_text("#!/bin/sh\necho ok\n")
            adapter.chmod(adapter.stat().st_mode | stat.S_IXUSR)
            digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
            plan = tr.compile_runtime_plan(
                contract,
                external_executable=str(adapter),
                external_executable_sha256=digest,
            )
            adapter.write_text("#!/bin/sh\necho changed\n")
            with self.assertRaisesRegex(tr.TokenizerRuntimeError, "changed after planning"):
                tr.run_external(plan, operation="encode", text="x")


if __name__ == "__main__":
    unittest.main()
