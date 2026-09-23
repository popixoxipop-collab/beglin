#!/usr/bin/env python3
import json
import unittest
from unittest.mock import patch

import precision_context as pc
import precision_evidence_v3 as ev


class FakeResponse:
    def __init__(self, body):
        self.body = body.encode() if isinstance(body, str) else body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return self.body


def h(ch):
    return ch * 64


def context():
    return pc.ExecutionContext(
        schema="precision-context-v3",
        model_id="m",
        architecture="arch",
        checkpoint_sha256=h("1"),
        tokenizer_sha256=h("2"),
        base_artifact_sha256=h("3"),
        backend="mlx_metal",
        device_fingerprint="dev",
        binary_sha256=h("4"),
        build_manifest_sha256=h("5"),
        kernel_revision="k",
        execution_mode="online",
        runtime_config_sha256=h("6"),
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        correction_mode="off",
    )


class EvidenceStoreTests(unittest.TestCase):
    @patch.object(ev, "_credentials", return_value=("https://x", "k"))
    @patch.object(ev.urllib.request, "urlopen")
    def test_context_upsert_includes_context_hash(self, urlopen, _creds):
        ctx = context()
        urlopen.return_value = FakeResponse(json.dumps([{
            **ctx.payload(), "context_hash": ctx.context_hash
        }]))
        got = ev.upsert_execution_context(ctx)
        self.assertEqual(got["context_hash"], ctx.context_hash)
        req = urlopen.call_args.args[0]
        body = json.loads(req.data)
        self.assertEqual(body["backend"], "mlx_metal")
        self.assertEqual(body["context_hash"], ctx.context_hash)
        self.assertEqual(body["schema_version"], "precision-context-v3")
        self.assertNotIn("schema", body)
        self.assertIn("on_conflict=context_hash", req.full_url)

    @patch.object(ev, "_credentials", return_value=("https://x", "k"))
    @patch.object(ev.urllib.request, "urlopen")
    def test_preflight_lookup_is_context_scoped(self, urlopen, _creds):
        urlopen.return_value = FakeResponse(json.dumps([{"id": 1}]))
        got = ev.fetch_latest_preflight("ctx", "shared_down_proj", 4, 6)
        self.assertEqual(got["id"], 1)
        req = urlopen.call_args.args[0]
        self.assertIn("context_hash=eq.ctx", req.full_url)
        self.assertIn("role=eq.shared_down_proj", req.full_url)


if __name__ == "__main__":
    unittest.main()
