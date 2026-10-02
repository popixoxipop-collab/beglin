#!/usr/bin/env python3
from pathlib import Path
import re
import unittest


SRC = Path(__file__).resolve().parents[1] / "qwen_infer.c"


class PersistentGpuSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SRC.read_text()

    def test_persistent_mode_is_opt_in(self):
        self.assertIn('getenv("QWEN_MOE_GPU_PERSIST_CONTROL")', self.text)
        self.assertIn("int persistent_on = env_persist_control && env_persist_control[0];", self.text)

    def test_original_gpu_gate_and_result_are_retained(self):
        self.assertEqual(
            self.text.count("static int run_moe_gpu_cbatch_online_gate(int argc, char **argv)"),
            1,
        )
        self.assertEqual(
            self.text.count("RESULT: MoE GPU V5h online cbatch gate complete"),
            1,
        )
        self.assertIn("QWEN_MOE_GPU_CBATCH_ONLINE", self.text)

    def test_persistent_transport_is_atomic_file_based(self):
        self.assertIn("moe_gpu_persist_wait_request", self.text)
        self.assertIn("moe_gpu_persist_publish_response", self.text)
        self.assertIn('rename(tmp, response_path)', self.text)
        self.assertIn('fsync(fileno(f))', self.text)
        self.assertIn('nanosleep(&idle, NULL)', self.text)

    def test_persistent_second_batch_skips_duplicate_warmup(self):
        self.assertIn(
            "int first_pass = (persistent_on && persist_warmed) ? 1 : 0;",
            self.text,
        )
        self.assertIn("persist_warmed = 1;", self.text)
        self.assertIn("goto persist_next_request;", self.text)

    def test_persistent_manifest_controls_request_count(self):
        self.assertIn("R = mf_n;", self.text)
        self.assertIn(
            'FATAL: [moe gpu persist] manifest request count R=%d out of [1,%d]',
            self.text,
        )

    def test_response_carries_runtime_evidence(self):
        for token in [
            "beglin-gpu-persistent-response-v1",
            '\\"duration_ms\\"',
            '\\"finite_logits\\"',
            '\\"weight_epoch\\"',
            '\\"generated_tokens\\"',
        ]:
            self.assertIn(token, self.text)

    def test_clean_shutdown_does_not_touch_default_path(self):
        self.assertIn('!strcmp(manifest, "QUIT")', self.text)
        self.assertIn("[moe gpu persist] clean shutdown requested", self.text)


if __name__ == "__main__":
    unittest.main()
