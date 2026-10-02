#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = str(ROOT / "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import backend_adapters as adapters
import precision_context as pc


def ctx(backend):
    return pc.build_context(
        model_id="deepseek-v2-lite",
        architecture="deepseek_v2",
        checkpoint_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        base_artifact_sha256="c" * 64,
        backend=backend,
        device_fingerprint="cpu-device" if backend == "cpu" else "mlx-device",
        binary_sha256="d" * 64,
        build_manifest_sha256="e" * 64,
        kernel_version="cpu-v1" if backend == "cpu" else "mlx-v1",
        precision_mode="role_layer",
        quant_format="qng64_g64_ef_v1",
        group_size=64,
        encoder_version="qng64-ef-v1",
        execution_mode="online_b1",
        runtime_config={"batch": 1},
    )


def evidence(context, status="passed", passed=True):
    row = pc.evidence_scope(
        context=context,
        run_id="run-1",
        role="shared_down_proj",
        layer=4,
        n=6,
        preimage_policy={("shared_down_proj", 4): 5},
        candidate_policy={("shared_down_proj", 4): 6},
    )
    row["status"] = status
    row["pass"] = passed
    return row


class TestBackendAdmission(unittest.TestCase):
    def test_cpu_pass_is_accepted_by_cpu_adapter(self):
        cpu = ctx("cpu")
        self.assertTrue(
            adapters.admission_evidence_ok(
                evidence(cpu), cpu,
                preimage_policy={("shared_down_proj", 4): 5},
            )
        )

    def test_cpu_pass_is_rejected_for_mlx_context(self):
        cpu = ctx("cpu")
        gpu = ctx("mlx_metal")
        self.assertFalse(
            adapters.admission_evidence_ok(
                evidence(cpu), gpu,
                preimage_policy={("shared_down_proj", 4): 5},
            )
        )

    def test_failed_row_is_rejected_even_when_context_matches(self):
        cpu = ctx("cpu")
        self.assertFalse(adapters.admission_evidence_ok(evidence(cpu, status="failed", passed=False), cpu))

    def test_backend_state_paths_are_separate(self):
        cpu = adapters.backend_scoped_paths("/ctl", "modelrev", "cpu")["root"]
        gpu = adapters.backend_scoped_paths("/ctl", "modelrev", "mlx_metal")["root"]
        self.assertNotEqual(cpu, gpu)
        self.assertTrue(cpu.endswith("/modelrev/cpu"))
        self.assertTrue(gpu.endswith("/modelrev/mlx_metal"))


if __name__ == "__main__":
    unittest.main()
