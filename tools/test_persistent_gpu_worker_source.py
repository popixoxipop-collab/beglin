#!/usr/bin/env python3
from pathlib import Path
import unittest

SRC=Path("qwen_infer.c").read_text()


class PersistentGpuSourceContract(unittest.TestCase):
    def test_persistent_env_is_opt_in(self):
        self.assertIn('QWEN_MOE_GPU_PERSISTENT_DIR', SRC)
        self.assertIn('persistent_mode = persistent_dir && persistent_dir[0]', SRC)

    def test_atomic_request_claim_protocol_exists(self):
        self.assertIn('BEGLIN_GPU_PERSISTENT_REQUEST_V1', SRC)
        self.assertIn('request.processing', SRC)
        self.assertIn('rename(request_path, processing_path)', SRC)

    def test_atomic_result_publish_protocol_exists(self):
        self.assertIn('BEGLIN_GPU_PERSISTENT_RESULT_V1', SRC)
        self.assertIn('fsync(fileno(f))', SRC)
        self.assertIn('rename(tmp_path, result_path)', SRC)

    def test_first_batch_warms_later_batches_skip_warmup(self):
        self.assertIn(
            'for (int pass = (persistent_mode && persistent_warmed) ? 1 : 0; pass < 2; pass++)',
            SRC,
        )
        self.assertIn('persistent_warmed = 1;', SRC)

    def test_worker_stays_resident_after_batch(self):
        self.assertIn('worker stays resident', SRC)
        self.assertIn('continue;', SRC)

    def test_shutdown_only_checked_while_idle(self):
        self.assertIn('shutdown_path', SRC)
        self.assertIn('access(shutdown_path, F_OK) == 0', SRC)


if __name__=="__main__":
    unittest.main()
