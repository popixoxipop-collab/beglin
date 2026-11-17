#!/usr/bin/env python3
"""Bounded source-attested real external pretrained reference forward receipt guards."""
import json
import hashlib
import math
import pathlib
import sys
import unittest

R=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/"tools"))
import je01_external_chat4bit_real_forward as run

class ExternalPretrainedReferenceEvidence(unittest.TestCase):
    def test_actual_pretrained_27layer_one_prompt_forward_pass_receipt(self):
        file=R/"docs/research/JE01_EXTERNAL_CHAT4BIT_REAL_PRETRAINED_FORWARD_20261117.json"
        record=json.loads(file.read_text())
        self.assertEqual(record["status"],"EXTERNAL_CHAT4BIT_REAL_PRETRAINED_FORWARD_PASS")
        self.assertEqual(record["schema"],run.SCHEMA)
        self.assertEqual(record["managed_job_id"],"job_c415161e5e2b0a13a09c52e209b944ce")
        self.assertEqual(record["managed_job_exit_code"],0)
        self.assertEqual(record["source_github_blob_sha"],"52630b35071d4e52f7afb6a4c0f2714a6c917326")
        self.assertEqual(record["reference_checkpoint_identity_sha256"],run.MODEL_SHA)
        self.assertEqual(record["reference_tokenizer_sha256"],run.TOK_SHA)
        self.assertEqual(record["reference_revision"],run.REV)
        self.assertEqual(record["real_pretrained_full_model_layers"],27)
        self.assertEqual(record["logit_vocabulary_size"],102400)
        self.assertEqual(record["input_token_count"],6)
        self.assertEqual(record["next_token_argmax"],8913)
        self.assertTrue(math.isfinite(record["max_logit"]))
        self.assertTrue(math.isfinite(record["min_logit"]))
        self.assertEqual(record["reference_mlx_hard_memory_limit"],run.MLX_LIMIT_BYTES)
        self.assertLess(record["reference_mlx_gpu_peak_bytes"],run.MLX_LIMIT_BYTES)
        self.assertGreater(record["reference_mlx_gpu_active_bytes"],8_800_000_000)
        self.assertFalse(record["filelock_imported"])
        self.assertFalse(record["JE01_frozen_training_or_holdout_used"])
        self.assertFalse(record["production_write_allowed"])
        self.assertEqual(record["source_model_original_bf16_identity"],"DIFFERENT_NOT_PARITY")
        self.assertEqual(record["BEGLIN_original_full_logits_parity"],"NOT_TESTED")
        self.assertEqual(record["JE01_real_routed_Expert_NLL"],"NOT_MEASURED")
        self.assertIn("ALL_REFERENCE_WEIGHTS_MATERIALIZED",record["all_run_stage_names"])
        self.assertIn("PRETRAINED_REFERENCE_WEIGHTS_BOUND_STRICT",record["all_run_stage_names"])

    def test_source_hard_memory_cap_and_no_model_equivalence_claim(self):
        self.assertEqual(run.MLX_LIMIT_BYTES,10_200_000_000)
        self.assertEqual(run.MODEL_SHA,"33730a3d5b3bec0ba72cea02f896a78ca367275d7fbf63e0e880e0e0c99b4d4c")
        self.assertNotEqual(run.TOK_SHA,
            "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669")
        self.assertEqual(run.SHARDS,{"model-00001-of-00002.safetensors":5316545126,
                                     "model-00002-of-00002.safetensors":3523543576})

if __name__=="__main__":
    unittest.main(verbosity=2)
