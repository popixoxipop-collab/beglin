#!/usr/bin/env python3
"""CPU-only fail-closed negative tests for the private holdout distribution audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_v11_sealed_holdout1_cached_vs_uncached_v1 import (
    stats, logits_from_receipt, need,
)


class AuditedFullVocabTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="beglin_v11_holdout_audit_")
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.name="BEGLIN_v11_holdout1_qa_logits_step0.npy"
        self.vals=np.linspace(-2.0,2.0,102400,dtype=np.float32)[None,:]
        self.path=self.root/self.name
        with self.path.open("xb") as file:
            np.save(file,self.vals,allow_pickle=False)
        self.stage={
            "private_logits_basename":self.name,
            "private_logits_npy_sha256":hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "logits_SHA256":hashlib.sha256(self.vals.tobytes()).hexdigest(),
        }

    def test_same_full_vocab_outputs(self):
        a=self.vals.astype(np.float64)[0]
        info=stats(a,a)
        self.assertTrue(info["top1_agreement"])
        self.assertEqual(info["cached_vs_uncached_vocab_logit_relative_L2"],0)
        self.assertAlmostEqual(info["KL_uncached_to_cached_nats"],0,places=10)
        self.assertAlmostEqual(info["JS_divergence_nats"],0,places=10)
        self.assertEqual(info["top5_overlap"],5)

    def test_different_argmax_detected(self):
        x=self.vals.astype(np.float64)[0]
        y=x.copy()
        y[0]=50.0
        info=stats(x,y)
        self.assertFalse(info["top1_agreement"])
        self.assertGreater(info["cached_vs_uncached_vocab_logit_relative_L2"],0)
        self.assertGreater(info["KL_uncached_to_cached_nats"],0)

    def test_reopen_full_102400_f32_logits(self):
        vals,sha=logits_from_receipt(self.root,self.stage)
        self.assertEqual(sha,self.stage["private_logits_npy_sha256"])
        self.assertEqual(len(vals),102400)
        self.assertTrue(np.array_equal(vals,self.vals[0].astype(np.float64)))

    def test_refuse_changed_manifest_hash(self):
        bad=dict(self.stage,private_logits_npy_sha256="0"*64)
        with self.assertRaisesRegex(ValueError,"SHA"):
            logits_from_receipt(self.root,bad)

    def test_refuse_path_traversal(self):
        bad=dict(self.stage,private_logits_basename="../../original.safetensors")
        with self.assertRaisesRegex(ValueError,"basename"):
            logits_from_receipt(self.root,bad)

    def test_refuse_wrong_dtype(self):
        self.path.unlink()
        values=self.vals.astype(np.float64)
        with self.path.open("xb") as file:
            np.save(file,values,allow_pickle=False)
        bad=dict(self.stage)
        bad["private_logits_npy_sha256"]=hashlib.sha256(self.path.read_bytes()).hexdigest()
        bad["logits_SHA256"]=hashlib.sha256(values.tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,"F32"):
            logits_from_receipt(self.root,bad)

    def test_refuse_wrong_shape(self):
        self.path.unlink()
        values=self.vals[:,:2048]
        with self.path.open("xb") as file:
            np.save(file,values,allow_pickle=False)
        bad=dict(self.stage)
        bad["private_logits_npy_sha256"]=hashlib.sha256(self.path.read_bytes()).hexdigest()
        bad["logits_SHA256"]=hashlib.sha256(values.tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,"F32"):
            logits_from_receipt(self.root,bad)

    def test_refuse_symlink(self):
        path=self.root/"original.npy"
        self.path.rename(path)
        self.path.symlink_to(path)
        with self.assertRaisesRegex(ValueError,"symlink"):
            logits_from_receipt(self.root,self.stage)

    def test_refuse_modified_binary_payload(self):
        data=bytearray(self.path.read_bytes())
        data[-3]^=0x01
        self.path.write_bytes(data)
        with self.assertRaisesRegex(ValueError,"SHA"):
            logits_from_receipt(self.root,self.stage)


if __name__=="__main__":
    unittest.main(verbosity=2)
