#!/usr/bin/env python3
"""Synthetic import/binding tests, no real checkpoint files required."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from precision_budget_collect import COLLECT_SCHEMA,collect
from precision_budget_search import PlanError


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.sumfile=self.root/'summary.json'
        self.manfile=self.root/'manifest.json'
        self.summary={"tensor":"model.layers.2.self_attn.q_proj.weight","shape":[2,64],"results":[
            {"n":2,"effective_bpw":2.5,"rel_l2":.2,"path":"sim_n2.safetensors"},
            {"n":5,"effective_bpw":5.5,"rel_l2":.07,"path":"sim_n5.safetensors"},
            {"n":7,"effective_bpw":7.5,"rel_l2":.02,"path":"sim_n7.safetensors"}]}
        self.manifest={"schema":COLLECT_SCHEMA,"model_id":"SYNTHETIC","checkpoint_sha256":"a"*64,
            "backend":"mlx_metal","budget_bytes":120,"tensors":[
                {"name":self.summary["tensor"],"role":"q_proj","layer":2,"summary_path":"summary.json","summary_sha256":""}]}
        self.save()

    def save(self):
        self.sumfile.write_text(json.dumps(self.summary))
        self.manifest["tensors"][0]["summary_sha256"]=hashlib.sha256(self.sumfile.read_bytes()).hexdigest()
        self.manfile.write_text(json.dumps(self.manifest))

    def test_valid_source_binding(self):
        x=collect(self.manfile)
        self.assertEqual(x["selections"][0]["n"],7)
        self.assertEqual(x["source_metric_provenance"]["summaries"][0]["sha256"],self.manifest["tensors"][0]["summary_sha256"])
        self.assertFalse(x["automatic_live_promotion"])

    def test_modified_summary_rejected(self):
        self.summary["results"][0]["rel_l2"]=.999
        self.sumfile.write_text(json.dumps(self.summary))
        with self.assertRaisesRegex(PlanError,"SHA mismatch"):collect(self.manfile)

    def test_legacy_wrong_overhead_rejected(self):
        self.summary["results"][0]["effective_bpw"]=2.25
        self.save()
        with self.assertRaisesRegex(PlanError,"legacy effective_bpw"):collect(self.manfile)

    def test_unaligned_shape_rejected(self):
        self.summary["shape"]=[2,63]
        self.save()
        with self.assertRaisesRegex(PlanError,"aligned"):collect(self.manfile)

    def test_wrong_tensor_name_rejected(self):
        self.manifest["tensors"][0]["name"]="different.tensor"
        self.manfile.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(PlanError,"identity mismatch"):collect(self.manfile)

    def test_directory_escape_rejected(self):
        for bad in ["/tmp/outside.json","../outside.json"]:
            self.manifest["tensors"][0]["summary_path"]=bad
            self.manfile.write_text(json.dumps(self.manifest))
            with self.assertRaisesRegex(PlanError,"relative|inside"):collect(self.manfile)

    def test_gpu_n8_rejected(self):
        self.summary["results"].append({"n":8,"effective_bpw":8.5,"rel_l2":.01})
        self.save()
        with self.assertRaisesRegex(PlanError,"does not have a declared"):collect(self.manfile)


if __name__=='__main__':
    unittest.main(verbosity=2)
