#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from qt_heatmap import build_q_heatmap
from qt_qwen25_group_perturb import build_events

H1="1"*64
H2="2"*64
H3="3"*64

def f32_to_bf16_bytes(values):
    out=bytearray()
    for x in values:
        bits=struct.unpack("<I",struct.pack("<f",float(x)))[0]
        out += struct.pack("<H", bits >> 16)
    return bytes(out)

def write_safetensors(path: Path, values):
    raw=f32_to_bf16_bytes(values)
    header={
        "model.layers.0.self_attn.q_proj.weight":{
            "dtype":"BF16","shape":[1,64],"data_offsets":[0,len(raw)]
        }
    }
    h=json.dumps(header,separators=(",",":")).encode()
    path.write_bytes(struct.pack("<Q",len(h))+h+raw)

class GroupPerturbTests(unittest.TestCase):
    def test_group_cpu_evidence_is_deterministic_and_unapproved(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"tiny.safetensors"
            values=[((i%9)-4)*0.003 for i in range(64)]
            write_safetensors(p,values)
            kwargs=dict(
                checkpoint=p,
                tensor="model.layers.0.self_attn.q_proj.weight",
                row=0,group_index=0,candidates=[4,5,6],windows=2,
                checkpoint_identity_sha256=H1,skeleton_sha256=H2,
                capability_bundle_sha256=H3,source_commit="79d98de2",
            )
            a=build_events(**kwargs)
            b=build_events(**kwargs)
            self.assertEqual(a,b)
            self.assertEqual(len(a),6)
            self.assertTrue(all(e["backend_parity_error"] is None for e in a))
            self.assertTrue(all(e["restore_diff"] is None for e in a))
            q=build_q_heatmap(a)
            cell=q["cells"][0]
            self.assertEqual(cell["target_key"],"qwen2.5/L0/q_proj/row=0/group64=0")
            self.assertEqual(cell["backend_verified_by_n"],{"4":False,"5":False,"6":False})
            self.assertIn(cell["state"],("CANDIDATE","QUARANTINED"))
            if cell["state"]=="CANDIDATE":
                self.assertIsNotNone(cell["recommended_n"])

if __name__=="__main__":
    unittest.main(verbosity=2)
