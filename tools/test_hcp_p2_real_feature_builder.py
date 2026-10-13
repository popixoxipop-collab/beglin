#!/usr/bin/env python3
"""Synthetic binary fixtures verify actual pre-FFN feature extraction causality."""
from __future__ import annotations
import pathlib,struct,tempfile,unittest
import numpy as np
from hcp_p2_real_feature_builder import extract_prompt,read_route,read_hidden,logits_entropy_file

class RealFeatureBuilderContractTests(unittest.TestCase):
    def make(self,root,n=6,vocab=100000):
        root=pathlib.Path(root)
        with (root/"true_h2.bin").open("wb") as f:
            for i in range(n):
                vec=np.full(2048,0.01+i*.0001,dtype="<f4")
                f.write(struct.pack("<iii",i,1,2048));f.write(vec.tobytes())
        with (root/"moe3a_c_routing.txt").open("w") as f:
            for i in range(n):
                weights=" ".join(f"{e}:{.1+.01*((i+e)%6):.5f}" for e in range(6))
                f.write(f"pos {i} layer 1 experts {weights}\n")
        logits=np.zeros((n,vocab),dtype="<f4")
        for i in range(n):logits[i,(i+12)%vocab]=10.0
        (root/"moe3a_c_logits.bin").write_bytes(logits.tobytes())
        return logits
    def test_causal_next_target_uses_only_future_labels(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d)
            base,_=extract_prompt(pathlib.Path(d),6)
            self.assertEqual(len(base),5)
            self.assertAlmostEqual(base[0]["target_next1"],base[1]["proxy"])
            self.assertAlmostEqual(base[0]["target_next4"],
                                   float(np.mean([r["proxy"] for r in
                                   [*base,{"proxy":base[-1]["target_next1"]}][1:5]])))
            # Changing SAME-position final logits cannot be fed into that token's
            # feature; only PRIOR token final logits may appear.
            p=pathlib.Path(d)/"moe3a_c_logits.bin"
            logits=np.frombuffer(p.read_bytes(),dtype="<f4").copy().reshape(6,100000)
            logits[2]=np.linspace(-9,9,100000,dtype=np.float32)
            p.write_bytes(logits.tobytes())
            shifted,_=extract_prompt(pathlib.Path(d),6)
            np.testing.assert_allclose(base[2]["fast"],shifted[2]["fast"])
            self.assertNotAlmostEqual(base[3]["fast"][-1],shifted[3]["fast"][-1],places=7)
    def test_duplicate_and_missing_route_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d)
            p=pathlib.Path(d)/"moe3a_c_routing.txt"
            p.write_text(p.read_text()+"pos 0 layer 1 experts "+" ".join(f"{i}:0.1" for i in range(6))+"\n")
            with self.assertRaisesRegex(ValueError,"duplicate"):
                read_route(p,6)
    def test_invalid_hidden_site_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d)
            p=pathlib.Path(d)/"true_h2.bin"
            b=bytearray(p.read_bytes());b[4:8]=struct.pack("<i",2)
            p.write_bytes(b)
            with self.assertRaisesRegex(ValueError,"wrong/duplicate"):
                read_hidden(p,6)
    def test_unexpected_vocab_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d,vocab=32)
            with self.assertRaisesRegex(ValueError,"vocabulary"):
                logits_entropy_file(pathlib.Path(d)/"moe3a_c_logits.bin",6)

if __name__=="__main__":unittest.main()
