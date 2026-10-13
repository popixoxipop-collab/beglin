#!/usr/bin/env python3
"""P2 real-contract and split-leakage tests using SYNTHETIC source-shaped fixtures."""
from __future__ import annotations
import copy,json,pathlib,tempfile,unittest
import numpy as np
from hcp_p2_multiscale import evaluate,load,P2Error,SCHEMA,STRIDE,ARMS

def fixture():
    rng=np.random.default_rng(20261014)
    data={"schema":"beglin-hcp-p2-causal-real-trace-v1",
        "source":"REAL_DEEPSEEK_QUANTIZED_AF_H2_AND_TOP6_ROUTING",
        "status":"PROXY_ONLY_NO_CERTIFIED_RUNTIME_PERMISSION",
        "prompt_counts":{"train":72,"validation":24,"holdout":24},
        "target_is_genuine_topK_boundary_margin":False,
        "input_uses_same_token_post_ffn_logits":False,
        "policy_change_allowed":False,"production_touched":False,"prompts":[]}
    for p in range(120):
        split="train" if p<72 else "validation" if p<96 else "holdout"
        latent=rng.normal(0,0.2)
        rows=[];slow=np.zeros(6)
        for t in range(9):
            latent=.93*latent+.07*rng.normal()
            f=np.array([latent+.05*rng.normal(),.2+rng.normal()*.05,.6+.10*latent,
                .2+.1*rng.normal(),.1+.1*rng.normal(),.3+rng.normal()*.04])
            slow=f if t==0 else .75*slow+.25*f
            rows.append({"position":t,"fast":f.tolist(),"slow":slow.tolist(),"proxy":.5+.15*latent})
        for t in range(8):
            tail=[r["proxy"] for r in rows[t+1:min(9,t+5)]]
            rows[t]["target_next4"]=float(np.mean(tail))
            rows[t]["target_next1"]=float(rows[t+1]["proxy"])
        data["prompts"].append({"id":f"p{p:03d}","split":split,"rows":rows[:-1]})
    return data

def run(data):
    with tempfile.TemporaryDirectory() as td:
        p=pathlib.Path(td)
        (p/"trace.json").write_text(json.dumps(data))
        result=evaluate(p/"trace.json",p/"result.json")
        assert json.loads((p/"result.json").read_text())["schema"]==SCHEMA
        return result

class HCP2Tests(unittest.TestCase):
    def test_all_four_arms_heldout_readonly(self):
        z=run(fixture())
        self.assertEqual(set(z["final_heldout"]),set(ARMS))
        self.assertEqual(z["holdout_prompt_count"],24)
        self.assertEqual(z["holdout_token_positions"],24*8)
        self.assertEqual(z["slow_stride"],STRIDE)
        self.assertEqual(z["status"],"REAL_SURROGATE_EVALUATED_NATIVE_POLICY_BLOCKED")
        self.assertFalse(z["native_policy_rebinding_allowed"])
        self.assertFalse(z["production_touched"])
        self.assertTrue(all(np.isfinite(t["prompt_macro_rmse"]) for t in z["final_heldout"].values()))
    def test_holdout_target_not_used_for_fit_or_validation_choice(self):
        d=fixture()
        a=run(d)
        other=copy.deepcopy(d)
        for doc in other["prompts"]:
            if doc["split"]=="holdout":
                for row in doc["rows"]:
                    row["target_next4"]=0.9
                    row["target_next1"]=0.9
        b=run(other)
        self.assertEqual(a["validation_hyperparameter_selection"],b["validation_hyperparameter_selection"])
        self.assertNotEqual(a["final_heldout"]["flat"]["rmse"],b["final_heldout"]["flat"]["rmse"])
    def test_false_routing_boundary_claim_blocks(self):
        d=fixture()
        d["target_is_genuine_topK_boundary_margin"]=True
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"input.json";p.write_text(json.dumps(d))
            with self.assertRaisesRegex(P2Error,"false certification"):
                load(p)
    def test_injected_p7_not_accepted_as_real_prompt_data(self):
        d=fixture()
        d["source"]="P7_OPERATOR_INJECTED_LOW_MARGIN"
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"input.json";p.write_text(json.dumps(d))
            with self.assertRaisesRegex(P2Error,"not real DeepSeek"):
                load(p)
    def test_duplicate_prompt_and_noncausal_position_fail(self):
        d=fixture()
        d["prompts"][1]["id"]=d["prompts"][0]["id"]
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"input.json";p.write_text(json.dumps(d))
            with self.assertRaisesRegex(P2Error,"duplicate"):
                load(p)
        d=fixture()
        d["prompts"][5]["rows"][4]["position"]=7
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"input.json";p.write_text(json.dumps(d))
            with self.assertRaisesRegex(P2Error,"timestamp"):
                load(p)

if __name__=="__main__":unittest.main()
