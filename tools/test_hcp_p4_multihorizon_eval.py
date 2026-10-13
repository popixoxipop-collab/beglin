#!/usr/bin/env python3
"""Synthetic learning fixtures test real-P4 data provenance and holdout protection."""
from __future__ import annotations
import copy,json,pathlib,tempfile,unittest
import numpy as np
from hcp_p4_multihorizon_eval import load,evaluate,ARMS,SCHEMA

def fixture():
    rng=np.random.default_rng(48)
    groups=("astro","phy","chem","bio","math","code",
            "ml","robot","energy","econ","history","language")
    obj={"schema":"beglin-hcp-p4-long-actual-af-risk-dataset-v1",
        "status":"ACTUAL_DEEPSEEK_LONGISH_CONTEXT_RISK_TRACES_ONLY",
        "train_val_holdout":{"train":12,"validation":12,"holdout":12},
        "record_count":36,"production_touched":False,"native_precision_changed":False,
        "full_logits_exported":False,"sequences":[]}
    for i in range(36):
        pid=f"p{i:03d}"
        split=("train","validation","holdout")[i%3]
        risk=np.zeros(110)
        latent=rng.uniform(.1,.9)
        for t in range(110):
            latent=.95*latent+.05*rng.random()
            risk[t]=.003+.012*latent
        ema=np.zeros(8)
        rows=[]
        for t in range(109):
            fast=np.array([risk[t],.2+rng.normal()*.03,.1,.4,.7,.2,risk[t],.3])
            ema=fast.copy() if t==0 else .9375*ema+.0625*fast
            target={h:(float(np.min(risk[t+1:t+1+int(h)])) if t+int(h)<110 else None)
                   for h in ("4","16","32")}
            rows.append({"position":t,"fast":fast.tolist(),"slow4":fast.tolist(),
                 "slow16":ema.tolist(),"observed_min_kth_gap":float(risk[t]),
                 "true_future_gap_min":target})
        obj["sequences"].append({"prompt_id":pid,"split":split,"domain":groups[i//3],"rows":rows})
    return obj

def run(data):
    with tempfile.TemporaryDirectory() as td:
        root=pathlib.Path(td)
        source=root/"source.json";out=root/"result.json"
        source.write_text(json.dumps(data))
        z=evaluate(source,out)
        selfcheck=json.loads(out.read_text())
        assert selfcheck["schema"]==SCHEMA
        return z

class P4ForecastTests(unittest.TestCase):
    def test_112_context_true_future16_predictor_ablation(self):
        z=run(fixture())
        self.assertEqual(set(z["heldout"]),set(ARMS))
        self.assertEqual(z["holdout_prompts"],12)
        self.assertFalse(z["native_precision_policy_change_allowed"])
        self.assertFalse(z["genuine_JEPA_training_performed"])
        self.assertEqual(len(z["predictive_depth_gate"]),2)
    def test_validation_does_not_learn_holdout_target(self):
        src=fixture();before=run(src)
        change=copy.deepcopy(src)
        for d in change["sequences"]:
            if d["split"]=="holdout":
                for row in d["rows"]:
                    if row["true_future_gap_min"]["16"] is not None:
                        row["true_future_gap_min"]["16"]=.8
        after=run(change)
        self.assertEqual(before["validation_hyperparameters"],after["validation_hyperparameters"])
        self.assertNotEqual(before["heldout"]["flat"]["rmse"],after["heldout"]["flat"]["rmse"])
    def test_short_or_untrusted_source_blocks(self):
        obj=fixture();obj["sequences"][0]["rows"]=obj["sequences"][0]["rows"][:50]
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"source.json";p.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,"genuine"):
                load(p)
        obj=fixture();obj["production_touched"]=True
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"source.json";p.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,"forbidden"):
                load(p)
    def test_nonfinite_causal_feature_blocks(self):
        obj=fixture();obj["sequences"][10]["rows"][20]["slow16"][0]=float("nan")
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/"source.json";p.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,"nonfinite"):
                load(p)

if __name__=="__main__":unittest.main()
