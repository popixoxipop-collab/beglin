#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS=Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0,str(TOOLS))

from qt_selective_training import (
    apply_selective_sgd,
    build_training_sensitivity_event,
    parse_group_target,
    requantize_changed_groups,
)

class SelectiveTrainingTests(unittest.TestCase):
    def test_masked_sgd_updates_only_selected_group(self):
        out_dim,in_dim=2,128
        master=[0.125*((i%7)-3) for i in range(out_dim*in_dim)]
        grad=[0.001*((i%5)+1) for i in range(out_dim*in_dim)]
        cells=[]
        for row in range(out_dim):
            for g in range(in_dim//64):
                target=f"m/L0/q_proj/row={row}/group64={g}"
                trainable=(row==0 and g==1)
                cells.append({
                    "target_key":target,"trainable":trainable,
                    "lr_scale":0.5 if trainable else 0.0,
                    "weight_decay_scale":1.0 if trainable else 0.0,
                    "master_precision":"BF16","confidence":1.0,
                })
        policy={
            "schema":"beglin-selective-training-policy-v1",
            "policy_id":"t-test","checkpoint_identity_sha256":"a"*64,
            "skeleton_sha256":"b"*64,"heatmap_sha256":"c"*64,
            "weight_epoch":0,"generation":0,"approved_by_beval":False,
            "cells":cells,
        }
        after,stats=apply_selective_sgd(master,grad,out_dim=out_dim,in_dim=in_dim,policy=policy,base_lr=2.0)
        self.assertEqual(stats.changed_group_count,1)
        self.assertEqual(stats.frozen_leakage_max_abs,0.0)
        self.assertGreater(stats.selected_update_max_abs,0.0)
        for row in range(out_dim):
            for g in range(in_dim//64):
                start=row*in_dim+g*64
                if row==0 and g==1:
                    self.assertTrue(any(float(after[start+p])!=master[start+p] for p in range(64)))
                else:
                    self.assertTrue(all(float(after[start+p])==master[start+p] for p in range(64)))

    def test_requantization_touches_only_changed_targets(self):
        out_dim,in_dim=2,128
        master=[0.002*((i%13)-6) for i in range(out_dim*in_dim)]
        changed=["m/L0/q_proj/row=0/group64=1","m/L0/q_proj/row=1/group64=0"]
        a=requantize_changed_groups(master,out_dim=out_dim,in_dim=in_dim,changed_targets=changed,
                                    n_by_target={changed[0]:4,changed[1]:6})
        b=requantize_changed_groups(master,out_dim=out_dim,in_dim=in_dim,changed_targets=reversed(changed),
                                    n_by_target={changed[0]:4,changed[1]:6})
        self.assertEqual(a,b)
        self.assertEqual(a["changed_group_count"],2)
        self.assertEqual([g["n"] for g in a["groups"]],[4,6])

    def test_training_sensitivity_event_uses_gradient_times_quant_error(self):
        target="m/L0/q_proj/row=0/group64=0"
        ident={
            "schema":"beglin-observation-identity-v2",
            "checkpoint_identity_sha256":"a"*64,"skeleton_sha256":"b"*64,
            "capability_bundle_sha256":"c"*64,"target_key":target,"model_id":"m",
            "layer":0,"tensor_role":"q_proj","expert_id":None,"row":0,"column":None,
            "qgroup_index":0,"group_size":64,"element_index":None,"backend":"cpu-train",
            "source_dtype":"BF16","runtime_dtype":"qNg64","policy_epoch":0,"weight_epoch":0,
            "observation_window_id":"w0","observation_seq":0,"source_commit":"1234567",
            "binary_sha256":"d"*64,
        }
        e=build_training_sensitivity_event(
            identity=ident,group_weights=[0.001*(i-32) for i in range(64)],
            group_gradients=[0.01*(1+(i%3)) for i in range(64)],n=5,
            optimizer_step=0,learning_rate=1e-3,loss_before=1.0)
        self.assertEqual(e["schema"],"beglin-training-sensitivity-v1")
        self.assertGreater(e["gradient_times_quant_error"],0.0)
        self.assertGreater(e["fisher_diag_ema"],0.0)
        self.assertTrue(e["finite"])

    def test_target_parser_rejects_non_group_identity(self):
        self.assertEqual(parse_group_target("m/L0/q_proj/row=9/group64=3"),(9,3))
        with self.assertRaises(ValueError):
            parse_group_target("m/L0/q_proj")

if __name__=="__main__":
    unittest.main(verbosity=2)
