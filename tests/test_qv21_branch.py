#!/usr/bin/env python3
"""Offline Qv21 policy/receipt tests. Synthetic fixtures, no model or heldout access."""
import hashlib
import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import qv21_policy as policy
import qv21_calibration_gate as gate
import qv21_fit as fitter

SPEC = policy.read_spec()

def bjson(o):
    return (json.dumps(o, sort_keys=True, indent=2) + "\n").encode()

def hashfile(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rows():
    return [{"prefix_len": n, "prefix_ids": [int((j * 8191 + i * 7) % 151936) for j in range(n)],
             "correct": "A"} for i in range(96) for n in ([45] if i % 2 == 0 else [115])]

class PolicyContracts(unittest.TestCase):
    def test_spec_sha(self):
        self.assertEqual(hashfile(ROOT / "qv21_spec.json"), policy.SPEC_SHA256)
        self.assertEqual(SPEC["qv20_eval24_arc_and_ob"], "DO_NOT_READ")

    def test_features_are_finite(self):
        for n in (18, 85, 208):
            v = policy.feature_vector(list(range(n)))
            self.assertEqual(v.shape, (13,))
            self.assertTrue(np.isfinite(v).all())

    def test_invalid_prefix_fails(self):
        for bad in ([], [4] * 17, [4] * 209, [3] * 20 + [True], [2] * 20 + [151936]):
            with self.assertRaises(ValueError): policy.feature_vector(bad)

    def test_repeated_prefix_features(self):
        x = policy.feature_vector([7] * 32)
        self.assertEqual(x[1], 1/32)
        self.assertEqual(x[4], 1)
        self.assertEqual(x[8], 0)
        self.assertEqual(x[11], 1)
        self.assertEqual(x[12], 1)

    def test_gold_labels_cannot_influence_features(self):
        r = {"prefix_ids": list(range(64)), "correct": "A", "leak": "foo"}
        before = policy.features([r])
        r.update(correct="D", leak="bar")
        self.assertTrue(np.array_equal(before, policy.features([r])))

    def test_folds_follow_length_without_labels(self):
        a = policy.fold_assignments(rows())
        for fold in range(5):
            indices = np.where(a == fold)[0]
            self.assertEqual(sum(i % 2 == 0 for i in indices), sum(i % 2 == 1 for i in indices))
        with self.assertRaises(ValueError):
            policy.fold_assignments([{"prefix_len": 120}] * 96)

    def test_risk_constant_prior(self):
        for delta in (-1., 1.):
            m = policy.fit_models(np.ones((60,13)), np.full(60,delta), SPEC)
            self.assertEqual(m["risk_model"]["kind"], "constant")
            self.assertGreater(m["risk_model"]["probability"], 0)

    def test_risk_two_class(self):
        x = np.random.default_rng(20261025).normal(size=(80,13))
        d = np.where(x[:,0]>0, 1.5, -0.8)
        m = policy.fit_models(x,d,SPEC)
        self.assertEqual(m["risk_model"]["kind"], "logistic")
        a,b = policy.predict_risk(x,m)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        self.assertTrue(((b>0)&(b<1)).all())

    def test_strict_thresholds(self):
        m = {"means":[0]*13,"scales":[1]*13,
             "ridge_coefficients":[-0.25]+[0]*13,
             "risk_model":{"kind":"constant","probability":0.12}}
        self.assertTrue(policy.choose(np.zeros((1,13)), m, SPEC)[0])
        m["risk_model"]["probability"] = 0.15
        self.assertFalse(policy.choose(np.zeros((1,13)), m, SPEC)[0])
        m["risk_model"]["probability"] = 0.14
        m["ridge_coefficients"][0] = -0.2
        self.assertFalse(policy.choose(np.zeros((1,13)), m, SPEC)[0])

    def test_successful_synthetic_gate(self):
        a = np.ones(96)
        b = np.r_[np.zeros(48),np.full(48,2.)]
        chosen = np.r_[np.ones(48, bool),np.zeros(48, bool)]
        res = policy.evaluate_gate(a,b,chosen,SPEC)
        self.assertTrue(res["accepted"])
        self.assertGreater(res["bootstrap_95_lower_bound"],0)

    def test_no_selection_fallback(self):
        a = np.ones(96); b = np.full(96, .7)
        x = policy.evaluate_gate(a,b,np.zeros(96,bool),SPEC)
        self.assertFalse(x["accepted"])
        self.assertEqual(x["best_static_index"], 1)

    def test_tail_over_limit_fails(self):
        a = np.ones(96); b = np.r_[np.zeros(48),np.full(48,2.)]
        c = np.r_[np.ones(55,bool),np.zeros(41,bool)]
        self.assertFalse(policy.evaluate_gate(a,b,c,SPEC)["accepted"])

    def test_nonfinite_loss_fails(self):
        a = np.ones(96); b = np.full(96,.5); b[8] = float('nan')
        with self.assertRaises(ValueError): policy.evaluate_gate(a,b,np.ones(96,bool),SPEC)

    def test_crossfit_deterministic(self):
        r=rows(); a=np.full(96,1.3); b=np.r_[np.full(48,.9),np.full(48,2.0)]
        x=policy.crossfit(r,a,b,SPEC)
        y=policy.crossfit(r,a,b,SPEC)
        self.assertEqual(x[0],y[0]); self.assertEqual(x[1],y[1]); self.assertTrue(np.array_equal(x[2],y[2]))

class ReceiptContracts(unittest.TestCase):
    def fixtures(self, folder):
        base=pathlib.Path(folder)
        exe=base/'fake-exe';exe.write_bytes(b'synthetic-not-native')
        rs=rows()
        private=base/'cal_arc_train96/frozen_inputs_private.json'
        private.parent.mkdir(parents=True);private.write_bytes(bjson({'selected_rows':rs}))
        public={'state':'FROZEN_BEFORE_ANY_QV21_NATIVE_SCORING',
                'study_spec_SHA256':policy.SPEC_SHA256,
                'frozen_private_input_SHA256':{'cal_arc_train96':hashfile(private)}}
        pub=base/'Qv21_PREEXEC_PUBLIC.json';pub.write_bytes(bjson(public))
        masks=base/'masks';masks.mkdir()
        for arm in ('static_source','static_random'):
            (masks/(arm+'.txt')).write_text('synthetic-mask-'+arm)
            dest=base/'native_original_C_static/cal_arc_train96'/arm;dest.mkdir(parents=True)
            losses=[1.0+i/100 for i in range(96)]
            lines=['projection_role_tensors: f32=0 n8=28 q4=168']
            for i,row in enumerate(rs):
                lines.append(f'[qv5-choice] window {i} NLL {losses[i]} expected 0 predicted 0 prefix {row["prefix_len"]}')
            total=sum(losses);mean=total/96
            lines.append(f'qv5_choice NLL {total} items 96 correct 96 mean {mean} ppl {math.exp(mean)}')
            log=dest/'native_original_C_Qwen28.log';log.write_text('\n'.join(lines)+'\n')
            obj={'schema':gate.SCHEMA,
                 'state':'ACTUAL_ORIGINAL_QWEN28_C_STATIC_'+arm.upper()+'_PASS',
                 'split':'cal_arc_train96','arm':arm,'n':96,
                 'original_Qwen28_C_executable_sha256':SPEC['original_native_c_sha256'],
                 'source_frozen_private_SHA256':hashfile(private),
                 'source_public_preexec_SHA256':hashfile(pub),
                 'mask_sha256':SPEC['static_masks'][arm], 'Q4_THREADS':1,
                 'real_native_Q4_N8_weights_active':True,
                 'real_native_original_C_log_SHA256':hashfile(log),
                 'original_full_vocab_gold_nexttoken_NLL':losses,
                 'original_choice_correct_each':[1]*96,
                 'mean_NLL':mean,'correct':96}
            priv=dest/'PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json';priv.write_bytes(bjson(obj))
            pubobj={k:v for k,v in obj.items() if k not in ('original_full_vocab_gold_nexttoken_NLL','original_choice_correct_each')}
            pubobj['private_scores_receipt_SHA256']=hashfile(priv)
            (dest/'PUBLIC_AGGREGATE.json').write_bytes(bjson(pubobj))
        return base,exe

    def fakehash(self, exe):
        real=hashfile
        def read(p):
            p=pathlib.Path(p)
            if p==exe:return SPEC['original_native_c_sha256']
            if p.parent.name=='masks':return SPEC['static_masks'][p.stem]
            return real(p)
        return read

    def test_synthetic_native_receipt_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root,exe=self.fixtures(td)
            with mock.patch.object(gate,'ROOT',root),mock.patch.object(gate,'EXE',exe),mock.patch.object(gate,'filehash',side_effect=self.fakehash(exe)):
                out=gate.audit()
            self.assertEqual(out['status'],'BOTH_QV21_ORIGINAL_C_CALIBRATION_ARMS_VERIFIED')
            self.assertEqual(out['aggregates']['static_source']['n'],96)
            self.assertFalse(out['eval_scores_accessed'])

    def test_tampered_native_receipt_denied(self):
        with tempfile.TemporaryDirectory() as td:
            root,exe=self.fixtures(td)
            bad=root/'native_original_C_static/cal_arc_train96/static_random/PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
            obj=json.loads(bad.read_text());obj['original_full_vocab_gold_nexttoken_NLL'][0]=44.
            bad.write_bytes(bjson(obj))
            with mock.patch.object(gate,'ROOT',root),mock.patch.object(gate,'EXE',exe),mock.patch.object(gate,'filehash',side_effect=self.fakehash(exe)):
                with self.assertRaises(RuntimeError):gate.audit()

    def test_missing_native_arm_denied(self):
        with tempfile.TemporaryDirectory() as td:
            root,exe=self.fixtures(td)
            missing=root/'native_original_C_static/cal_arc_train96/static_random/PRIVATE_ACTUAL_NATIVE_CASE_SCORES.json'
            missing.unlink()
            with mock.patch.object(gate,'ROOT',root),mock.patch.object(gate,'EXE',exe),mock.patch.object(gate,'filehash',side_effect=self.fakehash(exe)):
                with self.assertRaises(FileNotFoundError):gate.audit()

    def test_fitter_does_not_create_policy_without_audit(self):
        with tempfile.TemporaryDirectory() as td:
            base=pathlib.Path(td)
            with mock.patch.object(fitter,'ROOT',base),mock.patch.object(fitter,'audit',side_effect=RuntimeError('not verified')):
                with self.assertRaisesRegex(RuntimeError,'not verified'):fitter.fit_once()
            self.assertFalse((base/'policy').exists())

if __name__=='__main__':unittest.main(verbosity=2)
