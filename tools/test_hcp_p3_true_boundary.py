#!/usr/bin/env python3
"""P3 true-rank7 extraction, causality and no-policy-authority unit contracts."""
import copy,json,pathlib,tempfile,unittest,random
import numpy as np
from hcp_p3_router_instrument import patch
from hcp_p3_boundary_features import read_boundaries,from_one_prompt,ROUTED_LAYERS
from hcp_p3_true_boundary_eval import load,evaluate,auc,average_precision

def tiny_c_source():
    return '''
static void moe_forward_token(const uint8_t *af) {
    moe_attention(af, t, l, pos, h, x);
        float *w_postln = (float *)(g_moe_f32_blob + t->post_attn_ln->off);
        moe_rmsnorm(x, w_postln, h2, MOE_HIDDEN);
        float *mlp_out = g_mft_mlp_out;
            int *top_idx = g_mft_top_idx;
            moe_top_k_select(router_scores, MOE_N_EXPERTS, MOE_TOP_K, top_idx);
            if (MOE_NORM_TOPK_PROB) moe_topk_renorm(router_scores, top_idx, MOE_TOP_K);
}
'''

def synthetic_dataset():
    rng=np.random.default_rng(42)
    docs=[]
    for k in range(120):
        split="train" if k<72 else ("validation" if k<96 else "holdout")
        z=rng.uniform(.1,.4)
        rows=[]
        for t in range(8):
            z=.65*z+.35*rng.uniform(.05,.75)
            f=(np.array([.35,.20,.5,.08,.31,.10,z,.23])+
               rng.normal(scale=.03,size=8))
            f[6]=z
            if t==0:slow=f.copy()
            else:slow=.75*slow+.25*f
            rows.append({"fast":f.tolist(),"slow":slow.tolist(),"margin":z})
        v=[]
        for t in range(7):
            next4=[rows[i]["margin"] for i in range(t+1,min(t+5,8))]
            v.append({"position":t,"fast":rows[t]["fast"],"slow":rows[t]["slow"],
                 "observed_margin":rows[t]["margin"],
                 "target_next1":next4[0],"target_next4":min(next4)})
        docs.append({"id":f"p{k:03d}","split":split,"rows":v})
    return {"schema":"beglin-hcp-p3-true-router-boundary-features-v1",
        "status":"ACTUAL_TOP6_TOP7_RISK_PROXY_ONLY",
        "prompt_counts":{"train":72,"validation":24,"holdout":24},
        "source":"SYNTHETIC_UNIT_ONLY","true_rank7_measured":True,
        "target_is_certified_precision_risk":False,
        "precision_rebinding_allowed":False,"production_touched":False,
        "prompts":docs}

def run(raw):
    with tempfile.TemporaryDirectory() as td:
        root=pathlib.Path(td)
        (root/"source.json").write_text(json.dumps(raw))
        r=evaluate(root/"source.json",root/"result.json")
        assert json.loads((root/"result.json").read_text())["schema"]==r["schema"]
        return r

class RankBoundaryTest(unittest.TestCase):
    def test_patch_is_exactly_before_topk_renorm_and_not_in_sibling(self):
        z=patch(tiny_c_source())
        self.assertEqual(z.count('getenv("BEGLIN_P3_ROUTER_BOUNDARY")'),1)
        self.assertEqual(z.count("BEGLIN_CAPTURE_PRE_FFN_INPUT"),1)
        self.assertLess(z.index("moe_top_k_select"),z.index("P3 RESEARCH ONLY"))
        self.assertLess(z.index("P3 RESEARCH ONLY"),z.index("moe_topk_renorm"))
    def test_patch_refuses_wrong_source(self):
        with self.assertRaisesRegex(ValueError,"anchor"):
            patch("bad source")
    def test_rank6_7_all_layers_complete(self):
        with tempfile.TemporaryDirectory() as td:
            f=pathlib.Path(td)/"trace.txt"
            lines=[]
            for pos in range(3):
                for layer in ROUTED_LAYERS:
                    lines.append(f"pos {pos} layer {layer} kth 5 runnerup 6 p6 0.1 p7 0.09 relgap 0.1 logratio 0.105360516 entropy 0.7")
            f.write_text("\n".join(lines)+"\n")
            got=read_boundaries(f,3)
            self.assertEqual(len(got),3*26)
            self.assertAlmostEqual(got[0,1]["relative_gap"],.1)
            f.write_text("\n".join(lines[:-1])+"\n")
            with self.assertRaisesRegex(ValueError,"incomplete"):
                read_boundaries(f,3)
    def test_rank7_equal_top6_member_contract_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f=pathlib.Path(td)/"trace.txt"
            lines=[f"pos {p} layer {l} kth 5 runnerup 5 p6 0.1 p7 0.09 relgap 0.1 logratio 0.105360516 entropy 0.7"
                   for p in range(2) for l in ROUTED_LAYERS]
            f.write_text("\n".join(lines)+"\n")
            with self.assertRaisesRegex(ValueError,"incorrect rank"):
                read_boundaries(f,2)
    def test_heldout_leakage_guard(self):
        raw=synthetic_dataset()
        a=run(raw)
        changed=copy.deepcopy(raw)
        for p in changed["prompts"]:
            if p["split"]=="holdout":
                for row in p["rows"]:
                    row["target_next1"]=.001
                    row["target_next4"]=.001
        b=run(changed)
        self.assertEqual(a["per_arm_validation_choice"],b["per_arm_validation_choice"])
        self.assertNotEqual(a["heldout"]["flat"]["token_rmse"],b["heldout"]["flat"]["token_rmse"])
    def test_explicit_policy_gate_and_splits(self):
        x=run(synthetic_dataset())
        self.assertEqual(x["supervised_token_positions"],{"train":504,"validation":168,"holdout":168})
        self.assertFalse(x["native_policy_rebind_allowed"])
        self.assertFalse(x["context_specific_qng64_acceptance_verified"])
        self.assertEqual(x["train_prompts"],72)
        self.assertIn("persistence",x["heldout"])
        self.assertGreaterEqual(x["near_tie_relative_gap_threshold"],0)
    def test_policy_certificate_claim_cannot_enter_training(self):
        d=synthetic_dataset()
        d["target_is_certified_precision_risk"]=True
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/"src.json";path.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,"invalid router source"):
                load(path)
    def test_auc_and_average_precision(self):
        self.assertAlmostEqual(auc([0,0,1,1],[0.1,0.4,0.8,0.7]),1.)
        self.assertAlmostEqual(average_precision([0,0,1,1],[0.1,0.4,0.8,0.7]),1.)
        self.assertIsNone(auc([0,0],[1,2]))

if __name__=="__main__":unittest.main()
