#!/usr/bin/env python3
"""P4 source-only 112-token safety, exact scratch patch and raw export controls."""
from __future__ import annotations
import hashlib,json,pathlib,tempfile,unittest
from unittest.mock import patch
from hcp_p4_long_prompt_registry import prepare,public_manifest,BANK_FILES
from hcp_p4_long_router_instrument import prepare as instrument
from hcp_p4_native_policies_ab import check_manifest,percentile,SCHEMA
from hcp_p3_real_capture import EXPECTED_INDEX_SHA

class FakeTokenizer:
    def encode(self,prompt,add_special_tokens=True):
        seed=hashlib.sha256(prompt.encode()).digest()
        return [100000]+[int.from_bytes(seed[(i%12):(i%12)+2],"big")%102399 for i in range(130)]

def fake_bank(root):
    root=pathlib.Path(root)
    for i,filename in enumerate(BANK_FILES):
        obj={}
        for j in range(3):
            domain=f"topic_{i}_{j}"
            obj[domain]=[
              f"Explain the scientific methodology and possible experimental weaknesses in scenario {i} {j} {k} with technical details."
              for k in range(10)]
        (root/filename).write_text(json.dumps(obj))

def miniature_source():
    return ('''static void moe_forward_token(const uint8_t *af) {
 moe_attention(af, t, l, pos, h, x);
        float *w_postln = (float *)(g_moe_f32_blob + t->post_attn_ln->off);
        moe_rmsnorm(x, w_postln, h2, MOE_HIDDEN);
        float *mlp_out = g_mft_mlp_out;
            int *top_idx = g_mft_top_idx;
            moe_top_k_select(router_scores, MOE_N_EXPERTS, MOE_TOP_K, top_idx);
            if (MOE_NORM_TOPK_PROB) moe_topk_renorm(router_scores, top_idx, MOE_TOP_K);
}
''')

class LongContractTests(unittest.TestCase):
    def test_unique_domain_disjoint_train_validation_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);fake_bank(root)
            with patch("hcp_p4_long_prompt_registry.AutoTokenizer.from_pretrained",return_value=FakeTokenizer()):
                data=prepare(root,root)
            self.assertEqual(len(data["prompts"]),36)
            self.assertEqual(data["prompt_splits"],{"train":12,"validation":12,"holdout":12})
            self.assertEqual(len({x["token_sha256"] for x in data["prompts"]}),36)
            self.assertTrue(all(len(x["token_ids"])==112 for x in data["prompts"]))
            safe=public_manifest(data)
            self.assertTrue(all("token_ids" not in x for x in safe["prompts"]))
            self.assertEqual(safe["distinct_tokenized_input_count"],36)
    def test_scratch_instrument_patches_exactly_one_frozen_site(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            (root/"qwen_infer.c").write_text(miniature_source())
            (root/"CMakeLists.txt").write_text("add_library(qwen_infer_obj OBJECT\n  qwen_infer.c\n)\n")
            receipt=instrument(root)
            updated=(root/"qwen_infer.c").read_text()
            self.assertEqual(updated.count("pos < MOE_MAXPOS"),1)
            self.assertNotIn("pos < 40",updated)
            self.assertEqual(updated.count("BEGLIN_P3_ROUTER_BOUNDARY"),1)
            self.assertEqual(receipt["token_h2_cap"],160)
            self.assertTrue(receipt["production_touched"] is False)
            self.assertEqual((root/"CMakeLists.txt").read_text().count("qng64_dense.c"),1)
    def test_true_native_reference_match_requires_same_sha_and_count(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);features=root/"derived_features.json"
            refs=root/"references_INTERNAL_ONLY.json"
            rows=[];refs_data=[]
            for i in range(36):
                ids=[100000]+[i+10 for _ in range(95)]
                sha=hashlib.sha256(",".join(map(str,ids)).encode()).hexdigest()
                rows.append({"prompt_id":f"p{i:03d}","token_sha256":sha,"prompt_length":len(ids),
                    "split":("train" if i<12 else "validation" if i<24 else "holdout"),
                    "category":"topic","real_min_rank6_rank7_gap_min":0.001})
                refs_data.append({"prompt_id":f"p{i:03d}","token_sha256":sha,
                    "prompt_token_ids":ids,"reference_last_token":11})
            features.write_text(json.dumps({"schema":"beglin-hcp-p4-long-actual-af-risk-dataset-v1",
                "status":"ACTUAL_DEEPSEEK_LONGISH_CONTEXT_RISK_TRACES_ONLY",
                "record_count":36,"train_val_holdout":{"train":12,"validation":12,"holdout":12},
                "native_precision_changed":False,"production_touched":False,"runs":rows}))
            refs.write_text(json.dumps(refs_data))
            s,_=check_manifest(features,refs)
            self.assertEqual(len(s),36)
            refs_data[0]["token_sha256"]="f"*64
            refs.write_text(json.dumps(refs_data))
            with self.assertRaisesRegex(ValueError,"mismatched"):
                check_manifest(features,refs)
    def test_percentiles_and_no_source_import_side_effect(self):
        self.assertEqual(percentile([1,2,3,4],.5),2.5)
        self.assertEqual(percentile([1,2,3],.95),2.9)
        self.assertEqual(len(EXPECTED_INDEX_SHA),64)
        self.assertTrue(SCHEMA.endswith("-v1"))

if __name__=="__main__":unittest.main()
