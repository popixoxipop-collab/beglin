"""Independent verification of source-frozen Qv2 native eight-public-window results.

Recompute 151936-way logits/KL from private unmodified BEGLIN output files,
check native PPL/bench receipts + model binary hashes + identical inputs,
assert the new source-activation-aware MILP is NOT better than Q4/random NLL.
Only public synthetic text; never old V6/holdout, original weights or labels.
"""
import hashlib,json,math,pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/oml_acs"))
import q_v2_8window_native_evaluator as ev
NAMES=tuple(ev.CONDITIONS)
class NativeQv2PublicEvaluationQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.prompts,cls.lock=ev.validated_sources()
        cls.receipts={}
        for stage in ("ppl","bench","dump"):
            data=ev.LAB/f"all_{stage}_public_eval_receipt.json"
            assert data.is_file() and data.stat().st_mode&0o777==0o600
            obj=json.loads(data.read_text())
            assert obj["status"]=="PASS_ALL_FIVE_PUBLIC_EVALUATIONS"
            cls.receipts[stage]=obj["models"]
        cls.logits={}
        for name in NAMES:
            f=ev.LAB/name/"results/c_logits_last.f32"
            assert f.is_file() and (f.stat().st_mode&0o777)==0o600 and f.stat().st_size==151936*4
            b=f.read_bytes()
            assert hashlib.sha256(b).hexdigest()==cls.receipts["dump"][name]["full_vocab_logits_sha256"]
            v=np.frombuffer(b,dtype="<f4").astype(np.float64)
            assert np.isfinite(v).all()
            cls.logits[name]=v
    def test_exactly_eight_evaluation_texts_and_frozen_policy(self):
        self.assertEqual(len(self.prompts),8)
        self.assertEqual(len(set(self.prompts)),8)
        self.assertTrue(set(self.prompts).isdisjoint(set(self.cfg["calibration_public_only"])))
        self.assertEqual(self.lock["source_policy_tensor_mask_sha256"],ev.EXPECTED_V2_POLICY_SHA)
    def test_same_input_tokens_under_all_five_models(self):
        windows=tuple(ev.LAB/"original_BF16_widened_F32"/f"eval/ppl_work/w{i}.i32" for i in range(8))
        for name in NAMES:
            with self.subTest(name=name):
                for i,w in enumerate(windows):
                    self.assertEqual((ev.LAB/name/f"eval/ppl_work/w{i}.i32").read_bytes(),w.read_bytes())
                self.assertEqual(self.receipts["ppl"][name]["scored_tokens"],131)
    def test_no_model_or_binary_drift(self):
        for name,(binfile,precision,mask) in ev.CONDITIONS.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256(binfile.read_bytes()).hexdigest(),
                                 self.receipts["ppl"][name]["native_binary_sha256"])
                self.assertEqual(self.receipts["bench"][name]["native_binary_sha256"],
                                 self.receipts["dump"][name]["native_binary_sha256"])
                if mask is not None:
                    self.assertEqual(hashlib.sha256(mask.read_bytes()).hexdigest(),
                                     self.receipts["ppl"][name]["source_only_policy_sha256"])
    def test_quality_failure_ppl_preserved(self):
        ppl={name:self.receipts["ppl"][name]["perplexity"] for name in NAMES}
        self.assertAlmostEqual(ppl["v2_activation_async_MILP"],827.4999,places=3)
        self.assertGreater(ppl["v2_activation_async_MILP"],ppl["all_Q4G64"])
        self.assertGreater(ppl["v2_activation_async_MILP"],ppl["frozen_seed73_random"])
        self.assertLess(ppl["original_BF16_widened_F32"],ppl["frozen_seed73_random"])
    def test_candidate_native_speed_vs_controls(self):
        ms={name:self.receipts["bench"][name]["single_decode_best_of_three_ms"] for name in NAMES}
        self.assertLess(ms["v2_activation_async_MILP"],ms["frozen_seed73_random"])
        self.assertLess(ms["v2_activation_async_MILP"],ms["former_weight_kernel_aware"])
        self.assertGreater(ms["v2_activation_async_MILP"],ms["all_Q4G64"])
    def test_kl_is_distinct_from_nll(self):
        f32=self.logits["original_BF16_widened_F32"]
        p=np.exp(f32-f32.max());p/=p.sum()
        q4=self.logits["all_Q4G64"];v2=self.logits["v2_activation_async_MILP"]
        def kl(y):
            q=np.exp(y-y.max());q/=q.sum()
            return float(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300)))))
        self.assertAlmostEqual(kl(v2),0.4400130767598419,places=8)
        self.assertAlmostEqual(kl(q4),0.674379610663686,places=8)
        self.assertLess(kl(v2),kl(q4))
        # Earlier source-only BF16 gains were not actual teacher-forced NLL gains.
        self.assertGreater(self.receipts["ppl"]["v2_activation_async_MILP"]["nll_nats_per_token"],
                           self.receipts["ppl"]["all_Q4G64"]["nll_nats_per_token"])
    def test_actual_RSS_not_packed_weight_memory(self):
        rss={name:self.receipts["ppl"][name]["max_rss_bytes"] for name in NAMES}
        self.assertGreater(rss["v2_activation_async_MILP"],rss["all_Q4G64"])
        self.assertLess(rss["v2_activation_async_MILP"],rss["original_BF16_widened_F32"])
    def test_no_live_async_serving_claim(self):
        self.assertFalse(self.lock["full_model_parallel_runtime_claim"])
        self.assertTrue(self.lock["status"]=="OPTIMAL_HIGHS_SOURCE_ONLY_BEFORE_TEST")
        self.assertTrue(all(not data.get("production_touched",False) for data in []))
if __name__=="__main__":
    unittest.main(verbosity=2)
