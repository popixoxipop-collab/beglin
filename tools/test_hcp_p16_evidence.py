#!/usr/bin/env python3
"""P16 real native full-logit and pinned full-BF16-teacher adversarial audit tests."""
from __future__ import annotations
import json,pathlib,shutil,tempfile,unittest
import numpy as np
from hcp_p16_independent_compare import audit,sha
REPO=pathlib.Path(__file__).resolve().parents[1]
NATIVE=REPO.parent/"private_native_full_logits"
TEACHER=REPO.parent.parent/"hcp-p15-streamed-teacher-20261015/private_full_teacher_evidence"
INDEX_NAMES=("prompt0_bf16","prompt0_fp32","prompt1_bf16","prompt1_fp32")

class P16EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (NATIVE/"P16_NATIVE_8_CAPTURE.json").is_file() or not all((TEACHER/(x+".logits.npy")).is_file() for x in INDEX_NAMES):
            raise RuntimeError("P16 originals absent; refuse synthetic empty-source fixture")
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=pathlib.Path(self.tmp.name)
        self.native=self.root/"native";self.teacher=self.root/"teachers";self.repo=self.root/"repo"
        self.native.mkdir();self.teacher.mkdir();(self.repo/"experiments/hcp_p16").mkdir(parents=True)
        shutil.copy2(REPO/"experiments/hcp_p16/FROZEN_TEACHER_VS_NATIVE_V1.json",self.repo/"experiments/hcp_p16/FROZEN_TEACHER_VS_NATIVE_V1.json")
        for f in NATIVE.iterdir():
            if f.is_file() and (f.name.endswith(".f32") or f.name in ("P16_NATIVE_8_CAPTURE.json","P16_ORIGINAL_CAPTURE_COPIES_MANIFEST.json","P16_ORIGINAL_WORKER.log")):
                shutil.copy2(f,self.native/f.name)
        for name in INDEX_NAMES:
            for ext in (".logits.npy",".json"):
                shutil.copy2(TEACHER/(name+ext),self.teacher/(name+ext))
    def tearDown(self):self.tmp.cleanup()
    def runaudit(self):return audit(self.repo,self.native,self.teacher)
    def reject(self):
        with self.assertRaises((ValueError,RuntimeError,KeyError,TypeError,AssertionError,IndexError)):
            self.runaudit()
    def mutate(self,p,fn):
        obj=json.loads(p.read_text());fn(obj)
        p.write_text(json.dumps(obj,sort_keys=True)+"\n")
    def runfile(self):return self.native/"P16_NATIVE_8_CAPTURE.json"
    def firstcapture(self):
        j=json.loads(self.runfile().read_text())
        return self.native/j["capture_rows"][0]["native_full_logits_binary_filename"]
    def test_01_original_real_native_and_original_teachers_pass(self):
        r=self.runaudit()
        self.assertTrue(r["all_actual_C_prefill_token_SHA_and_position_verified"])
        self.assertEqual(r["native_8_requests"],8)
        self.assertEqual(len(r["unique_native_policy_prompt_metrics"]),4)
        self.assertFalse(r["quality_noninferiority_certified"])
    def test_02_wrong_actual_native_logits_bytes(self):
        p=self.firstcapture();a=np.fromfile(p,dtype="<f4");a[20]+=1.0;a.tofile(p);self.reject()
    def test_03_nonfinite_native_logits(self):
        p=self.firstcapture();a=np.fromfile(p,dtype="<f4");a[20]=np.nan;a.tofile(p);self.reject()
    def test_04_missing_native_logits_array(self):
        self.firstcapture().unlink();self.reject()
    def test_05_wrong_generated_token(self):
        self.mutate(self.runfile(),lambda z:z["capture_rows"][0].update(native_real_first_generated_token_id=8));self.reject()
    def test_06_wrong_frozen_token_SHA(self):
        self.mutate(self.runfile(),lambda z:z["capture_rows"][0].update(prompt_tokens_sha256="a"*64));self.reject()
    def test_07_wrong_policy_runtime_SHA(self):
        self.mutate(self.runfile(),lambda z:z["capture_rows"][0].update(actual_native_policy_sha256="b"*64));self.reject()
    def test_08_wrong_worker_original_release_SHA(self):
        self.mutate(self.runfile(),lambda z:z["original_stage_after"].update(binary_sha256="c"*64));self.reject()
    def test_09_wrong_request_cell_order(self):
        self.mutate(self.runfile(),lambda z:z["capture_rows"][0].update(request_index=7));self.reject()
    def test_10_wrong_original_binary_SHA(self):
        self.mutate(self.runfile(),lambda z:z.update(instrumented_native_binary_sha256="d"*64));self.reject()
    def test_11_false_native_quality_noninferiority_certification(self):
        self.mutate(self.runfile(),lambda z:z.update(quality_noninferiority_certified=True));self.reject()
    def test_12_missing_native_repeated_source_row(self):
        self.mutate(self.runfile(),lambda z:z["capture_rows"].pop());self.reject()
    def test_13_C_native_trace_input_SHA_mutated(self):
        p=self.native/"P16_ORIGINAL_WORKER.log"
        s=p.read_text(errors="replace")
        self.assertIn("P16_NATIVE_FULL_LOGITS",s)
        p.write_text(s.replace("inputsha=f3b3029", "inputsha=aaaaaaa",1));self.reject()
    def test_14_C_actual_prefill_position_forged(self):
        p=self.native/"P16_ORIGINAL_WORKER.log"
        s=p.read_text(errors="replace")
        p.write_text(s.replace("plen=7 pos=6","plen=7 pos=5",1));self.reject()
    def test_15_wrong_teacher_BF16_original_array(self):
        p=self.teacher/"prompt0_bf16.logits.npy";a=np.load(p,allow_pickle=False)
        a[0,3]+=1
        with p.open("wb") as f:np.save(f,a,allow_pickle=False)
        self.reject()
    def test_16_wrong_teacher_FP32_original_array(self):
        p=self.teacher/"prompt1_fp32.logits.npy";a=np.load(p,allow_pickle=False)
        a[0,5]+=1
        with p.open("wb") as f:np.save(f,a,allow_pickle=False)
        self.reject()
    def test_17_original_teacher_report_corrupted(self):
        self.mutate(self.teacher/"prompt1_bf16.json",lambda z:z.update(token_sha256="e"*64));self.reject()
    def test_18_frozen_teacher_reference_corrupted(self):
        p=self.repo/"experiments/hcp_p16/FROZEN_TEACHER_VS_NATIVE_V1.json"
        self.mutate(p,lambda z:z["full_teacher_refs"][0].update(token_sha256="f"*64));self.reject()
    def test_19_original_C_record_Capture_SHA_forged(self):
        self.mutate(self.runfile(),lambda z:z["capture_rows"][0].update(native_full_logits_sha256="0"*64));self.reject()
    def test_20_raw_copy_manifest_mismatch(self):
        p=self.native/"P16_ORIGINAL_CAPTURE_COPIES_MANIFEST.json"
        self.mutate(p,lambda z:z.update(original_signed_run_report_sha256="1"*64));self.reject()
    def test_21_duplicate_source_log_capture_id(self):
        p=self.native/"P16_ORIGINAL_WORKER.log"
        s=p.read_text(errors="replace")
        t=next(x for x in s.splitlines() if x.startswith("P16_NATIVE_FULL_LOGITS"))
        p.write_text(s+"\n"+t+"\n");self.reject()
    def test_22_run_observer_claims_production_mutation(self):
        self.mutate(self.runfile(),lambda z:z.update(production_touched=True));self.reject()
    def test_23_wrong_first_prompt_policy_cell(self):
        self.mutate(self.runfile(),lambda z:z["capture_rows"][0].update(policy="B"));self.reject()
    def test_24_wrong_checkpoint_index_under_preregister(self):
        p=self.repo/"experiments/hcp_p16/FROZEN_TEACHER_VS_NATIVE_V1.json"
        self.mutate(p,lambda z:z.update(original_bf16_teacher_full_model_index_SHA="2"*64));self.reject()

if __name__=="__main__":unittest.main(verbosity=2)
