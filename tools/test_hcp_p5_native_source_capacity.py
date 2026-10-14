#!/usr/bin/env python3
"""P5 test original C source SHA, capacity128 guard preservation, and isolated binary path rules."""
import hashlib,pathlib,tempfile,unittest
from hcp_p5_scratch_capacity128 import patch,apply,ALLOWED_SOURCE_HASH
from hcp_p5_scratch_rebind_ab import check_scratch_binary,validate_prior_stage
from hcp_p5_scratch_pair_capacity import patch_both,MLX_GIT_BLOB_SHA

SOURCE=pathlib.Path(__file__).resolve().parents[1]/"qwen_infer.c"

class ScratchP5NativeTests(unittest.TestCase):
    def test_native_approved_c_source_with_atomic_multi_target_txn(self):
        text=SOURCE.read_text()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),ALLOWED_SOURCE_HASH)
        self.assertIn("REBIND_SET_APPLIED",text)
        self.assertIn("#define MOE_CBATCH_MAXPOS 32",text)
        self.assertIn("#define MOE_MAXPOS 160",text)
    def test_capacity_patch_changes_exactly_one_macro(self):
        source=SOURCE.read_text()
        updated=patch(source)
        self.assertEqual(updated,source.replace("#define MOE_CBATCH_MAXPOS 32","#define MOE_CBATCH_MAXPOS 128"))
        self.assertNotIn("#define MOE_CBATCH_MAXPOS 32",updated)
        self.assertIn("moe_cb4b_admit_guard",updated)
        self.assertIn("REBIND_SET_APPLIED",updated)
        self.assertIn("#define MOE_MAXPOS 160",updated)
    def test_capacity_patch_rejects_modification_or_double_patch(self):
        source=SOURCE.read_text()
        with self.assertRaisesRegex(ValueError,"SHA"):
            patch(source+" ")
        with self.assertRaisesRegex(ValueError,"SHA"):
            patch(patch(source))
    def test_patched_source_only_under_explicit_scratch(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            (root/"qwen_infer.c").write_bytes(SOURCE.read_bytes())
            (root/"CMakeLists.txt").write_text("add_library(qwen_infer_obj OBJECT\n  qwen_infer.c\n)\n")
            receipt=apply(root)
            self.assertEqual(receipt["native_cbatch_maxpos_to"],128)
            self.assertFalse(receipt["production_touched"])
            self.assertEqual((root/"CMakeLists.txt").read_text().count("qng64_dense.c"),1)
            self.assertEqual((root/"qwen_infer.c").read_text().count("#define MOE_CBATCH_MAXPOS 128"),1)
    def test_dual_native_and_mlx_gpu_128_capacity_patch(self):
        backend=pathlib.Path(__file__).resolve().parents[1]/"mlx_moe.cpp"
        ml=backend.read_bytes()
        import hashlib
        gitblob=hashlib.sha1(b"blob "+str(len(ml)).encode()+bytes([0])+ml).hexdigest()
        self.assertEqual(gitblob,MLX_GIT_BLOB_SHA)
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            (root/"qwen_infer.c").write_bytes(SOURCE.read_bytes())
            (root/"mlx_moe.cpp").write_bytes(ml)
            (root/"CMakeLists.txt").write_text("add_library(qwen_infer_obj OBJECT\\n  qwen_infer.c\\n)\\n")
            result=patch_both(root)
            self.assertEqual(result["gpu_mla_KV_capacity"],128)
            self.assertEqual(result["native_c_capacity"],128)
            self.assertFalse(result["production_touched"])
            self.assertIn("#define MLA_L0_MAXPOS 128",(root/"mlx_moe.cpp").read_text())
            self.assertIn("#define GQA_L0_MAXPOS 32",(root/"mlx_moe.cpp").read_text())
            self.assertIn("#define MOE_CBATCH_MAXPOS 128",(root/"qwen_infer.c").read_text())

    def test_binary_path_and_old_stage_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);(root/"build").mkdir()
            binary=root/"build/qwen_infer_gpu";binary.write_bytes(b"SCRATCH")
            sh=check_scratch_binary(binary,run_root=root)
            self.assertEqual(sh,hashlib.sha256(b"SCRATCH").hexdigest())
            with self.assertRaisesRegex(ValueError,"stage binary identity"):
                validate_prior_stage(stage=binary,expected_sha="0"*64)
            self.assertEqual(validate_prior_stage(stage=binary,expected_sha=sh),sh)
            bad=root/"outside";bad.write_bytes(b"SCRATCH")
            with self.assertRaisesRegex(ValueError,"exact scratch"):
                check_scratch_binary(bad,run_root=root)

if __name__=="__main__":unittest.main()
