#!/usr/bin/env python3
"""P5 controls: fail closed on history mismatch or scratch KV128 source topology."""
import os,pathlib,tempfile,unittest
from unittest.mock import patch
from hcp_p5_scratch_128_patch import patch as patch_c,SCHEMA as PATCH_SCHEMA,CMAKE_START
from hcp_p5_scratch_binary_adapter import verify_scratch_binary
from hcp_p5_native_long128_shadow import run as run_long

MANDATORY=(
 "g_moe_cb_slots_cap*MOE_CBATCH_MAXPOS*MOE_KROW",
 "g_moe_cb_slots_cap*MOE_CBATCH_MAXPOS*MOE_VROW",
 "MOE_CB4C_LANES*MOE_CBATCH_MAXPOS*MOE_KROW",
 "MOE_CB4C_LANES*MOE_CBATCH_MAXPOS*MOE_VROW",
 "MOE_NEARTIE_LANES*MOE_CBATCH_MAXPOS*MOE_KROW",
 "MOE_NEARTIE_LANES*MOE_CBATCH_MAXPOS*MOE_VROW",
 "static int mf_ids[MOE_CB4B_RMAX][MOE_CBATCH_MAXPOS]",
 "rq_out[MOE_CB4B_RMAX][MOE_CBATCH_MAXPOS]",
 "if (plen[r] >= MOE_CBATCH_MAXPOS)",
 "if (plen[r] + maxnew[r] > MOE_CBATCH_MAXPOS)",
 "float scores[MOE_CBATCH_MAXPOS]",
 "REBIND_SET_APPLIED")

def mock_mlx():
    return "\n".join([
        "#define MLA_L0_MAXPOS 32", "#define GQA_L0_MAXPOS 32",
        "if (spos[i] < 0 || spos[i] >= MLA_L0_MAXPOS)",
        "mask_bytes[(size_t)m * MLA_L0_MAXPOS + j]",
        "g_fused_K.push_back(mx::zeros({B, H, MLA_L0_MAXPOS, QHD}",
        "g_fused_V.push_back(mx::zeros({B, H, MLA_L0_MAXPOS, VHD}",
        "if (spos[i] < 0 || spos[i] >= GQA_L0_MAXPOS)",
        "mask_bytes[(size_t)m * GQA_L0_MAXPOS + j]",
        "g_fused_gqa_cK.push_back(mx::zeros({B, KVH, GQA_L0_MAXPOS, HD}",
        "g_fused_gqa_cV.push_back(mx::zeros({B, KVH, GQA_L0_MAXPOS, HD}",
    ])+"\n"

def source():
    return ("#define MOE_MAXPOS 160\n#define MOE_CBATCH_MAXPOS 32\n"
            "#define MOE_CB4C_LANES 8\n#define MOE_NEARTIE_LANES 8\n"
            +"\n".join(MANDATORY)+"\n")

class P5ScratchContracts(unittest.TestCase):
    def test_scratch_patch_covers_all_C_KV_related_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            src=root/"my-run/src";src.mkdir(parents=True)
            (src/"qwen_infer.c").write_text(source())
            (src/"qng64_dense.c").write_text("/* fake, build not executed */")
            (src/"mlx_moe.cpp").write_text(mock_mlx())
            (src/"CMakeLists.txt").write_text(CMAKE_START+")\n")
            with patch.dict(os.environ,{"RUNNER_TEMP":str(root)}):
                outcome=patch_c(src)
            self.assertEqual(outcome["schema"],PATCH_SCHEMA)
            self.assertEqual(outcome["scratch_max_position"],128)
            self.assertTrue(outcome["all_reviewed_positioned_KV_buffers_use_same_cap"])
            self.assertNotEqual(outcome["original_C_sha256"],outcome["revised_C_sha256"])
            self.assertIn("#define MOE_CBATCH_MAXPOS 128",(src/"qwen_infer.c").read_text())
            self.assertIn("qng64_dense.c",(src/"CMakeLists.txt").read_text())
            self.assertIn("#define MLA_L0_MAXPOS 128",(src/"mlx_moe.cpp").read_text())
            self.assertIn("#define GQA_L0_MAXPOS 128",(src/"mlx_moe.cpp").read_text())
            self.assertTrue(outcome["all_reviewed_MLX_device_KV_and_attention_masks_use_same_cap"])
    def test_unreviewed_native_mirror_blocks_without_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            src=root/"my-run/src";src.mkdir(parents=True)
            unsafe=source().replace("MOE_NEARTIE_LANES*MOE_CBATCH_MAXPOS*MOE_KROW",
                                    "MOE_NEARTIE_LANES*32*MOE_KROW")
            (src/"qwen_infer.c").write_text(unsafe)
            (src/"qng64_dense.c").write_text("/* fake */")
            (src/"mlx_moe.cpp").write_text(mock_mlx())
            (src/"CMakeLists.txt").write_text(CMAKE_START+")\n")
            with patch.dict(os.environ,{"RUNNER_TEMP":str(root)}):
                with self.assertRaisesRegex(ValueError,"unreviewed long KV"):
                    patch_c(src)
            self.assertEqual(unsafe,(src/"qwen_infer.c").read_text())
    def test_no_permission_to_patch_other_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            foreign=root/"foreign/src";foreign.mkdir(parents=True)
            with patch.dict(os.environ,{"RUNNER_TEMP":str(root/"allowed")}):
                (root/"allowed").mkdir()
                with self.assertRaisesRegex(ValueError,"per-run scratch"):
                    patch_c(foreign)
    def test_shadow_old_stage_requires_REBIND_SET_and_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            owner=root/"hcp-p4-native-short-123";src=owner/"src";build=owner/"build"
            src.mkdir(parents=True);build.mkdir(parents=True)
            c=src/"qwen_infer.c";c.write_text("REBIND_SET_APPLIED\nREBIND_SET\n#define MOE_CBATCH_MAXPOS 32\n")
            binary=build/"qwen_infer_gpu";binary.write_bytes(b"fake isolated binary")
            with patch.dict(os.environ,{"RUNNER_TEMP":str(root)}):
                s=verify_scratch_binary(binary,c,owner)
                self.assertTrue(s["compiled_with_rebind_set_source"])
                self.assertEqual(len(s["binary_sha256"]),64)
                c.write_text("REBIND\n#define MOE_CBATCH_MAXPOS 32")
                with self.assertRaisesRegex(ValueError,"lacks multi-target"):
                    verify_scratch_binary(binary,c,owner)
    def test_shadow_long_rejects_unbounded_inputs_before_any_worker(self):
        with self.assertRaisesRegex(ValueError,"one or three"):
            run_long(None,None,None,None,None,None,None,None,count=4)

if __name__=="__main__":unittest.main()
