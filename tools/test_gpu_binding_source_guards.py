#!/usr/bin/env python3
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CPP = (ROOT / "mlx_moe.cpp").read_text()
HDR = (ROOT / "mlx_moe.h").read_text()
QWEN = (ROOT / "qwen_infer.c").read_text()


class BindingSourceGuardTests(unittest.TestCase):
    def test_dense_rebind_clears_custom_qng64(self):
        block = CPP.split("if (bits == 16 || bits == 32)", 1)[1]
        block = block.split("// D-metal-4:", 1)[0]
        self.assertIn("g_tensors.erase", block)
        self.assertIn("g_qng64_tensors.erase", block)

    def test_custom_qng64_rebind_clears_other_registries(self):
        block = CPP.split("if (bits == 7 || (bits >= 9 && bits <= 15))", 1)[1]
        block = block.split("if (ng <= 0 || (in % 8) != 0)", 1)[0]
        self.assertIn("g_tensors.erase", block)
        self.assertIn("g_dtensors.erase", block)

    def test_native_quant_rebind_clears_dense_and_custom(self):
        tail = CPP.split("// insert_or_assign, not operator[]=", 1)[1]
        block = tail.split("g_bound_count++;", 1)[0]
        self.assertIn("g_dtensors.erase", block)
        self.assertIn("g_qng64_tensors.erase", block)

    def test_binding_kind_probe_is_public(self):
        self.assertIn("int mlx_gpu_binding_kind(", CPP)
        self.assertIn("int mlx_gpu_binding_kind(", HDR)

    def test_snapshot_restore_api_is_public(self):
        for symbol in (
            "mlx_gpu_snapshot_binding",
            "mlx_gpu_restore_binding_snapshot",
            "mlx_gpu_drop_binding_snapshot",
            "mlx_gpu_binding_snapshot_count",
        ):
            self.assertIn(symbol, CPP)
            self.assertIn(symbol, HDR)

    def test_synchronize_api_is_public(self):
        self.assertIn("int mlx_gpu_synchronize(", CPP)
        self.assertIn("int mlx_gpu_synchronize(", HDR)
        self.assertIn("mx::synchronize()", CPP)

    def test_restore_clears_all_representation_maps_before_insert(self):
        block = CPP.split("int mlx_gpu_restore_binding_snapshot", 1)[1]
        block = block.split("int mlx_gpu_drop_binding_snapshot", 1)[0]
        self.assertIn("g_tensors.erase(key)", block)
        self.assertIn("g_dtensors.erase(key)", block)
        self.assertIn("g_qng64_tensors.erase(key)", block)

    def test_restore_requires_quiescence_by_contract_comment(self):
        block = CPP.split("int mlx_gpu_restore_binding_snapshot", 1)[0]
        self.assertIn("pause admission", block)
        self.assertIn("drain existing requests", block)
        self.assertIn("synchronize pending MLX/Metal work", block)

    def test_runtime_epoch_reset_clears_persistent_lazy_and_kv_state(self):
        self.assertIn("int mlx_gpu_reset_runtime_epoch(", CPP)
        self.assertIn("int mlx_gpu_reset_runtime_epoch(", HDR)
        block = CPP.split("int mlx_gpu_reset_runtime_epoch", 1)[1]
        self.assertIn("delete g_fused_x", block)
        self.assertIn("delete g_cbatch_x", block)
        self.assertIn("g_fused_K.clear()", block)
        self.assertIn("g_fused_V.clear()", block)
        self.assertIn("g_fused_gqa_K.clear()", block)
        self.assertIn("g_fused_gqa_V.clear()", block)
        self.assertIn("g_fused_gqa_cK.clear()", block)
        self.assertIn("g_fused_gqa_cV.clear()", block)

    def test_cpu_precision_mutation_is_outside_admission_loop(self):
        sched = QWEN.split("int qhead = 0, nact = 0, step = 0;", 1)[1]
        sched = sched.split("RESULT: MoE-4b", 1)[0]
        admission = sched.split("// 1. admission", 1)[1].split("if (nact == 0)", 1)[0]
        self.assertNotIn("moe_promotion_maybe_apply();", admission)
        self.assertNotIn("moe_demotion_nq_maybe_apply();", admission)
        self.assertIn("int cpu_precision_drain = 0;", sched)
        self.assertIn("moe_precision_control_pending_cpu()", sched)
        self.assertIn("if (cpu_precision_drain && nact == 0)", sched)

    def test_gpu_promotion_is_transactional_and_verified(self):
        block = QWEN.split("static void moe_promotion_nq_init_gpu(void)", 2)[2]
        block = block.split("#endif", 1)[0]
        self.assertLess(
            block.index("mlx_gpu_snapshot_binding"),
            block.index("mlx_gpu_bind_af"),
        )
        self.assertIn("mlx_gpu_binding_kind(base_ptr->name, &applied_bits)", block)
        self.assertIn("expected_kind = (n == 7 || (n >= 9 && n <= 15)) ? 3 : 1", block)
        self.assertIn("mlx_gpu_restore_binding_snapshot(attempt_snapshot)", block)
        self.assertNotIn("n=7 has no native MLX kernel", block)

    def test_gpu_demotion_requires_quiescent_sync_and_verified_restore(self):
        block = QWEN.split("static int moe_gpu_demotion_apply_quiescent(void)", 2)[2]
        block = block.split("// D-qNg64-gpu-1:", 1)[0]
        self.assertLess(block.index("mlx_gpu_synchronize()"), block.index("mlx_gpu_restore_binding_snapshot"))
        self.assertIn("mlx_gpu_binding_kind(base_ptr->name, &restored_bits)", block)
        self.assertIn("restored_kind != expected_kind || restored_bits != expected_bits", block)
        self.assertIn("mlx_gpu_reset_runtime_epoch()", block)
        self.assertIn("g_moe_gpu_weight_epoch++", block)
        self.assertIn('moe_gpu_write_applied_ack("ROLLBACK_APPLIED"', block)

    def test_both_online_gpu_schedulers_pause_admission_for_demotion(self):
        self.assertGreaterEqual(QWEN.count("int gpu_demotion_drain = 0;"), 2)
        self.assertGreaterEqual(QWEN.count("moe_gpu_demotion_pending()"), 4)
        self.assertGreaterEqual(QWEN.count("moe_gpu_demotion_apply_quiescent()"), 4)
        self.assertIn(
            "if (!gpu_demotion_drain) for (int s = 0; s < B && qhead < R; s++)",
            QWEN,
        )

    def test_gpu_ack_is_fsync_then_rename_and_fail_closed(self):
        block = QWEN.split("static int moe_gpu_write_applied_ack", 2)[2]
        block = block.split("static MoeAFTensor *moe_gpu_role_base_tensor", 1)[0]
        self.assertLess(block.index("fsync(fd)"), block.index("rename(tmp, path)"))
        self.assertIn("active_policy", block)
        self.assertIn("g_moe_promoted_nq_gpu[r][l]", block)
        demote = QWEN.split("static int moe_gpu_demotion_apply_quiescent", 2)[2]
        demote = demote.split("// D-qNg64-gpu-1:", 1)[0]
        self.assertIn("restore verified but durable ACK failed; admission remains stopped", demote)


if __name__ == "__main__":
    unittest.main()
