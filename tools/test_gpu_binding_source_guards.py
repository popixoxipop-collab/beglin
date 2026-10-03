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

    def test_gpu_startup_promotion_source_is_independent_from_correction(self):
        block = QWEN.split("static void moe_promotion_nq_init_gpu(void)", 2)[2]
        block = block.split("#endif", 1)[0]
        self.assertIn("QWEN_MOE_PROMOTION_SAFETENSORS", block)
        self.assertIn("moe_neartie_correct_load_attn_hi(promo_st)", block)
        branch = block.split(
            'const char *promo_st = getenv("QWEN_MOE_PROMOTION_SAFETENSORS")',
            1,
        )[1].split("} else {", 1)[0]
        self.assertIn("moe_neartie_correct_load_attn_hi(promo_st)", branch)
        self.assertNotIn("atoi(nt_on)", branch)

    def test_gpu_startup_ack_exists_even_for_empty_policy(self):
        block = QWEN.split("static void moe_promotion_nq_init_gpu(void)", 2)[2]
        block = block.split("#endif", 1)[0]
        self.assertIn('"STARTUP_STATE"', block)
        self.assertIn("ack_path && ack_path[0]", block)
        self.assertIn("moe_gpu_write_applied_ack(ack_status, n_applied, NULL)", block)
        self.assertNotIn("if (n_applied && !moe_gpu_write_applied_ack", block)

    def test_gpu_online_argmax_rejects_nonfinite_logits(self):
        self.assertIn("static int moe_gpu_argmax_finite(", QWEN)
        helper = QWEN.split("static int moe_gpu_argmax_finite(", 1)[1]
        helper = helper.split("static int run_moe_gpu_gqa_cbatch_online_gate", 1)[0]
        self.assertIn("isfinite(bm)", helper)
        self.assertIn("isfinite(lg[v])", helper)
        self.assertGreaterEqual(
            QWEN.count("moe_gpu_argmax_finite(lg, MOE_VOCAB"), 4
        )

    def test_gpu_validation_report_covers_both_online_schedulers(self):
        self.assertGreaterEqual(
            QWEN.count("QWEN_MOE_GPU_VALIDATION_REPORT"), 2
        )
        self.assertGreaterEqual(
            QWEN.count("GPU_VALIDATION_V1 backend=mlx_metal"), 2
        )
        self.assertIn("arch=gqa correction=%s finite_logits=%d", QWEN)
        self.assertIn("arch=mla correction=%s finite_logits=%d", QWEN)
        self.assertGreaterEqual(QWEN.count("validation_logits_checked"), 6)
        self.assertGreaterEqual(QWEN.count("isfinite(gpu_logits[vi])"), 2)

    def test_gpu_ack_is_fsync_then_rename_and_fail_closed(self):
        block = QWEN.split("static int moe_gpu_write_applied_ack", 2)[2]
        block = block.split("static MoeAFTensor *moe_gpu_role_base_tensor", 1)[0]
        self.assertLess(block.index("fsync(fd)"), block.index("rename(tmp, path)"))
        self.assertIn("active_policy", block)
        self.assertIn("g_moe_promoted_nq_gpu[r][l]", block)
        demote = QWEN.split("static int moe_gpu_demotion_apply_quiescent", 2)[2]
        demote = demote.split("// D-qNg64-gpu-1:", 1)[0]
        self.assertIn("restore verified but durable ACK failed; admission remains stopped", demote)

    def test_gpu_txn_protocol_keeps_legacy_and_adds_atomic_set_cas_fields(self):
        self.assertIn("QWEN_MOE_GPU_TXN_FILE", QWEN)
        self.assertIn(
            "DEMOTE <txn_id> <expected_epoch> <expected_n> <role> <layer> <policy_sha256>",
            QWEN,
        )
        parser = QWEN.split("static int moe_gpu_txn_read(", 1)[1]
        parser = parser.split("static int moe_gpu_ack_already_has_txn", 1)[0]
        self.assertIn('strcmp(op, "DEMOTE")', parser)
        self.assertIn('strcmp(op, "REBIND_SET")', parser)
        self.assertIn("MOE_GPU_TXN_MAX_TARGETS", QWEN)
        self.assertIn("out->expected_epoch", parser)
        self.assertIn("out->expected_n", parser)
        self.assertIn("out->expected_policy_hash", parser)
        self.assertIn("out->role", parser)
        self.assertIn("out->layer", parser)
        self.assertIn("out->target_count", parser)
        self.assertIn("out->targets[i]", parser)

    def test_gpu_txn_rejects_stale_before_drain_or_registry_mutation(self):
        poll = QWEN.split("static int moe_gpu_demotion_pending(void)", 2)[2]
        poll = poll.split("static int moe_gpu_demotion_apply_quiescent(void)", 1)[0]
        self.assertIn("cmd.expected_epoch != g_moe_gpu_weight_epoch", poll)
        self.assertIn("g_moe_promoted_nq_gpu[t->role][t->layer] != t->expected_n", poll)
        self.assertIn('"STALE_COMMAND"', poll)
        self.assertNotIn("mlx_gpu_restore_binding_snapshot", poll)
        self.assertNotIn("mlx_gpu_reset_runtime_epoch", poll)

    def test_gpu_rebind_set_is_atomic_one_epoch_with_full_rollback(self):
        block = QWEN.split("static int moe_gpu_rebind_set_apply_quiescent", 1)[1]
        block = block.split("static int moe_gpu_demotion_pending", 1)[0]
        self.assertIn("mlx_gpu_synchronize()", block)
        self.assertIn("mlx_gpu_snapshot_binding", block)
        self.assertIn("mlx_gpu_restore_binding_snapshot", block)
        self.assertIn("mlx_gpu_reset_runtime_epoch()", block)
        self.assertIn("g_moe_gpu_weight_epoch++", block)
        self.assertIn('"REBIND_SET_APPLIED"', block)
        self.assertIn('"REBIND_SET_FAILED"', block)
        self.assertIn("cmd->target_count", block)
        self.assertLess(
            block.index("mlx_gpu_snapshot_binding"),
            block.index("mlx_gpu_bind_af"),
        )
        self.assertLess(
            block.index("mlx_gpu_reset_runtime_epoch()"),
            block.index("g_moe_gpu_weight_epoch++"),
        )

    def test_gpu_txn_duplicate_is_idempotent_from_ack(self):
        dup = QWEN.split("static int moe_gpu_ack_already_has_txn", 1)[1]
        dup = dup.split("static int moe_gpu_txn_mark_terminal", 1)[0]
        self.assertIn("g_moe_gpu_last_txn_id", dup)
        self.assertIn("QWEN_MOE_GPU_APPLIED_ACK", dup)
        self.assertIn('\\\"txn_id\\\":\\\"%s\\\"', dup)
        poll = QWEN.split("static int moe_gpu_demotion_pending(void)", 2)[2]
        poll = poll.split("static int moe_gpu_demotion_apply_quiescent(void)", 1)[0]
        self.assertIn("moe_gpu_ack_already_has_txn(cmd.txn_id)", poll)

    def test_gpu_txn_ack_carries_expected_and_actual_epoch_context(self):
        ack = QWEN.split("static int moe_gpu_write_applied_ack", 2)[2]
        ack = ack.split("static int moe_gpu_txn_token_safe", 1)[0]
        for field in ("txn_id", "expected_epoch", "expected_n", "expected_policy_hash",
                      "target_role", "target_layer", "txn_targets", "weight_epoch",
                      "active_policy", "correction_mode"):
            self.assertIn(field, ack)


if __name__ == "__main__":
    unittest.main()
