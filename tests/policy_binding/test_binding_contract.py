#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP = (ROOT / "mlx_moe.cpp").read_text(encoding="utf-8")
HDR = (ROOT / "mlx_moe.h").read_text(encoding="utf-8")

def require(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing required binding contract fragment: {needle}")

def main() -> int:
    for needle in (
        "int mlx_gpu_get_binding_state(",
        "int mlx_gpu_assert_binding(",
        "int mlx_gpu_get_role_binding_state(",
        "int mlx_gpu_assert_role_binding(",
        "int mlx_gpu_qng64_batch_probe(",
    ):
        require(HDR, needle)

    qng = CPP.index("auto qng = g_qng64_tensors.find(name);")
    quant = CPP.index("auto quant = g_tensors.find(name);", qng)
    dense = CPP.index("auto dense = g_dtensors.find(name);", quant)
    assert qng < quant < dense

    require(CPP, "if (!present) return 0;")
    require(CPP, "return observed_n == requested_n ? 1 : 0;")
    require(CPP, "beglin_role_tensor_name")
    require(CPP, '"kv_a_proj_with_mqa"')
    require(CPP, '"shared_up_proj"')
    require(CPP, "if (!beglin_binding_state(name, &n, &representation)) return;")
    require(CPP, "BEGLIN_POLICY_BINDING_V1")
    require(CPP, "binding_present=1 binding_representation=%d")

    qng_start = CPP.index("static mx::array qng64_gemv_e0")
    qng_end = CPP.index("// D-metal-7: routed-FFN counterpart", qng_start)
    body = CPP[qng_start:qng_end]
    require(body, "int A = (int)x.shape(0);")
    require(body, "std::vector<mx::Shape> output_shapes = {{A, (int)t.out}};")
    require(body, '{"out_dim", (int)t.out}')
    require(body, "{64, (int)t.out, A}")
    require(CPP, "uint z = thread_position_in_grid.z;")
    require(CPP, "x[z * (ng * 64u) + g * 64 + p]")
    require(CPP, "out[z * (uint)out_dim + row]")

    require(CPP, "g_tensors.erase(std::string(name));")
    require(CPP, "g_dtensors.erase(std::string(name));")
    require(CPP, "g_qng64_tensors.erase(std::string(name));")
    require(CPP, "mlx_gpu_snapshot_binding")
    require(CPP, "mlx_gpu_reset_runtime_epoch")

    print(
        "P8_RECONCILED_BINDING_CONTRACT_PASS "
        "abi=PASS role_api=PASS lookup_priority=PASS fail_closed=PASS "
        "instrumentation_truth=PASS qng64_batch_source=PASS "
        "exclusive_maps=PASS certified_snapshot_epoch=PASS"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
