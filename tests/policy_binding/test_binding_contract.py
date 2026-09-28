#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CPP = (ROOT / "mlx_moe.cpp").read_text(encoding="utf-8")
HDR = (ROOT / "mlx_moe.h").read_text(encoding="utf-8")


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing required binding contract fragment: {needle}")


def main() -> int:
    # Public C ABI: callers can ask the runtime map, not infer application
    # from a promotion/debug string.
    require(HDR, "int mlx_gpu_get_binding_state(")
    require(HDR, "int mlx_gpu_assert_binding(")\n    require(HDR, "int mlx_gpu_get_role_binding_state(")\n    require(HDR, "int mlx_gpu_assert_role_binding(")

    # Canonical lookup order must match runtime dispatch priority.
    qng = CPP.index("auto qng = g_qng64_tensors.find(name);")
    quant = CPP.index("auto quant = g_tensors.find(name);", qng)
    dense = CPP.index("auto dense = g_dtensors.find(name);", quant)
    assert qng < quant < dense

    # Missing bindings are explicit and mismatches fail closed.
    require(CPP, "if (!present) return 0;")
    require(CPP, "return observed_n == requested_n ? 1 : 0;")\n    require(CPP, "beglin_role_tensor_name")\n    require(CPP, '"kv_a_proj_with_mqa"')\n    require(CPP, '"shared_up_proj"')

    # Instrumentation must be derived from the same binding-state helper.
    require(CPP, "if (!beglin_binding_state(name, &n, &representation)) return;")
    require(CPP, "BEGLIN_POLICY_BINDING_V1")
    require(CPP, "binding_present=1 binding_representation=%d")

    # Rebinding paths must erase competing representations before insertion.
    require(CPP, "g_tensors.erase(std::string(name));")
    require(CPP, "g_dtensors.erase(std::string(name));")
    require(CPP, "g_qng64_tensors.erase(std::string(name));")

    print(
        "P8-B binding contract PASS "
        "abi=PASS lookup_priority=PASS fail_closed=PASS "
        "instrumentation_truth=PASS exclusive_maps=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
