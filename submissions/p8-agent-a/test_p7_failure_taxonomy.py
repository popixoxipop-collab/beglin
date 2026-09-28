#!/usr/bin/env python3
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LOG=ROOT/"fixtures"/"p7_failure_logs"
QWEN=ROOT.parents[1]/"qwen_infer.c"

def text(name):
    return (LOG/name).read_text(encoding="utf-8",errors="replace")

def main():
    kva4=text("kva-l11_n4_slots4.log")
    sup4=text("shared-up-l3_n4_slots4.log")
    kva7=text("kva-l11_n7_slots4.log")
    source=QWEN.read_text(encoding="utf-8")

    assert "n=4 not in {2,3,5,6,7,8..15}" in kva4
    assert "n=4 uses a DIFFERENT format+registrar, q4g64" in kva4
    assert "n=4 not in {2,3,5,6,7,8..15}" in sup4
    assert "n=4 uses a DIFFERENT format+registrar, q4g64" in sup4

    assert "PROMOTED to qNg64(n=7) on GPU" in kva7
    assert "BEGLIN_INSTRUMENTATION_V2 kind=activation role=kv_a_proj_with_mqa layer=11 n=7" in kva7
    assert "mlx_gpu_cbatch_layer_step_lazy failed at layer 11 step 0" in kva7

    assert "BEGLIN_RUNTIME_UNSUPPORTED kind=quant_format" in source
    assert "n=7 is NOT categorically unsupported" in source
    print("P8 Agent A failure taxonomy PASS: n4=STRUCTURAL_UNSUPPORTED x2, kva_n7=CROSS_LANE_MLX_RUNTIME x1")

if __name__=="__main__":
    main()
