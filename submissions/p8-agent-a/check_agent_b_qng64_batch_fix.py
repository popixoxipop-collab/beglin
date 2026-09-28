#!/usr/bin/env python3
from pathlib import Path
import sys

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mlx_moe.cpp")
s = path.read_text(encoding="utf-8")
start = s.index("static mx::array qng64_gemv_e0")
end = s.index("// D-metal-7:", start)
block = s[start:end]
required = [
    "uint z = thread_position_in_grid.z;",
    "x[z * (ng * 64u) + g * 64 + p]",
    "out[z * (uint)out_dim + row]",
    "const int batch = x_shape.size() >= 2 ? (int)x_shape[0] : 1;",
    "std::vector<mx::Shape> output_shapes = {{batch, (int)t.out}};",
    '{"n", t.n}, {"ng", (int)t.ng}, {"out_dim", (int)t.out}',
    "{64, (int)t.out, batch}",
]
missing = [x for x in required if x not in s]
stale = [
    "std::vector<mx::Shape> output_shapes = {{1, (int)t.out}};",
    "{64, (int)t.out, 1}, {64, 1, 1}",
]
stale_present = [x for x in stale if x in block]
if missing or stale_present:
    print("P8 B batch-aware source gate FAIL")
    for x in missing:
        print("missing:", x)
    for x in stale_present:
        print("stale:", x)
    raise SystemExit(2)
print("P8 B batch-aware source gate PASS")
