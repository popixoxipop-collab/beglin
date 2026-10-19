#!/usr/bin/env python3
import hashlib,json,tempfile
from pathlib import Path
import qt_incumbent_map as m
def expect_fail(obj,**kw):
 p=Path(tempfile.mkstemp()[1]);p.write_text(json.dumps(obj,sort_keys=True,separators=(",",":"))+"\n")
 try:m.load_incumbent(p,**kw)
 except m.IncumbentMapError:return
 raise AssertionError("expected fail")
base={"schema":m.EXPECTED_SCHEMA,"bits":{f"{r}:{g}":6 for r in range(896) for g in range(14)},"automatic_live_promotion":False}
# Any locally constructed map must fail SHA pin before semantics can authorize runtime use.
expect_fail(base,tensor_name=m.EXPECTED_TENSOR,out_dim=896,in_dim=896)
expect_fail(base,tensor_name="wrong",out_dim=896,in_dim=896)
print("QT_INCUMBENT_MAP_FAIL_CLOSED_PASS")
