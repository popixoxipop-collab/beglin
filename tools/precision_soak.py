#!/usr/bin/env python3
"""P7 deterministic soak/failure-injection primitives for precision serving."""
from __future__ import annotations
import json, os, signal, time
from pathlib import Path
import precision_observability as po

class PrecisionSoakError(RuntimeError): pass

def rss_drift(samples):
    vals=[int(x) for x in samples]
    if not vals: return {"start":0,"end":0,"peak":0,"delta":0}
    return {"start":vals[0],"end":vals[-1],"peak":max(vals),"delta":vals[-1]-vals[0]}

def verify_epoch_sequence(rows):
    if not rows: return
    pid=rows[0]["pid"]; prior=None
    for i,row in enumerate(rows):
        if int(row["pid"])!=int(pid): raise PrecisionSoakError(f"PID drift at {i}")
        before=int(row["before_epoch"]); after=int(row["after_epoch"])
        if after < before or after-before > 1:
            raise PrecisionSoakError(f"invalid epoch step at {i}: {before}->{after}")
        if prior is not None and before != prior:
            raise PrecisionSoakError(f"epoch discontinuity at {i}: {prior}->{before}")
        prior=after

def verify_cache_plateau(rows, warmup=4):
    tail=rows[int(warmup):]
    if not tail: return
    added=sum(int(x.get("cache_bytes_added",0)) for x in tail)
    misses=sum(int(x.get("cache_misses",0)) for x in tail)
    if added or misses:
        raise PrecisionSoakError(f"cache did not plateau: misses={misses} bytes={added}")

def corrupt_json(path):
    p=Path(path); original=p.read_bytes()
    p.write_bytes(b'{"corrupt":')
    return original

def restore_bytes(path, data):
    p=Path(path); p.write_bytes(data)

def corrupt_lineage(path):
    p=Path(path); lines=p.read_text().splitlines()
    if not lines: raise PrecisionSoakError("lineage empty")
    row=json.loads(lines[-1]); row["request_count"]=int(row["request_count"])+1
    lines[-1]=json.dumps(row,sort_keys=True,separators=(",",":"))
    p.write_text("\n".join(lines)+"\n")

def verify_lineage_failure(path):
    try:
        po.verify_records(po.PrecisionObservability(lineage_path=path).read_records())
    except po.PrecisionObservabilityError:
        return True
    raise PrecisionSoakError("corrupt lineage unexpectedly verified")

def terminate_worker(proc):
    if proc is None or proc.poll() is not None: return
    os.kill(proc.pid, signal.SIGTERM)
    try: proc.wait(timeout=5)
    except Exception:
        os.kill(proc.pid, signal.SIGKILL); proc.wait(timeout=5)

def retention_files(path):
    p=Path(path)
    return sorted(p.parent.glob(p.name+".*.rotated"))

def assert_retention(path,max_files):
    files=retention_files(path)
    if len(files)>int(max_files):
        raise PrecisionSoakError(f"retention exceeded: {len(files)}>{max_files}")
    return files
