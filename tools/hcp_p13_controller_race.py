#!/usr/bin/env python3
"""P13 genuine separate-process controller ownership race on EOE scratch.

The P12 exclusive flock and stricter P13 archive semantics are applied before
a would-be worker can start. No GPU, no production process, no real requests.
Competing contenders are independent Python processes, not mocks/threads.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,shutil,subprocess,sys,time
from hcp_p13_strict_journal import P13StrictPrestartLease

ROOT=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/.tailnet-tmp/hcp-p11-inflight-p12-multiowner-20261014")
OUTPUT=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/hcp-p13-native-code-parity-20261014/evidence/P13_MULTI_CONTROLLER_RACE.json")
N=8

CHILD=r"""
import json,os,pathlib,sys,time
from hcp_p13_strict_journal import P13StrictPrestartLease
root=pathlib.Path(sys.argv[1]);worker=pathlib.Path(sys.argv[2]);gate=pathlib.Path(sys.argv[3]);name=sys.argv[4]
deadline=time.monotonic()+7
while str(gate)!="NO_BARRIER" and not gate.is_file() and time.monotonic()<deadline:
 time.sleep(.004)
if time.monotonic()>=deadline:raise RuntimeError("P13 competitor barrier timeout")
start=time.monotonic_ns()
try:
 with P13StrictPrestartLease(root,worker) as lease:
  verdict=lease.inspect_for_start()
  assert verdict["no_auto_replay"] and verdict["ready_for_fresh_start"]
  holdstart=time.monotonic_ns()
  time.sleep(.27)
  holdend=time.monotonic_ns()
  out={"child":name,"status":"acquired","attempt_ns":start,"exclusive_from":holdstart,"exclusive_until":holdend}
except RuntimeError as e:
 if "another controller" not in str(e):
  raise
 out={"child":name,"status":"denied","attempt_ns":start}
print(json.dumps(out,sort_keys=True))
"""
def run_children(root,worker,gate,count):
    env={**os.environ,"PYTHONPATH":str(pathlib.Path(__file__).resolve().parent)}
    processes=[]
    for i in range(count):
        p=subprocess.Popen([sys.executable,"-c",CHILD,str(root),str(worker),
             "NO_BARRIER" if gate is None else str(gate),f"child-{i:02d}"],
             env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
             close_fds=True)
        processes.append(p)
    if gate is not None:
        time.sleep(.18)
        with gate.open("x") as f:f.write("release\n")
    rows=[]
    for p in processes:
        stdout,stderr=p.communicate(timeout=12)
        if p.returncode!=0:raise RuntimeError(f"P13 child failed: {stderr[-800:]}")
        rows.append(json.loads(stdout.strip()))
    return rows
def main():
    if ROOT.exists() or OUTPUT.exists():raise RuntimeError("P13 run-owned race root or receipt already exists")
    ROOT.parent.mkdir(parents=True,exist_ok=True)
    worker=ROOT/"workers"/"p13-native-candidate"
    worker.mkdir(parents=True)
    try:
        with P13StrictPrestartLease(ROOT,worker) as owner:
            first=owner.inspect_for_start()
            assert first["state"]=="CLEAN_FRESH_START"
            blocked=run_children(ROOT,worker,None,N)
            if any(x["status"]!="denied" for x in blocked):
                raise RuntimeError("P13 second controller acquired lock while owner still held")
        # Same filesystem, independently started 8 OS processes compete at one barrier.
        independent=run_children(ROOT,worker,ROOT/"ALL_GO.marker",N)
        granted=[x for x in independent if x["status"]=="acquired"]
        denied=[x for x in independent if x["status"]=="denied"]
        if not granted or not denied:
            raise RuntimeError("P13 simultaneous controller race did not produce accepted and blocked contenders")
        ordered=sorted(granted,key=lambda x:x["exclusive_from"])
        for x,y in zip(ordered,ordered[1:]):
            if x["exclusive_until"]>y["exclusive_from"]:
                raise RuntimeError("P13 concurrent controllers overlapped ownership")
        with P13StrictPrestartLease(ROOT,worker) as owner:
            last=owner.inspect_for_start()
            if not last["ready_for_fresh_start"] or last["state"]!="CLEAN_FRESH_START":
                raise RuntimeError("P13 sequential controller cannot subsequently acquire lease")
        data={"schema":"beglin-hcp-p13-real-multiprocess-exclusive-admission-race-v1",
            "status":"REAL_MULTI_PROCESS_FLOCK_FAIL_CLOSED",
            "parent_locks_held_during_first_batch":1,
            "simultaneous_blocked_while_owner_active":len(blocked),
            "independent_competing_processes_after_release":N,
            "independent_successful_controllers":len(granted),
            "independent_refused_controllers":len(denied),
            "exclusive_owner_time_intervals_no_overlap":True,
            "later_serial_start_allowed":True,
            "prestart_no_auto_replay_verified":True,
            "recorded_competitors":independent,
            "real_GPU_admission_executed":False,
            "full_multi_GPU_worker_concurrency_certified":False,
            "P12_original_live_PIDs_untouched":True,"production_touched":False,
            "automatic_live_promotion":False}
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        with OUTPUT.open("x") as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
        print("P13_REAL_CONTROLLER_RACE_PASS",json.dumps({
            "blocked_when_owned":len(blocked),"independent_granted":len(granted),
            "independent_denied":len(denied),
            "source_SHA256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "receipt_SHA256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest()},sort_keys=True),flush=True)
    finally:
        if ROOT.is_dir() and not ROOT.is_symlink():
            shutil.rmtree(ROOT)

if __name__=="__main__":main()
