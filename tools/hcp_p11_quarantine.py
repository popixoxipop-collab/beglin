#!/usr/bin/env python3
"""Research-only durable quarantine of a crash-ambiguous REBIND_SET command.

Fail closed: requires a separately witnessed paused native crash, matching
preimage ACK, exact immutable two-target command and confirmed dead worker.
Never silently retries the ambiguous txn. Quarantine is journaled and fsynced;
any interrupted quarantine halts and requires independent manual reconciliation.
This is NOT installed on the production admission or boot path.
"""
from __future__ import annotations
import hashlib,json,os,pathlib,re,stat
import gpu_runtime_control as grc

VALID_ID=re.compile(r"^epoch-[0-9a-f]{32}$")
ALLOWED_WINDOWS=frozenset(("first_bind","before_ack"))
MAX_COMMAND_BYTES=2048

def hash_bytes(b):return hashlib.sha256(b).hexdigest()
def fsync_dir(p):
    fd=os.open(str(p),os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
    try:os.fsync(fd)
    finally:os.close(fd)
def durable_write_new(p,b):
    p=pathlib.Path(p)
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)
    fd=os.open(str(p),flags,0o600)
    try:
        with os.fdopen(fd,"wb",closefd=False) as f:
            f.write(b);f.flush()
        os.fsync(fd)
    finally:os.close(fd)
    fsync_dir(p.parent)
def assert_owned(owned,worker):
    owned=pathlib.Path(owned);worker=pathlib.Path(worker)
    if (not owned.is_dir() or owned.is_symlink() or
        not owned.name.startswith("hcp-p11-inflight-") or
        owned.parent.name!=".tailnet-tmp" or worker.parent!=owned/"workers" or
        worker.is_symlink()):
        raise ValueError("outside exact run-owned P11 scratch")
    return owned,worker
def inspect(owned,worker,marker,*,pid_is_dead):
    owned,worker=assert_owned(owned,worker)
    marker=pathlib.Path(marker)
    if not pid_is_dead:raise RuntimeError("native worker still active or unverified")
    if marker.parent!=owned or marker.is_symlink() or not marker.is_file():
        raise ValueError("P11 crash evidence marker absent/foreign")
    mbytes=marker.read_bytes()
    if len(mbytes)>200 or not mbytes.endswith(b"\n"):
        raise ValueError("P11 crash marker invalid")
    fields=mbytes.decode("utf-8").split()
    if len(fields)!=2 or fields[0] not in ALLOWED_WINDOWS or not VALID_ID.fullmatch(fields[1]):
        raise ValueError("P11 actual midtransaction marker invalid")
    path=worker/"txn.cmd"
    ackfile=worker/"applied_ack.json"
    if (not path.is_file() or path.is_symlink() or path.stat().st_size>MAX_COMMAND_BYTES or
        not ackfile.is_file() or ackfile.is_symlink() or
        path.parent!=worker):
        raise ValueError("missing precise durable command or old ACK")
    payload=path.read_bytes()
    if not payload.endswith(b"\n") or len(payload)>MAX_COMMAND_BYTES:
        raise ValueError("truncated/oversized crash-ambiguous command")
    parts=payload.decode("utf-8").strip().split()
    if (len(parts)!=13 or parts[0]!="REBIND_SET" or
        parts[1]!=fields[1] or parts[3]!="2"):
        raise ValueError("foreign, wrong target count or malformed transaction")
    try:
        epoch=int(parts[2])
        p1=(parts[5],int(parts[6]),int(parts[7]),int(parts[8]))
        p2=(parts[9],int(parts[10]),int(parts[11]),int(parts[12]))
    except (ValueError,IndexError) as e:
        raise ValueError("incompatible P11 command targets") from e
    if set((p1,p2)) not in (
        {("shared_down_proj",26,5,6),("shared_up_proj",3,6,5)},
        {("shared_down_proj",26,6,5),("shared_up_proj",3,5,6)},
    ):
        raise ValueError("not the expected immutable P11 two-target policy")
    ack=grc.read_runtime_ack(ackfile)
    if (ack["weight_epoch"]!=epoch or ack["active_policy_hash"]!=parts[4] or
        ack.get("txn_id")==fields[1] or ack["status"] not in ("PROMOTION_APPLIED","REBIND_SET_APPLIED")):
        raise ValueError("ACK inconsistent with pre-crash requested policy")
    return {"schema":"beglin-p11-crash-ambiguous-command-preimage-v1",
        "window":fields[0],"txn_id":fields[1],"expected_epoch":epoch,
        "source_command_sha256":hash_bytes(payload),
        "source_marker_sha256":hash_bytes(mbytes),
        "preimage_ack_sha256":ack["ack_sha256"],
        "preimage_policy_hash":ack["active_policy_hash"],
        "preimage_epoch":ack["weight_epoch"],
        "terminal_ACK_for_interrupted_transaction":False,
        "old_native_worker_verified_dead":True,
        "original_txn_file":str(path)}
def quarantine(owned,worker,marker,*,pid_is_dead):
    owned,worker=assert_owned(owned,worker)
    quarantine_root=worker/"quarantine"
    if quarantine_root.exists():raise RuntimeError("existing/partially completed quarantine; HALT")
    details=inspect(owned,worker,marker,pid_is_dead=pid_is_dead)
    if any(path.name.startswith("p11-reconcile-") for path in worker.iterdir()):
        raise RuntimeError("partially written reconciliation journal; HALT")
    quarantine_root.mkdir(mode=0o700)
    fsync_dir(worker)
    safeid=details["txn_id"];src=worker/"txn.cmd"
    dest=quarantine_root/(safeid+"."+details["source_command_sha256"]+".cmd")
    intent=quarantine_root/"INTENT.json"
    final=quarantine_root/"VERIFIED.json"
    payload=json.dumps(details,sort_keys=True,indent=2).encode()+b"\n"
    durable_write_new(intent,payload)
    os.replace(src,dest)
    fsync_dir(quarantine_root);fsync_dir(worker)
    if (src.exists() or hash_bytes(dest.read_bytes())!=details["source_command_sha256"]):
        raise RuntimeError("P11 cannot prove exact atomic command quarantine")
    verification={**details,"quarantine_cmd_path":str(dest),
      "status":"CRASH_AMBIGUOUS_UNCOMMITTED_TRANSACTION_QUARANTINED",
      "no_automatic_retry":True,"journaled_fsynced":True,
      "command_remains_retained_for_inspection":True,
      "next_restart_requires_fresh_STARTUP_ACK_and_policy_A":True,
      "production_touched":False}
    durable_write_new(final,json.dumps(verification,sort_keys=True,indent=2).encode()+b"\n")
    return verification
def verify_quarantine(worker):
    worker=pathlib.Path(worker)
    q=worker/"quarantine"
    final=q/"VERIFIED.json"
    intent=q/"INTENT.json"
    if (worker/"txn.cmd").exists() or not final.is_file() or not intent.is_file():
        raise RuntimeError("no finalized P11 quarantine; BOOT MUST HALT")
    f=json.loads(final.read_text());base=json.loads(intent.read_text())
    path=pathlib.Path(f["quarantine_cmd_path"])
    if (f.get("status")!="CRASH_AMBIGUOUS_UNCOMMITTED_TRANSACTION_QUARANTINED"
        or not f.get("journaled_fsynced") or path.parent!=q or path.is_symlink() or
        not path.is_file() or hash_bytes(path.read_bytes())!=f["source_command_sha256"]
        or f["source_command_sha256"]!=base["source_command_sha256"]
        or f["txn_id"]!=base["txn_id"]):
        raise RuntimeError("P11 retained command evidence tampered; BOOT MUST HALT")
    return f
