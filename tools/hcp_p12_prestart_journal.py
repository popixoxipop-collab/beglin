#!/usr/bin/env python3
"""P12 research-only fail-closed prestart journal for P9 precision worker.

A preflight gate, not an installed production supervisor or proof of exactly-once.
Canonical run-owned *.tailnet-tmp/hcp-p11-inflight-p12-* roots only; an exclusive
POSIX flock is held across native startup/admissions/shutdown by the caller.
Never retries or presumes completion of an ambiguous REBIND_SET.
"""
from __future__ import annotations
import contextlib,fcntl,hashlib,json,os,pathlib,re,subprocess
from hcp_p11_quarantine import verify_quarantine,fsync_dir,durable_write_new
import gpu_runtime_control as grc
import precision_context as pc

MODEL_SHA="d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"
SOURCE_SHA="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
BINARY_SHA="103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd"
POLICY_A=pc.normalize_policy([
 {"role":"shared_up_proj","layer":3,"n":6},
 {"role":"shared_down_proj","layer":26,"n":5}
])
POLICY_B=pc.normalize_policy([
 {"role":"shared_up_proj","layer":3,"n":5},
 {"role":"shared_down_proj","layer":26,"n":6}
])
SOURCE_POLICY_HASH=pc.policy_hash(POLICY_A)
TARGET_POLICY_HASH=pc.policy_hash(POLICY_B)
TXN=re.compile(r"^epoch-[0-9a-f]{32}$")
MAX_TXN_BYTES=2048
KNOWN_TARGETS={
 ("shared_up_proj",3,6,5),("shared_down_proj",26,5,6)
}
KNOWN_REVERSE={
 ("shared_up_proj",3,5,6),("shared_down_proj",26,6,5)
}
def sha_bytes(b:bytes):return hashlib.sha256(b).hexdigest()
def sha(path):return sha_bytes(pathlib.Path(path).read_bytes())

def safe_root(root,worker):
 root=pathlib.Path(root);worker=pathlib.Path(worker)
 if (root.name.find("hcp-p11-inflight-p12-")!=0
  or root.parent.name!=".tailnet-tmp"
  or not root.is_dir() or root.is_symlink() or root.resolve()!=root):
  raise ValueError("P12 prestart must remain under a resolved run-owned .tailnet-tmp root")
 if (worker.parent!=root/"workers" or not worker.is_dir() or worker.is_symlink()
  or worker.resolve()!=worker):
  raise ValueError("P12 foreign worker directory")
 return root,worker

def is_owned_binary_running(root):
 binary=str(root/"build/qwen_infer_gpu")
 res=subprocess.check_output(["/bin/ps","-axo","pid=,comm="],text=True,timeout=5)
 return any(binary in line for line in res.splitlines())

def parse_txn(cmd):
 if len(cmd)>MAX_TXN_BYTES or not cmd.endswith(b"\n") or b"\x00" in cmd:
  raise ValueError("P12 transaction invalid bytes")
 text=cmd.decode("ascii").strip().split()
 if (len(text)!=13 or text[0]!="REBIND_SET" or
  not TXN.fullmatch(text[1]) or text[3]!="2" or
  not re.fullmatch(r"[0-9a-f]{64}",text[4])):
  raise ValueError("P12 previous REBIND_SET transaction corrupted")
 try:
  epoch=int(text[2]);targets={
   (text[5],int(text[6]),int(text[7]),int(text[8])),
   (text[9],int(text[10]),int(text[11]),int(text[12]))}
 except (IndexError,ValueError) as exc:raise ValueError("P12 invalid requested target fields") from exc
 if epoch<0 or len(targets)!=2 or targets not in (KNOWN_TARGETS,KNOWN_REVERSE):
  raise ValueError("P12 rejects anything outside exact two-target 5/6-bit contract")
 before=SOURCE_POLICY_HASH if targets==KNOWN_TARGETS else TARGET_POLICY_HASH
 after=TARGET_POLICY_HASH if targets==KNOWN_TARGETS else SOURCE_POLICY_HASH
 if text[4]!=before:raise ValueError("P12 command policy preimage SHA mismatch")
 return {"txn_id":text[1],"epoch":epoch,"before_hash":before,
         "after_hash":after,"targets":sorted(list(targets))}

def verify_archive_folder(folder):
 folder=pathlib.Path(folder)
 if not folder.is_dir() or folder.is_symlink():raise ValueError("P12 journal not a real directory")
 names={i.name for i in folder.iterdir()}
 if names!={"INTENT.json","original.cmd","original_ack.json","VERIFIED.json"}:
  raise RuntimeError("P12 incomplete previous journal; HALT")
 intent=json.loads((folder/"INTENT.json").read_text())
 final=json.loads((folder/"VERIFIED.json").read_text())
 if (intent.get("schema")!="beglin-hcp-p12-acknowledged-prestart-intent-v1" or
     final.get("schema")!="beglin-hcp-p12-acknowledged-prestart-verified-v1" or
     intent.get("txn_id")!=folder.name or final.get("txn_id")!=folder.name or
     not TXN.fullmatch(folder.name)):
  raise RuntimeError("P12 previous journal wrong identity")
 if (final.get("cmd_sha256")!=sha(folder/"original.cmd") or
     final.get("ack_sha256")!=sha(folder/"original_ack.json") or
     intent.get("cmd_sha256")!=final.get("cmd_sha256") or
     intent.get("ack_sha256")!=final.get("ack_sha256") or
     final.get("decision") not in ("TERMINAL_COMMITTED", "TERMINAL_FAILED") or
     final.get("no_auto_replay") is not True):
  raise RuntimeError("P12 finalized journal incomplete or tampered")
 if parse_txn((folder/"original.cmd").read_bytes())["txn_id"]!=folder.name:
  raise RuntimeError("P12 archive identity not the original command")
 return final

class PrestartLease:
 """One-process prestart ownership + durable journal and startup admission gate.

 Caller MUST keep lease context alive for entire research worker lifecycle.
 This class never invokes worker.start or does automatic retry/recovery itself.
 """
 def __init__(self,root,worker,*,model_sha=MODEL_SHA,
              source_sha=SOURCE_SHA,binary_sha=BINARY_SHA):
  self.root,self.worker=safe_root(root,worker)
  if (model_sha!=MODEL_SHA or source_sha!=SOURCE_SHA or binary_sha!=BINARY_SHA):
   raise ValueError("P12 source/model/binary identity is not pinned")
  self.identity={"schema":"beglin-p12-owner-prestart-v1",
   "root":str(self.root),"worker":str(self.worker),
   "model_index_sha256":model_sha,"P9_mlx_source_sha256":source_sha,
   "P9_binary_sha256":binary_sha}
  self.fd=None
 def __enter__(self):
  lock=self.root/"P12_EXCLUSIVE_CONTROLLER.lock"
  if lock.is_symlink():raise RuntimeError("P12 lock symlink denied")
  fd=os.open(str(lock),os.O_RDWR|os.O_CREAT|getattr(os,"O_NOFOLLOW",0),0o600)
  try:
   fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except (OSError,BlockingIOError):
   os.close(fd)
   raise RuntimeError("P12 another controller holds prestart ownership lock")
  self.fd=fd
  try:
   self._owner()
   return self
  except BaseException:
   self.__exit__(None,None,None)
   raise
 def __exit__(self,*args):
  if self.fd is not None:
   fcntl.flock(self.fd,fcntl.LOCK_UN)
   os.close(self.fd);self.fd=None
 def _owner(self):
  name=self.root/"P12_OWNER_IDENTITY.json"
  if name.is_symlink():raise RuntimeError("P12 owner source metadata symlink")
  if not name.exists():
   if (self.worker/"txn.cmd").exists() or (self.worker/"applied_ack.json").exists():
    raise RuntimeError("P12 existing worker state cannot be adopted without prior owner receipt")
   durable_write_new(name,json.dumps(self.identity,sort_keys=True,indent=2).encode()+b"\n")
  if not name.is_file() or json.loads(name.read_text())!=self.identity:
   raise RuntimeError("P12 worker source identity mismatch")
 def inspect_for_start(self):
  if self.fd is None:raise RuntimeError("P12 missing exclusive owner lock")
  if is_owned_binary_running(self.root):
   raise RuntimeError("P12 run-owned native worker already running; do not start another")
  if (self.worker/"quarantine").exists():
   receipt=verify_quarantine(self.worker)
   if (self.worker/"txn.cmd").exists():
    raise RuntimeError("P12 quarantine and active txn both present")
  else:receipt=None
  journal=self.worker/"p12_journal"
  if journal.exists():
   if not journal.is_dir() or journal.is_symlink():raise RuntimeError("P12 invalid previous journal")
   seen=set()
   for sub in journal.iterdir():
    if not sub.is_dir() or sub.is_symlink() or sub.name in seen:raise RuntimeError("P12 bad archive")
    verify_archive_folder(sub);seen.add(sub.name)
  txn=self.worker/"txn.cmd"
  ack=self.worker/"applied_ack.json"
  if txn.is_symlink() or ack.is_symlink():raise RuntimeError("P12 source command/ACK symlink denied")
  if txn.exists():
   if not ack.is_file():raise RuntimeError("P12 txn without ACK; UNKNOWN: startup blocked")
   parsed=parse_txn(txn.read_bytes())
   current=grc.read_runtime_ack(ack)
   status=current.get("status")
   if (current.get("txn_id")!=parsed["txn_id"] or status not in
       ("REBIND_SET_APPLIED","REBIND_SET_FAILED")):
    raise RuntimeError("P12 UNKNOWN crash-ambiguous transaction: archive via P11 quarantine only, never retry")
   if (current["expected_epoch"]!=parsed["epoch"] or
       current["expected_policy_hash"]!=parsed["before_hash"]):
    raise RuntimeError("P12 previous terminal ACK source epoch or policy mismatch")
   applied=status=="REBIND_SET_APPLIED"
   expected_epoch=parsed["epoch"]+(1 if applied else 0)
   expected_hash=parsed["after_hash"] if applied else parsed["before_hash"]
   if (current["weight_epoch"]!=expected_epoch or
       current["active_policy_hash"]!=expected_hash or
       int(current.get("changed_targets",-1))!=(2 if applied else 0)):
    raise RuntimeError("P12 terminal ACK epoch/tensor result contradiction")
   if receipt:raise RuntimeError("P12 newly asserted terminal txn conflicts with existing quarantine")
   self._archive_terminal(parsed,txn,ack,status)
   return {"state":"KNOWN_TERMINAL_JOURNALED","old_txn_id_sha256":sha_bytes(parsed["txn_id"].encode()),
           "terminal":status,"no_auto_replay":True,"ready_for_fresh_start":True}
  if ack.exists():
   a=grc.read_runtime_ack(ack)
   if a.get("txn_id"):
    # A restart between VERIFIED fsync and native startup must be idempotent.
    # Old ACK can remain only if its exact bytes were archived with that txn.
    archived=journal/str(a["txn_id"])
    if (not journal.is_dir() or not archived.is_dir() or
        not TXN.fullmatch(str(a["txn_id"]))):
     raise RuntimeError("P12 unaccompanied terminal ACK; manual audit required")
    f=verify_archive_folder(archived)
    if (f["ack_sha256"]!=sha(ack) or
        a["status"] not in ("REBIND_SET_APPLIED","REBIND_SET_FAILED")):
     raise RuntimeError("P12 original ACK no longer matches durable journal")
   elif a.get("status") not in ("PROMOTION_APPLIED","STARTUP_STATE"):
    raise RuntimeError("P12 previous ACK has neither clean startup nor sealed terminal")
  return {"state":"QUARANTINED_UNKNOWN_FRESH_START" if receipt else "CLEAN_FRESH_START",
          "prior_unknown_original_outcome":bool(receipt),
          "ready_for_fresh_start":True,"no_auto_replay":True}
 def _archive_terminal(self,t,txn,ack,status):
  journal=self.worker/"p12_journal"
  if journal.is_symlink():raise RuntimeError("P12 journal symlink")
  if not journal.exists():journal.mkdir(mode=0o700)
  folder=journal/t["txn_id"]
  if folder.exists():raise RuntimeError("P12 duplicate/partial terminal archive; HALT")
  folder.mkdir(mode=0o700)
  fsync_dir(journal)
  cmd=txn.read_bytes()
  ackbytes=ack.read_bytes()
  intent={"schema":"beglin-hcp-p12-acknowledged-prestart-intent-v1",
      "txn_id":t["txn_id"],"source_epoch":t["epoch"],
      "source_policy_hash":t["before_hash"],"target_policy_hash":t["after_hash"],
      "terminal_status":status,"cmd_sha256":sha_bytes(cmd),
      "ack_sha256":sha_bytes(ackbytes)}
  durable_write_new(folder/"INTENT.json",json.dumps(intent,sort_keys=True,indent=2).encode()+b"\n")
  os.replace(txn,folder/"original.cmd")
  fsync_dir(self.worker);fsync_dir(folder)
  if sha(folder/"original.cmd")!=intent["cmd_sha256"]:
   raise RuntimeError("P12 old command changed while journaling")
  durable_write_new(folder/"original_ack.json",ackbytes)
  final={"schema":"beglin-hcp-p12-acknowledged-prestart-verified-v1",
         "txn_id":t["txn_id"],"cmd_sha256":intent["cmd_sha256"],
         "ack_sha256":intent["ack_sha256"],
         "decision":"TERMINAL_COMMITTED" if status=="REBIND_SET_APPLIED" else "TERMINAL_FAILED",
         "source_epoch":t["epoch"],
         "no_auto_replay":True,"no_double_commit":True,
         "new_worker_must_start_at_policy_A":True,
         "no_production_mutation":True}
  durable_write_new(folder/"VERIFIED.json",json.dumps(final,sort_keys=True,indent=2).encode()+b"\n")
  verify_archive_folder(folder)
  if txn.exists():raise RuntimeError("P12 old command still present after journal")
  fsync_dir(journal)
