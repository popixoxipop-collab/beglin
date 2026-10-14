#!/usr/bin/env python3
"""P13 strict semantic validation of previously SHA-journaled P12 native ACK.

P12 validates exact terminal source identity + fsync archive. This extension
also validates actual archived native ACK semantics and cross-checks both
tensor-role width transitions against the original immutable command.
No host-level or production binary mutation; no replay of old transactions.
"""
from __future__ import annotations
import hashlib,json,pathlib
from hcp_p12_prestart_journal import (
 PrestartLease,parse_txn,verify_archive_folder,POLICY_A,POLICY_B
)
import gpu_runtime_control as grc
import precision_context as pc

MAX_ACK_SIZE=16384
def verify_strict_terminal(folder):
    folder=pathlib.Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("P13 forged/symlink terminal journal folder")
    parts=(folder/"original.cmd",folder/"original_ack.json",
           folder/"INTENT.json",folder/"VERIFIED.json")
    for p in parts:
        if not p.is_file() or p.is_symlink():
            raise ValueError("P13 forged original terminal journal component")
    receipt=verify_archive_folder(folder)
    cmd=parse_txn(parts[0].read_bytes())
    if folder.name!=cmd["txn_id"]:raise ValueError("P13 old terminal ID not exact")
    if parts[1].stat().st_size>MAX_ACK_SIZE:raise ValueError("P13 unbounded old native ACK")
    ack=grc.read_runtime_ack(parts[1])
    intent=json.loads(parts[2].read_text())
    decision=receipt["decision"]
    status=ack["status"]
    if ((status=="REBIND_SET_APPLIED")!=(decision=="TERMINAL_COMMITTED") or
        status not in ("REBIND_SET_APPLIED","REBIND_SET_FAILED")):
        raise ValueError("P13 native ACK terminal status and committed receipt disagree")
    expected_epoch=cmd["epoch"]+(1 if status=="REBIND_SET_APPLIED" else 0)
    expected_policy=cmd["after_hash"] if status=="REBIND_SET_APPLIED" else cmd["before_hash"]
    if (ack["txn_id"]!=cmd["txn_id"] or
        ack.get("expected_epoch")!=cmd["epoch"] or
        ack.get("expected_policy_hash")!=cmd["before_hash"] or
        ack["weight_epoch"]!=expected_epoch or
        ack["active_policy_hash"]!=expected_policy or
        ack.get("changed_targets")!=(2 if status=="REBIND_SET_APPLIED" else 0) or
        intent.get("source_epoch")!=cmd["epoch"] or
        intent.get("source_policy_hash")!=cmd["before_hash"] or
        intent.get("target_policy_hash")!=cmd["after_hash"] or
        intent.get("terminal_status")!=status or
        receipt.get("source_epoch")!=cmd["epoch"]):
        raise ValueError("P13 original archived semantic epoch/preimage/after-image mismatch")
    targets=ack.get("txn_targets")
    expected_set={(name,layer,old,new) for (name,layer,old,new) in cmd["targets"]}
    if (not isinstance(targets,list) or len(targets)!=2 or
        {(t["role"],t["layer"],t["expected_n"],t["target_n"]) for t in targets}!=expected_set):
        raise ValueError("P13 actual two-target native ACK does not match source command")
    expected_policy_list=POLICY_B if expected_policy==pc.policy_hash(POLICY_B) else POLICY_A
    if ack["active_policy_hash"]!=pc.policy_hash(expected_policy_list):
        raise ValueError("P13 native terminal policy hash not recognized")
    if (receipt.get("no_auto_replay") is not True or
        receipt.get("new_worker_must_start_at_policy_A") is not True):
        raise ValueError("P13 unsafe replay/boot receipt")
    return {"transaction_id_sha256":hashlib.sha256(cmd["txn_id"].encode()).hexdigest(),
            "status":status,"source_epoch":cmd["epoch"],"new_epoch":expected_epoch,
            "two_targets_verified":True,"legacy_command_retained":True,
            "no_automatic_replay":True}

class P13StrictPrestartLease(PrestartLease):
    """Reusable P12 lock, additional native ACK semantics before admissions."""
    def inspect_for_start(self):
        result=super().inspect_for_start()
        root=self.worker/"p12_journal"
        if root.exists():
            if root.is_symlink() or not root.is_dir():
                raise RuntimeError("P13 terminal journal invalid")
            for item in root.iterdir():
                verify_strict_terminal(item)
        if not result.get("ready_for_fresh_start") or result.get("no_auto_replay") is not True:
            raise RuntimeError("P13 cannot safely restart from unverified prior state")
        return result
