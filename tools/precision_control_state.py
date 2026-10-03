#!/usr/bin/env python3
"""Durable desired/applied/quarantine state for backend-scoped precision control."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import precision_context as pc


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ControlStore:
    def __init__(self, root, model_revision_id, backend):
        if backend not in pc.SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported backend {backend}")
        self.base = Path(root) / str(model_revision_id) / backend
        self.backend = backend
        self.base.mkdir(parents=True, exist_ok=True)

    @property
    def desired_path(self):
        return self.base / "desired_policy.json"

    @property
    def applied_path(self):
        return self.base / "applied_state.json"

    @property
    def quarantine_path(self):
        return self.base / "quarantine.json"

    @property
    def journal_dir(self):
        p = self.base / "journal"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _read(self, path, default):
        if not Path(path).exists():
            return default
        with open(path) as f:
            return json.load(f)

    def set_desired(self, *, context_hash, policy, reason, source="autopilot"):
        rows = pc.normalize_policy(policy)
        value = {
            "version": 1,
            "backend": self.backend,
            "context_hash": context_hash,
            "policy": rows,
            "policy_hash": pc.policy_hash(rows),
            "reason": reason,
            "source": source,
            "updated_at": _now(),
        }
        _atomic_json(self.desired_path, value)
        return value

    def set_applied(
        self,
        *,
        context_hash,
        policy,
        epoch,
        txn_id,
        ack_sha256,
    ):
        rows = pc.normalize_policy(policy)
        value = {
            "version": 1,
            "backend": self.backend,
            "context_hash": context_hash,
            "policy": rows,
            "policy_hash": pc.policy_hash(rows),
            "epoch": int(epoch),
            "txn_id": txn_id,
            "ack_sha256": ack_sha256,
            "updated_at": _now(),
        }
        _atomic_json(self.applied_path, value)
        return value

    def desired(self):
        return self._read(self.desired_path, None)

    def applied(self):
        return self._read(self.applied_path, None)

    def quarantine(self):
        return self._read(self.quarantine_path, {"version": 1, "targets": []})

    def quarantine_target(
        self,
        *,
        context_hash,
        role,
        layer,
        n,
        reason,
        evidence_id=None,
    ):
        q = self.quarantine()
        target = {
            "context_hash": context_hash,
            "role": str(role),
            "layer": int(layer),
            "n": int(n),
            "reason": reason,
            "evidence_id": evidence_id,
            "quarantined_at": _now(),
        }
        kept = [
            x for x in q.get("targets", [])
            if not (
                x.get("context_hash") == context_hash
                and x.get("role") == str(role)
                and int(x.get("layer", -1)) == int(layer)
                and int(x.get("n", -1)) == int(n)
            )
        ]
        kept.append(target)
        q = {
            "version": 1,
            "backend": self.backend,
            "targets": kept,
            "updated_at": _now(),
        }
        _atomic_json(self.quarantine_path, q)
        return target

    def is_quarantined(self, *, context_hash, role, layer, n):
        return any(
            x.get("context_hash") == context_hash
            and x.get("role") == str(role)
            and int(x.get("layer", -1)) == int(layer)
            and int(x.get("n", -1)) == int(n)
            for x in self.quarantine().get("targets", [])
        )

    def reconcile(self):
        desired = self.desired()
        applied = self.applied()
        if desired is None or applied is None:
            return {
                "status": "INCOMPLETE",
                "desired": desired,
                "applied": applied,
            }
        if desired["context_hash"] != applied["context_hash"]:
            return {
                "status": "CONTEXT_MISMATCH",
                "desired": desired,
                "applied": applied,
            }
        if desired["policy_hash"] == applied["policy_hash"]:
            return {
                "status": "IN_SYNC",
                "desired": desired,
                "applied": applied,
            }
        return {
            "status": "DRIFT",
            "desired": desired,
            "applied": applied,
        }
