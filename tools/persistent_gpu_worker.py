#!/usr/bin/env python3
"""Client/supervisor-side protocol for Beglin's persistent GPU worker mode."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import threading
import time
from typing import Any


class PersistentWorkerError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}.{threading.get_ident()}")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def write_i32(path: Path, values: list[int]) -> None:
    with path.open("wb") as f:
        for value in values:
            f.write(struct.pack("<i",int(value)))
        f.flush()
        os.fsync(f.fileno())


def validate_batch(requests: Any, *, max_requests: int=12, max_prompt_tokens: int=4096, max_new_tokens: int=256) -> list[tuple[list[int],int]]:
    if not isinstance(requests,list) or not requests:
        raise PersistentWorkerError("requests must be a non-empty list")
    if len(requests)>max_requests:
        raise PersistentWorkerError("request batch exceeds limit")
    out=[]
    for row in requests:
        if not isinstance(row,dict):
            raise PersistentWorkerError("request row must be an object")
        tokens=row.get("prompt_tokens")
        if not isinstance(tokens,list) or not tokens:
            raise PersistentWorkerError("prompt_tokens must be a non-empty list")
        if len(tokens)>max_prompt_tokens:
            raise PersistentWorkerError("prompt_tokens exceeds limit")
        vals=[]
        for raw in tokens:
            try: token=int(raw)
            except (TypeError,ValueError) as exc:
                raise PersistentWorkerError("prompt token must be an integer") from exc
            if token<0 or token>2**31-1:
                raise PersistentWorkerError("prompt token outside int32 range")
            vals.append(token)
        try: max_new=int(row.get("max_new_tokens",10))
        except (TypeError,ValueError) as exc:
            raise PersistentWorkerError("max_new_tokens must be an integer") from exc
        if max_new<=0 or max_new>max_new_tokens:
            raise PersistentWorkerError("max_new_tokens outside limit")
        out.append((vals,max_new))
    return out


class PersistentGpuWorker:
    def __init__(
        self,
        *,
        name: str,
        binary: str | Path,
        repo: str | Path,
        state_dir: str | Path,
        moe_base: str | Path,
        safetensors: str | Path,
        promotion_line: str,
        expected_policy_hash: str,
        tools_dir: str | Path,
        slots: int=4,
        startup_timeout: float=120.0,
        request_timeout: float=90.0,
    ):
        self.name=str(name)
        self.binary=Path(binary)
        self.repo=Path(repo)
        self.state_dir=Path(state_dir)
        self.moe_base=Path(moe_base)
        self.safetensors=Path(safetensors)
        self.promotion_line=str(promotion_line)
        self.expected_policy_hash=str(expected_policy_hash)
        self.tools_dir=Path(tools_dir)
        self.slots=int(slots)
        self.startup_timeout=float(startup_timeout)
        self.request_timeout=float(request_timeout)
        self.proc: subprocess.Popen | None=None
        self._log_handle=None
        self._seq=0
        self._lock=threading.Lock()

    @property
    def pid(self) -> int | None:
        return int(self.proc.pid) if self.proc and self.proc.poll() is None else None

    def _env(self) -> dict[str,str]:
        env={k:v for k,v in os.environ.items() if k in ("PATH","HOME","USER","LOGNAME","LANG","LC_ALL","TMPDIR")}
        env.update({
            "QWEN_MOE_GPU_CBATCH_ONLINE":"1",
            "QWEN_MOE_GPU_PERSIST_DIR":str(self.state_dir),
            "QWEN_MOE_BASE":str(self.moe_base),
            "QWEN_MOE_NEARTIE_CORRECT":"0",
            "QWEN_MOE_CB_SLOTS":str(self.slots),
            "QWEN_MOE_CB_REQS":"12",
            "QWEN_MOE_GPU_VALIDATION_REPORT":"1",
            "QWEN_MOE_GPU_APPLIED_ACK":str(self.state_dir/"applied_ack.json"),
            "QWEN_MOE_GPU_TXN_FILE":str(self.state_dir/"txn.cmd"),
            "QWEN_MOE_PROMOTION_FILE_NQ":str(self.state_dir/"promotion_nq.txt"),
            "QWEN_MOE_PROMOTION_SAFETENSORS":str(self.safetensors),
        })
        return env

    def _reset_state(self) -> None:
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)
        self.state_dir.mkdir(parents=True,mode=0o700)
        (self.state_dir/"promotion_nq.txt").write_text(self.promotion_line)
        self._seq=0

    def _read_ack(self) -> dict:
        import sys
        tools=str(self.tools_dir)
        if tools not in sys.path:
            sys.path.insert(0,tools)
        import gpu_runtime_control as grc
        return grc.read_runtime_ack(self.state_dir/"applied_ack.json")

    def start(self) -> dict:
        if self.proc and self.proc.poll() is None:
            return self.status()
        for p in (self.binary,self.safetensors):
            if not p.exists():
                raise PersistentWorkerError(f"required path missing: {p}")
        self._reset_state()
        log_path=self.state_dir/"worker.log"
        self._log_handle=log_path.open("a",buffering=1)
        self.proc=subprocess.Popen(
            [str(self.binary)],
            cwd=self.repo,
            env=self._env(),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline=time.monotonic()+self.startup_timeout
        ready=self.state_dir/"ready.json"
        ack_path=self.state_dir/"applied_ack.json"
        while time.monotonic()<deadline:
            if self.proc.poll() is not None:
                raise PersistentWorkerError(
                    f"{self.name} exited during startup rc={self.proc.returncode}"
                )
            if ready.is_file() and ack_path.is_file():
                obj=json.loads(ready.read_text())
                ack=self._read_ack()
                if obj.get("status")=="READY":
                    if ack.get("active_policy_hash")!=self.expected_policy_hash:
                        self.stop()
                        raise PersistentWorkerError(
                            f"{self.name} policy hash mismatch "
                            f"expected={self.expected_policy_hash} actual={ack.get('active_policy_hash')}"
                        )
                    return {
                        "schema":"beglin-persistent-worker-start-v1",
                        "status":"READY",
                        "name":self.name,
                        "pid":self.pid,
                        "policy_hash":ack["active_policy_hash"],
                        "weight_epoch":int(ack["weight_epoch"]),
                        "ack_sha256":ack["ack_sha256"],
                    }
            time.sleep(0.05)
        self.stop()
        raise PersistentWorkerError(f"{self.name} startup timed out")

    def status(self) -> dict:
        alive=self.proc is not None and self.proc.poll() is None
        return {
            "name":self.name,
            "alive":alive,
            "pid":self.pid,
            "state_dir":str(self.state_dir),
            "binary":str(self.binary),
        }

    def submit(self, requests: list[dict]) -> dict:
        parsed=validate_batch(requests)
        with self._lock:
            if not self.proc or self.proc.poll() is not None:
                raise PersistentWorkerError(f"{self.name} is not running")
            seq=self._seq+1
            req_dir=self.state_dir/f"request-{seq}"
            req_dir.mkdir(parents=False,exist_ok=False)
            lines=[]
            for idx,(tokens,max_new) in enumerate(parsed):
                raw=req_dir/f"req-{idx}.i32"
                write_i32(raw,tokens)
                lines.append(f"{raw} {max_new}\n")
            atomic_text(self.state_dir/"request.manifest","".join(lines))
            started=time.monotonic()
            atomic_text(self.state_dir/"request.seq",f"{seq}\n")
            response_path=self.state_dir/f"response.{seq}.json"
            deadline=started+self.request_timeout
            while time.monotonic()<deadline:
                if self.proc.poll() is not None:
                    raise PersistentWorkerError(
                        f"{self.name} exited while processing seq={seq} rc={self.proc.returncode}"
                    )
                if response_path.is_file():
                    obj=json.loads(response_path.read_text())
                    if obj.get("schema")!="beglin-gpu-persistent-response-v1":
                        raise PersistentWorkerError("unexpected persistent response schema")
                    if int(obj.get("seq",-1))!=seq:
                        raise PersistentWorkerError("persistent response sequence mismatch")
                    if int(obj.get("requests",-1))!=len(parsed):
                        raise PersistentWorkerError("persistent response request count mismatch")
                    if obj.get("finite_logits") is not True:
                        raise PersistentWorkerError("persistent worker reported non-finite logits")
                    if int(obj.get("pid",-1)) != int(self.pid or -2):
                        raise PersistentWorkerError(
                            f"persistent response PID mismatch expected={self.pid} actual={obj.get('pid')}"
                        )
                    responses=obj.get("responses")
                    if not isinstance(responses,list) or len(responses)!=len(parsed):
                        raise PersistentWorkerError("persistent response rows mismatch")
                    self._seq=seq
                    total_ms=max(1,int((time.monotonic()-started)*1000))
                    return {
                        "schema":"beglin-persistent-worker-client-response-v1",
                        "status":"OK",
                        "name":self.name,
                        "pid":self.pid,
                        "seq":seq,
                        "engine_duration_ms":float(obj["duration_ms"]),
                        "roundtrip_duration_ms":total_ms,
                        "finite_logits":True,
                        "weight_epoch":int(obj["weight_epoch"]),
                        "peak_rss_bytes":int(obj.get("peak_rss_bytes",0)),
                        "responses":responses,
                    }
                time.sleep(0.01)
            raise PersistentWorkerError(f"{self.name} request seq={seq} timed out")

    def stop(self) -> None:
        proc=self.proc
        if not proc:
            return
        if proc.poll() is None:
            try:
                atomic_text(self.state_dir/"shutdown","1\n")
                proc.wait(timeout=5)
            except Exception:
                proc.terminate()
                try: proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                    try: proc.wait(timeout=5)
                    except Exception: pass
        self.proc=None
        if self._log_handle:
            try:self._log_handle.close()
            except Exception:pass
            self._log_handle=None


class PersistentWorkerPool:
    def __init__(self, *, workers: dict[str,PersistentGpuWorker]):
        self.workers=dict(workers)

    def start_all(self) -> dict:
        out={}
        for route_id,worker in self.workers.items():
            out[route_id]=worker.start()
        return out

    def stop_all(self) -> None:
        for worker in self.workers.values():
            worker.stop()

    def get(self, route_id: str) -> PersistentGpuWorker:
        try:return self.workers[str(route_id)]
        except KeyError as exc:
            raise PersistentWorkerError(f"no persistent worker for route {route_id!r}") from exc
