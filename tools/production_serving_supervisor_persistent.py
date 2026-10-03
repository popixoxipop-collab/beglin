#!/usr/bin/env python3
"""Persistent-worker Beglin localhost serving supervisor.

This is a successor to production_serving_supervisor.py.  It keeps TWO native
MLX workers resident at once:
- baseline q4g64 route
- approved shared_up_proj/L3 qNg64(n=6) candidate route

The C workers own model/MLX state for their full lifetime and exchange batches
with this supervisor through the atomic request/result protocol implemented by
QWEN_MOE_GPU_PERSISTENT_DIR.  Route changes therefore only change which warm
PID receives the next admitted HTTP batch; they do not reload the model.

The previous spawn-per-admission supervisor remains available for rollback.
"""
from __future__ import annotations

import argparse
from http import HTTPStatus
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

import production_serving_supervisor as base


PERSISTENT_BINARY = Path("/Users/xox/vdsp_serving/persistent-stage/qwen_infer_gpu")
PERSISTENT_PROBE = Path("/Users/xox/vdsp_serving/persistent-stage/probe.json")
EXPECTED_PERSISTENT_BINARY_SHA = "8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
EXPECTED_PERSISTENT_SOURCE_SHA = "8b46cdd38f5b39dcaeaa9d1c5dfbd9642601a30a"
DEFAULT_PERSISTENT_ROOT = Path("/Users/xox/vdsp_serving/persistent-workers")
WORKER_READY_TIMEOUT_SECONDS = 120
REQUEST_TIMEOUT_SECONDS = 60
POLL_SECONDS = 0.01
NEARTIE_TELEMETRY_THRESHOLD = 0.02
ADAPTIVE_L26_ENV = "BEGLIN_ADAPTIVE_L26_ENABLED"
ADAPTIVE_L26_ROLE = "shared_down_proj"
ADAPTIVE_L26_LAYER = 26
ADAPTIVE_L26_BASE_N = 5
ADAPTIVE_L26_RECOVERY_N = 6
ADAPTIVE_L26_MARGIN_MAX = 0.02
ADAPTIVE_L26_EVIDENCE_SHA256 = "de976ab12283673a8cf97638be9cf0b5c8f7ab3df2a0db87d4be96d9e67f6075"
ADAPTIVE_L26_REBIND_EVIDENCE_SHA256 = "de976ab12283673a8cf97638be9cf0b5c8f7ab3df2a0db87d4be96d9e67f6075"
ADAPTIVE_L26_STARTUP_POLICY_SHA256 = "dbee11614bb74073e0751bbf9fad0f67f0b99396256cb88589df80701f363ef1"
ADAPTIVE_L26_ACCEPTANCE_EVIDENCE = Path("/Users/xox/vdsp_serving/adaptive-isolated-final-48a7872-20261003/result.json")
ADAPTIVE_L26_ACCEPTANCE_SHA256 = "dc95591d79698369dd93903d6bf39bd0d66afaaf24300f708c4bbe3c5912792c"
ADAPTIVE_L26_ACCEPTANCE_SOURCE = "48a7872a9c837138a1556cfa3011459fcf47c530"
LAUNCHD_PROCESS_TYPE = "Interactive"


class PersistentSupervisorError(base.SupervisorError):
    pass


def verify_persistent_artifact() -> dict:
    if not PERSISTENT_BINARY.is_file():
        raise PersistentSupervisorError(f"persistent binary missing: {PERSISTENT_BINARY}")
    binary_sha = base._sha256_file(PERSISTENT_BINARY)
    if binary_sha != EXPECTED_PERSISTENT_BINARY_SHA:
        raise PersistentSupervisorError(
            f"persistent binary SHA mismatch: expected={EXPECTED_PERSISTENT_BINARY_SHA} actual={binary_sha}"
        )
    if not PERSISTENT_PROBE.is_file():
        raise PersistentSupervisorError(f"persistent probe evidence missing: {PERSISTENT_PROBE}")
    probe = json.loads(PERSISTENT_PROBE.read_text())
    if probe.get("status") != "PASS":
        raise PersistentSupervisorError("persistent GPU probe status is not PASS")
    if probe.get("binary_sha256") != EXPECTED_PERSISTENT_BINARY_SHA:
        raise PersistentSupervisorError("persistent probe binary identity mismatch")
    if probe.get("source_head") != EXPECTED_PERSISTENT_SOURCE_SHA:
        raise PersistentSupervisorError("persistent probe source identity mismatch")
    if probe.get("same_process_for_both_requests") is not True:
        raise PersistentSupervisorError("persistent probe did not prove same-process reuse")
    if int(probe.get("warm", {}).get("reference_hits", 0)) != 12:
        raise PersistentSupervisorError("persistent warm probe reference coverage mismatch")
    checkpoint = json.loads(base.CHECKPOINT_EVIDENCE.read_text())
    if checkpoint.get("status") != "VERIFIED":
        raise PersistentSupervisorError("checkpoint evidence is not VERIFIED")
    if checkpoint.get("checkpoint_sha256") != base.EXPECTED_CHECKPOINT_SHA:
        raise PersistentSupervisorError("checkpoint identity mismatch")
    return {
        "binary_sha256": binary_sha,
        "source_head": EXPECTED_PERSISTENT_SOURCE_SHA,
        "probe": probe,
    }


def verify_adaptive_l26_acceptance() -> dict:
    path = ADAPTIVE_L26_ACCEPTANCE_EVIDENCE
    if not path.is_file():
        raise PersistentSupervisorError(f"adaptive L26 acceptance evidence missing: {path}")
    actual_sha = base._sha256_file(path)
    if actual_sha != ADAPTIVE_L26_ACCEPTANCE_SHA256:
        raise PersistentSupervisorError(
            "adaptive L26 acceptance evidence SHA mismatch: "
            f"expected={ADAPTIVE_L26_ACCEPTANCE_SHA256} actual={actual_sha}"
        )
    try:
        evidence = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistentSupervisorError("adaptive L26 acceptance evidence is unreadable") from exc
    required = {
        "schema": "beglin-adaptive-isolated-final/1",
        "status": "PASS",
        "source_head": ADAPTIVE_L26_ACCEPTANCE_SOURCE,
        "binary_sha256": EXPECTED_PERSISTENT_BINARY_SHA,
        "adaptive_evidence_sha256": ADAPTIVE_L26_EVIDENCE_SHA256,
        "adaptive_startup_policy_sha256": ADAPTIVE_L26_STARTUP_POLICY_SHA256,
        "production_route_touched": False,
    }
    for key, expected in required.items():
        if evidence.get(key) != expected:
            raise PersistentSupervisorError(
                f"adaptive L26 acceptance evidence field mismatch: {key}"
            )
    rows = evidence.get("requests")
    if not isinstance(rows, list) or len(rows) < 2:
        raise PersistentSupervisorError("adaptive L26 acceptance requires two verified requests")
    pids = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PersistentSupervisorError("adaptive L26 acceptance request row is invalid")
        try:
            pids.add(int(row["pid"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PersistentSupervisorError("adaptive L26 acceptance PID is invalid") from exc
        if row.get("finite_logits") is not True or int(row.get("token8", -1)) != 1224:
            raise PersistentSupervisorError("adaptive L26 acceptance output mismatch")
        if row.get("action") != "RECOVERY_N6" or row.get("trigger_request_indices") != [0]:
            raise PersistentSupervisorError("adaptive L26 acceptance recovery contract mismatch")
        if int(row.get("worker_left_at_n", -1)) != ADAPTIVE_L26_RECOVERY_N:
            raise PersistentSupervisorError("adaptive L26 acceptance terminal precision mismatch")
        base_events = row.get("base_events")
        if not isinstance(base_events, list) or not base_events:
            raise PersistentSupervisorError("adaptive L26 acceptance trigger evidence missing")
        if any(float(event.get("margin", 1.0)) > ADAPTIVE_L26_MARGIN_MAX for event in base_events):
            raise PersistentSupervisorError("adaptive L26 acceptance margin exceeds certified bucket")
        if idx > 0 and row.get("restored_from_n6") is not True:
            raise PersistentSupervisorError("adaptive L26 acceptance did not prove n6->n5 restore")
    if len(pids) != 1:
        raise PersistentSupervisorError("adaptive L26 acceptance did not prove same-PID reuse")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "worker_pid": next(iter(pids)),
        "request_count": len(rows),
    }


def _runtime_tools_path() -> str:
    # Bind control helpers to the same reviewed/deployed source tree as this
    # supervisor. base.REPO may point at an older local engine checkout on XOX
    # and must not silently override the adaptive control contract.
    return str(Path(__file__).resolve().parent)


def _assert_local_helper(module, name: str):
    expected = Path(_runtime_tools_path()).resolve()
    actual_file = getattr(module, "__file__", None)
    if not actual_file or Path(actual_file).resolve().parent != expected:
        raise PersistentSupervisorError(
            f"{name} loaded from unreviewed path: {actual_file}"
        )
    return module


def _load_gpu_runtime_control():
    tools = _runtime_tools_path()
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import gpu_runtime_control as grc
    _assert_local_helper(grc, "gpu_runtime_control")
    required = ("read_runtime_ack", "prepare_rebind", "verify_terminal_ack")
    missing = [name for name in required if not callable(getattr(grc, name, None))]
    if missing:
        raise PersistentSupervisorError(
            f"gpu runtime control helper is stale/missing: {','.join(missing)}"
        )
    return grc


def _load_precision_epoch_scheduler():
    tools = _runtime_tools_path()
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import precision_epoch_scheduler as pes
    return _assert_local_helper(pes, "precision_epoch_scheduler")


def _load_precision_closed_loop():
    tools = _runtime_tools_path()
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import precision_closed_loop as pcl
    return _assert_local_helper(pcl, "precision_closed_loop")


def _read_runtime_ack(path: Path) -> dict:
    return _load_gpu_runtime_control().read_runtime_ack(path)


def _read_neartie_events_since(path: Path, offset: int) -> list[dict]:
    if offset < 0:
        raise PersistentSupervisorError("near-tie telemetry offset cannot be negative")
    if not path.is_file():
        return []
    events = []
    with path.open("rb") as handle:
        handle.seek(offset)
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if row.get("kind") != "event":
                continue
            try:
                event = {
                    "req": int(row["req"]),
                    "pos": int(row["pos"]),
                    "predicted_token": int(row["predicted_token"]),
                    "competing_token": int(row["competing_token"]),
                    "margin": float(row["margin"]),
                    "batch_size": int(row["batch_size"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            for key, caster in (
                ("entropy", float),
                ("routing_ambiguity_score", float),
                ("routing_ambiguity_layer", int),
                ("routing_boundary_selected", float),
                ("routing_boundary_next", float),
            ):
                if row.get(key) is not None:
                    try:
                        event[key] = caster(row[key])
                    except (TypeError, ValueError):
                        pass
            events.append(event)
    return events


def _policy_hash(rows: list[dict]) -> str:
    tools = _runtime_tools_path()
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import precision_context as pc
    _assert_local_helper(pc, "precision_context")
    return pc.policy_hash(rows)


def _policy_target_n(policy: list[dict], role: str, layer: int) -> int | None:
    for row in policy:
        if row.get("role") == role and int(row.get("layer", -1)) == int(layer):
            return int(row["n"])
    return None


def _adaptive_trigger_indices(events: list[dict], request_count: int) -> list[int]:
    triggered = set()
    for event in events:
        try:
            req = int(event["req"])
            margin = float(event["margin"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= req < int(request_count) and margin <= ADAPTIVE_L26_MARGIN_MAX:
            triggered.add(req)
    return sorted(triggered)


def _adaptive_enabled_from_env() -> bool:
    value = os.environ.get(ADAPTIVE_L26_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_persistent_result(text: str, *, request_id: str, expected_requests: int) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise PersistentSupervisorError("persistent worker returned an empty result")
    head = lines[0].split()
    if len(head) != 5 or head[0] != "BEGLIN_GPU_PERSISTENT_RESULT_V1":
        raise PersistentSupervisorError("invalid persistent response header")
    if head[1] != request_id:
        raise PersistentSupervisorError("persistent response request id mismatch")
    try:
        count = int(head[2])
        finite_logits = int(head[3]) == 1
        engine_wall_ms = float(head[4])
    except ValueError as exc:
        raise PersistentSupervisorError("invalid persistent response header values") from exc
    if count != expected_requests:
        raise PersistentSupervisorError(
            f"persistent response count mismatch: expected={expected_requests} actual={count}"
        )
    responses: list[list[int] | None] = [None] * count
    saw_end = False
    for line in lines[1:]:
        if line == "END":
            saw_end = True
            break
        parts = line.split()
        if len(parts) < 3 or parts[0] != "REQ":
            raise PersistentSupervisorError("invalid persistent response row")
        try:
            idx = int(parts[1]); n = int(parts[2]); tokens = [int(v) for v in parts[3:]]
        except ValueError as exc:
            raise PersistentSupervisorError("invalid persistent response token row") from exc
        if idx < 0 or idx >= count or responses[idx] is not None:
            raise PersistentSupervisorError("persistent response request index is invalid or duplicated")
        if len(tokens) != n:
            raise PersistentSupervisorError("persistent response token count mismatch")
        responses[idx] = tokens
    if not saw_end or any(row is None for row in responses):
        raise PersistentSupervisorError("persistent response is incomplete")
    return {
        "finite_logits": finite_logits,
        "engine_wall_ms": engine_wall_ms,
        "responses": responses,
    }


class PersistentRouteWorker:
    def __init__(self, *, route: dict, root: Path):
        self.route = base.routing.normalize_route(route)
        self.root = Path(root)
        self.queue_dir = self.root / "queue"
        self.requests_dir = self.root / "requests"
        self.promotion_path = self.root / "promotion_nq.txt"
        self.ack_path = self.root / "applied_ack.json"
        self.txn_path = self.root / "txn.cmd"
        self.log_path = self.root / "worker.log"
        self.neartie_path = self.root / "neartie.jsonl"
        self.proc: subprocess.Popen | None = None
        self.log_handle = None
        self.ack: dict | None = None
        self.lock = threading.RLock()
        self.precision_epoch_scheduler = None
        self.precision_closed_loop_engine = None

    @property
    def route_id(self) -> str:
        return str(self.route["route_id"])

    def expected_runtime_policy_hash(self) -> str:
        return str(self.route["policy_hash"])

    def _prepare_files(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        for stale in ("request.txt", "request.processing", "shutdown"):
            try:
                (self.queue_dir / stale).unlink()
            except FileNotFoundError:
                pass
        self.promotion_path.write_text(base._route_policy_line(self.route))
        try:
            self.ack_path.unlink()
        except FileNotFoundError:
            pass
        try:
            self.txn_path.unlink()
        except FileNotFoundError:
            pass
        self.neartie_path.write_text("")

    def _env(self) -> dict[str, str]:
        env = base._minimal_env()
        env.update({
            "QWEN_MOE_GPU_CBATCH_ONLINE": "1",
            "QWEN_MOE_BASE": str(base.MOE_BASE),
            "QWEN_MOE_NEARTIE_CORRECT": "0",
            "QWEN_MOE_CB_SLOTS": "4",
            "QWEN_MOE_CB_REQS": "12",
            "QWEN_MOE_GPU_VALIDATION_REPORT": "1",
            "QWEN_MOE_GPU_APPLIED_ACK": str(self.ack_path),
            "QWEN_MOE_GPU_TXN_FILE": str(self.txn_path),
            "QWEN_MOE_PROMOTION_FILE_NQ": str(self.promotion_path),
            "QWEN_MOE_PROMOTION_SAFETENSORS": str(base.SAFETENSORS),
            "QWEN_MOE_GPU_PERSISTENT_DIR": str(self.queue_dir),
            "QWEN_MOE_NEARTIE_LOG": "1",
            "QWEN_MOE_NEARTIE_THRESHOLD": str(NEARTIE_TELEMETRY_THRESHOLD),
            "QWEN_MOE_NEARTIE_EVENTS_LOG": str(self.neartie_path),
            "QWEN_MOE_NEARTIE_MODEL": "deepseek-v2-lite",
            "QWEN_MOE_NEARTIE_CORPUS": "production-persistent",
            "QWEN_MOE_RISK_SIGNALS": os.environ.get("QWEN_MOE_RISK_SIGNALS", "0"),
        })
        return env

    def start(self) -> dict:
        if self.proc is not None and self.proc.poll() is None:
            return self.health()
        self._prepare_files()
        self.log_handle = self.log_path.open("w")
        self.proc = subprocess.Popen(
            [str(PERSISTENT_BINARY)],
            cwd=base.REPO,
            env=self._env(),
            text=True,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + WORKER_READY_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.log_handle.flush()
                tail = self.log_path.read_text(errors="ignore")[-8000:]
                raise PersistentSupervisorError(
                    f"persistent worker {self.route_id} exited during startup rc={self.proc.returncode}: {tail}"
                )
            if self.log_path.exists() and "persistent admission enabled" in self.log_path.read_text(errors="ignore"):
                if self.ack_path.is_file():
                    ack = _read_runtime_ack(self.ack_path)
                    expected_policy_hash = self.expected_runtime_policy_hash()
                    if ack["active_policy_hash"] != expected_policy_hash:
                        raise PersistentSupervisorError(
                            f"persistent worker {self.route_id} policy hash mismatch: "
                            f"expected={expected_policy_hash} actual={ack['active_policy_hash']}"
                        )
                    self.ack = ack
                    return self.health()
            time.sleep(0.05)
        self.stop(force=True)
        raise PersistentSupervisorError(f"persistent worker {self.route_id} did not become ready")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _rss_bytes(self) -> int:
        if not self.is_alive():
            return 0
        try:
            out = subprocess.check_output(
                ["/bin/ps", "-o", "rss=", "-p", str(self.proc.pid)],
                text=True,
                timeout=5,
            ).strip()
            return int(out) * 1024 if out else 0
        except Exception:
            return 0

    def health(self) -> dict:
        return {
            "route_id": self.route_id,
            "alive": self.is_alive(),
            "pid": int(self.proc.pid) if self.is_alive() else None,
            "policy_hash": self.route["policy_hash"],
            "runtime_policy_hash": (
                self.ack.get("active_policy_hash") if self.ack
                else self.expected_runtime_policy_hash()
            ),
            "runtime_policy": self.ack.get("active_policy", []) if self.ack else [],
            "weight_epoch": int(self.ack["weight_epoch"]) if self.ack else None,
            "ack_sha256": self.ack.get("ack_sha256") if self.ack else None,
            "rss_bytes": self._rss_bytes(),
            "near_tie_telemetry": {
                "enabled": True,
                "threshold": NEARTIE_TELEMETRY_THRESHOLD,
            },
            "precision_epoch_scheduler": {
                "available": True,
                "max_atomic_targets": 8,
                "hot_policy_shape_change": False,
            },
            "precision_closed_loop": {
                "available": True,
                "configured": self.precision_closed_loop_engine is not None,
                "production_write_allowed": False,
            },
        }

    def _epoch_scheduler(self):
        if self.precision_epoch_scheduler is None:
            pes = _load_precision_epoch_scheduler()
            self.precision_epoch_scheduler = pes.PrecisionEpochScheduler(
                ack_path=self.ack_path,
                txn_path=self.txn_path,
                submit_fn=lambda parsed: PersistentRouteWorker.submit(self, parsed),
                admission_lock=self.lock,
            )
        return self.precision_epoch_scheduler

    def configure_precision_closed_loop(
        self,
        *,
        candidates: list[dict],
        trigger_evidence: list[dict],
        combined_policy_evidence: list[dict],
        cost_evidence: dict | None = None,
        memory_weight: float = 1.0,
        latency_weight: float = 0.0,
        rss_weight: float = 0.0,
        transition_weight: float = 0.0,
        cache_weight: float = 0.0,
        inference_pass_weight: float = 0.0,
        e2e_weight: float = 0.0,
        policy_e2e_weight: float = 0.0,
        policy_cache_weight: float = 0.0,
        policy_transition_weight: float = 0.0,
        policy_active_memory_weight: float = 0.0,
    ) -> dict:
        pcl = _load_precision_closed_loop()
        with self.lock:
            self.precision_closed_loop_engine = pcl.PrecisionClosedLoopEngine(
                candidates=candidates,
                trigger_evidence=trigger_evidence,
                combined_policy_evidence=combined_policy_evidence,
                cost_evidence=cost_evidence,
                memory_weight=memory_weight,
                latency_weight=latency_weight,
                rss_weight=rss_weight,
                transition_weight=transition_weight,
                cache_weight=cache_weight,
                inference_pass_weight=inference_pass_weight,
                e2e_weight=e2e_weight,
                policy_e2e_weight=policy_e2e_weight,
                policy_cache_weight=policy_cache_weight,
                policy_transition_weight=policy_transition_weight,
                policy_active_memory_weight=policy_active_memory_weight,
            )
            return {
                "schema": "beglin-precision-closed-loop-config-v1",
                "status": "CONFIGURED",
                "evidence_snapshot_sha256": (
                    self.precision_closed_loop_engine.snapshot_sha256
                ),
                "production_write_allowed": False,
            }

    def submit_with_closed_loop_precision(
        self,
        parsed: list[tuple[list[int], int]],
        *,
        signal: dict,
        admission_id: str | None = None,
    ) -> dict:
        """Allocate, select and apply one exact precision policy for admission."""
        with self.lock:
            engine = self.precision_closed_loop_engine
            if engine is None:
                raise PersistentSupervisorError(
                    "precision closed-loop engine is not configured"
                )
            before = _read_runtime_ack(self.ack_path)
            decision = engine.decide(
                current_policy=before["active_policy"],
                signal=signal,
                runtime_state=before,
            )
            result = self._epoch_scheduler().run(
                parsed,
                target_policy=decision["selected_policy"],
                admission_id=admission_id,
            )
            epoch = result.get("precision_epoch") or {}
            if epoch.get("before_policy_hash") != decision["current_policy_hash"]:
                raise PersistentSupervisorError(
                    "closed-loop scheduler preimage differs from decision preimage"
                )
            if epoch.get("after_policy_hash") != decision["selected_policy_hash"]:
                raise PersistentSupervisorError(
                    "closed-loop scheduler result differs from selected policy"
                )
            self.ack = _read_runtime_ack(self.ack_path)
            out = dict(result)
            out["precision_closed_loop"] = {
                "schema": decision["schema"],
                "status": decision["status"],
                "evidence_snapshot_sha256": decision[
                    "evidence_snapshot_sha256"
                ],
                "allocation_sha256": decision["allocation_sha256"],
                "selection_sha256": decision["selection_sha256"],
                "cost_evidence_sha256": decision.get("cost_evidence_sha256"),
                "objective_weights": decision.get("objective_weights"),
                "policy_cost_optimizer": decision.get("policy_cost_optimizer"),
                "policy_cost_weights": decision.get("policy_cost_weights"),
                "current_policy_hash": decision["current_policy_hash"],
                "selected_policy_hash": decision["selected_policy_hash"],
                "selected_policy": decision["selected_policy"],
                "changes": decision["changes"],
                "combined_policy_evidence": decision[
                    "combined_policy_evidence"
                ],
                "signal": decision["signal"],
                "production_write_allowed": False,
            }
            return out

    def submit_with_precision_policy(
        self,
        parsed: list[tuple[list[int], int]],
        *,
        target_policy: list[dict],
        admission_id: str | None = None,
    ) -> dict:
        """Run one admission under an exact resident-policy epoch.

        v1 can only change n values for targets already materialized at worker
        startup. The scheduler serializes read-CAS-transition-infer-ACK so a
        second caller cannot observe or modify a half-transitioned policy.
        """
        result = self._epoch_scheduler().run(
            parsed,
            target_policy=target_policy,
            admission_id=admission_id,
        )
        self.ack = _read_runtime_ack(self.ack_path)
        return result

    def submit(self, parsed: list[tuple[list[int], int]]) -> dict:
        if not parsed:
            raise PersistentSupervisorError("persistent batch cannot be empty")
        with self.lock:
            if not self.is_alive():
                raise PersistentSupervisorError(f"persistent worker {self.route_id} is not alive")
            request_id = uuid.uuid4().hex
            request_root = self.requests_dir / request_id
            request_root.mkdir(parents=True, exist_ok=False)
            manifest = request_root / "manifest.txt"
            result_path = request_root / "result.txt"
            lines = []
            for idx, (tokens, max_new) in enumerate(parsed):
                raw = request_root / f"req-{idx}.i32"
                base._write_i32(raw, tokens)
                lines.append(f"{raw} {max_new}\n")
            manifest.write_text("".join(lines))
            request_tmp = self.queue_dir / f"request.txt.tmp.{request_id}"
            request_ready = self.queue_dir / "request.txt"
            request_processing = self.queue_dir / "request.processing"
            if request_ready.exists() or request_processing.exists():
                shutil.rmtree(request_root, ignore_errors=True)
                raise PersistentSupervisorError(f"persistent queue {self.route_id} is busy")
            request_tmp.write_text(
                f"BEGLIN_GPU_PERSISTENT_REQUEST_V1 {request_id} {manifest} {result_path}\n"
            )
            neartie_offset = self.neartie_path.stat().st_size if self.neartie_path.exists() else 0
            started = time.monotonic()
            os.replace(request_tmp, request_ready)
            deadline = time.time() + REQUEST_TIMEOUT_SECONDS
            try:
                while time.time() < deadline:
                    if not self.is_alive():
                        raise PersistentSupervisorError(
                            f"persistent worker {self.route_id} died while serving request"
                        )
                    if result_path.is_file():
                        roundtrip_ms = (time.monotonic() - started) * 1000.0
                        parsed_result = parse_persistent_result(
                            result_path.read_text(),
                            request_id=request_id,
                            expected_requests=len(parsed),
                        )
                        parsed_result["roundtrip_ms"] = roundtrip_ms
                        parsed_result["inference_passes"] = 1
                        parsed_result["neartie_events"] = _read_neartie_events_since(
                            self.neartie_path, neartie_offset
                        )
                        return parsed_result
                    time.sleep(POLL_SECONDS)
                raise PersistentSupervisorError(
                    f"persistent worker {self.route_id} request timed out"
                )
            finally:
                try:
                    request_tmp.unlink()
                except FileNotFoundError:
                    pass
                if result_path.exists():
                    shutil.rmtree(request_root, ignore_errors=True)

    def stop(self, *, force: bool = False) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None and not force:
            try:
                (self.queue_dir / "shutdown").write_text("")
                proc.wait(timeout=15)
            except Exception:
                force = True
        if proc.poll() is None and force:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except Exception:
                pass
        self.proc = None
        self.log_handle = None


class AdaptivePersistentRouteWorker(PersistentRouteWorker):
    """Candidate worker with fail-closed L26 low-cost base + n6 recovery."""

    def __init__(self, *, route: dict, root: Path):
        super().__init__(route=route, root=root)
        reviewed = base.candidate_route()
        if (
            self.route_id != reviewed["route_id"]
            or self.route.get("role") != reviewed.get("role")
            or int(self.route.get("layer")) != int(reviewed.get("layer"))
            or int(self.route.get("n")) != int(reviewed.get("n"))
        ):
            raise PersistentSupervisorError(
                "adaptive L26 worker is allowed only on the reviewed candidate route"
            )
        self.startup_policy = [
            {
                "role": str(self.route["role"]),
                "layer": int(self.route["layer"]),
                "n": int(self.route["n"]),
            },
            {
                "role": ADAPTIVE_L26_ROLE,
                "layer": ADAPTIVE_L26_LAYER,
                "n": ADAPTIVE_L26_BASE_N,
            },
        ]
        self.startup_policy_hash = _policy_hash(self.startup_policy)
        if self.startup_policy_hash != ADAPTIVE_L26_STARTUP_POLICY_SHA256:
            raise PersistentSupervisorError(
                "adaptive L26 startup policy identity mismatch: "
                f"expected={ADAPTIVE_L26_STARTUP_POLICY_SHA256} "
                f"actual={self.startup_policy_hash}"
            )

    def expected_runtime_policy_hash(self) -> str:
        return self.startup_policy_hash

    def _prepare_files(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        for stale in ("request.txt", "request.processing", "shutdown"):
            try:
                (self.queue_dir / stale).unlink()
            except FileNotFoundError:
                pass
        self.promotion_path.write_text(
            base._route_policy_line(self.route)
            + f"{ADAPTIVE_L26_ROLE} {ADAPTIVE_L26_LAYER} {ADAPTIVE_L26_BASE_N}\n"
        )
        for path in (self.ack_path, self.txn_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self.neartie_path.write_text("")

    def health(self) -> dict:
        row = super().health()
        row["adaptive_precision"] = {
            "enabled": True,
            "role": ADAPTIVE_L26_ROLE,
            "layer": ADAPTIVE_L26_LAYER,
            "base_n": ADAPTIVE_L26_BASE_N,
            "recovery_n": ADAPTIVE_L26_RECOVERY_N,
            "margin_max": ADAPTIVE_L26_MARGIN_MAX,
            "evidence_sha256": ADAPTIVE_L26_EVIDENCE_SHA256,
            "rebind_evidence_sha256": ADAPTIVE_L26_REBIND_EVIDENCE_SHA256,
        }
        return row

    def _current_l26_n(self) -> int:
        ack = _read_runtime_ack(self.ack_path)
        self.ack = ack
        n = _policy_target_n(
            ack["active_policy"], ADAPTIVE_L26_ROLE, ADAPTIVE_L26_LAYER
        )
        if n not in {ADAPTIVE_L26_BASE_N, ADAPTIVE_L26_RECOVERY_N}:
            raise PersistentSupervisorError(
                f"adaptive L26 runtime is in unexpected state n={n}"
            )
        return int(n)

    def _prepare_rebind(self, *, expected_n: int, target_n: int, txn_id: str) -> None:
        grc = _load_gpu_runtime_control()
        ack = _read_runtime_ack(self.ack_path)
        self.ack = ack
        grc.prepare_rebind(
            ack_path=self.ack_path,
            txn_path=self.txn_path,
            txn_id=txn_id,
            expected_epoch=int(ack["weight_epoch"]),
            expected_policy_hash=str(ack["active_policy_hash"]),
            role=ADAPTIVE_L26_ROLE,
            layer=ADAPTIVE_L26_LAYER,
            expected_n=int(expected_n),
            target_n=int(target_n),
        )

    def _verify_rebind(self, *, txn_id: str, target_n: int) -> dict:
        grc = _load_gpu_runtime_control()
        ack = grc.verify_terminal_ack(
            ack_path=self.ack_path,
            txn_id=txn_id,
            allowed_statuses={"REBIND_APPLIED"},
        )
        actual_n = _policy_target_n(
            ack["active_policy"], ADAPTIVE_L26_ROLE, ADAPTIVE_L26_LAYER
        )
        if actual_n != int(target_n):
            raise PersistentSupervisorError(
                f"adaptive L26 rebind ACK target mismatch: expected={target_n} actual={actual_n}"
            )
        self.ack = ack
        return ack

    def submit_base(self, parsed: list[tuple[list[int], int]]) -> dict:
        """Bypass adaptive rerun; used only for startup prewarm evidence."""
        return super().submit(parsed)

    def submit(self, parsed: list[tuple[list[int], int]]) -> dict:
        if not parsed:
            raise PersistentSupervisorError("adaptive persistent batch cannot be empty")

        # A triggered request is intentionally left at n6.  The next request
        # itself applies 6->5 at the quiescent boundary before computing its
        # first/base pass, avoiding a third inference solely for restoration.
        restored_from_n6 = False
        if self._current_l26_n() == ADAPTIVE_L26_RECOVERY_N:
            txn_id = f"adaptive-restore-{uuid.uuid4().hex}"
            self._prepare_rebind(
                expected_n=ADAPTIVE_L26_RECOVERY_N,
                target_n=ADAPTIVE_L26_BASE_N,
                txn_id=txn_id,
            )
            first = super().submit(parsed)
            self._verify_rebind(txn_id=txn_id, target_n=ADAPTIVE_L26_BASE_N)
            restored_from_n6 = True
        else:
            first = super().submit(parsed)

        trigger_indices = _adaptive_trigger_indices(
            first.get("neartie_events", []), len(parsed)
        )
        if not trigger_indices:
            first["adaptive_precision"] = {
                "enabled": True,
                "action": "BASE_N5",
                "restored_from_n6": restored_from_n6,
                "role": ADAPTIVE_L26_ROLE,
                "layer": ADAPTIVE_L26_LAYER,
                "base_n": ADAPTIVE_L26_BASE_N,
                "recovery_n": ADAPTIVE_L26_RECOVERY_N,
                "trigger_request_indices": [],
                "worker_left_at_n": ADAPTIVE_L26_BASE_N,
                "evidence_sha256": ADAPTIVE_L26_EVIDENCE_SHA256,
                "rebind_evidence_sha256": ADAPTIVE_L26_REBIND_EVIDENCE_SHA256,
            }
            return first

        txn_id = f"adaptive-recover-{uuid.uuid4().hex}"
        self._prepare_rebind(
            expected_n=ADAPTIVE_L26_BASE_N,
            target_n=ADAPTIVE_L26_RECOVERY_N,
            txn_id=txn_id,
        )
        recovery_parsed = [parsed[idx] for idx in trigger_indices]
        recovery = super().submit(recovery_parsed)
        self._verify_rebind(txn_id=txn_id, target_n=ADAPTIVE_L26_RECOVERY_N)

        merged_responses = [list(tokens) for tokens in first["responses"]]
        for local_idx, original_idx in enumerate(trigger_indices):
            merged_responses[original_idx] = recovery["responses"][local_idx]

        base_events = []
        for event in first.get("neartie_events", []):
            row = dict(event)
            row["phase"] = "base_n5"
            base_events.append(row)
        recovery_events = []
        for event in recovery.get("neartie_events", []):
            row = dict(event)
            local_req = int(row.get("req", -1))
            if 0 <= local_req < len(trigger_indices):
                row["req"] = trigger_indices[local_req]
            row["phase"] = "recovery_n6"
            recovery_events.append(row)

        return {
            **first,
            "finite_logits": bool(first["finite_logits"] and recovery["finite_logits"]),
            "engine_wall_ms": float(first["engine_wall_ms"]) + float(recovery["engine_wall_ms"]),
            "roundtrip_ms": float(first["roundtrip_ms"]) + float(recovery["roundtrip_ms"]),
            "inference_passes": int(first.get("inference_passes", 1)) + int(recovery.get("inference_passes", 1)),
            "responses": merged_responses,
            "neartie_events": base_events + recovery_events,
            "adaptive_precision": {
                "enabled": True,
                "action": "RECOVERY_N6",
                "restored_from_n6": restored_from_n6,
                "role": ADAPTIVE_L26_ROLE,
                "layer": ADAPTIVE_L26_LAYER,
                "base_n": ADAPTIVE_L26_BASE_N,
                "recovery_n": ADAPTIVE_L26_RECOVERY_N,
                "trigger_request_indices": trigger_indices,
                "base_event_count": len(base_events),
                "recovery_event_count": len(recovery_events),
                "worker_left_at_n": ADAPTIVE_L26_RECOVERY_N,
                "evidence_sha256": ADAPTIVE_L26_EVIDENCE_SHA256,
            },
        }


class PersistentWorkerPool:
    def __init__(
        self,
        root: Path = DEFAULT_PERSISTENT_ROOT,
        *,
        adaptive_l26: bool = False,
    ):
        self.root = Path(root)
        self.adaptive_l26 = bool(adaptive_l26)
        candidate_cls = (
            AdaptivePersistentRouteWorker if self.adaptive_l26
            else PersistentRouteWorker
        )
        self.workers = {
            base.baseline_route()["route_id"]: PersistentRouteWorker(
                route=base.baseline_route(), root=self.root / "baseline"
            ),
            base.candidate_route()["route_id"]: candidate_cls(
                route=base.candidate_route(), root=self.root / "candidate"
            ),
        }
        self.prewarm_evidence: dict[str, Any] = {}

    def get(self, route: dict) -> PersistentRouteWorker:
        rid = str(route["route_id"])
        worker = self.workers.get(rid)
        if worker is None:
            raise PersistentSupervisorError(f"route is not backed by a persistent worker: {rid}")
        if route["policy_hash"] != worker.route["policy_hash"]:
            raise PersistentSupervisorError("route policy differs from persistent worker policy")
        return worker

    @staticmethod
    def _reference_token(route: dict) -> int:
        return 3268 if route.get("n") is None else 1224

    def _prewarm(self, worker: PersistentRouteWorker) -> dict:
        prompt = base._read_first_certified_prompt()
        batch = [(prompt, 10) for _ in range(12)]
        # Prewarm must validate the same end-to-end output users receive.
        # An adaptive worker may intentionally produce a wrong low-cost base pass
        # and recover only triggered requests at n6, so bypassing adaptive recovery
        # here would reject a valid fail-closed policy during startup.
        result = worker.submit(batch)
        if not result["finite_logits"]:
            raise PersistentSupervisorError(f"prewarm non-finite logits for {worker.route_id}")
        expected = self._reference_token(worker.route)
        hits = sum(
            len(tokens) > 8 and tokens[8] == expected
            for tokens in result["responses"]
        )
        if hits != 12:
            raise PersistentSupervisorError(
                f"prewarm reference mismatch for {worker.route_id}: {hits}/12"
            )
        return {
            "route_id": worker.route_id,
            "reference_token": expected,
            "reference_hits": hits,
            "engine_wall_ms": result["engine_wall_ms"],
            "roundtrip_ms": result["roundtrip_ms"],
            "pid": worker.health()["pid"],
        }

    def start_all(self) -> dict:
        verify_persistent_artifact()
        started = []
        try:
            for worker in self.workers.values():
                worker.start()
                started.append(worker)
            for worker in self.workers.values():
                self.prewarm_evidence[worker.route_id] = self._prewarm(worker)
            return self.health()
        except Exception:
            for worker in reversed(started):
                worker.stop(force=True)
            raise

    def health(self) -> dict:
        rows = {rid: worker.health() for rid, worker in self.workers.items()}
        return {
            "all_alive": all(row["alive"] for row in rows.values()),
            "workers": rows,
            "combined_rss_bytes": sum(int(row["rss_bytes"] or 0) for row in rows.values()),
            "prewarm": self.prewarm_evidence,
            "adaptive_l26_enabled": self.adaptive_l26,
        }

    def stop_all(self) -> None:
        for worker in self.workers.values():
            worker.stop()


class PersistentEngineExecutor:
    def __init__(self, route_manifest: str | Path, pool: PersistentWorkerPool):
        self.route_manifest = Path(route_manifest)
        self.store = base.routing.AtomicRouteManifestStore(self.route_manifest)
        self.pool = pool
        self.gpu_lock = threading.BoundedSemaphore(1)

    def shutdown(self) -> None:
        self.pool.stop_all()

    def read_route_snapshot(self) -> dict:
        manifest = self.store.read()
        route = base.routing.normalize_route(manifest["active_route"])
        base._route_policy_line(route)
        worker = self.pool.get(route)
        if not worker.is_alive():
            raise PersistentSupervisorError(f"active persistent worker is down: {route['route_id']}")
        return {
            "generation": int(manifest["generation"]),
            "manifest_sha256": str(manifest["manifest_sha256"]),
            "route": route,
        }

    def generate_batch(self, requests: list[dict]) -> dict:
        if not isinstance(requests, list) or not requests:
            raise PersistentSupervisorError("requests must be a non-empty list")
        if len(requests) > base.MAX_BATCH_REQUESTS:
            raise PersistentSupervisorError("batch exceeds supervisor request limit")
        parsed = [base._validate_request(row) for row in requests]
        snapshot = self.read_route_snapshot()
        if not self.gpu_lock.acquire(blocking=False):
            raise PersistentSupervisorError("GPU worker is busy")
        started = time.monotonic()
        try:
            worker = self.pool.get(snapshot["route"])
            result = worker.submit(parsed)
            if not result["finite_logits"]:
                raise PersistentSupervisorError("persistent worker reported non-finite logits")
            h = worker.health()
            return {
                "schema": "beglin-supervisor-persistent-batch-response-v1",
                "status": "OK",
                "route_generation": snapshot["generation"],
                "route_manifest_sha256": snapshot["manifest_sha256"],
                "route": snapshot["route"],
                "worker_instance_id": f"pid-{h['pid']}",
                "worker_ack_sha256": h["ack_sha256"],
                "neartie_events": result.get("neartie_events", []),
                "neartie_event_count": len(result.get("neartie_events", [])),
                "adaptive_precision": result.get(
                    "adaptive_precision", {"enabled": False}
                ),
                "worker_epoch": h["weight_epoch"],
                "finite_logits": True,
                "duration_ms": max(1, int((time.monotonic() - started) * 1000)),
                "engine_wall_ms": result["engine_wall_ms"],
                "peak_child_rss_bytes": h["rss_bytes"],
                "responses": [
                    {"request_index": idx, "generated_tokens": tokens}
                    for idx, tokens in enumerate(result["responses"])
                ],
                "persistent_worker": True,
                "production_write_allowed": False,
                "auto_promotion_enabled": False,
            }
        finally:
            self.gpu_lock.release()


class PersistentHTTPServer(base.SupervisorHTTPServer):
    def __init__(self, server_address, handler_cls, executor: PersistentEngineExecutor, pool: PersistentWorkerPool):
        super().__init__(server_address, handler_cls, executor)
        self.pool = pool


class PersistentHandler(base.Handler):
    server_version = "BeglinPersistentSupervisor/1"

    def do_GET(self) -> None:
        if self.path != "/healthz":
            return super().do_GET()
        try:
            snap = self.server.executor.read_route_snapshot()
            pool = self.server.pool.health()
            status = HTTPStatus.OK if pool["all_alive"] else HTTPStatus.SERVICE_UNAVAILABLE
            self._send(status, {
                "schema": "beglin-supervisor-persistent-health-v1",
                "status": "ok" if pool["all_alive"] else "degraded",
                "router_id": "beglin-local-http-persistent-supervisor-v1",
                "listen_host": self.server.server_address[0],
                "listen_port": self.server.server_address[1],
                "route_generation": snap["generation"],
                "active_route": snap["route"],
                "route_manifest_sha256": snap["manifest_sha256"],
                "workers": pool["workers"],
                "combined_worker_rss_bytes": pool["combined_rss_bytes"],
                "prewarm": pool["prewarm"],
                "uptime_seconds": int(time.time() - self.server.started_at),
                "persistent_workers": True,
                "adaptive_precision_enabled": pool["adaptive_l26_enabled"],
                "external_network_exposed": False,
                "production_write_allowed": False,
                "auto_promotion_enabled": False,
            })
        except Exception as exc:
            self._send(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": type(exc).__name__, "message": str(exc)},
            )


def make_server(
    *,
    route_manifest: str | Path,
    persistent_root: str | Path = DEFAULT_PERSISTENT_ROOT,
    host: str = "127.0.0.1",
    port: int = base.DEFAULT_PORT,
) -> PersistentHTTPServer:
    if host != "127.0.0.1":
        raise PersistentSupervisorError("persistent supervisor refuses non-loopback binding")
    verify_persistent_artifact()
    adaptive_l26 = _adaptive_enabled_from_env()
    if adaptive_l26:
        verify_adaptive_l26_acceptance()
    manifest = Path(route_manifest)
    base.ensure_route_manifest(manifest)
    pool = PersistentWorkerPool(
        Path(persistent_root),
        adaptive_l26=adaptive_l26,
    )
    pool.start_all()
    try:
        executor = PersistentEngineExecutor(manifest, pool)
        return PersistentHTTPServer((host, int(port)), PersistentHandler, executor, pool)
    except Exception:
        pool.stop_all()
        raise


def _batch_reference_probe(port: int) -> dict:
    tokens = base._read_first_certified_prompt()
    result = base._post_json(
        port,
        "/v1/batch_generate",
        {"requests": [{"prompt_tokens": tokens, "max_new_tokens": 10} for _ in range(12)]},
    )
    values = [
        row["generated_tokens"][8] if len(row["generated_tokens"]) > 8 else None
        for row in result["responses"]
    ]
    expected = 3268 if result["route"].get("n") is None else 1224
    hits = sum(value == expected for value in values)
    return {
        "schema": "beglin-persistent-supervisor-batch-reference-probe-v1",
        "status": "PASS" if hits == 12 and result["finite_logits"] else "FAIL",
        "route_generation": result["route_generation"],
        "route_id": result["route"]["route_id"],
        "policy_hash": result["route"]["policy_hash"],
        "expected_reference_token": expected,
        "reference_hits": hits,
        "requests": len(result["responses"]),
        "finite_logits": result["finite_logits"],
        "duration_ms": result["duration_ms"],
        "engine_wall_ms": result["engine_wall_ms"],
        "worker_instance_id": result["worker_instance_id"],
        "persistent_worker": True,
    }


def self_test() -> dict:
    verify_persistent_artifact()
    import tempfile
    with tempfile.TemporaryDirectory(prefix="beglin-persistent-supervisor-selftest-") as td:
        root = Path(td)
        route_path = root / "active-route.json"
        store = base.routing.AtomicRouteManifestStore(route_path)
        initial = store.initialize(base.baseline_route())
        server = make_server(
            route_manifest=route_path,
            persistent_root=root / "workers",
            port=0,
        )
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            baseline = _batch_reference_probe(port)
            if baseline["status"] != "PASS" or baseline["expected_reference_token"] != 3268:
                raise PersistentSupervisorError("persistent baseline self-test failed")
            switched = store.compare_and_swap(
                expected_generation=int(initial["generation"]),
                expected_route_id=base.baseline_route()["route_id"],
                new_route=base.candidate_route(),
                reason="persistent supervisor self-test candidate",
                authorization_digest="a" * 64,
            )
            candidate = _batch_reference_probe(port)
            if candidate["status"] != "PASS" or candidate["expected_reference_token"] != 1224:
                raise PersistentSupervisorError("persistent candidate self-test failed")
            rolled = store.compare_and_swap(
                expected_generation=int(switched["generation"]),
                expected_route_id=base.candidate_route()["route_id"],
                new_route=base.baseline_route(),
                reason="persistent supervisor self-test rollback",
                authorization_digest="b" * 64,
            )
            restored = _batch_reference_probe(port)
            if restored["status"] != "PASS" or restored["expected_reference_token"] != 3268:
                raise PersistentSupervisorError("persistent rollback self-test failed")
            health = base._get_json(port, "/healthz")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            server.pool.stop_all()
        out = {
            "schema": "beglin-persistent-serving-supervisor-selftest-v1",
            "status": "PASS",
            "route_generation_path": [
                int(initial["generation"]),
                int(switched["generation"]),
                int(rolled["generation"]),
            ],
            "baseline": baseline,
            "candidate": candidate,
            "rollback": restored,
            "health": health,
            "external_network_exposed": False,
            "production_write_allowed": False,
            "auto_promotion_enabled": False,
        }
        out["result_sha256"] = base.routing.sha256_json(out)
        return out


def install_user_launchd(
    *,
    script_path: Path,
    route_manifest: Path,
    port: int,
) -> dict:
    state_root = route_manifest.parent
    state_root.mkdir(parents=True, exist_ok=True)
    base.ensure_route_manifest(route_manifest)
    logs = state_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    label = "com.beglin.production-supervisor"
    plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    program = str(Path(sys.executable).resolve())
    script = str(script_path.resolve(strict=True))
    route = str(route_manifest.resolve())
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>{label}</string>
<key>ProgramArguments</key><array>
<string>{program}</string>
<string>{script}</string>
<string>--serve</string>
<string>--route-manifest</string><string>{route}</string>
<string>--port</string><string>{int(port)}</string>
</array>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
<key>ProcessType</key><string>{LAUNCHD_PROCESS_TYPE}</string>
<key>StandardOutPath</key><string>{logs / "supervisor.stdout.log"}</string>
<key>StandardErrorPath</key><string>{logs / "supervisor.stderr.log"}</string>
</dict></plist>
"""
    tmp = plist.with_name(plist.name + f".tmp.{os.getpid()}")
    tmp.write_text(xml)
    os.replace(tmp, plist)
    uid = os.getuid()
    domain = f"gui/{uid}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    proc = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise PersistentSupervisorError(f"launchctl bootstrap failed: {proc.stderr.strip()}")
    return {
        "schema": "beglin-persistent-supervisor-launchd-install-v1",
        "status": "INSTALLED",
        "label": label,
        "plist": str(plist),
        "python": program,
        "script": script,
        "route_manifest": route,
        "listen_host": "127.0.0.1",
        "listen_port": int(port),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--install-user-launchd", action="store_true")
    mode.add_argument("--probe-health", action="store_true")
    mode.add_argument("--probe-batch-reference", action="store_true")
    ap.add_argument("--route-manifest", default=str(base.DEFAULT_ROUTE_MANIFEST))
    ap.add_argument("--port", type=int, default=base.DEFAULT_PORT)
    args = ap.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    if args.probe_health:
        print(json.dumps(base._get_json(args.port, "/healthz"), indent=2, sort_keys=True))
        return 0
    if args.probe_batch_reference:
        result = _batch_reference_probe(args.port)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.install_user_launchd:
        result = install_user_launchd(
            script_path=Path(__file__),
            route_manifest=Path(args.route_manifest),
            port=args.port,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    server = make_server(
        route_manifest=args.route_manifest,
        host="127.0.0.1",
        port=args.port,
    )
    stop_once = threading.Event()
    def request_shutdown(signum, frame):
        if stop_once.is_set():
            return
        stop_once.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    print(json.dumps({
        "schema": "beglin-persistent-serving-supervisor-start-v1",
        "status": "SERVING",
        "listen_host": "127.0.0.1",
        "listen_port": int(server.server_address[1]),
        "workers": server.pool.health(),
        "external_network_exposed": False,
        "production_write_allowed": False,
        "auto_promotion_enabled": False,
    }, sort_keys=True), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        server.pool.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
