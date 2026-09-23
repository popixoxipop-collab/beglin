#!/usr/bin/env python3
"""Backend adapter contract for backend-scoped precision control.

The MLX/Metal adapter is intentionally fail-closed until Apple-Silicon runtime
validation completes. MockBackendAdapter exists for state-machine tests.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import precision_context as pc


class BackendError(RuntimeError):
    pass


class BackendUnverified(BackendError):
    pass


class StaleCommand(BackendError):
    pass


@dataclass(frozen=True)
class AppliedState:
    backend: str
    epoch: int
    policy: list[dict]
    policy_hash: str
    admission_paused: bool
    active_requests: int


class BackendAdapter:
    name = "abstract"

    def collect_context(self) -> pc.ExecutionContext:
        raise NotImplementedError

    def query_applied_state(self) -> AppliedState:
        raise NotImplementedError

    def pause_admission(self) -> None:
        raise NotImplementedError

    def drain(self) -> None:
        raise NotImplementedError

    def synchronize(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> Any:
        raise NotImplementedError

    def apply_policy(self, policy: list[dict]) -> AppliedState:
        raise NotImplementedError

    def restore(self, snapshot: Any) -> AppliedState:
        raise NotImplementedError

    def resume_admission(self) -> None:
        raise NotImplementedError

    def verify_policy(self, expected_hash: str) -> AppliedState:
        state = self.query_applied_state()
        if state.policy_hash != expected_hash:
            raise BackendError(
                f"applied policy mismatch: expected={expected_hash} "
                f"actual={state.policy_hash}"
            )
        return state


class MlxMetalBackendAdapter(BackendAdapter):
    name = "mlx_metal"

    def __init__(self, *, verified: bool = False):
        self.verified = bool(verified)

    def _deny(self):
        raise BackendUnverified(
            "mlx_metal runtime mutation is IMPLEMENTED_UNVERIFIED; "
            "Apple-Silicon transition tests must pass before enabling it"
        )

    def collect_context(self):
        self._deny()

    def query_applied_state(self):
        self._deny()

    def pause_admission(self):
        self._deny()

    def drain(self):
        self._deny()

    def synchronize(self):
        self._deny()

    def snapshot(self):
        self._deny()

    def apply_policy(self, policy):
        self._deny()

    def restore(self, snapshot):
        self._deny()

    def resume_admission(self):
        self._deny()


class MockBackendAdapter(BackendAdapter):
    """Deterministic backend used to validate G2/G3 control semantics."""

    def __init__(
        self,
        backend: str = "mlx_metal",
        policy: list[dict] | None = None,
        epoch: int = 0,
        active_requests: int = 0,
        context: pc.ExecutionContext | None = None,
    ):
        if backend not in pc.SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported mock backend {backend}")
        self.name = backend
        self._policy = pc.normalize_policy(policy or [])
        self._epoch = int(epoch)
        self._active_requests = int(active_requests)
        self._paused = False
        self._synced = False
        self._fail_apply = False
        self._fail_restore = False
        self._fail_sync = False
        self._context = context

    def inject_failure(self, *, apply=False, restore=False, sync=False):
        self._fail_apply = bool(apply)
        self._fail_restore = bool(restore)
        self._fail_sync = bool(sync)

    def collect_context(self):
        if self._context is None:
            raise BackendError("mock execution context not configured")
        return self._context

    def query_applied_state(self):
        return AppliedState(
            backend=self.name,
            epoch=self._epoch,
            policy=copy.deepcopy(self._policy),
            policy_hash=pc.policy_hash(self._policy),
            admission_paused=self._paused,
            active_requests=self._active_requests,
        )

    def pause_admission(self):
        self._paused = True

    def drain(self):
        if not self._paused:
            raise BackendError("cannot drain before admission pause")
        self._active_requests = 0

    def synchronize(self):
        if not self._paused or self._active_requests:
            raise BackendError("cannot synchronize before drain")
        if self._fail_sync:
            raise BackendError("injected synchronize failure")
        self._synced = True

    def snapshot(self):
        if not self._paused or self._active_requests or not self._synced:
            raise BackendError("snapshot requires paused, drained, synchronized backend")
        return {
            "policy": copy.deepcopy(self._policy),
            "epoch": self._epoch,
        }

    def apply_policy(self, policy):
        if not self._paused or self._active_requests or not self._synced:
            raise BackendError("apply requires paused, drained, synchronized backend")
        if self._fail_apply:
            raise BackendError("injected apply failure")
        self._policy = pc.normalize_policy(policy)
        self._epoch += 1
        self._synced = False
        return self.query_applied_state()

    def restore(self, snapshot):
        if not self._paused or self._active_requests:
            raise BackendError("restore requires paused and drained backend")
        if self._fail_restore:
            raise BackendError("injected restore failure")
        self._policy = pc.normalize_policy(snapshot["policy"])
        self._epoch += 1
        self._synced = False
        return self.query_applied_state()

    def resume_admission(self):
        if self._active_requests:
            raise BackendError("cannot resume with active requests from old epoch")
        self._paused = False
        self._synced = False


def backend_from_name(name: str, *, allow_unverified_gpu: bool = False):
    if name == "mlx_metal":
        return MlxMetalBackendAdapter(verified=allow_unverified_gpu)
    raise BackendUnverified(
        f"no generic mutable adapter registered for backend={name!r}; "
        "existing CPU production path remains on its proven controller"
    )
