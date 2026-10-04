#!/usr/bin/env python3
"""P12 isolated execution adapters for the symmetric backend v2 contract.

The planning layer in backend_adapters_v2.py is backend-neutral. This module
consumes those plans using the existing proven backend mechanisms while
returning one common result schema.

Safety boundary:
- MLX file execution is isolated-only in P12 v1.
- Known production persistent-worker paths are rejected.
- production_write_allowed stays false.
- VALIDATION_REQUIRED plans cannot execute.
- CPU precision changes execute only through an injected restart callback.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid

import backend_adapters_v2 as planv2
import gpu_runtime_control as grc
import model_capability as mc
import precision_context as pc


class BackendExecutionError(RuntimeError):
    pass


class RestartRequired(BackendExecutionError):
    pass


_PRODUCTION_PATH_PARTS = (
    "/vdsp_serving/persistent-workers/",
    "/vdsp_serving/active-route.json",
)


def _is_production_path(path: str | Path) -> bool:
    text = str(Path(path).expanduser().resolve())
    return any(marker in text for marker in _PRODUCTION_PATH_PARTS)


def _verify_plan(plan: Mapping[str, Any], *, backend: str, bundle_sha256: str) -> dict:
    value = dict(plan)
    if value.get("schema") != "beglin-backend-transition-plan-v1":
        raise BackendExecutionError("unsupported transition plan schema")
    if value.get("backend") != backend:
        raise BackendExecutionError(
            f"transition backend mismatch: expected={backend} actual={value.get('backend')}"
        )
    if value.get("model_capability_bundle_sha256") != bundle_sha256:
        raise BackendExecutionError("transition plan uses stale model capability bundle")
    if value.get("production_write_allowed") is not False:
        raise BackendExecutionError("transition plan unexpectedly permits production write")
    if value.get("automatic_live_promotion") is not False:
        raise BackendExecutionError("transition plan unexpectedly enables auto promotion")
    got = str(value.get("transition_plan_sha256") or "")
    actual = mc.stable_identity_sha256(value)
    if got != actual:
        raise BackendExecutionError(
            f"transition plan SHA mismatch: expected={got} actual={actual}"
        )
    return value


def _validated_bundle(value: Mapping[str, Any], *, backend: str) -> dict:
    bundle = copy.deepcopy(dict(value))
    if bundle.get("schema") != "beglin-model-capability-bundle-v1":
        raise BackendExecutionError("invalid model capability bundle")
    got = str(bundle.get("bundle_sha256") or "")
    actual = mc.stable_identity_sha256(bundle)
    if got != actual:
        raise BackendExecutionError(
            f"model capability bundle hash mismatch: expected={got} actual={actual}"
        )
    # Construction is also a capability-shape validation for this backend.
    planv2.adapter_v2(backend, bundle)
    return bundle


def _revalidate_capability_plan(
    plan: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    before: Mapping[str, Any],
) -> None:
    backend = str(plan["backend"])
    adapter = planv2.adapter_v2(backend, bundle)
    bindings = {}
    for row in plan.get("changes") or []:
        key = (str(row["role"]), int(row["layer"]))
        target_key = str(row.get("target_key") or "")
        if not target_key:
            raise BackendExecutionError(
                f"transition change missing target_key for {key}"
            )
        previous = bindings.get(key)
        if previous is not None and previous != target_key:
            raise BackendExecutionError(
                f"ambiguous target-key binding for {key}"
            )
        bindings[key] = target_key

    state = planv2.BackendStateV2.build(
        backend=backend,
        epoch=int(before["epoch"]),
        policy=list(before["policy"]),
        model_capability_bundle_sha256=str(bundle["bundle_sha256"]),
    )
    try:
        canonical = adapter.plan_transition(
            state=state,
            target_policy=list(plan["target_policy"]),
            target_keys=bindings,
        )
    except planv2.BackendV2Error as exc:
        raise BackendExecutionError(
            f"transition plan capability revalidation failed: {exc}"
        ) from exc
    if canonical["transition_plan_sha256"] != plan["transition_plan_sha256"]:
        raise BackendExecutionError(
            "transition plan differs from capability-derived canonical plan"
        )


def _normalize_state(value: Mapping[str, Any], *, backend: str) -> dict:
    try:
        epoch = int(value["epoch"])
        policy = pc.normalize_policy(value["policy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendExecutionError("backend state is incomplete") from exc
    policy_hash = str(value.get("policy_hash") or pc.policy_hash(policy))
    if policy_hash != pc.policy_hash(policy):
        raise BackendExecutionError("backend state policy hash mismatch")
    return {
        "backend": backend,
        "epoch": epoch,
        "policy": policy,
        "policy_hash": policy_hash,
    }


def _result(
    *,
    backend: str,
    plan: Mapping[str, Any],
    status: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    transitioned: bool,
    validation: Mapping[str, Any] | None = None,
    rollback: Mapping[str, Any] | None = None,
    transaction_id: str | None = None,
) -> dict:
    out = {
        "schema": "beglin-backend-transition-result-v1",
        "status": status,
        "backend": backend,
        "model_capability_bundle_sha256": plan["model_capability_bundle_sha256"],
        "transition_plan_sha256": plan["transition_plan_sha256"],
        "action": plan["action"],
        "before": dict(before),
        "after": dict(after),
        "transitioned": bool(transitioned),
        "transaction_id": transaction_id,
        "validation": dict(validation) if validation is not None else None,
        "rollback": dict(rollback) if rollback is not None else None,
        "production_write_allowed": False,
        "automatic_live_promotion": False,
    }
    out["result_sha256"] = mc.stable_identity_sha256(out)
    return out


class CpuRestartExecutionAdapterV2:
    """Execute CPU plans through an existing restart controller callback."""

    backend = "cpu"

    def __init__(
        self,
        *,
        model_capability_bundle: Mapping[str, Any],
        query_state: Callable[[], Mapping[str, Any]],
        restart: Callable[[list[dict]], Mapping[str, Any]],
        validate: Callable[[], Mapping[str, Any]] | None = None,
        rollback: Callable[[list[dict]], Mapping[str, Any]] | None = None,
    ):
        self.bundle = _validated_bundle(
            model_capability_bundle, backend=self.backend
        )
        self.bundle_sha = str(self.bundle["bundle_sha256"])
        self.query_state_fn = query_state
        self.restart_fn = restart
        self.validate_fn = validate
        self.rollback_fn = rollback

    def query_state(self) -> dict:
        return _normalize_state(self.query_state_fn(), backend=self.backend)

    def execute(self, plan: Mapping[str, Any]) -> dict:
        plan = _verify_plan(plan, backend=self.backend, bundle_sha256=self.bundle_sha)
        before = self.query_state()
        if before["epoch"] != int(plan["expected_epoch"]):
            raise BackendExecutionError("CPU runtime epoch drifted")
        if before["policy_hash"] != plan["expected_policy_hash"]:
            raise BackendExecutionError("CPU runtime policy preimage drifted")
        _revalidate_capability_plan(
            plan, bundle=self.bundle, before=before
        )

        action = str(plan["action"])
        if action == "NOOP":
            validation = self.validate_fn() if self.validate_fn else None
            return _result(
                backend=self.backend, plan=plan, status="NOOP_VERIFIED",
                before=before, after=before, transitioned=False,
                validation=validation,
            )
        if action != "RESTART_REQUIRED":
            raise BackendExecutionError(
                f"CPU execution refuses non-restart transition action={action}"
            )

        transitioned = False
        try:
            after = _normalize_state(
                self.restart_fn(copy.deepcopy(plan["target_policy"])),
                backend=self.backend,
            )
            transitioned = True
            if after["policy_hash"] != plan["target_policy_hash"]:
                raise BackendExecutionError("CPU restart applied wrong target policy")
            if after["epoch"] <= before["epoch"]:
                raise BackendExecutionError("CPU restart did not advance runtime epoch")
            validation = self.validate_fn() if self.validate_fn else None
            if validation is not None and validation.get("status") not in {None, "PASS"}:
                raise BackendExecutionError(
                    f"CPU restart validation failed: {validation!r}"
                )
            return _result(
                backend=self.backend, plan=plan, status="RESTART_VERIFIED",
                before=before, after=after, transitioned=True,
                validation=validation,
            )
        except Exception as exc:
            rollback_result = None
            if transitioned and self.rollback_fn is not None:
                try:
                    restored = _normalize_state(
                        self.rollback_fn(copy.deepcopy(before["policy"])),
                        backend=self.backend,
                    )
                    if restored["policy_hash"] != before["policy_hash"]:
                        raise BackendExecutionError("CPU rollback restored wrong policy")
                    rollback_result = {
                        "status": "ROLLBACK_VERIFIED",
                        "epoch": restored["epoch"],
                        "policy_hash": restored["policy_hash"],
                    }
                except Exception as rollback_exc:
                    raise BackendExecutionError(
                        f"CPU transition failed and rollback failed: {rollback_exc}"
                    ) from rollback_exc
            raise BackendExecutionError(
                f"CPU transition failed: {type(exc).__name__}: {exc}; "
                f"rollback={rollback_result}"
            ) from exc


class MlxFileExecutionAdapterV2:
    """Execute verified MLX hot-rebind plans on an isolated file-queue runtime."""

    backend = "mlx_metal"

    def __init__(
        self,
        *,
        model_capability_bundle: Mapping[str, Any],
        ack_path: str | Path,
        txn_path: str | Path,
        submit: Callable[[], Mapping[str, Any]],
    ):
        self.bundle = _validated_bundle(
            model_capability_bundle, backend=self.backend
        )
        self.bundle_sha = str(self.bundle["bundle_sha256"])
        self.ack_path = Path(ack_path).expanduser().resolve()
        self.txn_path = Path(txn_path).expanduser().resolve()
        if _is_production_path(self.ack_path) or _is_production_path(self.txn_path):
            raise BackendExecutionError(
                "P12 isolated MLX adapter refuses production persistent-worker paths"
            )
        self.submit_fn = submit

    def query_state(self) -> dict:
        ack = grc.read_runtime_ack(self.ack_path)
        return {
            "backend": self.backend,
            "epoch": int(ack["weight_epoch"]),
            "policy": pc.normalize_policy(ack["active_policy"]),
            "policy_hash": str(ack["active_policy_hash"]),
            "ack_sha256": str(ack["ack_sha256"]),
        }

    def _publish(self, plan: Mapping[str, Any], txn_id: str) -> str:
        changes = list(plan.get("changes") or [])
        if plan["action"] == "HOT_REBIND_SINGLE":
            if len(changes) != 1:
                raise BackendExecutionError("single-rebind plan must contain one change")
            row = changes[0]
            grc.prepare_rebind(
                ack_path=self.ack_path,
                txn_path=self.txn_path,
                txn_id=txn_id,
                expected_epoch=int(plan["expected_epoch"]),
                expected_policy_hash=plan["expected_policy_hash"],
                role=row["role"],
                layer=int(row["layer"]),
                expected_n=int(row["expected_n"]),
                target_n=int(row["target_n"]),
            )
            return "REBIND_APPLIED"
        if plan["action"] == "HOT_REBIND_MULTI":
            if len(changes) < 2:
                raise BackendExecutionError("multi-rebind plan needs at least two changes")
            grc.prepare_rebind_set(
                ack_path=self.ack_path,
                txn_path=self.txn_path,
                txn_id=txn_id,
                expected_epoch=int(plan["expected_epoch"]),
                expected_policy_hash=plan["expected_policy_hash"],
                changes=[
                    {
                        "role": row["role"],
                        "layer": int(row["layer"]),
                        "expected_n": int(row["expected_n"]),
                        "target_n": int(row["target_n"]),
                    }
                    for row in changes
                ],
            )
            return "REBIND_SET_APPLIED"
        raise BackendExecutionError(
            f"MLX hot execution refuses action={plan['action']}"
        )

    def _inverse_changes(self, plan: Mapping[str, Any]) -> list[dict]:
        return [
            {
                "role": row["role"],
                "layer": int(row["layer"]),
                "expected_n": int(row["target_n"]),
                "target_n": int(row["expected_n"]),
            }
            for row in plan.get("changes") or []
        ]

    def _rollback(self, plan: Mapping[str, Any]) -> dict:
        current = self.query_state()
        if current["policy_hash"] == plan["expected_policy_hash"]:
            return {
                "status": "PREIMAGE_ALREADY_ACTIVE",
                "epoch": current["epoch"],
                "policy_hash": current["policy_hash"],
            }
        if current["policy_hash"] != plan["target_policy_hash"]:
            raise BackendExecutionError(
                "cannot rollback unknown MLX runtime policy state"
            )
        inverse = self._inverse_changes(plan)
        txn_id = "p12-rollback-" + uuid.uuid4().hex
        if len(inverse) == 1:
            row = inverse[0]
            grc.prepare_rebind(
                ack_path=self.ack_path, txn_path=self.txn_path,
                txn_id=txn_id, expected_epoch=current["epoch"],
                expected_policy_hash=current["policy_hash"],
                role=row["role"], layer=row["layer"],
                expected_n=row["expected_n"], target_n=row["target_n"],
            )
            expected_status = "REBIND_APPLIED"
        else:
            grc.prepare_rebind_set(
                ack_path=self.ack_path, txn_path=self.txn_path,
                txn_id=txn_id, expected_epoch=current["epoch"],
                expected_policy_hash=current["policy_hash"], changes=inverse,
            )
            expected_status = "REBIND_SET_APPLIED"
        validation = dict(self.submit_fn())
        ack = grc.verify_terminal_ack(
            ack_path=self.ack_path, txn_id=txn_id,
            allowed_statuses={expected_status},
        )
        if ack["active_policy_hash"] != plan["expected_policy_hash"]:
            raise BackendExecutionError("MLX rollback policy hash mismatch")
        return {
            "status": "ROLLBACK_VERIFIED",
            "txn_id": txn_id,
            "epoch": int(ack["weight_epoch"]),
            "policy_hash": ack["active_policy_hash"],
            "validation": validation,
        }

    def execute(self, plan: Mapping[str, Any]) -> dict:
        plan = _verify_plan(plan, backend=self.backend, bundle_sha256=self.bundle_sha)
        action = str(plan["action"])
        if action == "VALIDATION_REQUIRED":
            raise BackendExecutionError(
                "MLX transition requires runtime validation before execution"
            )
        if action == "RESTART_REQUIRED":
            raise RestartRequired("MLX transition requires isolated worker restart")

        before = self.query_state()
        if before["epoch"] != int(plan["expected_epoch"]):
            raise BackendExecutionError("MLX runtime epoch drifted")
        if before["policy_hash"] != plan["expected_policy_hash"]:
            raise BackendExecutionError("MLX runtime policy preimage drifted")
        _revalidate_capability_plan(
            plan, bundle=self.bundle, before=before
        )

        if action == "NOOP":
            validation = dict(self.submit_fn())
            after = self.query_state()
            if after["epoch"] != before["epoch"] or after["policy_hash"] != before["policy_hash"]:
                raise BackendExecutionError("MLX NOOP admission changed runtime state")
            return _result(
                backend=self.backend, plan=plan, status="NOOP_VERIFIED",
                before=before, after=after, transitioned=False,
                validation=validation,
            )

        txn_id = "p12-" + uuid.uuid4().hex
        expected_status = self._publish(plan, txn_id)
        try:
            validation = dict(self.submit_fn())
            grc.verify_terminal_ack(
                ack_path=self.ack_path, txn_id=txn_id,
                allowed_statuses={expected_status},
            )
            after = self.query_state()
            if after["epoch"] != before["epoch"] + 1:
                raise BackendExecutionError(
                    "MLX precision transition did not advance exactly one epoch"
                )
            if after["policy_hash"] != plan["target_policy_hash"]:
                raise BackendExecutionError("MLX transition target policy mismatch")
            return _result(
                backend=self.backend, plan=plan, status="TRANSITION_VERIFIED",
                before=before, after=after, transitioned=True,
                validation=validation, transaction_id=txn_id,
            )
        except Exception as exc:
            try:
                rollback = self._rollback(plan)
            except Exception as rollback_exc:
                raise BackendExecutionError(
                    f"MLX transition failed and rollback failed: {rollback_exc}"
                ) from rollback_exc
            raise BackendExecutionError(
                f"MLX transition failed: {type(exc).__name__}: {exc}; "
                f"rollback={rollback}"
            ) from exc
