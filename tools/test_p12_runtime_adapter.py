#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend_adapters_v2 as planv2
import backend_execution_v2 as execv2
import gpu_runtime_control as grc
import model_capability as mc
import model_evidence_registry as mer
import precision_context as pc
from test_model_capability_p12 import qwen_fixture


def verification_evidence(
    source, descriptor, *, component, marker, backend=None, target_key=None,
    supported_n=None, mutation_mode=None,
):
    row = {
        "schema": mc.VERIFICATION_EVIDENCE_SCHEMA,
        "status": "VERIFIED",
        "component": component,
        "architecture_id": descriptor["architecture_id"],
        "checkpoint_identity_sha256": source["checkpoint_identity_sha256"],
        "evidence_sha256": marker * 64,
        "run_id": f"p12-{component}-{marker}",
        "kind": "UNIT_TEST",
    }
    if backend is not None:
        row["backend"] = backend
    if target_key is not None:
        row["target_key"] = str(target_key)
    if supported_n is not None:
        row["supported_n"] = list(supported_n)
    if mutation_mode is not None:
        row["mutation_mode"] = str(mutation_mode)
    return row


def compile_verified_bundle(root: Path):
    source = mc.inspect_model_source(root)
    descriptor = mc.build_architecture_descriptor(source)
    provisional = mc.compile_model_capabilities(root)
    hot_targets = [
        row["canonical_target_key"]
        for row in provisional["tensor_role_graph"]["nodes"]
        if row["role"] in {"Q_PROJ", "K_PROJ"} and row.get("layer") == 0
    ]
    cpu = verification_evidence(
        source, descriptor, component="backend_runtime", marker="1", backend="cpu"
    )
    mlx = verification_evidence(
        source, descriptor, component="backend_runtime", marker="2", backend="mlx_metal"
    )
    mlx_qng64 = [
        verification_evidence(
            source, descriptor, component="qng64_runtime", marker=str(3 + idx),
            backend="mlx_metal", target_key=target, supported_n=[5, 6],
        )
        for idx, target in enumerate(hot_targets)
    ]
    mlx_mutation = [
        verification_evidence(
            source, descriptor, component="mutation_runtime", marker=str(5 + idx),
            backend="mlx_metal", target_key=target, supported_n=[5, 6],
            mutation_mode="HOT_REBIND_SINGLE",
        )
        for idx, target in enumerate(hot_targets)
    ]
    bundle = mc.compile_model_capabilities(
        root, cpu_runtime_evidence=cpu, mlx_runtime_evidence=mlx,
        mlx_qng64_evidence=mlx_qng64, mlx_mutation_evidence=mlx_mutation,
    )
    return bundle, source, descriptor


def target_key(bundle, role, layer=0):
    return next(
        row["canonical_target_key"]
        for row in bundle["tensor_role_graph"]["nodes"]
        if row["role"] == role and row["layer"] == layer
    )


def write_ack(path: Path, *, epoch: int, policy: list[dict], status="PROMOTION_APPLIED",
              txn_id=None, expected_epoch=None, expected_policy_hash=None):
    raw = {
        "schema": grc.ACK_SCHEMA,
        "status": status,
        "backend": "mlx_metal",
        "correction_mode": "off",
        "weight_epoch": int(epoch),
        "changed_targets": 0 if txn_id is None else 1,
        "snapshot_count": len(policy),
        "txn_id": txn_id,
        "expected_epoch": expected_epoch,
        "expected_n": None,
        "expected_policy_hash": expected_policy_hash,
        "target_role": None,
        "target_layer": None,
        "active_policy": pc.normalize_policy(policy),
        "qng64_cache": [],
    }
    path.write_text(json.dumps(raw, sort_keys=True) + "\n")


class SimulatedMlxRuntime:
    def __init__(self, ack_path: Path, txn_path: Path, *, fail_once=False):
        self.ack_path = ack_path
        self.txn_path = txn_path
        self.fail_once = bool(fail_once)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        fields = self.txn_path.read_text().strip().split()
        before = grc.read_runtime_ack(self.ack_path)
        if fields[0] == "REBIND":
            _, txn_id, expected_epoch, expected_n, target_n, role, layer, expected_hash = fields
            changes = [{
                "role": role,
                "layer": int(layer),
                "expected_n": int(expected_n),
                "target_n": int(target_n),
            }]
            status = "REBIND_APPLIED"
        elif fields[0] == "REBIND_SET":
            txn_id = fields[1]
            expected_epoch = fields[2]
            count = int(fields[3])
            expected_hash = fields[4]
            changes = []
            cursor = 5
            for _ in range(count):
                role = fields[cursor]
                layer = int(fields[cursor + 1])
                expected_n = int(fields[cursor + 2])
                target_n = int(fields[cursor + 3])
                cursor += 4
                changes.append({
                    "role": role,
                    "layer": layer,
                    "expected_n": expected_n,
                    "target_n": target_n,
                })
            status = "REBIND_SET_APPLIED"
        else:
            raise AssertionError(fields[0])

        policy = [dict(row) for row in before["active_policy"]]
        by_key = {(row["role"], int(row["layer"])): row for row in policy}
        for change in changes:
            key = (change["role"], change["layer"])
            self.assert_target(by_key, key, change["expected_n"])
            by_key[key]["n"] = change["target_n"]

        raw = {
            "schema": grc.ACK_SCHEMA,
            "status": status,
            "backend": "mlx_metal",
            "correction_mode": "off",
            "weight_epoch": int(before["weight_epoch"]) + 1,
            "changed_targets": len(changes),
            "snapshot_count": len(policy),
            "txn_id": txn_id,
            "expected_epoch": int(expected_epoch),
            "expected_n": changes[0]["expected_n"] if len(changes) == 1 else None,
            "expected_policy_hash": expected_hash,
            "target_role": changes[0]["role"] if len(changes) == 1 else None,
            "target_layer": changes[0]["layer"] if len(changes) == 1 else None,
            "active_policy": pc.normalize_policy(policy),
            "qng64_cache": [],
        }
        self.ack_path.write_text(json.dumps(raw, sort_keys=True) + "\n")
        if self.fail_once and self.calls == 1:
            raise RuntimeError("injected post-apply validation failure")
        return {"finite_logits": True, "simulated": True, "call": self.calls}

    @staticmethod
    def assert_target(by_key, key, expected_n):
        if key not in by_key:
            raise AssertionError(f"missing target {key}")
        if int(by_key[key]["n"]) != int(expected_n):
            raise AssertionError(
                f"target preimage mismatch {key}: "
                f"{by_key[key]['n']} != {expected_n}"
            )


class EvidenceRegistryTests(unittest.TestCase):
    def test_registry_resolves_exact_checkpoint_backend_and_component(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "model")
            source = mc.inspect_model_source(root)
            descriptor = mc.build_architecture_descriptor(source)
            evidence_dir = Path(td) / "evidence"
            evidence_dir.mkdir()
            rows = [
                verification_evidence(
                    source, descriptor, component="backend_runtime",
                    marker="1", backend="cpu"
                ),
                verification_evidence(
                    source, descriptor, component="backend_runtime",
                    marker="2", backend="mlx_metal"
                ),
                verification_evidence(
                    source, descriptor, component="tokenizer", marker="3"
                ),
                verification_evidence(
                    source, descriptor, component="loader", marker="4"
                ),
            ]
            for idx, row in enumerate(rows):
                (evidence_dir / f"{idx}.json").write_text(json.dumps(row))
            registry = mer.VerificationEvidenceRegistry.from_paths([evidence_dir])
            resolved = registry.resolve_model_set(
                architecture_id=descriptor["architecture_id"],
                checkpoint_identity_sha256=source["checkpoint_identity_sha256"],
            )
            self.assertEqual(resolved["cpu_runtime_evidence"]["evidence_sha256"], "1" * 64)
            self.assertEqual(resolved["mlx_runtime_evidence"]["evidence_sha256"], "2" * 64)
            self.assertEqual(resolved["tokenizer_evidence"]["evidence_sha256"], "3" * 64)
            self.assertEqual(resolved["loader_evidence"]["evidence_sha256"], "4" * 64)

    def test_registry_resolves_target_scoped_qng64_and_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "model")
            source = mc.inspect_model_source(root)
            descriptor = mc.build_architecture_descriptor(source)
            provisional = mc.compile_model_capabilities(root)
            target = next(
                row["canonical_target_key"]
                for row in provisional["tensor_role_graph"]["nodes"]
                if row["role"] == "Q_PROJ" and row["layer"] == 0
            )
            qng = verification_evidence(
                source, descriptor, component="qng64_runtime", marker="8",
                backend="mlx_metal", target_key=target, supported_n=[5, 6],
            )
            mut = verification_evidence(
                source, descriptor, component="mutation_runtime", marker="9",
                backend="mlx_metal", target_key=target, supported_n=[5, 6],
                mutation_mode="HOT_REBIND_SINGLE",
            )
            registry = mer.VerificationEvidenceRegistry([qng, mut])
            resolved = registry.resolve_model_set(
                architecture_id=descriptor["architecture_id"],
                checkpoint_identity_sha256=source["checkpoint_identity_sha256"],
            )
            self.assertEqual(resolved["mlx_qng64_evidence"][0]["target_key"], target)
            self.assertEqual(resolved["mlx_qng64_evidence"][0]["supported_n"], [5, 6])
            self.assertEqual(resolved["mlx_mutation_evidence"][0]["target_key"], target)
            self.assertEqual(
                resolved["mlx_mutation_evidence"][0]["mutation_mode"],
                "HOT_REBIND_SINGLE",
            )

    def test_conflicting_exact_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = qwen_fixture(Path(td) / "model")
            source = mc.inspect_model_source(root)
            descriptor = mc.build_architecture_descriptor(source)
            a = verification_evidence(
                source, descriptor, component="backend_runtime",
                marker="1", backend="cpu"
            )
            b = verification_evidence(
                source, descriptor, component="backend_runtime",
                marker="2", backend="cpu"
            )
            registry = mer.VerificationEvidenceRegistry([a, b])
            with self.assertRaisesRegex(mer.EvidenceRegistryError, "ambiguous"):
                registry.resolve(
                    component="backend_runtime",
                    architecture_id=descriptor["architecture_id"],
                    checkpoint_identity_sha256=source["checkpoint_identity_sha256"],
                    backend="cpu",
                )

    def test_inspect_model_cli_accepts_registry_directory(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(__file__).resolve().parents[1]
            root = qwen_fixture(Path(td) / "model")
            source = mc.inspect_model_source(root)
            descriptor = mc.build_architecture_descriptor(source)
            evidence_dir = Path(td) / "evidence"
            evidence_dir.mkdir()
            row = verification_evidence(
                source, descriptor, component="backend_runtime",
                marker="1", backend="cpu"
            )
            (evidence_dir / "cpu.json").write_text(json.dumps(row))
            proc = subprocess.run(
                [
                    sys.executable, str(repo / "tools" / "inspect_model.py"),
                    str(root), "--backend", "cpu", "--evidence",
                    str(evidence_dir), "--json",
                ],
                cwd=repo, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            bundle = json.loads(proc.stdout)
            statuses = {
                row["inference_status"]
                for row in bundle["backend_capability_matrix"]["rows"]
            }
            self.assertEqual(statuses, {"VERIFIED"})


class BackendExecutionTests(unittest.TestCase):
    def bundle(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = qwen_fixture(Path(td.name) / "model")
        return compile_verified_bundle(root)[0]

    def test_cpu_shape_change_validates_added_targets_before_restart(self):
        bundle = self.bundle()
        qkey = target_key(bundle, "Q_PROJ")

        empty = planv2.BackendStateV2.build(
            backend="cpu", epoch=1, policy=[],
            model_capability_bundle_sha256=bundle["bundle_sha256"],
        )
        good = planv2.CpuBackendAdapterV2(bundle).plan_transition(
            state=empty,
            target_policy=[{"role":"Q_PROJ","layer":0,"n":5}],
            target_keys={("Q_PROJ",0):qkey},
        )
        self.assertEqual(good["action"], "RESTART_REQUIRED")
        self.assertEqual(good["reason"]["code"], "POLICY_SHAPE_CHANGE")
        self.assertEqual(good["changes"][0]["target_key"], qkey)
        self.assertIsNone(good["changes"][0]["expected_n"])

        with self.assertRaisesRegex(planv2.BackendV2Error, "missing target-key"):
            planv2.CpuBackendAdapterV2(bundle).plan_transition(
                state=empty,
                target_policy=[{"role":"BOGUS","layer":999,"n":5}],
                target_keys={},
            )

        with self.assertRaisesRegex(planv2.BackendV2Error, "unsupported"):
            planv2.CpuBackendAdapterV2(bundle).plan_transition(
                state=empty,
                target_policy=[{"role":"Q_PROJ","layer":0,"n":123}],
                target_keys={("Q_PROJ",0):qkey},
            )

    def test_cpu_restart_and_mlx_rebind_share_result_schema(self):
        bundle = self.bundle()
        qkey = target_key(bundle, "Q_PROJ")
        before_policy = [{"role": "Q_PROJ", "layer": 0, "n": 6}]
        after_policy = [{"role": "Q_PROJ", "layer": 0, "n": 5}]
        keys = {("Q_PROJ", 0): qkey}

        cpu_state = {"epoch": 2, "policy": before_policy}
        def cpu_query():
            return dict(cpu_state)
        def cpu_restart(policy):
            cpu_state["epoch"] += 1
            cpu_state["policy"] = [dict(row) for row in policy]
            return dict(cpu_state)

        cpu_plan = planv2.CpuBackendAdapterV2(bundle).plan_transition(
            state=planv2.BackendStateV2.build(
                backend="cpu", epoch=2, policy=before_policy,
                model_capability_bundle_sha256=bundle["bundle_sha256"],
            ),
            target_policy=after_policy, target_keys=keys,
        )
        cpu_result = execv2.CpuRestartExecutionAdapterV2(
            model_capability_bundle_sha256=bundle["bundle_sha256"],
            query_state=cpu_query, restart=cpu_restart,
            validate=lambda: {"finite_logits": True},
        ).execute(cpu_plan)

        with tempfile.TemporaryDirectory() as td:
            ack = Path(td) / "ack.json"
            txn = Path(td) / "txn.cmd"
            write_ack(ack, epoch=2, policy=before_policy)
            sim = SimulatedMlxRuntime(ack, txn)
            mlx_plan = planv2.MlxMetalBackendAdapterV2(bundle).plan_transition(
                state=planv2.BackendStateV2.build(
                    backend="mlx_metal", epoch=2, policy=before_policy,
                    model_capability_bundle_sha256=bundle["bundle_sha256"],
                ),
                target_policy=after_policy, target_keys=keys,
            )
            mlx_result = execv2.MlxFileExecutionAdapterV2(
                model_capability_bundle_sha256=bundle["bundle_sha256"],
                ack_path=ack, txn_path=txn, submit=sim,
            ).execute(mlx_plan)

        self.assertEqual(cpu_result["schema"], mlx_result["schema"])
        self.assertEqual(cpu_result["status"], "RESTART_VERIFIED")
        self.assertEqual(mlx_result["status"], "TRANSITION_VERIFIED")
        self.assertEqual(cpu_result["after"]["policy_hash"], mlx_result["after"]["policy_hash"])
        self.assertFalse(cpu_result["production_write_allowed"])
        self.assertFalse(mlx_result["production_write_allowed"])

    def test_mlx_multi_target_rebind_is_atomic_one_epoch(self):
        bundle = self.bundle()
        qkey = target_key(bundle, "Q_PROJ")
        kkey = target_key(bundle, "K_PROJ")
        before = [
            {"role": "Q_PROJ", "layer": 0, "n": 6},
            {"role": "K_PROJ", "layer": 0, "n": 6},
        ]
        after = [
            {"role": "Q_PROJ", "layer": 0, "n": 5},
            {"role": "K_PROJ", "layer": 0, "n": 5},
        ]
        plan = planv2.MlxMetalBackendAdapterV2(bundle).plan_transition(
            state=planv2.BackendStateV2.build(
                backend="mlx_metal", epoch=4, policy=before,
                model_capability_bundle_sha256=bundle["bundle_sha256"],
            ),
            target_policy=after,
            target_keys={("Q_PROJ",0):qkey, ("K_PROJ",0):kkey},
        )
        self.assertEqual(plan["action"], "HOT_REBIND_MULTI")
        with tempfile.TemporaryDirectory() as td:
            ack = Path(td) / "ack.json"
            txn = Path(td) / "txn.cmd"
            write_ack(ack, epoch=4, policy=before)
            sim = SimulatedMlxRuntime(ack, txn)
            result = execv2.MlxFileExecutionAdapterV2(
                model_capability_bundle_sha256=bundle["bundle_sha256"],
                ack_path=ack, txn_path=txn, submit=sim,
            ).execute(plan)
            self.assertEqual(result["after"]["epoch"], 5)
            self.assertEqual(result["after"]["policy_hash"], pc.policy_hash(after))

    def test_mlx_failure_rolls_back_preimage(self):
        bundle = self.bundle()
        qkey = target_key(bundle, "Q_PROJ")
        before = [{"role": "Q_PROJ", "layer": 0, "n": 6}]
        after = [{"role": "Q_PROJ", "layer": 0, "n": 5}]
        plan = planv2.MlxMetalBackendAdapterV2(bundle).plan_transition(
            state=planv2.BackendStateV2.build(
                backend="mlx_metal", epoch=8, policy=before,
                model_capability_bundle_sha256=bundle["bundle_sha256"],
            ),
            target_policy=after, target_keys={("Q_PROJ",0):qkey},
        )
        with tempfile.TemporaryDirectory() as td:
            ack = Path(td) / "ack.json"
            txn = Path(td) / "txn.cmd"
            write_ack(ack, epoch=8, policy=before)
            sim = SimulatedMlxRuntime(ack, txn, fail_once=True)
            adapter = execv2.MlxFileExecutionAdapterV2(
                model_capability_bundle_sha256=bundle["bundle_sha256"],
                ack_path=ack, txn_path=txn, submit=sim,
            )
            with self.assertRaisesRegex(execv2.BackendExecutionError, "rollback"):
                adapter.execute(plan)
            restored = grc.read_runtime_ack(ack)
            self.assertEqual(restored["active_policy_hash"], pc.policy_hash(before))
            self.assertEqual(restored["weight_epoch"], 10)

    def test_mlx_adapter_refuses_known_production_path(self):
        with self.assertRaisesRegex(execv2.BackendExecutionError, "refuses production"):
            execv2.MlxFileExecutionAdapterV2(
                model_capability_bundle_sha256="a" * 64,
                ack_path="/Users/xox/vdsp_serving/persistent-workers/candidate/applied_ack.json",
                txn_path="/Users/xox/vdsp_serving/persistent-workers/candidate/txn.cmd",
                submit=lambda: {},
            )


if __name__ == "__main__":
    unittest.main()
