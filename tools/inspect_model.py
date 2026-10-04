#!/usr/bin/env python3
"""Read-only P12 model capability front door.

Current vertical slice:
source identity -> architecture descriptor -> tokenizer/loader contracts ->
operator graph -> safetensors tensor-role graph -> model skeleton.

Inference and P8-P11 remain denied until backend/quant/mutation capability
contracts are compiled with verified evidence. No runtime mutation occurs here.
"""
from __future__ import annotations

import argparse
import json

import architecture_registry as ar
import loader_registry as loaders
import model_source_inspector as inspector
import tensor_role_mapper as tensor_roles
import tokenizer_registry as tokenizers


def inspect(path: str, *, model_id: str | None = None, model_revision: str = "local") -> dict:
    source = inspector.inspect_model_source(
        path, model_id=model_id, model_revision=model_revision
    )
    manifest = source["manifest"]
    config = source["config"]
    resolved_model_id = str(manifest["model_id"])

    architecture = ar.compile_descriptor(
        source["architecture_source_name"],
        config,
    )
    tokenizer = tokenizers.resolve_tokenizer_contract(
        model_id=resolved_model_id,
        architecture_id=architecture["architecture_id"],
        source_files=manifest["files"],
        vocab_size=config.get("vocab_size"),
    )
    loader = loaders.resolve_loader_contract(
        source_format=manifest["source_format"],
        architecture_id=architecture["architecture_id"],
        source_files=manifest["files"],
    )
    operator_graph = ar.build_operator_graph_from_config(
        model_id=resolved_model_id,
        descriptor=architecture,
        config=config,
    )

    tensor_graph = None
    if manifest["source_format"] in {"SAFETENSORS_SINGLE", "SAFETENSORS_SHARDED"}:
        tensor_graph = tensor_roles.build_tensor_role_graph_from_safetensors(
            model_id=resolved_model_id,
            architecture_id=architecture["architecture_id"],
            path=path,
        )

    skeleton = ar.build_model_skeleton_from_config(
        model_id=resolved_model_id,
        model_source_sha256=manifest["checkpoint_identity_sha256"],
        descriptor=architecture,
        config=config,
        operator_graph=operator_graph,
        tensor_role_graph_ref=(
            tensor_graph["graph_sha256"] if tensor_graph is not None else None
        ),
        tokenizer_contract_ref=tokenizer["contract_sha256"],
        loader_contract_ref=loader["contract_sha256"],
    )

    next_required = [
        "backend-capability-v1",
        "quant-capability-v1",
        "runtime-mutation-v1",
        "model-capability-bundle-v1",
    ]
    if tensor_graph is None or tensor_graph["unclaimed_tensor_count"] > 0:
        next_required.insert(0, "tensor-role-graph-v1")

    return {
        "schema": "beglin-inspect-model-p12-v1",
        "phase": "P12_TENSOR_IR_SLICE",
        "model_source": manifest,
        "architecture": architecture,
        "operator_graph": operator_graph,
        "tensor_role_graph": tensor_graph,
        "model_skeleton": skeleton,
        "tokenizer": tokenizer,
        "loader": loader,
        # Deliberately fail closed. Architecture/tensor recognition alone
        # cannot authorize inference or the precision pipeline.
        "inference_allowed": False,
        "p8_p11_eligibility": "DENIED",
        "next_required_contracts": next_required,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--model-id")
    ap.add_argument("--model-revision", default="local")
    args = ap.parse_args()
    print(
        json.dumps(
            inspect(
                args.path,
                model_id=args.model_id,
                model_revision=args.model_revision,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
