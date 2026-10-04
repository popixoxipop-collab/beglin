#!/usr/bin/env python3
"""Read-only P12 model capability front door.

Current vertical slice:
source identity -> architecture descriptor -> tokenizer/loader contracts ->
operator graph -> model skeleton.

Inference and P8-P11 remain denied until tensor-role and backend capability
contracts are compiled. No runtime mutation is performed here.
"""
from __future__ import annotations

import argparse
import json

import architecture_registry as ar
import loader_registry as loaders
import tokenizer_registry as tokenizers
import model_source_inspector as inspector


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
    skeleton = ar.build_model_skeleton_from_config(
        model_id=resolved_model_id,
        model_source_sha256=manifest["checkpoint_identity_sha256"],
        descriptor=architecture,
        config=config,
        operator_graph=operator_graph,
        tokenizer_contract_ref=tokenizer["contract_sha256"],
        loader_contract_ref=loader["contract_sha256"],
    )

    return {
        "schema": "beglin-inspect-model-p12-v1",
        "phase": "P12_ARCHITECTURE_IR_SLICE",
        "model_source": manifest,
        "architecture": architecture,
        "operator_graph": operator_graph,
        "model_skeleton": skeleton,
        "tokenizer": tokenizer,
        "loader": loader,
        # Deliberately fail closed. Tensor-role and backend capability are not
        # compiled yet, so architecture recognition alone cannot authorize
        # inference or the precision pipeline.
        "inference_allowed": False,
        "p8_p11_eligibility": "DENIED",
        "next_required_contracts": [
            "tensor-role-graph-v1",
            "backend-capability-v1",
            "quant-capability-v1",
            "runtime-mutation-v1",
            "model-capability-bundle-v1",
        ],
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
