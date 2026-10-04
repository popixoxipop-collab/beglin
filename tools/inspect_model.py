#!/usr/bin/env python3
"""Read-only P12 model capability front door (Phase 1).

Current scope: source identity + architecture identification.  Tensor/operator
capability compilation lands in subsequent P12 work packages.
"""
from __future__ import annotations

import argparse
import json

import model_capability as mc
import model_source_inspector as inspector


def inspect(path: str, *, model_id: str | None = None, model_revision: str = "local") -> dict:
    source = inspector.inspect_model_source(path, model_id=model_id, model_revision=model_revision)
    config = source["config"]
    facts = {
        key: config[key]
        for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
            "num_experts", "num_experts_per_tok",
        )
        if key in config
    }
    architecture = mc.identify_architecture(source["architecture_source_name"], facts=facts)
    return {
        "schema": "beglin-inspect-model-p12-v1",
        "phase": "P12_PHASE1_CONTRACT_SLICE",
        "model_source": source["manifest"],
        "architecture": architecture,
        "inference_allowed": False,
        "p8_p11_eligibility": "DENIED",
        "next_required_contracts": [
            "tensor-role-graph-v1",
            "operator-graph-v1",
            "tokenizer-contract-v1",
            "loader-contract-v1",
            "backend-capability-v1",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--model-id")
    ap.add_argument("--model-revision", default="local")
    args = ap.parse_args()
    print(json.dumps(inspect(args.path, model_id=args.model_id, model_revision=args.model_revision), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
