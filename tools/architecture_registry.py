#!/usr/bin/env python3
"""P12 architecture registry and canonical skeleton compiler.

This module turns already-parsed model metadata into Beglin's model IR.
It is intentionally conservative: registry membership means that Beglin knows
the architecture shape, not that every backend/tokenizer/loader path is
production-verified for every checkpoint.
"""
from __future__ import annotations

from typing import Any, Mapping

import model_capability as mc


class ArchitectureRegistryError(RuntimeError):
    pass


_PROFILES = {
    "qwen2": {
        "attention_operator": "ATTENTION_GQA",
        "ffn_kind": "DENSE",
        "qk_norm_kind": "NONE",
        "required_primitives": [
            "EMBEDDING_LOOKUP", "RMS_NORM", "ROPE", "ATTENTION_GQA",
            "DENSE_FFN", "LM_HEAD",
        ],
    },
    "llama": {
        "attention_operator": "ATTENTION_GQA",
        "ffn_kind": "DENSE",
        "qk_norm_kind": "NONE",
        "required_primitives": [
            "EMBEDDING_LOOKUP", "RMS_NORM", "ROPE", "ATTENTION_GQA",
            "DENSE_FFN", "LM_HEAD",
        ],
    },
    "deepseek_v2": {
        "attention_operator": "ATTENTION_MLA",
        "ffn_kind": "MOE",
        "qk_norm_kind": "NONE",
        "required_primitives": [
            "EMBEDDING_LOOKUP", "RMS_NORM", "ROPE", "ATTENTION_MLA",
            "ROUTER_TOPK", "MOE_EXPERT", "MOE_SHARED_EXPERT", "LM_HEAD",
        ],
    },
    "qwen3_moe": {
        "attention_operator": "ATTENTION_GQA",
        "ffn_kind": "MOE",
        "qk_norm_kind": "PER_HEAD",
        "required_primitives": [
            "EMBEDDING_LOOKUP", "RMS_NORM", "ROPE", "ATTENTION_GQA",
            "ROUTER_TOPK", "MOE_EXPERT", "LM_HEAD",
        ],
    },
    "olmoe": {
        "attention_operator": "ATTENTION_GQA",
        "ffn_kind": "MOE",
        "qk_norm_kind": "WHOLE_VECTOR",
        "required_primitives": [
            "EMBEDDING_LOOKUP", "RMS_NORM", "ROPE", "ATTENTION_GQA",
            "ROUTER_TOPK", "MOE_EXPERT", "LM_HEAD",
        ],
    },
    "gpt-oss": {
        "attention_operator": "ATTENTION_GQA",
        "ffn_kind": "MOE",
        "qk_norm_kind": "NONE",
        "required_primitives": [
            "EMBEDDING_LOOKUP", "RMS_NORM", "ROPE", "ATTENTION_GQA",
            "SLIDING_WINDOW_ATTENTION", "ATTENTION_SINK", "ROUTER_TOPK",
            "MOE_EXPERT", "LM_HEAD",
        ],
    },
}


_FACT_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "num_experts_per_token",
    "n_shared_experts",
    "sliding_window",
    "first_k_dense_replace",
    "moe_layer_freq",
    "norm_topk_prob",
)


def architecture_facts(config: Mapping[str, Any]) -> dict:
    """Extract facts without inventing aliases that are not present."""
    return {key: config[key] for key in _FACT_KEYS if key in config}


def compile_descriptor(
    source_name: str | None,
    config: Mapping[str, Any],
    *,
    adapter_version: str = "p12-v1",
) -> dict:
    descriptor = mc.identify_architecture(
        source_name,
        facts=architecture_facts(config),
        adapter_version=adapter_version,
    )
    profile = _PROFILES.get(str(descriptor["architecture_id"]))
    if profile:
        descriptor = dict(descriptor)
        facts = dict(descriptor.get("facts") or {})
        facts["qk_norm_kind"] = profile["qk_norm_kind"]
        descriptor["facts"] = facts
        descriptor["descriptor_sha256"] = mc.sha256_json({
            key: value
            for key, value in descriptor.items()
            if key != "descriptor_sha256"
        })
    return descriptor


def _int_fact(config: Mapping[str, Any], *names: str, default: int | None = None) -> int | None:
    for name in names:
        if name in config and config[name] is not None:
            try:
                return int(config[name])
            except (TypeError, ValueError) as exc:
                raise ArchitectureRegistryError(f"{name} must be an integer") from exc
    return default


def _layer_count(config: Mapping[str, Any]) -> int:
    value = _int_fact(config, "num_hidden_layers", "n_layer", default=0)
    if value is None or value < 0:
        raise ArchitectureRegistryError("layer count must be non-negative")
    return value


def _expert_count(config: Mapping[str, Any]) -> int | None:
    value = _int_fact(config, "num_experts", "n_routed_experts", "num_local_experts")
    if value is not None and value <= 0:
        raise ArchitectureRegistryError("expert count must be positive")
    return value


def _experts_per_token(config: Mapping[str, Any]) -> int | None:
    value = _int_fact(config, "num_experts_per_tok", "num_experts_per_token")
    if value is not None and value <= 0:
        raise ArchitectureRegistryError("experts_per_token must be positive")
    return value



def _layer_ffn_kind(
    arch: str,
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
    layer: int,
) -> str:
    """Return the real FFN topology for one layer."""
    if profile["ffn_kind"] == "DENSE":
        return "DENSE"
    if arch == "deepseek_v2":
        first_dense = int(config.get("first_k_dense_replace", 0) or 0)
        frequency = max(int(config.get("moe_layer_freq", 1) or 1), 1)
        if layer < first_dense:
            return "DENSE"
        return "MOE" if (layer - first_dense) % frequency == 0 else "DENSE"
    return "MOE"


def build_operator_graph_from_config(
    *,
    model_id: str,
    descriptor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict:
    arch = str(descriptor["architecture_id"])
    profile = _PROFILES.get(arch)
    if profile is None or descriptor.get("status") != "KNOWN":
        return mc.build_operator_graph(
            model_id=model_id,
            operators=[],
            unsupported_primitives=["UNKNOWN_ARCHITECTURE"],
        )

    operators = [
        {
            "operator_id": "global/embedding",
            "operator_type": "EMBEDDING_LOOKUP",
            "layer": None,
            "precision_sensitive": True,
        }
    ]
    expert_count = _expert_count(config)
    experts_per_token = _experts_per_token(config)

    for layer in range(_layer_count(config)):
        ffn_kind = _layer_ffn_kind(arch, profile, config, layer)
        operators.extend([
            {
                "operator_id": f"L{layer}/pre_attention_norm",
                "operator_type": "RMS_NORM",
                "layer": layer,
                "precision_sensitive": False,
            },
            {
                "operator_id": f"L{layer}/attention",
                "operator_type": profile["attention_operator"],
                "layer": layer,
                "precision_sensitive": True,
                "numeric_semantics": {
                    "attention_kind": descriptor.get("attention_kind"),
                    "qk_norm_kind": profile["qk_norm_kind"],
                },
            },
        ])
        if arch == "gpt-oss":
            operators.append({
                "operator_id": f"L{layer}/attention_window",
                "operator_type": "SLIDING_WINDOW_ATTENTION",
                "layer": layer,
                "precision_sensitive": False,
                "numeric_semantics": {
                    "sliding_window": config.get("sliding_window"),
                    "attention_sink": True,
                },
            })
        if ffn_kind == "DENSE":
            operators.append({
                "operator_id": f"L{layer}/ffn",
                "operator_type": "DENSE_FFN",
                "layer": layer,
                "precision_sensitive": True,
            })
        else:
            operators.extend([
                {
                    "operator_id": f"L{layer}/router",
                    "operator_type": "ROUTER_TOPK",
                    "layer": layer,
                    "precision_sensitive": True,
                    "numeric_semantics": {
                        "expert_count": expert_count,
                        "experts_per_token": experts_per_token,
                        "normalize_top_k": bool(
                            config.get("norm_topk_prob", False)
                        ),
                    },
                },
                {
                    "operator_id": f"L{layer}/experts",
                    "operator_type": "MOE_EXPERT",
                    "layer": layer,
                    "precision_sensitive": True,
                    "numeric_semantics": {"expert_count": expert_count},
                },
            ])
            if arch == "deepseek_v2" and config.get("n_shared_experts"):
                operators.append({
                    "operator_id": f"L{layer}/shared_experts",
                    "operator_type": "MOE_SHARED_EXPERT",
                    "layer": layer,
                    "precision_sensitive": True,
                    "numeric_semantics": {
                        "shared_expert_count": config.get("n_shared_experts"),
                    },
                })

    operators.extend([
        {
            "operator_id": "global/final_norm",
            "operator_type": "RMS_NORM",
            "layer": None,
            "precision_sensitive": False,
        },
        {
            "operator_id": "global/lm_head",
            "operator_type": "LM_HEAD",
            "layer": None,
            "precision_sensitive": True,
        },
    ])
    return mc.build_operator_graph(model_id=model_id, operators=operators)


def build_model_skeleton_from_config(
    *,
    model_id: str,
    model_source_sha256: str,
    descriptor: Mapping[str, Any],
    config: Mapping[str, Any],
    operator_graph: Mapping[str, Any],
    tensor_role_graph_ref: str | None = None,
    tokenizer_contract_ref: str | None = None,
    loader_contract_ref: str | None = None,
) -> dict:
    arch = str(descriptor["architecture_id"])
    profile = _PROFILES.get(arch)
    layer_count = _layer_count(config) if profile else 0
    expert_count = _expert_count(config) if profile else None
    experts_per_token = _experts_per_token(config) if profile else None

    layers = []
    for layer in range(layer_count):
        ffn_kind = _layer_ffn_kind(arch, profile, config, layer)
        row = {
            "layer_index": layer,
            "attention": {
                "kind": descriptor.get("attention_kind", "UNKNOWN"),
                "qk_norm_kind": profile["qk_norm_kind"],
                "operator_ref": f"L{layer}/attention",
            },
            "norm": {"kind": config.get("norm_kind", "RMS_NORM")},
            "ffn": {"kind": ffn_kind},
            "operator_refs": [
                f"L{layer}/pre_attention_norm",
                f"L{layer}/attention",
            ],
        }
        if profile["ffn_kind"] == "DENSE":
            row["operator_refs"].append(f"L{layer}/ffn")
        else:
            row["moe"] = {
                "routed_experts": expert_count,
                "experts_per_token": experts_per_token,
                "shared_experts": config.get("n_shared_experts"),
                "normalize_top_k": bool(
                    config.get("norm_topk_prob", False)
                ),
            }
            row["operator_refs"].extend([f"L{layer}/router", f"L{layer}/experts"])
            if arch == "deepseek_v2" and config.get("n_shared_experts"):
                row["operator_refs"].append(f"L{layer}/shared_experts")
        if arch == "gpt-oss":
            row["attention"]["sliding_window"] = config.get("sliding_window")
            row["attention"]["attention_sink"] = True
            row["operator_refs"].append(f"L{layer}/attention_window")
        layers.append(row)

    required = list(profile["required_primitives"]) if profile else []
    unsupported = [] if profile else ["UNKNOWN_ARCHITECTURE"]
    return mc.build_model_skeleton(
        model_source_sha256=model_source_sha256,
        architecture_descriptor=descriptor,
        layer_skeletons=layers,
        global_tensors=[
            {"role": "EMBEDDING"},
            {"role": "FINAL_NORM"},
            {"role": "LM_HEAD"},
        ] if profile else [],
        operator_graph_ref=operator_graph["graph_sha256"],
        tensor_role_graph_ref=tensor_role_graph_ref,
        tokenizer_contract_ref=tokenizer_contract_ref,
        loader_contract_ref=loader_contract_ref,
        required_primitives=required,
        unsupported_primitives=unsupported,
    )


def registered_architectures() -> list[str]:
    return sorted(_PROFILES)
