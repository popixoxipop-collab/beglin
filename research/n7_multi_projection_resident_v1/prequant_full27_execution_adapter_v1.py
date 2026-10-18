#!/usr/bin/env python3
"""Research-only sealed prequant qNg64 n7 -> native C++/Metal model modules.

Drops into the immutable original EOE full27 DeepSeek research path as an
opt-in hook and replaces its on-the-fly C quantizer per source role. A role
must be present in the FULL frozen 5,181 matrix manifest; BF16 exclusions
still use original source model implementations.

This does NOT yet eliminate BF16 tensor reads in the *separate original-model
reference* path. It does eliminate fresh C qNg64 quantization on decoding
passes and never alters original model files or serving.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from torch import nn

from prequant_n7_sealed_v1 import PackReader
from full27_sealed_materializer_v1 import (
    AUTHORITY_SCHEMA, BOUNDARY_SHA, INDEX_SHA, SHARD_SHAS,
    FULL_ROLES, FULL_Q_BYTES, geometry, gate,
)

# Specific original EOE read-only BF16 checkpoint and qNg64 n7 offline export.
# These are source-identity receipts, not a portable production loader policy.
SOURCE_RUN_AUTHORITY_SHA256 = "3b13bc1d98c5843f146afa428b6dfff45352f7def698d6a5a156d32d0fae136f"
SOURCE_EXPORTER_SHA256 = "ca1872c6ea41861d6db97ed29010e68d2d794c500cb1909e16a5fb9e40b1acf8"
SOURCE_NATIVE_C_SHA256 = "6d76b36ef23341e7ac90ecda79541e93133e701e5538ab215e757c334526d786"


class SealedPrequantFull27:
    def __init__(self, root, manifest_sha256, known, cpp_class, model_package):
        authority_file = Path(root)/"RUN_AUTHORITY.json"
        gate(not authority_file.is_symlink() and authority_file.is_file() and
             authority_file.stat().st_size < 16384,
             "sealed full27 original source authority file missing or unsafe")
        authority_raw = authority_file.read_bytes()
        gate(hashlib.sha256(authority_raw).hexdigest()==SOURCE_RUN_AUTHORITY_SHA256,
             "externally pinned original BF16 full27 run authority SHA mismatch")
        authority=json.loads(authority_raw)
        gate(type(authority) is dict and
             authority.get("schema")==AUTHORITY_SCHEMA and
             authority.get("index_sha256")==INDEX_SHA and
             authority.get("boundary_tsv_sha256")==BOUNDARY_SHA and
             authority.get("source_shard_sha256")==SHARD_SHAS and
             authority.get("native_quantizer_sha256")==SOURCE_NATIVE_C_SHA256 and
             authority.get("exporter_source_sha256")==SOURCE_EXPORTER_SHA256 and
             authority.get("qng64_role_count")==FULL_ROLES and
             authority.get("BF16_exclusion_count")==110 and
             authority.get("quantized_expected_bytes")==FULL_Q_BYTES,
             "sealed full27 source shard/transform/n7 policy authority drift")
        self.original_source_authority_sha256=SOURCE_RUN_AUTHORITY_SHA256
        self.reader = PackReader(Path(root), manifest_sha256, INDEX_SHA)
        gate(len(self.reader.roles)==FULL_ROLES and
             self.reader.total_packed_bytes == FULL_Q_BYTES,
             "frozen 5181-role full model prequant manifest required")
        gate(type(known) is dict and len(known) == FULL_ROLES,
             "prior full27 uniform7 model policy unavailable")
        policy_keys = {
            k for k, v in known.items()
            if v.get("bits",{}).get("uniform7") == 7
        }
        gate(policy_keys == set(self.reader.roles),
             "sealed pack does not match model's exact 5181 source role policy")
        for k, row in self.reader.roles.items():
            policy = known[k]
            out, input_ = row["out"], row["in_features"]
            expected,_scales = geometry(out,input_)
            gate(expected + _scales ==
                 policy["packed_bytes"]["uniform7"], "source role policy packed byte mismatch")
        self.model_package = model_package
        self.cpp_class = cpp_class
        self.known = known
        self.loads = 0
        self.logical_bytes_loaded = 0
        self.unique_roles = set()
        self.load_counts = {}
        self.cacheless = True
        self.input_BF16_weights_read_for_separate_reference = True
        self.quantize_calls_during_decode = 0

    def module(self, role, native, cpp_class=None, expected_shape=None):
        gate(cpp_class is None or cpp_class is self.cpp_class,
             "attempted wrong native source bridge class")
        gate(role in self.known and
             self.known[role]["bits"]["uniform7"] == 7,
             "unapproved qNg64 role")
        sealed = self.reader.load(role)
        if expected_shape is not None:
            gate(tuple(expected_shape)==(sealed.out,sealed.in_features),
                 "original BF16/packed model role shape drift")
        # Bypass __init__: the original class __init__ would re-run native
        # qNg64 quantization. Initialize its PyTorch base and *reuse* the
        # audited native qNg64 packed module's normal C/GPU forward method.
        comp = self.cpp_class.__new__(self.cpp_class)
        nn.Module.__init__(comp)
        comp.n = 7
        comp.out = sealed.out
        comp.in_features = sealed.in_features
        comp.native = native
        comp.packed = sealed.packed
        comp.scales = sealed.scales
        comp.serialized_bytes = sealed.serialized_bytes
        comp.calls = 0
        comp.tokens = 0
        comp.C_checked = False
        comp.max_C_vs_gpu_L2 = 0.0
        comp.gpu_mlx_call_wall_seconds = 0.0
        comp.cpu_to_gpu_rows = 0
        comp._beglin_sealed_packed_source = sealed  # holds both mmap handles
        self.cpp_class.n_constructed += 1
        self.loads += 1
        self.logical_bytes_loaded += sealed.serialized_bytes
        self.unique_roles.add(role)
        self.load_counts[role] = self.load_counts.get(role, 0) + 1
        return comp

    def prepare_weights(self, model, layer_number, policy, known, native, kernel):
        gate(policy == "uniform7" and known is self.known and
             type(layer_number) is int and 0<=layer_number<27,
             "frozen source policy mismatch")
        paths = self.model_package.ATTENTION + (
            tuple("mlp." + role for role in self.model_package.FFN)
            if layer_number == 0 else
            tuple("mlp.shared_experts." + role for role in self.model_package.FFN)
        )
        gate(len(paths)==7, "original attention + FFN fixed model role count")
        modules = []
        packed_total = bf16_total = 0
        for p in paths:
            key = "model.layers.%d.%s.weight" % (layer_number,p)
            source = model.get_submodule(p)
            gate(hasattr(source, "weight") and
                 hasattr(source.weight, "shape") and
                 len(source.weight.shape)==2,
                 "original source fixed projection expected")
            dims = tuple(map(int, source.weight.shape))
            module = self.module(key, native, expected_shape=dims)
            parent, name = p.rsplit(".",1)
            setattr(model.get_submodule(parent), name, module)
            packed_total += module.serialized_bytes
            bf16_total += 2*module.out*module.in_features
            modules.append(module)
        return len(modules),packed_total,bf16_total,modules

    def summary(self):
        return {
            "schema": "beglin.v11.full27.sealed-prequant-inference-hook/1",
            "original_source_run_authority_sha256": self.original_source_authority_sha256,
            "sealed_manifest_sha256": self.reader.manifest_sha256,
            "frozen_qng64_roles_in_manifest": FULL_ROLES,
            "modelwide_uniform7_compressed_bytes": FULL_Q_BYTES,
            "on_demand_model_role_mmap_loads": self.loads,
            "logical_packed_source_bytes_read": self.logical_bytes_loaded,
            "unique_runtime_roles": len(self.unique_roles),
            "all_source_roles_from_sealed_manifest": True,
            "native_C_quantize_calls_during_decode": self.quantize_calls_during_decode,
            "separate_BF16_reference_still_reads_source": self.input_BF16_weights_read_for_separate_reference,
            "full_model_gpu_resident": False,
            "production_promotion_allowed": False,
        }
