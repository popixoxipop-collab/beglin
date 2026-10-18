#!/usr/bin/env python3
"""EOE research-only BEGLIN device-resident n7 linear-source delegate.

Attaches a per-instance forward adapter to SHA-pinned original
NativeCppPackedLinear module, using a shared bounded GPU role cache.
First actual invocation is independently compared with the original native
BEGLIN C++ Metal gather GEMV and its native C oracle; the trusted original
callback then sets native_cpp_module.C_checked and its per-instance usage
counter. Subsequent uses of this module go through only device-resident
GPU GEMV with original (type-based) C++ call counters maintained for
source-boundary proof, never silently falling back to original GPU path.

Compressed source GPU LRU internally keeps immutable CPU backing buffers:
this is NOT a GPU-exclusive memory reduction or end-to-end speed certificate.
"""
from __future__ import annotations

import types
from collections import defaultdict

import torch


def gate(ok,why):
    if not ok:
        raise RuntimeError("BEGLIN_RESIDENT_N7_DELEGATE_FAIL_CLOSED: "+why)


class ResidentN7SourceDelegate:
    def __init__(self,gpu_registry):
        self.cache=gpu_registry
        self.roles=set()
        self.registered_module_instances=0
        self.original_native_Cpp_reference_checked=0
        self.actual_resident_GPU_role_forward_calls=0
        self.actual_native_cpp_original_reference_calls=0
        self.max_BF16_relative_L2=0.0
        self.max_BF16_abs=0.0
        self.has_discarded_originals=False
        self.phase="UNKNOWN"
        self.calls_by_phase=defaultdict(int)

    def bind(self,module,role,expected_bytes):
        gate(type(role) is str and role.startswith("model.layers.") and
             role.endswith(".weight") and not role.endswith(".kv_b_proj.weight"),
             "only original fixed n7 source role except fused kv_b")
        gate(type(expected_bytes) is int and expected_bytes>0 and
             getattr(module,"serialized_bytes",None)==expected_bytes and
             getattr(module,"n",None)==7 and
             getattr(module,"C_checked",None) is False and
             getattr(module,"calls",None)==0,
             "source compressed role was modified/called before admission")
        gate(role not in getattr(module,"__dict__",{}).get("_beglin_gpu_role",""),
             "source role already attached")
        gate("model.layers." in role and role.count(".weight")==1,
             "source role name not frozen format")
        gate(not hasattr(module,"_beglin_gpu_role"),"refuse rebinding an already patched role")
        original=type(module).forward
        # Do NOT capture the module itself in a closure; thus model layers can
        # be released after each source decoder pass.
        delegate=self
        def actual_forward(bound,x):
            gate(bound._beglin_gpu_role==role and
                 isinstance(x,torch.Tensor) and
                 x.dtype==torch.bfloat16 and x.device.type=="cpu" and
                 x.shape[-1]==bound.in_features,
                 "source original BF16 geometry or role identity")
            y=delegate.cache.call(role,bound,x)
            gate(y.dtype==torch.bfloat16 and
                 y.shape==(*x.shape[:-1],bound.out)
                 and torch.isfinite(y).all().item(),
                 "GPU n7 device GEMV output invalid")
            if not bound.C_checked:
                # Independent C++ backend + CPU C oracle: only once for each
                # actual model native packed instance to bound GPU forwarding.
                reference=original(bound,x)
                gate(torch.isfinite(reference).all().item() and reference.shape==y.shape,
                     "original model source native reference invalid")
                delta=(y.float()-reference.float()).reshape(-1)
                denominator=torch.linalg.vector_norm(reference.float().reshape(-1)).clamp_min(1e-30)
                rel=float(torch.linalg.vector_norm(delta)/denominator)
                absmax=float(delta.abs().max())
                gate(rel<3e-4,
                     "device-resident n7 role numerically diverged from original native C++: "+
                     role+" relL2="+str(rel))
                delegate.max_BF16_relative_L2=max(delegate.max_BF16_relative_L2,rel)
                delegate.max_BF16_abs=max(delegate.max_BF16_abs,absmax)
                delegate.original_native_Cpp_reference_checked+=1
                delegate.actual_native_cpp_original_reference_calls+=1
            else:
                # Original NativeCppPackedLinear.forward increments both of
                # these metrics every time, while its "n_executed" grows
                # only on the first true oracle validation of each instance.
                bound.calls+=1
                bound.tokens+=int(x.numel()/bound.in_features)
            delegate.actual_resident_GPU_role_forward_calls+=1
            delegate.calls_by_phase[delegate.phase]+=1
            return y

        module._beglin_gpu_role=role
        module.forward=types.MethodType(actual_forward,module)
        self.registered_module_instances+=1
        self.roles.add(role)
        return module

    def stats(self):
        return {
          "status":"RESEARCH_ONLY_SOURCE_VERIFIED_N7_GPU_DELEGATE",
          "unique_source_roles_seen":len(self.roles),
          "new_instance_GPU_dispatch_bindings":self.registered_module_instances,
          "source_modules_independently_original_Cpp_C_oracle_checked":
               self.original_native_Cpp_reference_checked,
          "source_original_native_Cpp_reference_calls":
               self.actual_native_cpp_original_reference_calls,
          "actual_device_resident_GPU_GEMV_forward_calls":
               self.actual_resident_GPU_role_forward_calls,
          "max_individual_BF16_relative_L2_vs_old_native_Cpp":self.max_BF16_relative_L2,
          "max_individual_BF16_absolute_diff_vs_old_native_Cpp":self.max_BF16_abs,
          "phase_call_counts":dict(self.calls_by_phase),
          "original_model_role_forced_dense_fallback":False,
          "actual_original_source_checked_per_module_instance":True,
          "source_module_source_op_immutable":True
        }
